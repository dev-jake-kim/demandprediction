#!/usr/bin/env python3
"""Run structural, gradient, and causality checks for the merged model.

    python validate_merged.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandConfig, MergedDemandModel
from models.merged.modules import (
    BranchAttention,
    CausalRetrieval,
    LocalHistoryEncoder,
    PeriodicLSTMEncoder,
)
from train import build_dataset_kwargs, select_node_adaptive_indices

ROOT = Path(__file__).resolve().parent


def _compose(city: str):
    # Select the city-specific root config; dataset overrides do not change its model group.
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name=f'config_{city}')


def _check_dataset(city: str, device: torch.device) -> dict:
    cfg = _compose(city)
    path = resolve_dataset_path(city, cfg.dataset.npy_path)
    kwargs = build_dataset_kwargs(cfg)
    train_set = UnifiedDemandDataset(path, 'train', **kwargs)
    val_set = UnifiedDemandDataset(path, 'val', **kwargs)
    test_set = UnifiedDemandDataset(path, 'test', **kwargs)
    train_weather = train_set.weather[train_set.time_step : train_set.train_end]
    weather_stats = {
        'temperature_min': float(train_weather[:, 0].min()),
        'temperature_max': float(train_weather[:, 0].max()),
        'precipitation_max': float((train_weather[:, 1] + train_weather[:, 2]).max()),
    }
    node_adaptive_indices = (
        select_node_adaptive_indices(train_set, float(cfg.model.node_adaptive_min_demand))
        if cfg.model.node_adaptive else None
    )
    if not (train_set.train_end == val_set.train_end == test_set.train_end):
        raise AssertionError(f'{city}: split bounds differ between dataset views')
    if not (train_set.val_end == val_set.val_end == test_set.val_end):
        raise AssertionError(f'{city}: validation bounds differ between dataset views')

    # 모든 daily·weekly lag의 원천 시점이 0 이상인 샘플로 검사한다.
    probe = int(train_set.weekly_lag_values[0])
    sample = next(iter(DataLoader(Subset(train_set, [probe]), batch_size=1, shuffle=False)))
    sample = {key: value.to(device) for key, value in sample.items()}
    if bool(sample['daily_mask'].any()) or bool(sample['weekly_mask'].any()):
        raise AssertionError(f'{city}: probe sample should have all periodic lags valid')
    model_config = MergedDemandConfig(
        height=train_set.height,
        width=train_set.width,
        time_step=train_set.time_step,
        retrieval_grid_path=str(path),
        retrieval_train_end=train_set.train_end,
        **weather_stats,
        node_adaptive_indices=node_adaptive_indices,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )
    model = MergedDemandModel(model_config).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward_debug(**sample)
    prediction = output['logits']
    weights = output['attention_weights']
    if tuple(prediction.shape) != (1, train_set.height, train_set.width):
        raise AssertionError(f'{city}: unexpected prediction shape {tuple(prediction.shape)}')
    if not torch.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError(f'{city}: prediction is not finite and non-negative')
    if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5):
        raise AssertionError(f'{city}: branch attention weights do not sum to one')

    # Retrieval is intentionally non-differentiable.
    model.train()
    model.zero_grad(set_to_none=True)
    model.forward_debug(**sample)['loss'].backward()

    def _has_gradient(parameter: torch.Tensor) -> bool:
        return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)

    gradient_checks = {
        'neural_head': model.output_gate.neural_head.weight.grad is not None,
        'branch_attention': model.branch_attention.query_projection.weight.grad is not None,
        'local_history': model.local_history.scalar_embedding.projection.weight.grad is not None,
        # Verify calendar embeddings reach the LSTM inputs.
        'weekday_embedding': _has_gradient(model.weekday_embedding.weight),
        'hour_embedding': _has_gradient(model.hour_embedding.weight),
    }
    if not all(gradient_checks.values()):
        raise AssertionError(f'{city}: end-to-end gradient check failed: {gradient_checks}')

    # Weather is a constant input, so verify its effect directly.
    model.eval()
    with torch.no_grad():
        baseline = model(**{k: v for k, v in sample.items() if k != 'labels'})['logits']
        perturbed_sample = {k: v for k, v in sample.items() if k != 'labels'}
        for key in ('weather', 'daily_weather', 'weekly_weather'):
            perturbed_sample[key] = sample[key] + 10.0
        perturbed = model(**perturbed_sample)['logits']
    weather_sensitivity = float((baseline - perturbed).abs().max())
    if weather_sensitivity <= 1e-6:
        raise AssertionError(
            f'{city}: 날씨를 바꿔도 출력이 그대로임 — 날씨 채널이 LSTM 입력에 반영되지 않음'
        )

    target_time = int(sample['sample_idx'][0].item())
    if target_time < train_set.time_step:
        raise AssertionError(f'{city}: sample index is before local history boundary')
    expected_target = torch.from_numpy(
        np.array(train_set.grid[target_time], dtype=np.float32, copy=True)
    ).to(device)
    if not torch.equal(sample['labels'][0], expected_target):
        raise AssertionError(f'{city}: labels is not read from the same temporal grid')
    daily_lag = int(train_set.daily_lag_values[-1])
    expected_daily = train_set.grid[target_time - daily_lag]
    if not np.array_equal(
        sample['daily_demand'][0, -1, :, 0].cpu().numpy(), expected_daily.reshape(-1)
    ):
        raise AssertionError(f'{city}: daily lag does not match its absolute source time')
    return {
        'dataset': city,
        'data_path': str(path),
        'shape': [train_set.total_steps, train_set.height, train_set.width],
        'split': {'train_end': train_set.train_end, 'val_end': train_set.val_end},
        'node_adaptive_nodes': len(node_adaptive_indices) if node_adaptive_indices else 0,
        'samples': {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)},
        'target_time_probe': target_time,
        'attention_shape': list(weights.shape),
        'gradient_checks': gradient_checks,
        'lstm_input_sizes': {
            'history': model.local_history.history_lstm.input_size,
            'daily': model.daily_branch.lstm.input_size,
            'weekly': model.weekly_branch.lstm.input_size,
        },
        'weather_sensitivity': weather_sensitivity,
        'loss_type': model.loss_type,
        'all_pass': True,
    }


def _check_invalid_mask() -> dict:
    encoder = PeriodicLSTMEncoder(hidden_size=8)
    values = torch.ones(2, 4, 3, 1)
    invalid = torch.tensor([[True, True, True, True], [True, False, True, False]])
    hidden, valid = encoder(values, invalid)
    if valid.tolist() != [False, True] or not torch.isfinite(hidden).all():
        raise AssertionError('periodic invalid-mask handling failed')
    attention = BranchAttention(history_hidden=8, periodic_hidden=8, fusion_dim=8)
    fused, weights = attention(torch.ones(2, 3, 8), hidden, hidden, valid, valid)
    if not torch.isfinite(fused).all() or not torch.allclose(
        weights[0, :, 2], torch.ones(3), atol=1e-6
    ):
        raise AssertionError('all-invalid periodic branches did not fall back to neural token')
    return {'all_invalid_periodic_falls_back_to_neural': True}


def _check_periodic_extra_channels() -> dict:
    """Verify weather/calendar channels survive invalid-lag compaction."""

    torch.manual_seed(0)
    extra_dim = 15
    encoder = PeriodicLSTMEncoder(hidden_size=8, extra_dim=extra_dim)
    encoder.eval()

    length, nodes = 4, 2
    values = torch.rand(1, length, nodes, 1)
    extra = torch.randn(1, length, extra_dim)
    invalid = torch.tensor([[False, True, False, False]])  # 두 번째 lag만 무효

    with torch.no_grad():
        hidden, valid = encoder(values, invalid, extra)

        # Reference: compact valid positions before the LSTM.
        keep = [0, 2, 3]
        sequence = torch.log1p(torch.clamp(values, min=0.0))
        sequence = sequence.permute(0, 2, 1, 3).reshape(nodes, length, 1)
        expanded = (
            extra[:, None, :, :].expand(1, nodes, length, extra_dim).reshape(nodes, length, extra_dim)
        )
        reference_input = torch.cat([sequence, expanded], dim=-1)[:, keep, :]
        _, (reference_hidden, _) = encoder.lstm(reference_input)
        reference = reference_hidden[-1].reshape(1, nodes, -1)

    if not bool(valid.item()):
        raise AssertionError('periodic extra-channel check: sequence should be valid')
    if not torch.allclose(hidden, reference, atol=1e-6):
        raise AssertionError(
            'periodic compaction dropped concatenated channels: '
            f'max|diff|={float((hidden - reference).abs().max()):.3e}'
        )
    if encoder.lstm.input_size != 1 + extra_dim:
        raise AssertionError(f'unexpected periodic LSTM input size {encoder.lstm.input_size}')
    return {
        'periodic_extra_channels_survive_compaction': True,
        'periodic_lstm_input_size': encoder.lstm.input_size,
    }


def _check_node_adaptive_identity() -> dict:
    """Verify zero node offsets match ``nn.LSTM`` and untouched nodes remain identical."""

    torch.manual_seed(0)
    height, width, time_step = 4, 3, 6
    num_nodes = height * width
    adaptive = [1, 4, 7]
    encoder = LocalHistoryEncoder(
        height=height,
        width=width,
        time_step=time_step,
        local_radius=1,
        d_model=8,
        num_fourier_bands=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_ffn=16,
        history_hidden=5,
        dropout=0.0,
        node_adaptive_indices=adaptive,
    )
    encoder.eval()
    # Node offsets are initialized at zero.
    if any(float(param.detach().abs().sum()) != 0.0 for param in encoder.node_delta_parameters()):
        raise AssertionError('node_delta 파라미터가 0으로 초기화되지 않음')

    demands = torch.rand(2, time_step, height, width) * 5
    with torch.no_grad():
        adaptive_hidden = encoder(demands)
        encoder.node_adaptive = False  # Run the shared nn.LSTM path with identical weights.
        reference_hidden = encoder(demands)
        encoder.node_adaptive = True

    gap = float((adaptive_hidden - reference_hidden).abs().max())
    if gap > 1e-5:
        raise AssertionError(f'ΔW=0인데 nn.LSTM과 결과가 다름: max|diff|={gap:.3e}')

    # Unselected nodes must remain bit-identical.
    others = [node for node in range(num_nodes) if node not in adaptive]
    untouched = float(
        (adaptive_hidden[:, others] - reference_hidden[:, others]).abs().max()
    )
    if untouched != 0.0:
        raise AssertionError(
            f'ΔW 대상이 아닌 노드의 값이 바뀜: max|diff|={untouched:.3e}'
        )

    n_gates = 4 * 5
    expected = len(adaptive) * (n_gates * (8 + 5) + n_gates)
    actual = sum(param.numel() for param in encoder.node_delta_parameters())
    if actual != expected:
        raise AssertionError(f'node_delta 파라미터 수가 예상과 다름: {actual} != {expected}')

    return {
        'node_adaptive_zero_delta_matches_nn_lstm': True,
        'max_abs_diff': gap,
        'non_adaptive_nodes_bit_identical': untouched == 0.0,
        'adaptive_nodes': len(adaptive),
        'node_delta_params': actual,
    }


def _check_retrieval_boundary() -> dict:
    with tempfile.TemporaryDirectory(prefix='merged_retrieval_') as temp_dir:
        path = Path(temp_dir) / 'grid.npy'
        grid = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
        np.save(path, grid, allow_pickle=False)
        retrieval = CausalRetrieval(
            height=1,
            width=1,
            time_step=2,
            local_radius=0,
            retrieval_grid_path=path,
            retrieval_k=1,
            retrieval_chunk_size=4,
            retrieval_scope='observed_past',
            retrieval_train_end=None,
        )
        query = retrieval._grid[3:5].clone().reshape(1, 2, 1, 1)
        before = retrieval(query, torch.tensor([5]))
        # Target and future values must be outside the candidate interval
        # [time_step, target_time), so changing them cannot affect retrieval.
        retrieval._grid[5:] = 10_000.0
        retrieval._crops[5:] = 10_000.0
        retrieval._cache.fill_(float('nan'))
        after = retrieval(query, torch.tensor([5]))
        if not torch.equal(before, after):
            raise AssertionError('retrieval used target or future values')
    return {'retrieval_candidate_boundary': 'tau < target_time'}


def _check_encoder_retrieval() -> dict:
    """retrieval_encoder_path: encoder latent 거리 검색이 브루트포스와 같고, t 이후 값과 무관하며,
    eval에서 잡음이 꺼지고, 메인 모델 체크포인트 복원 후에도 같은 예측을 내는지 확인한다."""

    from models.retrieval_encoder import RetrievalEncoder, encode_all, save_retrieval_encoder, window_table

    torch.manual_seed(0)
    height = width = 3
    k, top, target = 2, 3, 9
    grid = np.random.default_rng(1).integers(0, 4, size=(14, height, width)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='merged_encoder_') as temp_dir:
        grid_path = Path(temp_dir) / 'grid.npy'
        np.save(grid_path, grid, allow_pickle=False)
        encoder = RetrievalEncoder(torch.randn(9, 4), time_step=k, num_neighbors=9, input_noise_std=0.5)
        encoder_path = Path(temp_dir) / 'encoder.pt'
        save_retrieval_encoder(encoder, encoder_path)

        # eval에서는 잡음을 끈다: 같은 입력을 두 번 넣으면 같은 μ.
        windows = torch.rand(4, k, 9)
        nodes = torch.arange(4)
        encoder.eval()
        if not torch.equal(encoder(windows, nodes)[0], encoder(windows, nodes)[0]):
            raise AssertionError('encoder eval output is noisy')
        encoder.train()
        if torch.equal(encoder(windows, nodes)[0], encoder(windows, nodes)[0]):
            raise AssertionError('encoder train output has no input noise')
        encoder.eval()

        def build(values: np.ndarray) -> CausalRetrieval:
            np.save(grid_path, values, allow_pickle=False)
            return CausalRetrieval(
                height=height, width=width, time_step=k, local_radius=1, retrieval_grid_path=grid_path,
                retrieval_k=top, retrieval_chunk_size=4, retrieval_scope='observed_past',
                retrieval_train_end=None, retrieval_encoder_path=encoder_path,
            )

        history = torch.from_numpy(grid[target - k:target]).unsqueeze(0)
        actual = build(grid)(history, torch.tensor([target]))[0]
        # 브루트포스: μ 표로 같은 노드의 τ ∈ [k, t) 거리 top-k, softmax(s) 가중평균.
        mu = encode_all(encoder, window_table(torch.from_numpy(grid), 1), k)
        flat = torch.from_numpy(grid).reshape(len(grid), -1)
        expected = torch.zeros(height * width)
        for node in range(height * width):
            scores = encoder.similarity(mu[target, node], mu[k:target, node])
            top_scores, top_index = scores.topk(top)
            expected[node] = (torch.softmax(top_scores, 0) * flat[k + top_index, node]).sum()
        if not torch.allclose(actual, expected, atol=1e-5):
            raise AssertionError('encoder retrieval differs from brute-force latent-distance search')
        changed = grid.copy()
        changed[target:] = 100.0
        if not torch.equal(build(changed)(history, torch.tensor([target]))[0], actual):
            raise AssertionError('encoder retrieval used target or future values')

        np.save(grid_path, grid, allow_pickle=False)
        config = MergedDemandConfig(
            height=height, width=width, time_step=k, local_radius=1, d_model=8, transformer_heads=2,
            transformer_layers=1, retrieval_grid_path=str(grid_path), retrieval_k=top,
            retrieval_encoder_path=str(encoder_path),
            temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
        )
        batch = {
            'demand_history': history, 'daily_demand': torch.ones(1, 2, 9, 1),
            'daily_mask': torch.zeros(1, 2, dtype=torch.bool), 'weekly_demand': torch.ones(1, 2, 9, 1),
            'weekly_mask': torch.zeros(1, 2, dtype=torch.bool), 'sample_idx': torch.tensor([target]),
            'weather': torch.zeros(1, k, 3), 'hour_of_day': torch.zeros(1, k, dtype=torch.long),
            'day_of_week': torch.zeros(1, k, dtype=torch.long), 'daily_weather': torch.zeros(1, 2, 3),
            'daily_hour': torch.zeros(1, 2, dtype=torch.long), 'daily_day_of_week': torch.zeros(1, 2, dtype=torch.long),
            'weekly_weather': torch.zeros(1, 2, 3), 'weekly_hour': torch.zeros(1, 2, dtype=torch.long),
            'weekly_day_of_week': torch.zeros(1, 2, dtype=torch.long),
        }
        model = MergedDemandModel(config).eval()
        checkpoint = Path(temp_dir) / 'checkpoint'
        with torch.no_grad():
            before = model.forward_debug(**batch)
            model.save_pretrained(checkpoint)
            after = MergedDemandModel.from_pretrained(checkpoint).eval().forward_debug(**batch)
        if not torch.allclose(before['ir_out'][0], expected, atol=1e-5):
            raise AssertionError('main model ir_out does not use the encoder similarity')
        if not torch.equal(before['logits'], after['logits']):
            raise AssertionError('encoder-retrieval checkpoint round trip changed predictions')
    return {'encoder_retrieval_matches_bruteforce': True, 'encoder_eval_noise_off': True,
            'encoder_retrieval_roundtrip': True}


def _check_independent_retrieval_radius() -> dict:
    """Verify retrieval radius is independent of the Transformer radius."""
    with tempfile.TemporaryDirectory(prefix='merged_window_') as temp_dir:
        path = Path(temp_dir) / 'grid.npy'
        grid = (np.arange(12 * 3 * 3, dtype=np.float32) % 11).reshape(12, 3, 3)
        np.save(path, grid, allow_pickle=False)
        history = torch.from_numpy(grid[5:7].copy()).unsqueeze(0)
        zeros = torch.zeros(1, 2, 9, 1)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        weather = torch.zeros(1, 2, 3)
        time = torch.zeros(1, 2, dtype=torch.long)
        sample_idx = torch.tensor([7])
        for radius in (1, 2):
            config = MergedDemandConfig(
                height=3, width=3, time_step=2, local_radius=1, retrieval_local_radius=radius,
                d_model=8, transformer_heads=2, transformer_layers=1,
                retrieval_grid_path=str(path), retrieval_k=2,
                temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
            )
            model = MergedDemandModel(config).eval()
            with torch.no_grad():
                output = model._compute(
                    demand_history=history, daily_demand=zeros, daily_mask=mask,
                    weekly_demand=zeros, weekly_mask=mask, sample_idx=sample_idx,
                    weather=weather, hour_of_day=time, day_of_week=time,
                    daily_weather=weather, daily_hour=time, daily_day_of_week=time,
                    weekly_weather=weather, weekly_hour=time, weekly_day_of_week=time,
                )
                model.retrieval._cache.fill_(float('nan'))
                expected = model.retrieval(history, sample_idx)
            if model.local_history.num_neighbors != 9:
                raise AssertionError('Transformer window is not 3×3')
            if model.retrieval._crops.shape[-1] != (2 * radius + 1) ** 2:
                raise AssertionError(f'Retrieval candidates do not use radius {radius}')
            if not torch.equal(output['ir_out'], expected):
                raise AssertionError(f'Retrieval query does not use radius {radius}')
    return {'transformer_neighbors': 9, 'retrieval_neighbors_checked': [9, 25]}


def _check_retrieval_pass() -> dict:
    """검색을 끈 모델은 없는 raw grid로도 예측·체크포인트 복원이 가능해야 한다."""
    with tempfile.TemporaryDirectory(prefix='merged_no_retrieval_') as temp_dir:
        grid_path = Path(temp_dir) / 'absent.npy'
        config = MergedDemandConfig(
            height=3, width=3, time_step=2, local_radius=1, use_retrieval=False,
            d_model=8, transformer_heads=2, transformer_layers=1,
            retrieval_grid_path=str(grid_path),
            temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
        )
        model = MergedDemandModel(config).eval()
        if model.retrieval is not None:
            raise AssertionError('disabled retrieval instantiated a grid search')
        batch = {
            'demand_history': torch.ones(1, 2, 3, 3),
            'daily_demand': torch.ones(1, 2, 9, 1),
            'daily_mask': torch.zeros(1, 2, dtype=torch.bool),
            'weekly_demand': torch.ones(1, 2, 9, 1),
            'weekly_mask': torch.zeros(1, 2, dtype=torch.bool),
            'sample_idx': torch.tensor([7]),
            'weather': torch.zeros(1, 2, 3),
            'hour_of_day': torch.zeros(1, 2, dtype=torch.long),
            'day_of_week': torch.zeros(1, 2, dtype=torch.long),
            'daily_weather': torch.zeros(1, 2, 3),
            'daily_hour': torch.zeros(1, 2, dtype=torch.long),
            'daily_day_of_week': torch.zeros(1, 2, dtype=torch.long),
            'weekly_weather': torch.zeros(1, 2, 3),
            'weekly_hour': torch.zeros(1, 2, dtype=torch.long),
            'weekly_day_of_week': torch.zeros(1, 2, dtype=torch.long),
        }
        with torch.no_grad():
            expected = model.forward_debug(**batch)
        checkpoint = Path(temp_dir) / 'checkpoint'
        model.save_pretrained(checkpoint)
        restored = MergedDemandModel.from_pretrained(checkpoint).eval()
        with torch.no_grad():
            actual = restored.forward_debug(**batch)
        if restored.retrieval is not None or not torch.equal(expected['logits'], actual['logits']):
            raise AssertionError('search-free checkpoint did not reload without its raw grid')
        if not torch.equal(actual['logits'].flatten(1), actual['neural_pred']):
            raise AssertionError('disabled retrieval did not pass through neural prediction')
        if actual['ir_out'] is not None or not torch.all(actual['lambda_weight'] == 1):
            raise AssertionError('disabled retrieval passed through the output gate')
    return {'retrieval_pass_missing_grid_and_reload': True}


def _check_masked_grid_receptive_field() -> dict:
    """L층 masked Transformer에서 노드는 반경 a·L 밖 수요의 영향을 받지 않는다."""

    torch.manual_seed(0)
    height = width = 7
    center = 3 * width + 3
    for layers in (1, 2):
        encoder = LocalHistoryEncoder(
            height=height, width=width, time_step=2, local_radius=1, d_model=8,
            num_fourier_bands=2, transformer_layers=layers, transformer_heads=2,
            transformer_ffn=16, history_hidden=5, dropout=0.0,
        ).eval()
        with torch.no_grad():
            encoder.direction_bias.normal_()
            demands = torch.rand(1, 2, height, width) * 3
            base = encoder(demands)[0, center]
            responses = {}
            for distance in range(1, 4):
                moved = demands.clone()
                moved[:, :, 3 - distance, 3 - distance] += 5.0
                responses[distance] = bool((encoder(moved)[0, center] - base).abs().max() > 1e-6)
        expected = {distance: distance <= layers for distance in responses}
        if responses != expected:
            raise AssertionError(f'{layers}층 수용 영역이 다름: {responses} != {expected}')
    return {'masked_grid_receptive_radius': 'local_radius * transformer_layers'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    results = [_check_dataset(city, device) for city in ('ulsan', 'porto')]
    results.append(_check_invalid_mask())
    results.append(_check_periodic_extra_channels())
    results.append(_check_node_adaptive_identity())
    results.append(_check_retrieval_boundary())
    results.append(_check_encoder_retrieval())
    results.append(_check_independent_retrieval_radius())
    results.append(_check_retrieval_pass())
    results.append(_check_masked_grid_receptive_field())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
