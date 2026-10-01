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
    RetrievalFusion,
    demand_bucket_bounds,
)
from train import build_dataset_kwargs, select_node_adaptive_indices, select_retrieval_value_cap

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
        retrieval_value_cap=select_retrieval_value_cap(train_set),
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


def _reference_retrieval(grid: np.ndarray, k: int, radius: int, top: int, allowed) -> dict:
    """브루트포스 기준: (t, 노드)마다 허용 후보 τ를 직접 훑어 유사도 내림차순 top-k ``[(s, τ)]``."""

    total, height, width = grid.shape
    padded = np.pad(grid, ((0, 0), (radius, radius), (radius, radius)))
    size = 2 * radius + 1
    crops = np.stack(
        [padded[:, y:y + size, x:x + size].reshape(total, -1) for y in range(height) for x in range(width)],
        axis=1,
    ).astype(np.float64)  # [T, N, P]
    out = {}
    for t in range(k, total):
        for node in range(crops.shape[1]):
            query = crops[t - k:t, node].T.reshape(-1)
            query = query / max(np.linalg.norm(query), 1e-12)
            scored = []
            for tau in range(k, total):
                if allowed(t, tau):
                    key = crops[tau - k:tau, node].T.reshape(-1)
                    scored.append((float(query @ (key / max(np.linalg.norm(key), 1e-12))), tau))
            out[t, node] = sorted(scored, reverse=True)[:top]
    return out


def _table_gap(table: tuple[torch.Tensor, torch.Tensor], reference: dict) -> float:
    """표의 (유사도, τ)가 기준과 같은지 비교하고 최대 유사도 차이를 반환한다."""

    scores, times = table
    gap = 0.0
    for (t, node), expected in reference.items():
        row_scores, row_times = scores[t, node], times[t, node]
        valid = torch.isfinite(row_scores)
        if int(valid.sum()) != len(expected):
            raise AssertionError(f'retrieval candidate count differs at t={t}, node={node}')
        if row_times[valid].tolist() != [tau for _, tau in expected]:
            raise AssertionError(f'retrieval candidates differ at t={t}, node={node}')
        for actual, (score, _) in zip(row_scores[valid].tolist(), expected):
            gap = max(gap, abs(actual - score))
    return gap


def _check_retrieval_boundary() -> dict:
    """검색 결과가 브루트포스 기준과 같고, 누수 경계 밖 값을 바꿔도 결과가 그대로인지 확인한다."""

    k, mask_hours, train_end, top = 2, 5, 30, 3
    rng = np.random.default_rng(0)
    grid = rng.random((40, 3, 3)).astype(np.float32)

    def build(values: np.ndarray, temp_dir: str) -> CausalRetrieval:
        path = Path(temp_dir) / 'grid.npy'
        np.save(path, values, allow_pickle=False)
        return CausalRetrieval(
            height=3, width=3, time_step=k, local_radius=1, retrieval_grid_path=path,
            retrieval_k=top, retrieval_chunk_size=4, retrieval_train_end=train_end,
            future_mask_hours=mask_hours,
        )

    cpu = torch.device('cpu')
    with tempfile.TemporaryDirectory(prefix='merged_retrieval_') as temp_dir:
        retrieval = build(grid, temp_dir)
        train_table = retrieval.table(True, cpu)
        eval_table = retrieval.table(False, cpu)
        try:
            CausalRetrieval(
                height=3, width=3, time_step=k, local_radius=1, retrieval_grid_path=None,
                retrieval_k=top, retrieval_chunk_size=4, retrieval_train_end=train_end,
                future_mask_hours=k - 1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError('future mask shorter than time_step was accepted')

        def train_allowed(t: int, tau: int) -> bool:
            return tau < train_end and not (t <= tau <= t + mask_hours)

        gap = max(
            _table_gap(train_table, _reference_retrieval(grid, k, 1, top, train_allowed)),
            _table_gap(eval_table, _reference_retrieval(grid, k, 1, top, lambda t, tau: tau < t)),
        )
        if gap > 1e-5:
            raise AssertionError(f'retrieval differs from brute-force reference: {gap}')

        def query(values: np.ndarray, training: bool, times: list[int]) -> tuple[torch.Tensor, ...]:
            module = build(values, temp_dir).train(training)
            return module(torch.tensor(times))

        def same(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> bool:
            return all(torch.equal(x, y) for x, y in zip(a, b))

        # 정답 y_t를 바꿔도 train 모드 결과[t]는 같아야 한다(τ=t의 V, τ∈(t,t+k]의 K가 가려짐).
        target = 12
        changed = grid.copy()
        changed[target] = 1_000.0
        if not same(query(changed, True, [target]), query(grid, True, [target])):
            raise AssertionError('train-mode retrieval at t depends on y_t')
        # train 구간 밖 값을 바꿔도 train 구간 t의 train 모드 결과(유사도·V)는 같아야 한다.
        changed = grid.copy()
        changed[train_end:] = 1_000.0
        times = list(range(k, train_end + 1))
        if not same(query(changed, True, times), query(grid, True, times)):
            raise AssertionError('train-mode retrieval used values outside the train split')
        # eval 모드는 t 이후를 바꿔도 결과[t]가 같아야 한다.
        changed = grid.copy()
        changed[target:] = 1_000.0
        times = list(range(k, target + 1))
        if not same(query(changed, False, times), query(grid, False, times)):
            raise AssertionError('eval-mode retrieval used target or future values')
    return {
        'retrieval_matches_bruteforce': gap,
        'train_candidates': 'tau < train_end, tau not in [t, t+mask]',
        'eval_candidates': 'tau < t',
    }


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
        inputs = dict(
            demand_history=history, daily_demand=zeros, daily_mask=mask,
            weekly_demand=zeros, weekly_mask=mask, sample_idx=sample_idx,
            weather=weather, hour_of_day=time, day_of_week=time,
            daily_weather=weather, daily_hour=time, daily_day_of_week=time,
            weekly_weather=weather, weekly_hour=time, weekly_day_of_week=time,
        )
        for radius in (1, 2):
            config = MergedDemandConfig(
                height=3, width=3, time_step=2, local_radius=1, retrieval_local_radius=radius,
                d_model=8, transformer_heads=2, transformer_layers=1,
                retrieval_grid_path=str(path), retrieval_k=2, retrieval_train_end=10, retrieval_value_cap=7,
                temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
            )
            model = MergedDemandModel(config).eval()
            with torch.no_grad():
                output = model._compute(**inputs)
            scores, values, query_mean = model.retrieval(sample_idx)
            if model.local_history.num_neighbors != 9:
                raise AssertionError('Transformer window is not 3×3')
            if tuple(output['retrieval_values'].shape) != (1, 9, 2, (2 * radius + 1) ** 2):
                raise AssertionError(f'Retrieval values do not use radius {radius}')
            if not (torch.equal(output['retrieval_scores'], scores) and torch.equal(output['retrieval_values'], values)):
                raise AssertionError('eval forward did not use the eval-mode table')
            # m은 입력 demand_history의 [t-k, t) × 반경 r 창(격자 밖 0) 평균과 같아야 한다.
            window = 2 * radius + 1
            expected_mean = torch.nn.functional.unfold(
                history.reshape(-1, 1, 3, 3), window, padding=radius
            ).mean(dim=1).mean(dim=0)  # [N]
            if not torch.allclose(output['retrieval_query_mean'][0], expected_mean, atol=1e-6):
                raise AssertionError('retrieval query mean does not match the demand_history window')
            model.train()
            model._compute(**inputs)['logits'].sum().backward()
            fusion = model.retrieval_fusion
            for name, param in (
                ('embedding', fusion.value_embedding.weight),
                ('score_scale', fusion.score_scale),
                ('fallback_embedding', fusion.fallback_embedding),
            ):
                if param.grad is None or not bool(param.grad.abs().sum() > 0):
                    raise AssertionError(f'retrieval fusion {name} receives no gradient')
            # 검색 켬 체크포인트 복원(bucket 경계 포함)이 같은 예측을 내는지 확인한다.
            checkpoint = Path(temp_dir) / f'checkpoint_r{radius}'
            model.save_pretrained(checkpoint)
            restored = MergedDemandModel.from_pretrained(checkpoint).eval()
            model.eval()
            with torch.no_grad():
                if not torch.equal(model._compute(**inputs)['logits'], restored._compute(**inputs)['logits']):
                    raise AssertionError('retrieval checkpoint round trip changed predictions')
    return {
        'transformer_neighbors': 9, 'retrieval_neighbors_checked': [9, 25], 'fusion_gradient': True,
        'retrieval_checkpoint_roundtrip': True,
    }


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
        if actual['retrieval_values'] is not None or not torch.equal(actual['h_local'], actual['h_neural']):
            raise AssertionError('disabled retrieval changed the local representation')
    return {'retrieval_pass_missing_grid_and_reload': True}


def _check_retrieval_fusion() -> dict:
    """수요 bucket 경계, bucket 배정, 후보 없는 칸이 융합 결과에 영향을 주지 않는지 확인한다."""

    if demand_bucket_bounds(7) != [0, 1, 2, 3, 4, 5, 7]:
        raise AssertionError(f'bucket bounds for cap 7: {demand_bucket_bounds(7)}')
    if demand_bucket_bounds(30) != [0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 25, 30]:
        raise AssertionError(f'bucket bounds for cap 30: {demand_bucket_bounds(30)}')
    torch.manual_seed(0)
    fusion = RetrievalFusion(num_neighbors=2, history_hidden=4, embedding_dim=3, value_cap=7)
    buckets = fusion.bucketize(torch.tensor([0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 100.0])).tolist()
    if buckets != [0, 1, 4, 5, 5, 6, 6]:
        raise AssertionError(f'bucket assignment wrong: {buckets}')
    h_neural = torch.randn(1, 1, 4)
    scores = torch.tensor([[[0.9, float('-inf')]]])
    values = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    other = values.clone()
    other[..., 1, :] = 100.0
    busy = torch.tensor([[2.0]])
    if not torch.equal(fusion(h_neural, scores, values, busy), fusion(h_neural, scores, other, busy)):
        raise AssertionError('candidate slot without a valid neighbor changed the fusion output')
    # m=0(질의 전부 0)이면 검색 결과와 무관하게 O만 쓴다.
    with torch.no_grad():
        fusion.fallback_embedding.normal_()
    idle = torch.zeros(1, 1)
    expected = fusion.fuse(torch.cat([h_neural, fusion.fallback_embedding.expand(1, 1, -1)], dim=-1))
    other[..., 0, :] = 5.0
    if not (
        torch.allclose(fusion(h_neural, scores, values, idle), expected)
        and torch.equal(fusion(h_neural, scores, values, idle), fusion(h_neural, scores, other, idle))
    ):
        raise AssertionError('zero query did not fall back to O')
    return {
        'retrieval_buckets_cap7': demand_bucket_bounds(7), 'invalid_slot_ignored': True,
        'zero_query_uses_O': True,
    }


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
    results.append(_check_independent_retrieval_radius())
    results.append(_check_retrieval_fusion())
    results.append(_check_retrieval_pass())
    results.append(_check_masked_grid_receptive_field())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
