#!/usr/bin/env python3
"""Fast structural and causal checks for the merged model.

원본 ``merged_model/validate.py``를 새 구조(``models.merged`` + ``dataset_frame`` + Hydra)에
맞춰 갱신한 것이다. 실제 배치를 두 도시에서 한 번씩 흘리고 합성 검색 케이스를 하나 돌려서,
shape / mask / gradient / 시간 경계 회귀를 2,000-epoch 학습 전에 잡는다.

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
from torch.utils.data import DataLoader

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
    # 도시마다 최적 하이퍼파라미터가 달라 루트 config가 도시별로 나뉘어 있다
    # (configs/config_ulsan.yaml / configs/config_porto.yaml) — dataset= 오버라이드만으로는
    # model: 그룹(따라서 하이퍼파라미터)이 안 바뀌므로 config_name 자체를 골라야 한다.
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
    weather_mean = train_weather.mean(axis=0)
    weather_std = train_weather.std(axis=0).clip(min=1e-6)
    node_adaptive_indices = (
        select_node_adaptive_indices(train_set, float(cfg.model.node_adaptive_min_demand))
        if cfg.model.node_adaptive else None
    )
    if not (train_set.train_end == val_set.train_end == test_set.train_end):
        raise AssertionError(f'{city}: split bounds differ between dataset views')
    if not (train_set.val_end == val_set.val_end == test_set.val_end):
        raise AssertionError(f'{city}: validation bounds differ between dataset views')

    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False)))
    sample = {key: value.to(device) for key, value in sample.items()}
    model_config = MergedDemandConfig(
        height=train_set.height,
        width=train_set.width,
        time_step=train_set.time_step,
        retrieval_grid_path=str(path),
        retrieval_train_end=train_set.train_end,
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
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

    # 두 번째 pass는 손실 그래프가 neural head와 브랜치 투영에 모두 닿는지 본다.
    # 검색기는 의도적으로 미분 불가능하다.
    model.train()
    model.zero_grad(set_to_none=True)
    model.forward_debug(**sample)['loss'].backward()

    def _has_gradient(parameter: torch.Tensor) -> bool:
        return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)

    gradient_checks = {
        'neural_head': model.output_gate.neural_head.weight.grad is not None,
        'branch_attention': model.branch_attention.query_projection.weight.grad is not None,
        'local_history': model.local_history.scalar_embedding.projection.weight.grad is not None,
        # 캘린더 임베딩이 세 LSTM 입력에 실제로 연결돼 있는지 — concat을 빠뜨리면 여기서 잡힌다.
        'weekday_embedding': _has_gradient(model.weekday_embedding.weight),
        'hour_embedding': _has_gradient(model.hour_embedding.weight),
    }
    if not all(gradient_checks.values()):
        raise AssertionError(f'{city}: end-to-end gradient check failed: {gradient_checks}')

    # 날씨는 임베딩 층이 없어 gradient로 연결을 확인할 수 없다(정규화 후 그대로 concat되는 상수 입력).
    # 대신 날씨만 흔들어 출력이 실제로 달라지는지 본다 — concat을 빠뜨리면 출력이 그대로다.
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
    """concat한 날씨/캘린더 채널이 compaction(gather+pack)을 그대로 통과하는지 확인한다.

    ``PeriodicLSTMEncoder``는 유효 lag만 앞으로 당겨 packing하는데, gather의 feature 폭을
    하드코딩하면 예외 없이 첫 채널만 남고 나머지가 조용히 사라진다. 여기서는 같은 입력을
    직접 compaction해 LSTM에 넣은 결과와 비교해서 전 채널 보존을 증명한다.
    """

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

        # 참조 구현: 유효 위치(0,2,3)만 순서대로 모아 직접 LSTM에 통과시킨다.
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
    """ΔW=0인 노드별 LSTM 경로가 nn.LSTM(cuDNN fused)과 같은 값을 내는지 확인한다.

    이게 깨지면 "baseline에 노드별 offset만 추가"라는 node_adaptive 실험의 전제가 무너진다 —
    stage 1(ΔW 고정)의 결과가 node_adaptive를 끈 학습과 달라져서, 측정된 차이가 ΔW 때문인지
    수동 셀 루프 때문인지 분리되지 않는다. 게이트 순서(i,f,g,o)나 bias_ih+bias_hh 합산을
    틀리면 여기서 잡힌다.

    선택되지 않은 노드는 nn.LSTM 결과를 그대로 써야 하므로 그쪽도 함께 확인한다.
    """

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
    # delta는 0으로 시작해야 한다 — LocalHistoryEncoder 생성자의 계약.
    if any(float(param.detach().abs().sum()) != 0.0 for param in encoder.node_delta_parameters()):
        raise AssertionError('node_delta 파라미터가 0으로 초기화되지 않음')

    demands = torch.rand(2, time_step, height, width) * 5
    with torch.no_grad():
        _, adaptive_hidden = encoder(demands)
        encoder.node_adaptive = False  # 같은 가중치로 nn.LSTM 경로만 태운다
        _, reference_hidden = encoder(demands)
        encoder.node_adaptive = True

    gap = float((adaptive_hidden - reference_hidden).abs().max())
    if gap > 1e-5:
        raise AssertionError(f'ΔW=0인데 nn.LSTM과 결과가 다름: max|diff|={gap:.3e}')

    # 선택되지 않은 노드는 정확히 동일해야 한다(수동 루프를 아예 타지 않으므로).
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
        query = retrieval._crops[3:5].clone().reshape(1, 2, 1, 1)
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


def _check_retrieval_pass() -> dict:
    """검색을 끈 모델은 없는 raw grid로도 예측·체크포인트 복원이 가능해야 한다."""
    with tempfile.TemporaryDirectory(prefix='merged_no_retrieval_') as temp_dir:
        grid_path = Path(temp_dir) / 'absent.npy'
        config = MergedDemandConfig(
            height=3, width=3, time_step=2, local_radius=1, use_retrieval=False,
            d_model=8, transformer_heads=2, transformer_layers=1,
            retrieval_grid_path=str(grid_path),
            weather_mean=[0.0] * 3, weather_std=[1.0] * 3,
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
    results.append(_check_retrieval_pass())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
