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
    GATE_INIT,
    LinearTrendForecaster,
    LocalHistoryEncoder,
    NodeHourGate,
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

    # weekly lag까지 모두 유효한 샘플을 쓴다.
    probe = min(len(train_set) - 1, int(train_set.weekly_lag_values[0]))
    sample = next(iter(DataLoader(Subset(train_set, [probe]), batch_size=1)))
    sample = {key: value.to(device) for key, value in sample.items()}
    model_config = MergedDemandConfig(
        height=train_set.height,
        width=train_set.width,
        time_step=train_set.time_step,
        node_adaptive_indices=node_adaptive_indices,
        **weather_stats,
        **OmegaConf.to_container(cfg.model, resolve=True),
    )
    model = MergedDemandModel(model_config).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward_debug(**sample)
    prediction = output['logits']
    weights = output['gate_weights']
    if tuple(prediction.shape) != (1, train_set.height, train_set.width):
        raise AssertionError(f'{city}: unexpected prediction shape {tuple(prediction.shape)}')
    if not torch.isfinite(prediction).all():
        raise AssertionError(f'{city}: prediction is not finite')
    if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5):
        raise AssertionError(f'{city}: gate weights do not sum to one')
    if not (bool(output['daily_valid'].all()) and bool(output['weekly_valid'].all())):
        raise AssertionError(f'{city}: probe sample should have all periodic lags valid')

    model.train()
    model.zero_grad(set_to_none=True)
    model.forward_debug(**sample)['loss'].backward()

    def _has_gradient(parameter: torch.Tensor) -> bool:
        return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)

    gradient_checks = {
        'predict_layer': _has_gradient(model.predict_layer.weight),
        'gate_logit': _has_gradient(model.gate.gate_logit),
        'local_history': _has_gradient(model.local_history.scalar_embedding.projection.weight),
        'weekday_embedding': _has_gradient(model.weekday_embedding.weight),
        'hour_embedding': _has_gradient(model.hour_embedding.weight),
        # 적설이 0인 샘플에서는 값이 0일 수 있어 연결 여부만 본다.
        'snow_scale_connected': model.snow_scale.grad is not None,
    }
    if not all(gradient_checks.values()):
        raise AssertionError(f'{city}: end-to-end gradient check failed: {gradient_checks}')

    # Weather is a constant input, so verify its effect directly.
    model.eval()
    with torch.no_grad():
        baseline = model(**{k: v for k, v in sample.items() if k != 'labels'})['logits']
        perturbed_sample = {k: v for k, v in sample.items() if k != 'labels'}
        perturbed_sample['weather'] = sample['weather'] + 10.0
        perturbed = model(**perturbed_sample)['logits']
    weather_sensitivity = float((baseline - perturbed).abs().max())
    if weather_sensitivity <= 1e-6:
        raise AssertionError(
            f'{city}: 날씨를 바꿔도 출력이 그대로임 — 날씨 채널이 LSTM 입력에 반영되지 않음'
        )

    target_time = int(train_set.indices[probe])
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
        'gate_shape': list(weights.shape),
        'gradient_checks': gradient_checks,
        'history_lstm_input_size': model.local_history.history_lstm.input_size,
        'weather_sensitivity': weather_sensitivity,
        'loss_type': model.loss_type,
        'all_pass': True,
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


def _check_trend_forecaster() -> dict:
    """직선 외삽(numpy.polyfit), 0 clamp, 무효 lag 평균, 전부 무효 처리를 확인한다."""

    forecaster = LinearTrendForecaster()
    rows = [[3, 5, 1, 2, 4, 3], [5, 4, 3, 2, 1, 0], [2, 9, 4, 7, 1, 6], [1, 2, 3, 4, 5, 6]]
    values = torch.tensor(rows, dtype=torch.float64).T[None, :, :, None]  # [1,6,4,1]
    pred, valid = forecaster(values, torch.zeros(1, 6, dtype=torch.bool))
    position = np.arange(1, 7)
    expected = [max(np.polyval(np.polyfit(position, row, 1), 7), 0.0) for row in rows]
    if not np.allclose(pred[0].numpy(), expected, atol=1e-10) or not bool(valid.all()):
        raise AssertionError(f'trend extrapolation mismatch: {pred[0].tolist()} != {expected}')

    invalid = torch.tensor([[True, True, False, False, False, False],
                            [True, True, True, True, True, True]])
    pred, valid = forecaster(values.expand(2, -1, -1, -1), invalid)
    expected_mean = [float(np.mean(row[2:])) for row in rows]
    if not np.allclose(pred[0].numpy(), expected_mean, atol=1e-10):
        raise AssertionError('partially invalid lags must use the mean of valid lags')
    if valid.tolist() != [True, False] or bool(pred[1].abs().sum()):
        raise AssertionError('all-invalid lags must be invalid with zero prediction')
    return {'trend_extrapolation_matches_polyfit': True, 'example_[3,5,1,2,4,3]': expected[0]}


def _check_gate() -> dict:
    """초기 가중치, 무효 브랜치 제외, 합 1을 확인한다."""

    gate = NodeHourGate(num_nodes=3)
    ones = torch.ones(2, 3)
    hour = torch.tensor([0, 23])
    _, weights = gate(ones, 2 * ones, 3 * ones, hour, torch.tensor([True, True]),
                      torch.tensor([True, False]))
    if not torch.allclose(weights[0], torch.tensor(GATE_INIT).expand(3, 3), atol=1e-6):
        raise AssertionError(f'gate init is not {GATE_INIT}: {weights[0]}')
    expected = torch.tensor([GATE_INIT[0], GATE_INIT[1], 0.0]) / (GATE_INIT[0] + GATE_INIT[1])
    if not torch.allclose(weights[1], expected.expand(3, 3), atol=1e-6):
        raise AssertionError(f'invalid weekly branch is not excluded: {weights[1]}')
    return {'gate_init': list(GATE_INIT), 'invalid_branch_excluded': True}


def _check_roundtrip() -> dict:
    """save_pretrained -> from_pretrained 후 예측이 같아야 한다."""

    torch.manual_seed(0)
    config = MergedDemandConfig(
        height=3, width=3, time_step=2, local_radius=1, d_model=8, transformer_heads=2,
        transformer_layers=1, temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
        node_adaptive=True, node_adaptive_indices=[0, 4],
    )
    model = MergedDemandModel(config).eval()
    batch = {
        'demand_history': torch.rand(2, 2, 3, 3) * 3,
        'daily_demand': torch.rand(2, 6, 9, 1) * 3,
        'daily_mask': torch.zeros(2, 6, dtype=torch.bool),
        'weekly_demand': torch.rand(2, 4, 9, 1) * 3,
        'weekly_mask': torch.tensor([[False] * 4, [True] * 4]),
        'weather': torch.tensor([[[10.0, 1.0, 0.5], [12.0, 0.0, 0.0]]] * 2),
        'hour_of_day': torch.tensor([[7, 8], [22, 23]]),
        'day_of_week': torch.tensor([[1, 1], [5, 5]]),
    }
    with tempfile.TemporaryDirectory(prefix='merged_roundtrip_') as temp_dir:
        with torch.no_grad():
            expected = model(**batch)['logits']
        model.save_pretrained(temp_dir)
        restored = MergedDemandModel.from_pretrained(temp_dir).eval()
        with torch.no_grad():
            actual = restored(**batch)['logits']
    if not torch.equal(expected, actual):
        raise AssertionError('checkpoint round trip changed predictions')
    return {'checkpoint_roundtrip_identical': True}


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
    results.append(_check_trend_forecaster())
    results.append(_check_gate())
    results.append(_check_node_adaptive_identity())
    results.append(_check_masked_grid_receptive_field())
    results.append(_check_roundtrip())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
