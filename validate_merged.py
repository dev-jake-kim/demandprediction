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
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandConfig, MergedDemandModel
from models.merged.modules import (
    BranchAttention,
    LocalHistoryEncoder,
    PeriodicLSTMEncoder,
)
from train import build_dataset_kwargs

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
        **weather_stats,
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

    model.train()
    model.zero_grad(set_to_none=True)
    model.forward_debug(**sample)['loss'].backward()

    def _has_gradient(parameter: torch.Tensor) -> bool:
        return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)

    gradient_checks = {
        'neural_head': model.output_head.neural_head.weight.grad is not None,
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


def _check_checkpoint_reload() -> dict:
    """``save_pretrained`` → ``from_pretrained``로 복원한 모델이 같은 예측을 내야 한다."""

    with tempfile.TemporaryDirectory(prefix='merged_reload_') as temp_dir:
        config = MergedDemandConfig(
            height=3, width=3, time_step=2, local_radius=1,
            d_model=8, transformer_heads=2, transformer_layers=1,
            temperature_min=-5.0, temperature_max=30.0, precipitation_max=20.0,
        )
        model = MergedDemandModel(config).eval()
        batch = {
            'demand_history': torch.ones(1, 2, 3, 3),
            'daily_demand': torch.ones(1, 2, 9, 1),
            'daily_mask': torch.zeros(1, 2, dtype=torch.bool),
            'weekly_demand': torch.ones(1, 2, 9, 1),
            'weekly_mask': torch.zeros(1, 2, dtype=torch.bool),
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
        if not torch.equal(expected['logits'], actual['logits']):
            raise AssertionError('checkpoint reload changed predictions')
    return {'checkpoint_reload_identical': True}


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


def _check_event_mask() -> dict:
    """이벤트 표: [h,h+1)∩[start,end) 겹침 규칙, 샘플 창 [t−k+1, t+1), 0 임베딩 불변, 이벤트 노드에만 영향."""

    cfg = _compose('ulsan')
    path = resolve_dataset_path('ulsan', cfg.dataset.npy_path)
    kwargs = build_dataset_kwargs(cfg)
    origin = pd.Timestamp(kwargs['start_time'])
    node = 40
    with tempfile.TemporaryDirectory(prefix='merged_event_') as temp_dir:
        csv = Path(temp_dir) / 'events.csv'
        start, end = origin + pd.Timedelta(hours=100, minutes=30), origin + pd.Timedelta(hours=102)
        pd.DataFrame({'node': [node], 'start': [f'{start:%Y-%m-%d %H:%M}'], 'end': [f'{end:%Y-%m-%d %H:%M}']}).to_csv(csv, index=False)
        dataset = UnifiedDemandDataset(path, 'train', **{**kwargs, 'event_csv_path': csv})
    active = np.flatnonzero(dataset.event_table[:, node])
    if active.tolist() != [100, 101] or dataset.event_table.sum() != 2:
        raise AssertionError(f'이벤트 시각 표가 [100, 101]이 아님: {active.tolist()}')
    target = 101
    sample = dataset[target - int(dataset.indices[0])]
    window = sample['event_mask'].numpy()
    if not (int(dataset.indices[target - int(dataset.indices[0])]) == target and window[-1, node] == 1 and window[-2, node] == 1 and window.sum() == 2):
        raise AssertionError('event_mask 창이 [t−k+1, t+1)이 아님')

    torch.manual_seed(0)
    common = dict(height=4, width=3, time_step=6, local_radius=1, d_model=8, num_fourier_bands=2,
                  transformer_layers=1, transformer_heads=2, transformer_ffn=16, history_hidden=5, dropout=0.0)
    encoder = LocalHistoryEncoder(**common, use_event_embedding=True).eval()
    demands = torch.rand(2, 6, 4, 3) * 5
    mask = torch.zeros(2, 6, 12)
    mask[:, -1, 7] = 1.0
    with torch.no_grad():
        reference = encoder(demands, event_mask=torch.zeros_like(mask))
        zero_gap = float((encoder(demands, event_mask=mask) - reference).abs().max())
        encoder.event_embedding.fill_(1.0)
        changed = (encoder(demands, event_mask=mask) - reference).abs().amax(dim=(0, 2))
    if zero_gap != 0.0:
        raise AssertionError(f'0 이벤트 임베딩인데 출력이 바뀜: {zero_gap:.3e}')
    # 1층 3×3 창: 노드 7(행 2, 열 1)의 창 밖 노드는 바뀌지 않아야 한다.
    outside = [n for n in range(12) if abs(n // 3 - 2) > 1 or abs(n % 3 - 1) > 1]
    if float(changed[7]) == 0.0 or float(changed[outside].max()) != 0.0:
        raise AssertionError('이벤트 임베딩 영향이 이벤트 노드 창에 국한되지 않음')
    return {'event_mask_window': '[t-k+1, t+1)', 'event_zero_embedding_identity': True}



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
    results.append(_check_checkpoint_reload())
    results.append(_check_masked_grid_receptive_field())
    results.append(_check_event_mask())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
