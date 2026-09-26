"""Five-hour neighborhoods around each causal daily/weekly lag."""

from tempfile import TemporaryDirectory

import torch

from dataset_frame.unified_demand_dataset import _chronological_lags
from models.merged import MergedDemandConfig
from models.merged.modeling import MergedDemandModel, PeriodicViewEncoder


WEIGHTS = [0.05, 0.1, 0.7, 0.1, 0.05]


def test_periodic_window_centers_on_exact_lag_and_excludes_partial_groups():
    assert _chronological_lags(24, 2, 2).tolist() == [50, 49, 48, 47, 46, 26, 25, 24, 23, 22]
    assert _chronological_lags(168, 1, 2).tolist() == [170, 169, 168, 167, 166]
    encoder = PeriodicViewEncoder(MergedDemandConfig(
        periodic_mode='ma', height=1, width=1, periodic_window_weights=WEIGHTS,
    ))
    assert encoder.window_projection.bias is None
    torch.testing.assert_close(encoder.window_projection.weight[0], torch.tensor(WEIGHTS))
    values = torch.tensor([
        [2., 4., 10., 4., 2., 5., 5., 5., 5., 5.],
        [100., 100., 100., 100., 100., 1., 2., 3., 4., 5.],
        [100., 100., 100., 100., 100., 100., 100., 100., 100., 100.],
    ]).reshape(3, 10, 1, 1)
    mask = torch.tensor([
        [False] * 10,
        [True, False, False, False, False] + [False] * 5,
        [True] * 10,
    ])
    output, valid = encoder(values, mask)
    torch.testing.assert_close(output[:, 0], torch.tensor([6.5, 3.0, 0.0]))
    assert valid.tolist() == [True, True, False]
    output.sum().backward()
    assert encoder.window_projection.weight.grad is not None
    assert encoder.window_projection.weight.grad.abs().sum() > 0


def test_daily_weekly_window_weights_restore_and_predict_independently():
    config = MergedDemandConfig(
        periodic_mode='ma', height=1, width=1, periodic_window_weights=WEIGHTS,
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )
    model = MergedDemandModel(config)
    with torch.no_grad():
        model.daily_view.window_projection.weight.copy_(torch.tensor([[0., 0., 1., 0., 0.]]))
        model.weekly_view.window_projection.weight.copy_(torch.tensor([[1., 0., 0., 0., 0.]]))
    values = torch.tensor([1., 2., 3., 4., 5.]).reshape(1, 5, 1, 1)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    daily, _ = model.daily_view(values, mask)
    weekly, _ = model.weekly_view(values, mask)
    torch.testing.assert_close(daily, torch.tensor([[3.]]))
    torch.testing.assert_close(weekly, torch.tensor([[1.]]))
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path)
    torch.testing.assert_close(restored.daily_view(values, mask)[0], daily)
    torch.testing.assert_close(restored.weekly_view(values, mask)[0], weekly)


def test_window_lstm_uses_central_context_and_skips_incomplete_group():
    encoder = PeriodicViewEncoder(MergedDemandConfig(
        periodic_mode='lstm', height=1, width=1, d_model=4,
        weekday_dim=1, hour_dim=1, periodic_window_weights=WEIGHTS,
    ))
    values = torch.tensor([1., 2., 3., 4., 5., 90., 90., 90., 90., 90.]).reshape(1, 10, 1, 1)
    mask = torch.tensor([[False] * 5 + [True, False, False, False, False]])
    context = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4)
    output, valid = encoder(values, mask, context)
    reduced = torch.tensor([0.05 + 0.2 + 2.1 + 0.4 + 0.25])
    expected = torch.cat([torch.log1p(reduced), context[0, 2]]).reshape(1, 1, 5)
    _, (hidden, _) = encoder.lstm(expected)
    torch.testing.assert_close(output[0, 0], hidden[0, 0])
    assert valid.tolist() == [True]
