"""Scalar LSTM replacement for the periodic lag mean after five-hour windows."""

from tempfile import TemporaryDirectory

import torch

from models.merged import MergedDemandConfig
from models.merged.modeling import MergedDemandModel, PeriodicViewEncoder


WEIGHTS = [0.05, 0.1, 0.7, 0.1, 0.05]


def _config():
    return MergedDemandConfig(
        periodic_mode='lag_lstm', periodic_window_weights=WEIGHTS,
        height=1, width=1, d_model=4,
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )


def test_scalar_lstm_replaces_mean_and_compacts_valid_periodic_lags():
    torch.manual_seed(18)
    encoder = PeriodicViewEncoder(_config())
    values = torch.tensor([
        [2., 4., 10., 4., 2., 1., 2., 3., 4., 5.],
        [90., 90., 90., 90., 90., 1., 2., 3., 4., 5.],
        [90.] * 10,
    ]).reshape(3, 10, 1, 1)
    mask = torch.tensor([
        [False] * 10,
        [True, False, False, False, False] + [False] * 5,
        [True] * 10,
    ])
    output, valid = encoder(values, mask)
    # Older lag 8, newer lag 3; second sample contains only the newer lag.
    _, (first_hidden, _) = encoder.lag_lstm(torch.tensor([[[8.], [3.]]]))
    _, (second_hidden, _) = encoder.lag_lstm(torch.tensor([[[3.]]]))
    expected = torch.stack([
        encoder.lag_projection(first_hidden[-1])[0, 0],
        encoder.lag_projection(second_hidden[-1])[0, 0],
        torch.tensor(0.),
    ])
    torch.testing.assert_close(output[:, 0], expected)
    assert valid.tolist() == [True, True, False]
    output.sum().backward()
    assert encoder.lag_lstm.weight_ih_l0.grad.abs().sum() > 0
    assert encoder.lag_projection.weight.grad.abs().sum() > 0
    assert encoder.window_projection.weight.grad.abs().sum() > 0


def test_scalar_lstm_checkpoint_retains_direct_gate_prediction():
    model = MergedDemandModel(_config()).eval()
    values = torch.tensor([1., 2., 3., 4., 5.]).reshape(1, 5, 1, 1)
    weekly_values = torch.tensor([4., 3., 2., 1., 0.]).reshape(1, 5, 1, 1)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    local = torch.zeros(1, 1, model.config.d_model)
    with torch.no_grad():
        model.fusion.daily_gate.fill_(2.)
        model.fusion.weekly_gate.fill_(-1.)
        daily, daily_valid = model.daily_view(values, mask)
        weekly, weekly_valid = model.weekly_view(weekly_values, mask)
        predicted = model.fusion(local, daily, weekly, daily_valid, weekly_valid)
        expected = torch.nn.functional.softplus(
            model.fusion.local_projection(local).squeeze(-1)
            + model.fusion.daily_gate.sigmoid() * daily
            + model.fusion.weekly_gate.sigmoid() * weekly
        )
    assert daily_valid.item() and weekly_valid.item()
    torch.testing.assert_close(predicted, expected)
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path).eval()
    with torch.no_grad():
        restored_daily, restored_daily_valid = restored.daily_view(values, mask)
        restored_weekly, restored_weekly_valid = restored.weekly_view(weekly_values, mask)
        actual = restored.fusion(
            local, restored_daily, restored_weekly, restored_daily_valid, restored_weekly_valid,
        )
    torch.testing.assert_close(actual, expected)
