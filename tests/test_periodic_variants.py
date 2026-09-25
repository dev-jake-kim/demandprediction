"""Behavior of valid-lag periodic summaries and node-gated predictions."""

from tempfile import TemporaryDirectory

import torch

from models.merged import MergedDemandConfig
from models.merged.modeling import MergedDemandModel, PeriodicViewEncoder, ViewFusion


def _config(mode: str) -> MergedDemandConfig:
    return MergedDemandConfig(
        periodic_mode=mode, d_model=4, height=1, width=2, weekday_dim=1, hour_dim=1
    )


def test_ma_ignores_missing_lags_and_all_missing_is_zero():
    encoder = PeriodicViewEncoder(_config('ma'))
    values = torch.tensor([
        [[[100.0], [100.0]], [[2.0], [4.0]], [[6.0], [8.0]]],
        [[[100.0], [100.0]], [[50.0], [50.0]], [[25.0], [25.0]]],
    ])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    summary, valid = encoder(values, mask)
    torch.testing.assert_close(summary, torch.tensor([[4.0, 6.0], [0.0, 0.0]]))
    assert valid.tolist() == [True, False]


def test_ema_favors_recent_valid_lags_and_preserves_constant_signal():
    encoder = PeriodicViewEncoder(_config('ema'))
    values = torch.tensor([
        [[[100.0]], [[2.0]], [[8.0]]],
        [[[5.0]], [[5.0]], [[5.0]]],
    ])
    mask = torch.tensor([[True, False, False], [False, True, False]])
    summary, valid = encoder(values, mask)
    # alpha=2/(L+1)=0.5; the two valid weights are 0.5 and 1 before normalization.
    torch.testing.assert_close(summary[:, 0], torch.tensor([6.0, 5.0]))
    assert valid.all()


def test_lstm_compacts_valid_lags_in_chronological_order():
    torch.manual_seed(7)
    config = _config('lstm')
    encoder = PeriodicViewEncoder(config)
    values = torch.tensor([
        [[[100.0]], [[2.0]], [[100.0]], [[8.0]]],
        [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
    ])
    mask = torch.tensor([[True, False, True, False], [True, True, True, True]])
    context = torch.zeros(2, 4, config.context_dim)
    summary, valid = encoder(values, mask, context)
    expected_sequence = torch.cat(
        [torch.log1p(values[0, [1, 3], 0]), context[0, [1, 3]]], dim=-1
    )
    _, (hidden, _) = encoder.lstm(expected_sequence.unsqueeze(0))
    torch.testing.assert_close(summary[0, 0], hidden[0, 0])
    torch.testing.assert_close(summary[1], torch.zeros_like(summary[1]))
    assert valid.tolist() == [True, False]


def test_lstm_fusion_uses_both_periodic_vectors_only_when_valid():
    fusion = ViewFusion(_config('lstm'), num_views=3)
    with torch.no_grad():
        fusion.projection.weight.zero_()
        fusion.projection.bias.zero_()
        fusion.projection.weight[0, 4] = 1.0
        fusion.projection.weight[0, 8] = 2.0
    local = torch.zeros(2, 1, 4)
    daily = torch.ones_like(local)
    weekly = torch.ones_like(local)
    fused = fusion(
        local, daily, weekly, torch.tensor([True, False]), torch.tensor([True, False])
    )
    torch.testing.assert_close(fused[:, 0, 0], torch.tensor([3.0, 0.0]))


def test_node_gates_and_local_projection_both_affect_prediction():
    config = _config('ma')
    fusion = ViewFusion(config, num_views=3)
    with torch.no_grad():
        fusion.local_projection.weight.zero_()
        fusion.local_projection.bias.zero_()
        fusion.daily_gate.copy_(torch.tensor([0.0, 2.0]))
        fusion.weekly_gate.zero_()
    local = torch.zeros(1, 2, config.d_model, requires_grad=True)
    daily = torch.tensor([[2.0, 2.0]])
    weekly = torch.tensor([[10.0, 10.0]])
    pred = fusion(local, daily, weekly, torch.tensor([True]), torch.tensor([False]))
    expected = torch.nn.functional.softplus(
        torch.tensor([1.0, 2.0 * torch.sigmoid(torch.tensor(2.0))])
    )
    torch.testing.assert_close(pred[0], expected)
    pred.sum().backward()
    assert fusion.daily_gate.grad.abs().sum() > 0
    assert fusion.weekly_gate.grad.abs().sum() == 0
    assert fusion.local_projection.weight.grad is not None


def test_average_checkpoint_restores_node_gates_without_prediction_head():
    config = MergedDemandConfig(
        periodic_mode='ma', d_model=4, height=1, width=2,
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )
    model = MergedDemandModel(config)
    assert not hasattr(model, 'head')
    with torch.no_grad():
        model.fusion.daily_gate.copy_(torch.tensor([-1.0, 2.0]))
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path)
    assert restored.config.periodic_mode == 'ma'
    assert not hasattr(restored, 'head')
    torch.testing.assert_close(restored.fusion.daily_gate, model.fusion.daily_gate)
