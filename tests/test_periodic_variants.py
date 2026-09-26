"""Behavior of valid-lag periodic summaries and node-wise fusion."""

import math
from tempfile import TemporaryDirectory
import torch

from models.merged import MergedDemandConfig
from models.merged.modeling import (
    ContextEncoder, LocalViewEncoder, MergedDemandModel, PeriodicViewEncoder, ViewFusion,
)


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


def test_ma_gates_form_initialized_convex_mixture_and_learn():
    config = _config('ma')
    fusion = ViewFusion(config, num_views=3)
    with torch.no_grad():
        fusion.local_projection.weight.zero_()
        fusion.local_projection.bias.zero_()
    local = torch.ones(1, 2, config.d_model)
    local_mean = torch.tensor([[4., 4.]])
    daily = torch.tensor([[2., 2.]])
    weekly = torch.tensor([[10., 10.]])
    initial = fusion(
        local, daily, weekly, torch.tensor([True]), torch.tensor([True]),
        local_mean=local_mean,
    )
    torch.testing.assert_close(
        initial, torch.nn.functional.softplus(torch.tensor([[4.2, 4.2]])),
    )
    with torch.no_grad():
        fusion.daily_mix_logit[1] = math.log(0.6 / 0.3)
        fusion.weekly_mix_logit[1] = math.log(0.1 / 0.3)
    pred = fusion(
        local, daily, weekly, torch.tensor([True]), torch.tensor([False]),
        local_mean=local_mean,
    )
    torch.testing.assert_close(
        pred, torch.nn.functional.softplus(torch.tensor([[3.2, 2.4]])),
    )
    pred.sum().backward()
    assert fusion.daily_mix_logit.grad.abs().sum() > 0
    assert fusion.weekly_mix_logit.grad.abs().sum() > 0
    assert fusion.local_projection.weight.grad.abs().sum() > 0


def test_no_local_ma_matches_independent_sigmoid_fusion_and_restores():
    config = MergedDemandConfig(
        periodic_mode='ma_no_local', d_model=4, height=1, width=2,
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )
    model = MergedDemandModel(config).eval()
    with torch.no_grad():
        model.fusion.local_projection.weight.zero_()
        model.fusion.local_projection.bias.fill_(0.25)
        model.fusion.daily_gate.copy_(torch.tensor([0., 2.]))
        model.fusion.weekly_gate.copy_(torch.tensor([0., -2.]))
    local = torch.zeros(1, 2, config.d_model)
    daily = torch.tensor([[4., 9.]])
    weekly = torch.tensor([[10., 11.]])
    with torch.no_grad():
        prediction = model.fusion(
            local, daily, weekly, torch.tensor([True]), torch.tensor([False]),
        )
    expected = torch.nn.functional.softplus(
        torch.tensor([[0.25 + 0.5 * 4, 0.25 + torch.sigmoid(torch.tensor(2.)).item() * 9]])
    )
    torch.testing.assert_close(prediction, expected)
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path).eval()
    with torch.no_grad():
        actual = restored.fusion(
            local, daily, weekly, torch.tensor([True]), torch.tensor([False]),
        )
    torch.testing.assert_close(actual, expected)


def test_ma_uses_raw_center_node_history_and_restores_prediction():
    config = MergedDemandConfig(
        periodic_mode='ma', d_model=4, height=1, width=2, zero_node_indices=[0],
        history_weights=[0.1, 0.2, 0.7],
        temperature_min=0.0, temperature_max=1.0, precipitation_max=1.0,
    )
    model = MergedDemandModel(config).eval()
    history = torch.zeros(1, 24, 1, 2)
    history[:, :, :, 0] = 100.
    history[:, :12, :, 1] = 0.
    history[:, 12:, :, 1] = 4.
    with torch.no_grad():
        model.fusion.local_projection.weight.zero_()
        model.fusion.local_projection.bias.zero_()
    batch = {
        'demand_history': history,
        'daily_demand': torch.tensor([[[[20.], [5.]]]]),
        'daily_mask': torch.tensor([[False]]),
        'weekly_demand': torch.tensor([[[[20.], [9.]]]]),
        'weekly_mask': torch.tensor([[False]]),
        'weather': torch.zeros(1, 24, 3),
        'hour_of_day': torch.zeros(1, 24, dtype=torch.long),
        'day_of_week': torch.zeros(1, 24, dtype=torch.long),
        'daily_weather': torch.zeros(1, 1, 3),
        'daily_hour': torch.zeros(1, 1, dtype=torch.long),
        'daily_day_of_week': torch.zeros(1, 1, dtype=torch.long),
        'weekly_weather': torch.zeros(1, 1, 3),
        'weekly_hour': torch.zeros(1, 1, dtype=torch.long),
        'weekly_day_of_week': torch.zeros(1, 1, dtype=torch.long),
    }
    with torch.no_grad():
        prediction = model.forward_views(**batch)['logits']
    expected = torch.nn.functional.softplus(torch.tensor(0.7 * 2 + 0.2 * 5 + 0.1 * 9))
    torch.testing.assert_close(prediction[0, 0, 1], expected)
    assert prediction[0, 0, 0].item() == 0.
    with TemporaryDirectory() as path:
        model.save_pretrained(path)
        restored = MergedDemandModel.from_pretrained(path).eval()
    with torch.no_grad():
        actual = restored.forward_views(**batch)['logits']
    torch.testing.assert_close(actual, prediction)


def test_module_ablation_excludes_local_embedding_and_local_mean_independently():
    local = torch.ones(1, 2, 4)
    daily = torch.zeros(1, 2)
    weekly = torch.zeros(1, 2)
    valid = torch.tensor([True])
    with torch.no_grad():
        no_embedding = ViewFusion(
            MergedDemandConfig(periodic_mode='ma', d_model=4, height=1, width=2,
                               use_local_view=False), num_views=3,
        )
        no_embedding.local_projection.weight.fill_(3.)
        no_embedding.local_projection.bias.fill_(5.)
        without_embedding = no_embedding(local, daily, weekly, valid, valid,
                                         local_mean=torch.zeros(1, 2))
        torch.testing.assert_close(
            without_embedding, torch.nn.functional.softplus(torch.zeros(1, 2)),
        )

        no_mean = ViewFusion(
            MergedDemandConfig(periodic_mode='ma', d_model=4, height=1, width=2,
                               use_local_mean=False), num_views=3,
        )
        no_mean.local_projection.weight.zero_()
        no_mean.local_projection.bias.zero_()
        without_mean = no_mean(local, daily, weekly, valid, valid,
                               local_mean=torch.full((1, 2), 40.))
        torch.testing.assert_close(
            without_mean, torch.nn.functional.softplus(torch.zeros(1, 2)),
        )


def test_weather_and_calendar_ablation_independently_mask_context_inputs():
    kwargs = dict(periodic_mode='ma', d_model=4, height=1, width=1,
                  weekday_dim=2, hour_dim=2, temperature_min=0.,
                  temperature_max=20., precipitation_max=10.)
    weather = torch.tensor([[[10., 5., 0.]]])
    empty_weather = torch.zeros_like(weather)
    hour = torch.tensor([[3]])
    day = torch.tensor([[2]])

    no_weather = ContextEncoder(MergedDemandConfig(**kwargs, use_weather=False)).eval()
    with torch.no_grad():
        first = no_weather(weather, hour, day)
        second = no_weather(empty_weather, hour, day)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[..., :2], torch.zeros_like(first[..., :2]))

    no_calendar = ContextEncoder(MergedDemandConfig(**kwargs, use_calendar=False)).eval()
    with torch.no_grad():
        first = no_calendar(weather, hour, day)
        second = no_calendar(weather, hour + 1, day + 1)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[..., 2:], torch.zeros_like(first[..., 2:]))


def test_no_neighbors_keeps_center_history_while_removing_neighbor_signal():
    torch.manual_seed(45)
    config = MergedDemandConfig(
        periodic_mode='ma', d_model=4, height=1, width=2, local_radius=1,
        use_neighbors=False,
    )
    encoder = LocalViewEncoder(config).eval()
    first = torch.tensor([[[[2., 0.]], [[3., 0.]]]])
    second = first.clone()
    second[:, :, :, 1] = 100.
    context = torch.zeros(1, 2, config.context_dim)
    with torch.no_grad():
        a = encoder(first, context)
        b = encoder(second, context)
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)


def test_no_softplus_ablation_keeps_signed_scalar_output():
    fusion = ViewFusion(
        MergedDemandConfig(periodic_mode='ma', d_model=4, height=1, width=1,
                           use_softplus=False), num_views=3,
    )
    with torch.no_grad():
        fusion.local_projection.weight.zero_()
        fusion.local_projection.bias.fill_(-2.)
        prediction = fusion(
            torch.zeros(1, 1, 4), torch.zeros(1, 1), torch.zeros(1, 1),
            torch.tensor([True]), torch.tensor([True]), local_mean=torch.zeros(1, 1),
        )
    torch.testing.assert_close(prediction, torch.full((1, 1), -2.))
