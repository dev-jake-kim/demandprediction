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

    # gated: s=0이면 공유 경로와 같고, begin_node_adaptation(V←W, s←1) 직후에도 출력이 같아야 한다.
    torch.manual_seed(0)
    gated = LocalHistoryEncoder(
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
        node_adaptive_mode='gated',
    )
    gated.eval()
    with torch.no_grad():
        gated_before = gated(demands)
        gated.begin_node_adaptation()
        gated_after = gated(demands)
        gated.node_adaptive = False
        gated_reference = gated(demands)
        gated.node_adaptive = True
    gated_gap = max(
        float((gated_before - gated_reference).abs().max()),
        float((gated_after - gated_reference).abs().max()),
    )
    if gated_gap > 1e-5:
        raise AssertionError(f'gated: s=0 또는 V=W 직후 출력이 공유 경로와 다름: max|diff|={gated_gap:.3e}')
    if float(gated.node_delta_started) != 1.0:
        raise AssertionError('gated: begin_node_adaptation 후 s가 1이 아님')
    gated_params = sum(param.numel() for param in gated.node_delta_parameters())
    if gated_params != expected + len(adaptive):
        raise AssertionError(f'gated 파라미터 수가 예상과 다름: {gated_params} != {expected + len(adaptive)}')

    # basis: 생성 시 0, begin_node_adaptation(C 난수, B=0) 직후 출력 불변. B를 채우면 같은 ΔW의 additive와 일치.
    torch.manual_seed(0)
    descriptors = torch.randn(len(adaptive), 3).tolist()
    basis = LocalHistoryEncoder(
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
        node_adaptive_mode='basis',
        node_delta_rank=2,
        node_descriptors=descriptors,
    )
    basis.eval()
    if any(float(param.detach().abs().sum()) != 0.0 for param in basis.node_delta_parameters()):
        raise AssertionError('basis: node_delta 파라미터가 0으로 생성되지 않음')
    with torch.no_grad():
        basis.begin_node_adaptation()
        basis_after = basis(demands)
        basis.node_adaptive = False
        basis_reference = basis(demands)
        basis.node_adaptive = True
        basis_gap = float((basis_after - basis_reference).abs().max())
        if basis_gap > 1e-5:
            raise AssertionError(f'basis: begin_node_adaptation 직후 출력이 공유 경로와 다름: {basis_gap:.3e}')
        for param in (basis.node_delta_basis_ih, basis.node_delta_basis_hh, basis.node_delta_basis_bias):
            param.normal_(0.0, 0.3)
        coef = basis.node_delta_coefficients()
        encoder.node_delta_weight_ih.copy_(torch.einsum('am,mgf->agf', coef, basis.node_delta_basis_ih))
        encoder.node_delta_weight_hh.copy_(torch.einsum('am,mgh->agh', coef, basis.node_delta_basis_hh))
        encoder.node_delta_bias.copy_(coef @ basis.node_delta_basis_bias)
        encoder.load_state_dict({k: v for k, v in basis.state_dict().items() if 'node_delta' not in k}, strict=False)
        basis_gap_full = float((basis(demands) - encoder(demands)).abs().max())
        encoder.node_delta_weight_ih.zero_()
        encoder.node_delta_weight_hh.zero_()
        encoder.node_delta_bias.zero_()
    if basis_gap_full > 1e-5:
        raise AssertionError(f'basis: 같은 ΔW의 additive와 출력이 다름: {basis_gap_full:.3e}')
    basis_params = sum(param.numel() for param in basis.node_delta_parameters())
    basis_expected = 2 * (n_gates * (8 + 5) + n_gates) + len(adaptive) * 2 + 3 * 2
    if basis_params != basis_expected:
        raise AssertionError(f'basis 파라미터 수가 예상과 다름: {basis_params} != {basis_expected}')

    # periodic: ΔW=0이면 공유 LSTM과 같고, ΔW≠0이면 노드마다 W+ΔW[n]인 nn.LSTM을 따로 돌린 결과와 같아야 한다.
    from models.merged.modules.fusion import NeuralRetrievalGate
    from models.merged.modules.periodic import PeriodicLSTMEncoder

    torch.manual_seed(1)
    periodic = PeriodicLSTMEncoder(5, extra_dim=2, node_adaptive_indices=adaptive)
    values = torch.rand(2, 4, num_nodes, 1) * 3
    invalid = torch.tensor([[False, True, False, False], [True, True, True, True]])
    extra = torch.rand(2, 4, 2)
    with torch.no_grad():
        periodic_zero, _ = periodic(values, invalid, extra)
        shared = PeriodicLSTMEncoder(5, extra_dim=2)
        shared.load_state_dict({k: v for k, v in periodic.state_dict().items() if 'node_delta' not in k})
        periodic_shared, _ = shared(values, invalid, extra)
        periodic_zero_gap = float((periodic_zero - periodic_shared).abs().max())
        for param in periodic.node_delta_parameters():
            param.normal_(0.0, 0.3)
        periodic_out, _ = periodic(values, invalid, extra)
        weight_ih, weight_hh, bias = periodic.node_delta.weights(periodic.lstm)
        manual_gap = 0.0
        for slot, node in enumerate(adaptive):
            single = PeriodicLSTMEncoder(5, extra_dim=2)
            single.lstm.weight_ih_l0.copy_(weight_ih[slot])
            single.lstm.weight_hh_l0.copy_(weight_hh[slot])
            single.lstm.bias_ih_l0.copy_(bias[slot])
            single.lstm.bias_hh_l0.zero_()
            reference, _ = single(values[:, :, node : node + 1], invalid, extra)
            manual_gap = max(manual_gap, float((periodic_out[:, node] - reference[:, 0]).abs().max()))
        others_gap = float((periodic_out[:, others] - periodic_shared[:, others]).abs().max())
    if periodic_zero_gap > 1e-6 or manual_gap > 1e-5 or others_gap != 0.0:
        raise AssertionError(
            f'periodic ΔW: zero {periodic_zero_gap:.2e}, 노드별 LSTM {manual_gap:.2e}, 비대상 {others_gap:.2e}'
        )

    # head: ΔW=0이면 공유 head와 같고, 대상 노드만 h·Δw[n] + Δb[n]만큼 raw가 바뀐다.
    gate = NeuralRetrievalGate(6, use_softplus=False, node_adaptive_indices=adaptive)
    h_attn = torch.randn(2, num_nodes, 6)
    with torch.no_grad():
        head_zero, _, _ = gate(h_attn, None, bypass_gate=True)
        reference = gate.neural_head(h_attn).squeeze(-1)
        for param in gate.node_delta_parameters():
            param.normal_()
        head_delta, _, _ = gate(h_attn, None, bypass_gate=True)
        expected_head = reference.clone()
        expected_head[:, adaptive] += (h_attn[:, adaptive] * gate.node_delta_head_weight).sum(-1) + gate.node_delta_head_bias
    head_gap = max(float((head_zero - reference).abs().max()), float((head_delta - expected_head).abs().max()))
    if head_gap > 1e-5:
        raise AssertionError(f'head ΔW 결과가 예상과 다름: {head_gap:.2e}')

    # fusion: ΔW=0이면 공유 BranchAttention과 같고, 비대상 노드는 ΔW와 무관하다.
    from models.merged.modules.attention import BranchAttention

    torch.manual_seed(2)
    fusion = BranchAttention(5, 4, 6, node_adaptive_indices=adaptive)
    plain = BranchAttention(5, 4, 6)
    plain.load_state_dict({k: v for k, v in fusion.state_dict().items() if 'node_delta' not in k})
    inputs = (torch.randn(2, num_nodes, 5), torch.randn(2, num_nodes, 4), torch.randn(2, num_nodes, 4),
              torch.tensor([True, False]), torch.tensor([True, True]))
    with torch.no_grad():
        fusion_zero_gap = float((fusion(*inputs)[0] - plain(*inputs)[0]).abs().max())
        for param in fusion.node_delta_parameters():
            param.normal_()
        fused_delta = fusion(*inputs)[0]
        fusion_others_gap = float((fused_delta[:, others] - plain(*inputs)[0][:, others]).abs().max())
        fusion_changed = float((fused_delta[:, adaptive] - plain(*inputs)[0][:, adaptive]).abs().max())
    if fusion_zero_gap > 1e-6 or fusion_others_gap > 1e-6 or fusion_changed < 1e-3:
        raise AssertionError(f'fusion ΔW: zero {fusion_zero_gap:.2e}, 비대상 {fusion_others_gap:.2e}, 대상 변화 {fusion_changed:.2e}')

    # dynamic basis: W_dyn=0이면 정적 basis와 같고, W_dyn≠0이면 샘플별 계수로 만든 가중치의 노드별 LSTM과 같다.
    from models.merged.modules.node_delta import run_node_lstm

    torch.manual_seed(3)
    dynamic = LocalHistoryEncoder(
        height=height, width=width, time_step=time_step, local_radius=1, d_model=8, num_fourier_bands=2,
        transformer_layers=1, transformer_heads=2, transformer_ffn=16, history_hidden=5, dropout=0.0,
        node_adaptive_indices=adaptive, node_adaptive_mode='basis', node_delta_rank=2, node_delta_dynamic=True,
    )
    dynamic.eval()
    sequence = torch.randn(2, len(adaptive), time_step, 8)
    with torch.no_grad():
        dynamic.begin_node_adaptation()
        for param in (dynamic.node_delta_basis_ih, dynamic.node_delta_basis_hh, dynamic.node_delta_basis_bias):
            param.normal_(0.0, 0.3)
        coef = dynamic.node_delta_coefficients()
        lstm = dynamic.history_lstm
        static = run_node_lstm(
            sequence,
            lstm.weight_ih_l0 + torch.einsum('am,mgf->agf', coef, dynamic.node_delta_basis_ih),
            lstm.weight_hh_l0 + torch.einsum('am,mgh->agh', coef, dynamic.node_delta_basis_hh),
            lstm.bias_ih_l0 + lstm.bias_hh_l0 + coef @ dynamic.node_delta_basis_bias,
        )[:, :, -1]
        dynamic_static_gap = float((dynamic._node_adaptive_hidden(sequence) - static).abs().max())
        dynamic.node_delta_dynamic_weight.normal_(0.0, 0.5)
        dyn_out = dynamic._node_adaptive_hidden(sequence)
        scaled = coef[None] * (1.0 + torch.tanh(sequence[:, :, -1] @ dynamic.node_delta_dynamic_weight))
        dynamic_gap = 0.0
        for b in range(sequence.shape[0]):
            reference = run_node_lstm(
                sequence[b : b + 1],
                lstm.weight_ih_l0 + torch.einsum('am,mgf->agf', scaled[b], dynamic.node_delta_basis_ih),
                lstm.weight_hh_l0 + torch.einsum('am,mgh->agh', scaled[b], dynamic.node_delta_basis_hh),
                lstm.bias_ih_l0 + lstm.bias_hh_l0 + scaled[b] @ dynamic.node_delta_basis_bias,
            )[:, :, -1]
            dynamic_gap = max(dynamic_gap, float((dyn_out[b : b + 1] - reference).abs().max()))
    if dynamic_static_gap > 1e-5 or dynamic_gap > 1e-5:
        raise AssertionError(f'dynamic basis: W_dyn=0 {dynamic_static_gap:.2e}, 샘플별 {dynamic_gap:.2e}')

    # moe: 생성·begin 직후 공유 경로와 같고, 전문가 ΔW·router를 채우면 Σ_k g_k·LSTM(W+ΔW_k)와 같다.
    torch.manual_seed(4)
    moe = LocalHistoryEncoder(
        height=height, width=width, time_step=time_step, local_radius=1, d_model=8, num_fourier_bands=2,
        transformer_layers=1, transformer_heads=2, transformer_ffn=16, history_hidden=5, dropout=0.0,
        node_adaptive_indices=adaptive, node_adaptive_mode='moe', node_moe_experts=3, node_moe_router='dynamic',
        node_moe_top_k=2,
    )
    moe.eval()
    if any(float(param.detach().abs().sum()) != 0.0 for param in moe.node_delta_parameters()):
        raise AssertionError('moe: node_delta 파라미터가 0으로 생성되지 않음')
    with torch.no_grad():
        moe.begin_node_adaptation()
        moe_after = moe(demands)
        moe.node_adaptive = False
        moe_reference = moe(demands)
        moe.node_adaptive = True
        moe_identity_gap = float((moe_after - moe_reference).abs().max())
        experts = moe.node_moe
        for param in (experts.node_delta_expert_ih, experts.node_delta_expert_hh, experts.node_delta_expert_bias,
                      experts.node_delta_router_input):
            param.normal_(0.0, 0.3)
        moe_out = moe._node_adaptive_hidden(sequence)
        gate = experts.gates(sequence)
        if int((gate > 0).sum(-1).max()) > 2 or float((gate.sum(-1) - 1).abs().max()) > 1e-5:
            raise AssertionError('moe: top-k 게이트가 2개를 넘거나 합이 1이 아님')
        lstm = moe.history_lstm
        expected_moe = torch.zeros_like(moe_out)
        for k in range(3):
            single = torch.nn.LSTM(8, 5, batch_first=True)
            single.weight_ih_l0.copy_(lstm.weight_ih_l0 + experts.node_delta_expert_ih[k])
            single.weight_hh_l0.copy_(lstm.weight_hh_l0 + experts.node_delta_expert_hh[k])
            single.bias_ih_l0.copy_(lstm.bias_ih_l0 + experts.node_delta_expert_bias[k])
            single.bias_hh_l0.copy_(lstm.bias_hh_l0)
            _, (h_k, _) = single(sequence.reshape(-1, time_step, 8))
            expected_moe += gate[..., k, None] * h_k[-1].reshape(moe_out.shape)
        moe_gap = float((moe_out - expected_moe).abs().max())
    if moe_identity_gap > 1e-5 or moe_gap > 1e-5:
        raise AssertionError(f'moe: begin 직후 {moe_identity_gap:.2e}, 전문가 혼합 {moe_gap:.2e}')

    # moe residual: 같은 전문가·게이트에서 ΔW_n을 더하면 노드별 Σ_k g_k·LSTM(W + ΔW_k + ΔW_n)과 같다.
    from models.merged.modules.node_delta import NodeLSTMMoE

    torch.manual_seed(5)
    lstm_small = torch.nn.LSTM(8, 5, batch_first=True)
    residual_moe = NodeLSTMMoE(len(adaptive), 8, 5, num_experts=3, router='static', residual=True)
    with torch.no_grad():
        residual_moe.begin()
        zero_gap = float((residual_moe(lstm_small, sequence)[:, :, -1]
                          - lstm_small(sequence.reshape(-1, time_step, 8))[1][0][-1].reshape(2, len(adaptive), 5)).abs().max())
        for param in residual_moe.node_delta_parameters():
            param.normal_(0.0, 0.3)
        residual_out = residual_moe(lstm_small, sequence)[:, :, -1]
        gate = residual_moe.gates(sequence)
        residual_gap = 0.0
        for slot in range(len(adaptive)):
            expected = torch.zeros(2, 5)
            for k in range(3):
                single = torch.nn.LSTM(8, 5, batch_first=True)
                single.weight_ih_l0.copy_(lstm_small.weight_ih_l0 + residual_moe.node_delta_expert_ih[k] + residual_moe.node_delta_node_ih[slot])
                single.weight_hh_l0.copy_(lstm_small.weight_hh_l0 + residual_moe.node_delta_expert_hh[k] + residual_moe.node_delta_node_hh[slot])
                single.bias_ih_l0.copy_(lstm_small.bias_ih_l0 + residual_moe.node_delta_expert_bias[k] + residual_moe.node_delta_node_bias[slot])
                single.bias_hh_l0.copy_(lstm_small.bias_hh_l0)
                expected += gate[:, slot, k, None] * single(sequence[:, slot])[1][0][-1]
            residual_gap = max(residual_gap, float((residual_out[:, slot] - expected).abs().max()))
    if zero_gap > 1e-5 or residual_gap > 1e-5:
        raise AssertionError(f'moe residual: 0일 때 {zero_gap:.2e}, 노드별 {residual_gap:.2e}')

    return {
        'node_adaptive_zero_delta_matches_nn_lstm': True,
        'max_abs_diff': gap,
        'non_adaptive_nodes_bit_identical': untouched == 0.0,
        'adaptive_nodes': len(adaptive),
        'node_delta_params': actual,
        'gated_identity_max_abs_diff': gated_gap,
        'basis_identity_max_abs_diff': basis_gap,
        'basis_vs_additive_max_abs_diff': basis_gap_full,
        'periodic_delta_vs_per_node_lstm_max_abs_diff': manual_gap,
        'head_delta_max_abs_diff': head_gap,
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
    if not (int(sample['sample_idx']) == target and window[-1, node] == 1 and window[-2, node] == 1 and window.sum() == 2):
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
    results.append(_check_node_adaptive_identity())
    results.append(_check_retrieval_boundary())
    results.append(_check_independent_retrieval_radius())
    results.append(_check_retrieval_pass())
    results.append(_check_masked_grid_receptive_field())
    results.append(_check_event_mask())
    report = {'device': str(device), 'all_pass': True, 'checks': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
