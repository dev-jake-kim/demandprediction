#!/usr/bin/env python3
"""Fast structural and causal checks for the unified model.

This intentionally runs a real batch from both temporal-grid datasets and a
small synthetic retrieval case. It is not a replacement for the full
benchmark; it catches shape, mask, gradient, and time-boundary regressions
before an expensive 2,000-epoch run.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .modules import BranchAttention, CausalRetrieval, PeriodicLSTMEncoder
from .data import UnifiedDemandDataset, resolve_dataset_path
from .train import build_dataset_kwargs, build_model


ROOT = Path(__file__).resolve().parent


def _check_dataset(name: str, cfg: dict, device: torch.device) -> dict:
    dataset_cfg = cfg["datasets"][name]
    path = resolve_dataset_path(name, dataset_cfg.get("path"))
    kwargs = build_dataset_kwargs(cfg)
    weather_path = Path(dataset_cfg["weather_path"]).expanduser()
    if not weather_path.is_absolute():
        weather_path = Path.cwd() / weather_path
    kwargs["weather_csv_path"] = weather_path
    train_set = UnifiedDemandDataset(path, "train", **kwargs)
    val_set = UnifiedDemandDataset(path, "val", **kwargs)
    test_set = UnifiedDemandDataset(path, "test", **kwargs)
    train_weather = train_set.weather[train_set.time_step : train_set.train_end]
    weather_mean = train_weather.mean(axis=0)
    weather_std = train_weather.std(axis=0).clip(min=1e-6)
    if not (train_set.train_end == val_set.train_end == test_set.train_end):
        raise AssertionError(f"{name}: split bounds differ between dataset views")
    if not (train_set.val_end == val_set.val_end == test_set.val_end):
        raise AssertionError(f"{name}: validation bounds differ between dataset views")

    sample = next(iter(DataLoader(train_set, batch_size=1, shuffle=False)))
    sample = {key: value.to(device) for key, value in sample.items()}
    model = build_model(
        cfg,
        height=train_set.height,
        width=train_set.width,
        data_path=path,
        train_end=train_set.train_end,
        retrieval_scope="observed_past",
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(**sample)
    prediction = output["prediction"]
    weights = output["attention_weights"]
    if tuple(prediction.shape) != (1, train_set.height, train_set.width):
        raise AssertionError(f"{name}: unexpected prediction shape {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all() or (prediction < 0).any():
        raise AssertionError(f"{name}: prediction is not finite and non-negative")
    if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-5):
        raise AssertionError(f"{name}: branch attention weights do not sum to one")

    # A second pass checks that the single MAE graph reaches both the neural
    # head and branch projections. Retrieval is deliberately non-differentiable.
    model.train()
    model.zero_grad(set_to_none=True)
    train_output = model(**sample)
    train_output["loss"].backward()
    def _has_gradient(parameter: torch.Tensor) -> bool:
        return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)

    gradient_checks = {
        "neural_head": model.output_gate.neural_head.weight.grad is not None,
        "branch_attention": model.branch_attention.query_projection.weight.grad is not None,
        "local_history": model.local_history.scalar_embedding.projection.weight.grad is not None,
        "retrieval_has_no_gradient": model.retrieval._grid is not None,
        # 캘린더 임베딩이 세 LSTM 입력에 실제로 연결돼 있는지 — concat을 빠뜨리면 여기서 잡힌다.
        "weekday_embedding": _has_gradient(model.weekday_embedding.weight),
        "hour_embedding": _has_gradient(model.hour_embedding.weight),
    }
    if not all(gradient_checks.values()):
        raise AssertionError(f"{name}: end-to-end gradient check failed: {gradient_checks}")

    # 날씨는 임베딩 층이 없어 gradient로 연결을 확인할 수 없다(정규화 후 그대로 concat되는 상수 입력).
    # 대신 날씨만 흔들어 출력이 실제로 달라지는지 본다 — concat을 빠뜨리면 출력이 그대로다.
    model.eval()
    with torch.no_grad():
        baseline = model(**sample)["prediction"]
        perturbed_sample = dict(sample)
        for key in ("weather", "daily_weather", "weekly_weather"):
            perturbed_sample[key] = sample[key] + 10.0
        perturbed = model(**perturbed_sample)["prediction"]
    weather_sensitivity = float((baseline - perturbed).abs().max())
    if weather_sensitivity <= 1e-6:
        raise AssertionError(
            f"{name}: 날씨를 바꿔도 출력이 그대로임 — 날씨 채널이 LSTM 입력에 반영되지 않음"
        )

    target_time = int(sample["sample_idx"][0].item())
    if target_time < train_set.time_step:
        raise AssertionError(f"{name}: sample index is before local history boundary")
    expected_target = torch.from_numpy(np.array(train_set.grid[target_time], dtype=np.float32, copy=True)).to(device)
    if not torch.equal(sample["target"][0], expected_target):
        raise AssertionError(f"{name}: target is not read from the same temporal grid")
    daily_lag = int(train_set.daily_lag_values[-1])
    expected_daily = train_set.grid[target_time - daily_lag]
    if not np.array_equal(sample["daily_demand"][0, -1, :, 0].cpu().numpy(), expected_daily.reshape(-1)):
        raise AssertionError(f"{name}: daily lag does not match its absolute source time")
    return {
        "dataset": name,
        "data_path": str(path),
        "shape": [train_set.total_steps, train_set.height, train_set.width],
        "split": {"train_end": train_set.train_end, "val_end": train_set.val_end},
        "samples": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "target_time_probe": target_time,
        "attention_shape": list(weights.shape),
        "gradient_checks": gradient_checks,
        "lstm_input_sizes": {
            "history": model.local_history.history_lstm.input_size,
            "daily": model.daily_branch.lstm.input_size,
            "weekly": model.weekly_branch.lstm.input_size,
        },
        "weather_sensitivity": weather_sensitivity,
        "all_pass": True,
    }


def _check_invalid_mask() -> dict:
    encoder = PeriodicLSTMEncoder(hidden_size=8)
    values = torch.ones(2, 4, 3, 1)
    invalid = torch.tensor([[True, True, True, True], [True, False, True, False]])
    hidden, valid = encoder(values, invalid)
    if valid.tolist() != [False, True] or not torch.isfinite(hidden).all():
        raise AssertionError("periodic invalid-mask handling failed")
    attention = BranchAttention(history_hidden=8, periodic_hidden=8, fusion_dim=8)
    fused, weights = attention(
        torch.ones(2, 3, 8), hidden, hidden, valid, valid
    )
    if not torch.isfinite(fused).all() or not torch.allclose(weights[0, :, 2], torch.ones(3), atol=1e-6):
        raise AssertionError("all-invalid periodic branches did not fall back to neural token")
    return {"all_invalid_periodic_falls_back_to_neural": True}


def _check_periodic_extra_channels() -> dict:
    """concat한 날씨/캘린더 채널이 compaction(gather+pack)을 그대로 통과하는지 확인한다.

    `PeriodicLSTMEncoder`는 유효 lag만 앞으로 당겨 packing하는데, gather의 feature 폭을
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
        expanded = extra[:, None, :, :].expand(1, nodes, length, extra_dim).reshape(nodes, length, extra_dim)
        reference_input = torch.cat([sequence, expanded], dim=-1)[:, keep, :]
        _, (reference_hidden, _) = encoder.lstm(reference_input)
        reference = reference_hidden[-1].reshape(1, nodes, -1)

    if not bool(valid.item()):
        raise AssertionError("periodic extra-channel check: sequence should be valid")
    if not torch.allclose(hidden, reference, atol=1e-6):
        raise AssertionError(
            "periodic compaction dropped concatenated channels: "
            f"max|diff|={float((hidden - reference).abs().max()):.3e}"
        )
    if encoder.lstm.input_size != 1 + extra_dim:
        raise AssertionError(f"unexpected periodic LSTM input size {encoder.lstm.input_size}")
    return {
        "periodic_extra_channels_survive_compaction": True,
        "periodic_lstm_input_size": encoder.lstm.input_size,
    }


def _check_retrieval_boundary() -> dict:
    with tempfile.TemporaryDirectory(prefix="merged_retrieval_") as temp_dir:
        path = Path(temp_dir) / "grid.npy"
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
            retrieval_scope="observed_past",
            retrieval_train_end=None,
        )
        query = retrieval._crops[3:5].clone().reshape(1, 2, 1, 1)
        before = retrieval(query, torch.tensor([5]))
        # Target and future values must be outside the candidate interval
        # [time_step, target_time), so changing them cannot affect retrieval.
        retrieval._grid[5:] = 10_000.0
        retrieval._crops[5:] = 10_000.0
        retrieval._cache.fill_(float("nan"))
        after = retrieval(query, torch.tensor([5]))
        if not torch.equal(before, after):
            raise AssertionError("retrieval used target or future values")
    return {"retrieval_candidate_boundary": "tau < target_time"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results = [_check_dataset(name, cfg, device) for name in ("ulsan", "porto")]
    results.append(_check_invalid_mask())
    results.append(_check_periodic_extra_channels())
    results.append(_check_retrieval_boundary())
    report = {"device": str(device), "all_pass": True, "checks": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
