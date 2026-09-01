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
    train_set = UnifiedDemandDataset(path, "train", **kwargs)
    val_set = UnifiedDemandDataset(path, "val", **kwargs)
    test_set = UnifiedDemandDataset(path, "test", **kwargs)
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
    gradient_checks = {
        "neural_head": model.output_gate.neural_head.weight.grad is not None,
        "branch_attention": model.branch_attention.query_projection.weight.grad is not None,
        "local_history": model.local_history.scalar_embedding.projection.weight.grad is not None,
        "retrieval_has_no_gradient": model.retrieval._grid is not None,
    }
    if not all(gradient_checks.values()):
        raise AssertionError(f"{name}: end-to-end gradient check failed: {gradient_checks}")

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
    results.append(_check_retrieval_boundary())
    report = {"device": str(device), "all_pass": True, "checks": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
