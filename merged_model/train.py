"""Train and validate the unified model with one MAE objective."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

if __package__ in {None, ""}:  # Allow both ``python train.py`` and ``python -m ...``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from comparison_models.merged_model.data import UnifiedDemandDataset, resolve_dataset_path
    from comparison_models.merged_model.model import UnifiedDemandModel
else:
    from .data import UnifiedDemandDataset, resolve_dataset_path
    from .model import UnifiedDemandModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dataset", choices=("ulsan", "porto"), default="ulsan")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--retrieval-scope", choices=("observed_past", "train_prefix"), default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Useful for smoke tests")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.index not in (None, 0):
            raise RuntimeError("This entry point intentionally supports GPU 0 only")
    return device


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model: UnifiedDemandModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    batches = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(**batch)
            loss = output["loss"]
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        error = output["prediction"] - batch["target"]
        total_abs += error.detach().abs().sum().item()
        total_sq += error.detach().square().sum().item()
        total_count += error.numel()
        batches += 1

    if total_count == 0:
        raise RuntimeError("No batches were processed")
    return {
        "mae": total_abs / total_count,
        "rmse": float(np.sqrt(total_sq / total_count)),
        "batches": float(batches),
    }


def build_dataset_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    return {
        "time_step": int(data_cfg.get("time_step", 24)),
        "daily_period": int(data_cfg.get("daily_period", 24)),
        "daily_lags": int(data_cfg.get("daily_lags", 6)),
        "weekly_period": int(data_cfg.get("weekly_period", 24 * 7)),
        "weekly_lags": int(data_cfg.get("weekly_lags", 4)),
        "lag_radius": int(data_cfg.get("lag_radius", 0)),
        "train_ratio": float(data_cfg.get("train_ratio", 0.70)),
        "val_ratio": float(data_cfg.get("val_ratio", 0.15)),
    }


def build_model(
    cfg: dict[str, Any],
    *,
    height: int,
    width: int,
    data_path: Path,
    train_end: int,
    retrieval_scope: str,
) -> UnifiedDemandModel:
    model_cfg = cfg.get("model", {})
    data_kwargs = build_dataset_kwargs(cfg)
    return UnifiedDemandModel(
        height=height,
        width=width,
        time_step=data_kwargs["time_step"],
        local_radius=int(model_cfg.get("local_radius", 2)),
        d_model=int(model_cfg.get("d_model", 64)),
        num_fourier_bands=int(model_cfg.get("num_fourier_bands", 8)),
        transformer_layers=int(model_cfg.get("transformer_layers", 2)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_ffn=int(model_cfg.get("transformer_ffn", 128)),
        history_hidden=int(model_cfg.get("history_hidden", 64)),
        periodic_hidden=int(model_cfg.get("periodic_hidden", 64)),
        fusion_dim=int(model_cfg.get("fusion_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        retrieval_grid_path=data_path,
        retrieval_k=int(model_cfg.get("retrieval_k", 20)),
        retrieval_chunk_size=int(model_cfg.get("retrieval_chunk_size", 256)),
        retrieval_scope=retrieval_scope,
        retrieval_train_end=train_end,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config.resolve())
    train_cfg = cfg.get("training", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 2026))
    set_seed(seed)
    device = choose_device(args.device)

    dataset_cfg = cfg.get("datasets", {}).get(args.dataset, {})
    configured_path = args.data_path if args.data_path is not None else dataset_cfg.get("path")
    data_path = resolve_dataset_path(args.dataset, configured_path)
    dataset_kwargs = build_dataset_kwargs(cfg)
    if args.train_ratio is not None:
        dataset_kwargs["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        dataset_kwargs["val_ratio"] = args.val_ratio
    train_set = UnifiedDemandDataset(data_path, "train", **dataset_kwargs)
    val_set = UnifiedDemandDataset(data_path, "val", **dataset_kwargs)
    test_set = UnifiedDemandDataset(data_path, "test", **dataset_kwargs)

    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2))
    workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)

    retrieval_scope = args.retrieval_scope or train_cfg.get("retrieval_scope", "observed_past")
    model = build_model(
        cfg,
        height=train_set.height,
        width=train_set.width,
        data_path=data_path,
        train_end=train_set.train_end,
        retrieval_scope=retrieval_scope,
    ).to(device)
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 2000))
    patience = int(args.patience if args.patience is not None else train_cfg.get("patience", 50))

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_path": str(data_path),
                "shape": [train_set.total_steps, train_set.height, train_set.width],
                "split": {"train_end": train_set.train_end, "val_end": train_set.val_end},
                "device": str(device),
                "retrieval_scope": retrieval_scope,
                "objective": "MAE",
            },
            ensure_ascii=False,
        )
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer, max_batches=args.max_batches
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, max_batches=args.max_batches)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(json.dumps({"early_stop": True, "epoch": epoch}, ensure_ascii=False))
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced")
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, max_batches=args.max_batches)
    result = {
        "dataset": args.dataset,
        "data_path": str(data_path),
        "device": str(device),
        "retrieval_scope": retrieval_scope,
        "objective": "MAE",
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "test": test_metrics,
        "history": history,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False))

    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).with_name("runs") / f"{args.dataset}_result.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result_json={output_path}")


if __name__ == "__main__":
    main()
