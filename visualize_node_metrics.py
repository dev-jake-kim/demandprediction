from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_frame import GridDemandDataset
from models import GridDemandModel


METRICS = ("rmse", "mae", "mape_plus1")
METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "mape_plus1": "MAPE(+1) [%]",
}


@dataclass(frozen=True)
class Run:
    path: Path
    city: str
    seed: int
    npy_path: Path
    weather_csv_path: Path
    time_step: int
    train_ratio: float
    val_ratio: float


def resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def discover_runs(root: Path, project_root: Path) -> dict[tuple[str, int], Run]:
    runs: dict[tuple[str, int], Run] = {}
    for config_path in sorted(root.glob("*/.hydra/config.yaml")):
        run_dir = config_path.parent.parent
        if not (run_dir / "config.json").is_file() or not (run_dir / "model.safetensors").is_file():
            continue
        cfg = OmegaConf.load(config_path)
        run = Run(
            path=run_dir,
            city=str(cfg.dataset.city),
            seed=int(cfg.seed),
            npy_path=resolve_project_path(str(cfg.dataset.npy_path), project_root),
            weather_csv_path=resolve_project_path(str(cfg.dataset.weather_csv_path), project_root),
            time_step=int(cfg.dataset.time_step),
            train_ratio=float(cfg.dataset.train_ratio),
            val_ratio=float(cfg.dataset.val_ratio),
        )
        key = (run.city, run.seed)
        if key in runs:
            raise ValueError(f"Duplicate run for city={run.city}, seed={run.seed}: {run.path}")
        runs[key] = run
    return runs


def predict(run: Run, device: torch.device, batch_size: int, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.load(run.npy_path, mmap_mode="r")
    split2 = int(grid.shape[0] * (run.train_ratio + run.val_ratio))
    dataset = GridDemandDataset(
        run.npy_path,
        time_step=run.time_step,
        weather_csv_path=run.weather_csv_path,
        t_start=split2,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = GridDemandModel.from_pretrained(run.path).to(device).eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{run.city}/seed{run.seed}/{run.path.parent.parent.parent.name}"):
            model_inputs = {
                name: batch[name].to(device, non_blocking=True)
                for name in ("demands", "weather", "hour_of_day", "day_of_week")
            }
            predictions.append(model(**model_inputs)["logits"].cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def node_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    error = labels.astype(np.float64) - predictions.astype(np.float64)
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "mape_plus1": np.mean(np.abs(error) / (np.abs(labels) + 1.0), axis=0) * 100.0,
    }


def annotate_delta(axis: plt.Axes, values: np.ndarray) -> None:
    fontsize = 4.3 if values.size >= 190 else 4.8
    threshold = np.nanmax(np.abs(values)) * 0.55
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        color = "white" if abs(value) > threshold else "black"
        axis.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=fontsize, color=color)


def add_colorbar(fig: plt.Figure, image, axis: plt.Axes) -> None:
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_city(
    city: str,
    seed: int,
    baseline: dict[str, np.ndarray],
    adaptive: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(23, 15), constrained_layout=True)
    for metric_idx, metric in enumerate(METRICS):
        base = baseline[metric]
        adapt = adaptive[metric]
        delta = adapt - base
        common_min = float(min(base.min(), adapt.min()))
        common_max = float(max(base.max(), adapt.max()))
        if common_min == common_max:
            common_max = common_min + 1e-12

        for column, (values, title) in enumerate(
            ((base, "baseline-tmp"), (adapt, "node-adaptive"))
        ):
            axis = axes[metric_idx, column]
            image = axis.imshow(values, cmap="viridis", vmin=common_min, vmax=common_max, origin="upper")
            axis.set_title(f"{title} — {METRIC_LABELS[metric]}")
            add_colorbar(fig, image, axis)

        limit = float(np.max(np.abs(delta)))
        if limit == 0.0:
            limit = 1e-12
        delta_axis = axes[metric_idx, 2]
        delta_image = delta_axis.imshow(
            delta,
            cmap="RdYlGn_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            origin="upper",
        )
        improved = int(np.count_nonzero(delta < 0))
        degraded = int(np.count_nonzero(delta > 0))
        delta_axis.set_title(
            f"adaptive - baseline — {METRIC_LABELS[metric]}\n"
            f"green: decreased ({improved}), red: increased ({degraded})"
        )
        annotate_delta(delta_axis, delta)
        add_colorbar(fig, delta_image, delta_axis)

        for axis in axes[metric_idx]:
            axis.set_xlabel("grid column")
            axis.set_ylabel("grid row")
            axis.set_xticks(np.arange(base.shape[1]))
            axis.set_yticks(np.arange(base.shape[0]))
            axis.tick_params(labelsize=7)

    fig.suptitle(f"Per-node test metrics: {city} (matched seed={seed})", fontsize=18)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_node_csv(
    path: Path,
    rows: Iterable[tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> None:
    fields = ["city", "seed", "node_id", "row", "column"]
    for metric in METRICS:
        fields.extend((f"baseline_{metric}", f"adaptive_{metric}", f"delta_{metric}", f"status_{metric}"))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for city, seed, baseline, adaptive in rows:
            shape = baseline[METRICS[0]].shape
            for row, column in np.ndindex(shape):
                record: dict[str, str | int | float] = {
                    "city": city,
                    "seed": seed,
                    "node_id": row * shape[1] + column,
                    "row": row,
                    "column": column,
                }
                for metric in METRICS:
                    base = float(baseline[metric][row, column])
                    adapt = float(adaptive[metric][row, column])
                    delta = adapt - base
                    status = "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged"
                    record.update(
                        {
                            f"baseline_{metric}": base,
                            f"adaptive_{metric}": adapt,
                            f"delta_{metric}": delta,
                            f"status_{metric}": status,
                        }
                    )
                writer.writerow(record)


def write_summary_csv(
    path: Path,
    rows: Iterable[tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> None:
    fields = [
        "city",
        "seed",
        "metric",
        "baseline_global",
        "adaptive_global",
        "global_delta",
        "nodes_decreased",
        "nodes_increased",
        "nodes_unchanged",
        "mean_node_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for city, seed, baseline, adaptive in rows:
            for metric in METRICS:
                base = baseline[metric]
                adapt = adaptive[metric]
                delta = adapt - base
                if metric == "rmse":
                    baseline_global = float(np.sqrt(np.mean(base**2)))
                    adaptive_global = float(np.sqrt(np.mean(adapt**2)))
                else:
                    baseline_global = float(np.mean(base))
                    adaptive_global = float(np.mean(adapt))
                writer.writerow(
                    {
                        "city": city,
                        "seed": seed,
                        "metric": metric,
                        "baseline_global": baseline_global,
                        "adaptive_global": adaptive_global,
                        "global_delta": adaptive_global - baseline_global,
                        "nodes_decreased": int(np.count_nonzero(delta < 0)),
                        "nodes_increased": int(np.count_nonzero(delta > 0)),
                        "nodes_unchanged": int(np.count_nonzero(delta == 0)),
                        "mean_node_delta": float(np.mean(delta)),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and node-adaptive checkpoints per grid node.")
    parser.add_argument("--baseline-root", type=Path, default=Path("output/baseline-tmp/multirun/batch"))
    parser.add_argument(
        "--adaptive-root", type=Path, default=Path("output/baseline/multirun/node_adaptive_seed245")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("node_metric_comparison"))
    parser.add_argument("--batch-size", type=int, default=8)
    # 0 also works in restricted/container environments where multiprocessing sockets are unavailable.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    baseline_runs = discover_runs(args.baseline_root, project_root)
    adaptive_runs = discover_runs(args.adaptive_root, project_root)
    matched_keys = sorted(baseline_runs.keys() & adaptive_runs.keys())
    if not matched_keys:
        raise ValueError("No runs with matching (city, seed) were found.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Matched runs: {matched_keys}; device={device}; output={output_dir}")

    results: list[tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for city, seed in matched_keys:
        baseline_predictions, baseline_labels = predict(
            baseline_runs[(city, seed)], device, args.batch_size, args.num_workers
        )
        adaptive_predictions, adaptive_labels = predict(
            adaptive_runs[(city, seed)], device, args.batch_size, args.num_workers
        )
        if not np.array_equal(baseline_labels, adaptive_labels):
            raise ValueError(f"The test labels differ for city={city}, seed={seed}.")
        baseline_metrics = node_metrics(baseline_predictions, baseline_labels)
        adaptive_metrics = node_metrics(adaptive_predictions, adaptive_labels)
        results.append((city, seed, baseline_metrics, adaptive_metrics))
        plot_city(
            city,
            seed,
            baseline_metrics,
            adaptive_metrics,
            output_dir / f"{city}_seed{seed}_node_metrics.png",
        )

    write_node_csv(output_dir / "node_metrics.csv", results)
    write_summary_csv(output_dir / "summary.csv", results)
    print(f"Wrote {len(results)} plots and 2 CSV files to {output_dir}")


if __name__ == "__main__":
    main()
