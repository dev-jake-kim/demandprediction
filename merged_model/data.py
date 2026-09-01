"""Temporal-grid dataset and causal daily/weekly lag construction.

The dataset deliberately works with raw demand values.  The model owns the
log1p transformations for the neural, daily, and weekly branches so that the
retrieval branch can remain on the raw scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_FILES = {
    "ulsan": ("ulsan", "ulsan_temporal_grid.npy"),
    "porto": ("prtu", "porto_temporal_grid.npy"),
}


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path not in result and path.is_file():
            result.append(path)
    return result


def resolve_dataset_path(dataset: str, explicit_path: str | Path | None = None) -> Path:
    """Resolve a temporal-grid file without falling back to legacy demand.npy.

    The first location is the target ``comparison_models/data`` layout.  The
    second location keeps this new implementation runnable in the current
    workspace, where temporal grids are still stored in the previous research
    directory.  Once the files are copied to the target layout, the first
    location wins automatically.
    """

    dataset = dataset.lower()
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(DATASET_FILES)}")
    directory, filename = DATASET_FILES[dataset]

    module_root = Path(__file__).resolve().parents[1]  # comparison_models
    project_root = module_root.parent
    candidates: list[Path] = []
    if explicit_path is not None:
        requested = Path(explicit_path).expanduser()
        candidates.append(requested if requested.is_absolute() else (Path.cwd() / requested))
        candidates.append(requested if requested.is_absolute() else (module_root / requested))
    candidates.extend(
        [
            module_root / "data" / directory / filename,
            module_root / "data" / ("porto" if dataset == "porto" else directory) / filename,
            project_root / "research_700x700" / "comparison_models" / "data" / directory / filename,
        ]
    )
    existing = _unique_existing(candidates)
    if existing:
        return existing[0]

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No temporal-grid file found for {dataset!r}. Searched:\n  {searched}\n"
        "The merged model intentionally does not fall back to legacy demand.npy."
    )


def _chronological_lags(period: int, count: int, radius: int) -> np.ndarray:
    """Return source lags from oldest to newest in chronological order."""

    if period <= 0 or count <= 0 or radius < 0:
        raise ValueError("period and count must be positive; radius must be non-negative")
    lags = {
        base + offset
        for base in (period * i for i in range(1, count + 1))
        for offset in range(-radius, radius + 1)
        if base + offset > 0
    }
    return np.asarray(sorted(lags, reverse=True), dtype=np.int64)


@dataclass(frozen=True)
class SplitBounds:
    train_end: int
    val_end: int
    total: int


class UnifiedDemandDataset(Dataset):
    """One sample per target time from a ``[T, H, W]`` temporal grid.

    ``daily_mask`` and ``weekly_mask`` use the convention required by the
    model: ``True`` means that the lag is invalid.  ``sample_idx`` is always
    the absolute index in the original temporal grid, never a split-local
    index.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: Literal["train", "val", "test"] = "train",
        *,
        time_step: int = 24,
        daily_period: int = 24,
        daily_lags: int = 6,
        weekly_period: int = 24 * 7,
        weekly_lags: int = 4,
        lag_radius: int = 0,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio and val_ratio must leave a non-empty test split")
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        self.data_path = Path(data_path).expanduser().resolve()
        grid = np.load(self.data_path, mmap_mode="r")
        if grid.ndim != 3:
            raise ValueError(f"Expected temporal grid [T,H,W], got {grid.shape} from {self.data_path}")
        if not np.issubdtype(grid.dtype, np.number):
            raise TypeError(f"Temporal grid must be numeric, got {grid.dtype}")
        if not np.isfinite(grid).all():
            raise ValueError(f"Temporal grid contains non-finite values: {self.data_path}")
        if np.min(grid) < 0:
            raise ValueError("Demand must be non-negative for log1p and Softplus output")

        self.grid = np.asarray(grid, dtype=np.float32)
        self.total_steps, self.height, self.width = self.grid.shape
        self.num_nodes = self.height * self.width
        self.time_step = int(time_step)
        self.daily_lag_values = _chronological_lags(daily_period, daily_lags, lag_radius)
        self.weekly_lag_values = _chronological_lags(weekly_period, weekly_lags, lag_radius)

        train_end = int(self.total_steps * train_ratio)
        val_end = int(self.total_steps * (train_ratio + val_ratio))
        self.bounds = SplitBounds(train_end=train_end, val_end=val_end, total=self.total_steps)
        split_starts = {
            "train": self.time_step,
            "val": max(self.time_step, train_end),
            "test": max(self.time_step, val_end),
        }
        split_ends = {"train": train_end, "val": val_end, "test": self.total_steps}
        if split not in split_starts:
            raise ValueError(f"Unsupported split {split!r}")
        start, end = split_starts[split], split_ends[split]
        if end <= start:
            raise ValueError(f"Split {split!r} is empty: [{start}, {end})")
        self.split = split
        self.indices = np.arange(start, end, dtype=np.int64)

        self.daily_values, self.daily_mask = self._make_lag_table(self.daily_lag_values)
        self.weekly_values, self.weekly_mask = self._make_lag_table(self.weekly_lag_values)

    def _make_lag_table(self, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_times = np.arange(self.total_steps, dtype=np.int64)[:, None] - lags[None, :]
        valid = source_times >= 0
        safe_times = np.clip(source_times, 0, self.total_steps - 1)
        # [T, L, H, W] -> [T, L, N, 1].
        values = self.grid[safe_times].reshape(self.total_steps, len(lags), self.num_nodes)
        values = np.where(valid[..., None], values, 0.0).astype(np.float32, copy=False)
        values = values[..., None]
        return np.ascontiguousarray(values), np.ascontiguousarray(~valid)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_time = int(self.indices[item])
        history = self.grid[target_time - self.time_step : target_time]
        target = self.grid[target_time]
        return {
            "demand_history": torch.from_numpy(np.array(history, dtype=np.float32, copy=True)),
            "daily_demand": torch.from_numpy(self.daily_values[target_time]),
            "daily_mask": torch.from_numpy(self.daily_mask[target_time]),
            "weekly_demand": torch.from_numpy(self.weekly_values[target_time]),
            "weekly_mask": torch.from_numpy(self.weekly_mask[target_time]),
            "target": torch.from_numpy(np.array(target, dtype=np.float32, copy=True)),
            "sample_idx": torch.tensor(target_time, dtype=torch.long),
        }

    @property
    def train_end(self) -> int:
        return self.bounds.train_end

    @property
    def val_end(self) -> int:
        return self.bounds.val_end


__all__ = ["UnifiedDemandDataset", "resolve_dataset_path"]
