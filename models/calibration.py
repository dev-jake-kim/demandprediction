from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class CalibrationBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    a: float
    b: float
    raw_rmse: float
    affine_rmse: float
    final_rmse: float
    clamped_count: int


@dataclass(frozen=True)
class CalibrationTable:
    bin_width: float
    bins: tuple[CalibrationBin, ...]

    @property
    def bin_indices(self) -> list[int]:
        return [row.bin_index for row in self.bins]

    @property
    def slopes(self) -> list[float]:
        return [row.a for row in self.bins]

    @property
    def intercepts(self) -> list[float]:
        return [row.b for row in self.bins]

    def apply_numpy(
        self,
        predictions: np.ndarray,
        *,
        clamp: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        return apply_calibration_numpy(
            predictions,
            bin_width=self.bin_width,
            bin_indices=np.asarray(self.bin_indices, dtype=np.int64),
            slopes=np.asarray(self.slopes, dtype=np.float64),
            intercepts=np.asarray(self.intercepts, dtype=np.float64),
            clamp=clamp,
        )

    def write_csv(self, output_path: str | Path) -> None:
        path = Path(output_path)
        fieldnames = list(asdict(self.bins[0]).keys()) if self.bins else [
            'bin_index',
            'lower_bound',
            'upper_bound',
            'count',
            'a',
            'b',
            'raw_rmse',
            'affine_rmse',
            'final_rmse',
            'clamped_count',
        ]
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in self.bins)


def _validate_bin_width(bin_width: float) -> float:
    width = float(bin_width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError(f'bin_width는 유한한 양수여야 함: got {bin_width}')
    return width


def _as_finite_flat_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f'{name}가 비어 있음')
    if not np.isfinite(array).all():
        raise ValueError(f'{name}에 NaN 또는 inf가 포함됨')
    return array


def prediction_bin_indices(predictions: np.ndarray, bin_width: float) -> np.ndarray:
    width = _validate_bin_width(bin_width)
    values = _as_finite_flat_array(predictions, 'predictions')
    scaled_values = values / width
    scaled = np.floor(np.nextafter(scaled_values, np.inf))
    int64_info = np.iinfo(np.int64)
    if scaled.min() < int64_info.min or scaled.max() > int64_info.max:
        raise ValueError('prediction/bin_width가 int64 bin index 범위를 벗어남')
    return scaled.astype(np.int64)


def _rmse(predictions: np.ndarray, labels: np.ndarray) -> float:
    diff = labels - predictions
    return float(np.sqrt(np.mean(diff * diff)))


def _fit_affine_rmse(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    x_mean = float(predictions.mean())
    y_mean = float(labels.mean())
    centered_x = predictions - x_mean

    design = np.column_stack((centered_x, np.ones_like(centered_x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, labels, rcond=None)
    if rank < 2:
        return 0.0, y_mean

    slope = float(coefficients[0])
    intercept = float(coefficients[1] - slope * x_mean)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError('RMSE affine fitting 결과가 유한하지 않음')
    return slope, intercept


def fit_rmse_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    bin_width: float = 0.1,
) -> CalibrationTable:
    width = _validate_bin_width(bin_width)
    flat_predictions = _as_finite_flat_array(predictions, 'predictions')
    flat_labels = _as_finite_flat_array(labels, 'labels')
    if flat_predictions.shape != flat_labels.shape:
        raise ValueError(
            f'predictions와 labels shape이 다름: {flat_predictions.shape} != {flat_labels.shape}'
        )
    if np.any(flat_labels < 0):
        raise ValueError('0 clamp의 RMSE 보장을 위해 labels는 0 이상이어야 함')

    sample_bin_indices = prediction_bin_indices(flat_predictions, width)
    rows: list[CalibrationBin] = []
    for bin_index in np.unique(sample_bin_indices):
        mask = sample_bin_indices == bin_index
        bin_predictions = flat_predictions[mask]
        bin_labels = flat_labels[mask]
        slope, intercept = _fit_affine_rmse(bin_predictions, bin_labels)

        affine_predictions = slope * bin_predictions + intercept
        final_predictions = np.maximum(affine_predictions, 0.0)
        raw_rmse = _rmse(bin_predictions, bin_labels)
        affine_rmse = _rmse(affine_predictions, bin_labels)
        final_rmse = _rmse(final_predictions, bin_labels)

        tolerance = max(1e-12, raw_rmse * 1e-10)
        if affine_rmse > raw_rmse + tolerance or final_rmse > affine_rmse + tolerance:
            raise RuntimeError(
                f'bin {int(bin_index)} RMSE 최적화 검증 실패: '
                f'raw={raw_rmse}, affine={affine_rmse}, final={final_rmse}'
            )

        rows.append(
            CalibrationBin(
                bin_index=int(bin_index),
                lower_bound=float(bin_index * width),
                upper_bound=float((bin_index + 1) * width),
                count=int(mask.sum()),
                a=slope,
                b=intercept,
                raw_rmse=raw_rmse,
                affine_rmse=affine_rmse,
                final_rmse=final_rmse,
                clamped_count=int(np.count_nonzero(affine_predictions < 0)),
            )
        )

    return CalibrationTable(bin_width=width, bins=tuple(rows))


def _validate_lookup_arrays(
    bin_indices: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> None:
    if not (bin_indices.ndim == slopes.ndim == intercepts.ndim == 1):
        raise ValueError('calibration lookup arrays는 모두 1차원이어야 함')
    if not (len(bin_indices) == len(slopes) == len(intercepts)):
        raise ValueError('calibration lookup arrays 길이가 서로 다름')
    if len(bin_indices) > 1 and np.any(bin_indices[1:] <= bin_indices[:-1]):
        raise ValueError('calibration bin_indices는 중복 없이 오름차순이어야 함')
    if not np.isfinite(slopes).all() or not np.isfinite(intercepts).all():
        raise ValueError('calibration coefficient에 NaN 또는 inf가 포함됨')


def apply_calibration_numpy(
    predictions: np.ndarray,
    *,
    bin_width: float,
    bin_indices: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    clamp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    original_shape = np.asarray(predictions).shape
    flat_predictions = _as_finite_flat_array(predictions, 'predictions')
    lookup_indices = np.asarray(bin_indices, dtype=np.int64).reshape(-1)
    lookup_slopes = np.asarray(slopes, dtype=np.float64).reshape(-1)
    lookup_intercepts = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    _validate_lookup_arrays(lookup_indices, lookup_slopes, lookup_intercepts)

    corrected = flat_predictions.copy()
    matched = np.zeros(flat_predictions.shape, dtype=np.bool_)
    if lookup_indices.size:
        sample_indices = prediction_bin_indices(flat_predictions, bin_width)
        positions = np.searchsorted(lookup_indices, sample_indices)
        in_range = positions < lookup_indices.size
        in_range_locations = np.flatnonzero(in_range)
        exact_locations = in_range_locations[
            lookup_indices[positions[in_range]] == sample_indices[in_range]
        ]
        matched[exact_locations] = True
        matched_positions = positions[exact_locations]
        corrected[exact_locations] = (
            lookup_slopes[matched_positions] * flat_predictions[exact_locations]
            + lookup_intercepts[matched_positions]
        )

    if clamp:
        corrected = np.maximum(corrected, 0.0)
    return corrected.reshape(original_shape), matched.reshape(original_shape)


def apply_calibration_torch(
    predictions: torch.Tensor,
    *,
    bin_width: float,
    bin_indices: torch.Tensor,
    slopes: torch.Tensor,
    intercepts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_bin_width(bin_width)
    if not (bin_indices.ndim == slopes.ndim == intercepts.ndim == 1):
        raise ValueError('calibration lookup tensors는 모두 1차원이어야 함')
    if not (bin_indices.numel() == slopes.numel() == intercepts.numel()):
        raise ValueError('calibration lookup tensor 길이가 서로 다름')

    flat_predictions = predictions.reshape(-1)
    corrected = flat_predictions.clone()
    matched = torch.zeros_like(flat_predictions, dtype=torch.bool)
    if bin_indices.numel():
        scaled_values = flat_predictions.to(torch.float64) / bin_width
        stabilized_values = torch.nextafter(
            scaled_values,
            torch.full_like(scaled_values, torch.inf),
        )
        sample_indices = torch.floor(stabilized_values).to(torch.int64)
        positions = torch.searchsorted(bin_indices, sample_indices)
        in_range = positions < bin_indices.numel()
        safe_positions = positions.clamp(max=bin_indices.numel() - 1)
        matched = in_range & (bin_indices[safe_positions] == sample_indices)
        matched_positions = safe_positions[matched]
        corrected[matched] = (
            slopes[matched_positions].to(predictions.dtype) * flat_predictions[matched]
            + intercepts[matched_positions].to(predictions.dtype)
        )

    return corrected.clamp_min(0).reshape_as(predictions), matched.reshape_as(predictions)
