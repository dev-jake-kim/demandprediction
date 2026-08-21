from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_frame import GridDemandDataset
from models import GridDemandModel, compute_regression_metrics, fit_rmse_calibration

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def default_output_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f'{checkpoint_path.name}-calibrated')


def validate_output_path(checkpoint_path: Path, output_path: Path) -> None:
    source = checkpoint_path.resolve()
    destination = output_path.resolve()
    if source == destination:
        raise ValueError('원본 checkpoint와 calibration output 경로가 같을 수 없음')
    if destination.is_relative_to(source):
        raise ValueError('calibration output을 원본 checkpoint 내부에 만들 수 없음')
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f'calibration output이 디렉터리가 아님: {output_path}')
        if any(output_path.iterdir()):
            raise ValueError(f'calibration output 디렉터리가 비어 있지 않음: {output_path}')


def resolve_train_end(
    npy_path: str | Path,
    *,
    time_step: int,
    train_ratio: float | None,
    t_end: int | None,
) -> tuple[int, int]:
    if time_step < 1:
        raise ValueError(f'time_step은 1 이상이어야 함: got {time_step}')

    grid = np.load(npy_path, mmap_mode='r')
    if grid.ndim != 3:
        raise ValueError(f'temporal grid는 3차원이어야 함: got shape={grid.shape}')
    total_timesteps = int(grid.shape[0])

    if t_end is None:
        ratio = 0.7 if train_ratio is None else float(train_ratio)
        if not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError(f'train_ratio는 0보다 크고 1보다 작아야 함: got {ratio}')
        resolved_end = int(total_timesteps * ratio)
    else:
        resolved_end = int(t_end)

    if resolved_end > total_timesteps:
        raise ValueError(f't_end가 전체 timestep보다 큼: {resolved_end} > {total_timesteps}')
    if resolved_end <= time_step:
        raise ValueError(
            f'유효한 train target 구간이 없음: [{time_step}, {resolved_end})'
        )
    return resolved_end, total_timesteps


def collect_raw_predictions(
    model: GridDemandModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with model.calibration_disabled(), torch.no_grad():
        for batch in tqdm(loader, desc='collect train predictions'):
            demands = batch['demands'].to(device)
            output = model(demands=demands, apply_calibration=False)
            all_predictions.append(output['logits'].cpu().numpy())
            all_labels.append(batch['labels'].numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='기존 checkpoint의 train 예측으로 구간별 RMSE affine calibration을 fitting',
    )
    parser.add_argument('checkpoint_path', type=Path)
    parser.add_argument('--npy_path', type=Path, required=True)
    parser.add_argument('--time_step', type=int, default=24)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--train_ratio', type=float, default=None)
    split_group.add_argument('--t_end', type=int, default=None)
    parser.add_argument('--bin_width', type=float, default=0.1)
    parser.add_argument('--output_path', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    checkpoint_path = args.checkpoint_path
    if not checkpoint_path.is_dir():
        raise ValueError(f'checkpoint 디렉터리를 찾을 수 없음: {checkpoint_path}')
    output_path = args.output_path or default_output_path(checkpoint_path)
    validate_output_path(checkpoint_path, output_path)
    if not math.isfinite(args.bin_width) or args.bin_width <= 0:
        raise ValueError(f'bin_width는 유한한 양수여야 함: got {args.bin_width}')

    t_end, total_timesteps = resolve_train_end(
        args.npy_path,
        time_step=args.time_step,
        train_ratio=args.train_ratio,
        t_end=args.t_end,
    )
    logger.info(
        'Calibration train targets: t=[%d,%d), T=%d, samples=%d',
        args.time_step,
        t_end,
        total_timesteps,
        t_end - args.time_step,
    )

    device = resolve_device(args.device)
    model = GridDemandModel.from_pretrained(checkpoint_path).to(device)
    if model.has_calibration:
        raise ValueError('이미 calibration table이 포함된 checkpoint는 다시 calibration할 수 없음')

    train_dataset = GridDemandDataset(
        args.npy_path,
        time_step=args.time_step,
        t_end=t_end,
    )
    if (train_dataset.X, train_dataset.Y) != (model.H, model.W):
        raise ValueError(
            'checkpoint와 temporal grid의 공간 shape이 다름: '
            f'model={(model.H, model.W)}, data={(train_dataset.X, train_dataset.Y)}'
        )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    raw_predictions, labels = collect_raw_predictions(model, loader, device)

    table = fit_rmse_calibration(raw_predictions, labels, bin_width=args.bin_width)
    affine_predictions, affine_matched = table.apply_numpy(raw_predictions, clamp=False)
    final_predictions, final_matched = table.apply_numpy(raw_predictions, clamp=True)
    if not affine_matched.all() or not final_matched.all():
        raise RuntimeError('train prediction 중 fitted calibration bin과 매칭되지 않은 값이 있음')

    raw_metrics = compute_regression_metrics(raw_predictions, labels)
    affine_metrics = compute_regression_metrics(affine_predictions, labels)
    final_metrics = compute_regression_metrics(final_predictions, labels)
    tolerance = max(1e-10, raw_metrics['rmse'] * 1e-8)
    if final_metrics['rmse'] > raw_metrics['rmse'] + tolerance:
        raise RuntimeError(
            'aggregate train RMSE가 calibration 후 증가함: '
            f"raw={raw_metrics['rmse']}, final={final_metrics['rmse']}"
        )

    logger.info('Raw train metrics: %s', raw_metrics)
    logger.info('Affine train metrics: %s', affine_metrics)
    logger.info('Clamped final train metrics: %s', final_metrics)
    logger.info('Fitted calibration bins: %d', len(table.bins))

    model.install_calibration(table)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    table.write_csv(output_path / 'calibration_table.csv')

    summary = {
        'source_checkpoint': str(checkpoint_path.resolve()),
        'npy_path': str(args.npy_path.resolve()),
        'time_step': args.time_step,
        't_start': args.time_step,
        't_end': t_end,
        'total_timesteps': total_timesteps,
        'target_samples': len(train_dataset),
        'scalar_samples': int(labels.size),
        'bin_width': table.bin_width,
        'fitted_bins': len(table.bins),
        'raw_metrics': raw_metrics,
        'affine_metrics': affine_metrics,
        'final_metrics': final_metrics,
    }
    with (output_path / 'calibration_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    logger.info('Calibrated checkpoint saved to: %s', output_path)


if __name__ == '__main__':
    main()
