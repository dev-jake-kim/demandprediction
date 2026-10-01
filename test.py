"""저장된 MergedDemandModel 체크포인트를 데이터 split에서 평가하는 스크립트.

사용법과 평가 계약은 ``docs/SPEC.md`` §9를 참고한다.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandModel, compute_merged_metrics

logger = logging.getLogger(__name__)

# forward가 받지 않는 키(labels는 따로 모은다).
INPUT_KEYS = (
    'demand_history',
    'daily_demand',
    'daily_mask',
    'weekly_demand',
    'weekly_mask',
    'weather',
    'hour_of_day',
    'day_of_week',
)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def evaluate(model: MergedDemandModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='evaluate'):
            inputs = {key: batch[key].to(device) for key in INPUT_KEYS}
            out = model(**inputs)
            all_preds.append(out['logits'].float().cpu().numpy())
            all_labels.append(batch['labels'].numpy())

    return compute_merged_metrics(np.concatenate(all_preds), np.concatenate(all_labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'checkpoint_path', type=str, help='train.py가 save_pretrained로 저장한 디렉터리'
    )
    parser.add_argument('--city', type=str, required=True, choices=('ulsan', 'porto'))
    parser.add_argument('--npy_path', type=str, default=None, help='예: data/raw/ulsan_temporal_grid.npy')
    parser.add_argument('--weather_csv_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--time_step', type=int, default=24)
    parser.add_argument('--daily_period', type=int, default=24)
    parser.add_argument('--daily_lags', type=int, default=6)
    parser.add_argument('--weekly_period', type=int, default=168)
    parser.add_argument('--weekly_lags', type=int, default=4)
    parser.add_argument('--lag_radius', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.80)
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_file', type=str, default=None, help='기본값: <checkpoint_path>/test.log')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    log_path = Path(args.log_file) if args.log_file else checkpoint_path / 'test.log'
    setup_logging(log_path)
    logger.info(f'Logging to: {log_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MergedDemandModel.from_pretrained(checkpoint_path).to(device)

    data_path = resolve_dataset_path(args.city, args.npy_path)
    dataset = UnifiedDemandDataset(
        data_path,
        args.split,
        weather_csv_path=args.weather_csv_path,
        time_step=args.time_step,
        daily_period=args.daily_period,
        daily_lags=args.daily_lags,
        weekly_period=args.weekly_period,
        weekly_lags=args.weekly_lags,
        lag_radius=args.lag_radius,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    logger.info(
        f'Evaluating {args.city}/{args.split} on {len(dataset):,} samples '
        f'(checkpoint={checkpoint_path})'
    )
    metrics = evaluate(model, loader, device)
    for name, value in metrics.items():
        logger.info(f'{name}: {value:.4f}')


if __name__ == '__main__':
    main()
