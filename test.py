from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_frame import GridDemandDataset
from models import ADFormerModel, compute_regression_metrics

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def evaluate(model: ADFormerModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            demands = batch["demands"].to(device)
            hour_of_day = batch["hour_of_day"].to(device)
            day_of_week = batch["day_of_week"].to(device)

            out = model(demands=demands, hour_of_day=hour_of_day, day_of_week=day_of_week)
            all_preds.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return compute_regression_metrics(preds, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="checkpoint_path만 주면 학습 프로세스와 무관하게 단독 실행되는 평가 스크립트")
    parser.add_argument("checkpoint_path", type=str, help="model.save_pretrained()로 저장된 체크포인트 디렉토리")
    parser.add_argument("--npy_path", type=str, required=True, help="예: data/raw/ulsan_temporal_grid.npy")
    parser.add_argument("--time_step", type=int, default=24)
    parser.add_argument("--t_start", type=int, default=None, help="평가에 사용할 target 시간 구간 시작 (예: test split 경계)")
    parser.add_argument("--t_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_file", type=str, default=None, help="기본값: <checkpoint_path>/test.log")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    log_path = Path(args.log_file) if args.log_file else checkpoint_path / "test.log"
    setup_logging(log_path)
    logger.info(f"Logging to: {log_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADFormerModel.from_pretrained(checkpoint_path).to(device)

    test_ds = GridDemandDataset(
        args.npy_path,
        time_step=args.time_step,
        t_start=args.t_start,
        t_end=args.t_end,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"Evaluating on {len(test_ds):,} samples (checkpoint={checkpoint_path})")
    metrics = evaluate(model, loader, device)
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
