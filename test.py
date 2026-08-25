from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_frame import GridDemandDataset
from models import GridDemandModel, compute_regression_metrics

logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def evaluate(model: GridDemandModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            demands = batch["demands"].to(device)
            sample_idx = batch["sample_idx"].to(device)
            weather = batch["weather"].to(device)
            hour_of_day = batch["hour_of_day"].to(device)
            day_of_week = batch["day_of_week"].to(device)

            out = model(
                demands=demands,
                sample_idx=sample_idx,
                weather=weather,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
            )
            all_preds.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return compute_regression_metrics(preds, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="checkpoint_path만 주면 학습 프로세스와 무관하게 단독 실행되는 평가 스크립트")
    parser.add_argument("checkpoint_path", type=str, help="model.save_pretrained()로 저장된 체크포인트 디렉토리")
    parser.add_argument("--npy_path", type=str, required=True, help="예: data/raw/ulsan_temporal_grid.npy")
    parser.add_argument(
        "--weather_csv_path", type=str, required=True, help="예: data/raw/ulsan_meteorological_data.csv"
    )
    parser.add_argument("--time_step", type=int, default=24)
    parser.add_argument("--t_start", type=int, default=None, help="평가에 사용할 target 시간 구간 시작 (예: test split 경계)")
    parser.add_argument("--t_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)  # 이제 한 샘플이 H*W개 노드를 전부 예측
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_file", type=str, default=None, help="기본값: <checkpoint_path>/test.log")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    log_path = Path(args.log_file) if args.log_file else checkpoint_path / "test.log"
    setup_logging(log_path)
    logger.info(f"Logging to: {log_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GridDemandModel.from_pretrained(checkpoint_path)

    # 체크포인트의 config.npy_path/time_step(검색 DB를 만드는 데 쓰임)과 --npy_path/--time_step
    # (평가 데이터셋을 만드는 데 쓰임)이 어긋나면 검색 브랜치가 엉뚱한 grid를 검색하거나(조용히 틀린
    # 결과) time_step이 다르면 einsum에서 shape 에러가 남 — 미리 명확한 에러로 막는다.
    resolved_npy_path = str(Path(args.npy_path).resolve())
    if model.config.npy_path != resolved_npy_path:
        raise ValueError(
            f"체크포인트 config.npy_path({model.config.npy_path})와 --npy_path({resolved_npy_path})가 "
            f"다름 — 검색 DB와 평가 데이터셋이 같은 grid를 가리켜야 함"
        )
    if model.config.time_step != args.time_step:
        raise ValueError(
            f"체크포인트 config.time_step({model.config.time_step})과 --time_step({args.time_step})이 다름"
        )

    # retrieval_keys/values/norms는 persistent=False 버퍼라 from_pretrained의 meta-device
    # fast-init 후 자동 복원되지 않음(값이 깨져 있음) — npy로부터 명시적으로 다시 만들어야 함.
    # 반드시 .to(device) 전에(= CPU 상태에서) 재구성한다: 그렇지 않으면 GPU로 옮겨진 깨진 buffer +
    # 재구성 중간 텐서 + 새 buffer가 동시에 GPU에 존재하는 메모리 스파이크가 생김(특히 porto에서
    # OOM 위험).
    model.build_retrieval_db()
    model = model.to(device)

    test_ds = GridDemandDataset(
        args.npy_path,
        time_step=args.time_step,
        weather_csv_path=args.weather_csv_path,
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
