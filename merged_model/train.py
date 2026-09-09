"""Train and validate the unified model with one end-to-end objective.

The objective is selectable: ``combined`` (this repository's shared
``CombinedLoss``, so merged_model is comparable to every other branch) or
``mae`` (the original raw-scale L1 this model was first trained with).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# ablation 방식: 모듈을 지우지 않고, 그 모듈이 결과로 이어지는 텐서만 0으로 바꾼다.
# 구조/파라미터 수/텐서 shape가 전부 보존되므로 측정된 차이가 "그 모듈의 정보" 때문임이
# 분리된다. 예외는 no-branch-attn — 어텐션은 출력 텐서가 아니라 선택 메커니즘이라
# 0-치환이 성립하지 않아(0으로 만들면 세 브랜치가 통째로 사라짐) 균등 가중으로 대체한다.
ABLATION_MODE = "zero"

# 이름 붙인 ablation 조합. 값은 UnifiedDemandModel의 use_* 플래그를 덮어쓴다.
# "full"은 아무것도 끄지 않은 기본 모델 — 기존 *_mae_seed*.json 런과 동일한 설정이다.
ABLATIONS: dict[str, dict[str, object]] = {
    "full": {},
    "no-ir": {"use_retrieval": False},
    "no-periodic": {"use_daily": False, "use_weekly": False},
    "no-daily": {"use_daily": False},
    "no-weekly": {"use_weekly": False},
    "no-weather": {"use_weather": False},
    "no-calendar": {"use_calendar": False},
    "no-extra": {"use_weather": False, "use_calendar": False},
    "no-branch-attn": {"use_branch_attention": False},
    # (2a+1)^2 로컬 창에서 중앙(자기 노드)만 남기고 이웃 공간 정보를 끈다.
    # 검색기 질의는 원래 크롭을 그대로 쓴다 — 인코더의 공간 정보만 분리해서 재려는 것이다.
    "no-neighbors": {"use_neighbors": False},
    # 임베딩 방식 변형: 날씨를 LSTM concat 대신 ir-weather식으로 CLS에 더한다.
    "weather-cls-add": {"weather_injection": "cls_add"},
    # 최종 예측의 softplus(x)=log(1+exp(x)) 양수 보정을 없애고 raw 선형 출력을 그대로 쓴다.
    # 음수 예측이 나올 수 있다 — loss/지표 계산 자체는 부호에 무관해 문제없이 돈다.
    "no-softplus": {"use_softplus": False},
}

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
    parser.add_argument("--weather-path", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--loss-type",
        choices=("combined", "mae"),
        default=None,
        help="combined: 저장소 공용 CombinedLoss(다른 모델과 동일) | mae: 원래의 raw 스케일 L1",
    )
    parser.add_argument(
        "--ablation",
        choices=tuple(ABLATIONS),
        default="full",
        help="끌 모듈 조합의 이름. full은 아무것도 끄지 않은 기본 모델",
    )
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
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_relative_plus1 = 0.0
    total_relative_nonzero = 0.0
    total_count = 0
    nonzero_count = 0
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

        total_loss += output["loss_sum"].item()
        target_values = batch["target"]
        error = output["prediction"] - target_values
        absolute_error = error.detach().abs()
        total_abs += absolute_error.sum().item()
        total_sq += error.detach().square().sum().item()
        # MAPE(+1): 저장소 공용 지표(models/metrics.py의 compute_regression_metrics)와 같은 식.
        # 수요 0인 셀이 많아 분모에 +1 스무딩을 넣은 변형이라 표준 MAPE가 아니다.
        total_relative_plus1 += (absolute_error / (target_values.abs() + 1.0)).sum().item()
        # MAPE(0제외): 실제 수요가 0인 셀을 분자/분모 양쪽에서 빼고 계산한 것.
        nonzero = target_values != 0
        if bool(nonzero.any()):
            total_relative_nonzero += (
                absolute_error[nonzero] / target_values[nonzero].abs()
            ).sum().item()
            nonzero_count += int(nonzero.sum().item())
        total_count += error.numel()
        batches += 1

    if total_count == 0:
        raise RuntimeError("No batches were processed")
    return {
        # 학습에 실제로 쓰인 목적함수의 원소 평균. 조기 종료/최적 체크포인트 선택 기준이며,
        # loss_type=mae일 때는 아래 "mae"와 정확히 같은 값이 된다.
        "loss": total_loss / total_count,
        "mae": total_abs / total_count,
        "rmse": float(np.sqrt(total_sq / total_count)),
        "mape_plus1": total_relative_plus1 / total_count * 100.0,
        "mape_excl_zero": (
            total_relative_nonzero / nonzero_count * 100.0 if nonzero_count else float("nan")
        ),
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
    weather_mean: Sequence[float],
    weather_std: Sequence[float],
    loss_type: str,
    ablation: str = "full",
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
        weather_mean=weather_mean,
        weather_std=weather_std,
        weekday_dim=int(model_cfg.get("weekday_dim", 7)),
        hour_dim=int(model_cfg.get("hour_dim", 5)),
        loss_type=loss_type,
        loss_gamma=float(cfg.get("training", {}).get("loss_gamma", 1.0)),
        loss_eps=float(cfg.get("training", {}).get("loss_eps", 0.5)),
        **ABLATIONS[ablation],
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
    weather_path = args.weather_path if args.weather_path is not None else dataset_cfg.get("weather_path")
    if weather_path is None:
        raise ValueError(
            f"datasets.{args.dataset}.weather_path가 설정에 없음 — --weather-path로 주거나 config.yaml에 추가할 것"
        )
    weather_path = Path(weather_path).expanduser()
    if not weather_path.is_absolute():
        weather_path = Path.cwd() / weather_path
    dataset_kwargs = build_dataset_kwargs(cfg)
    dataset_kwargs["weather_csv_path"] = weather_path
    if args.train_ratio is not None:
        dataset_kwargs["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        dataset_kwargs["val_ratio"] = args.val_ratio
    train_set = UnifiedDemandDataset(data_path, "train", **dataset_kwargs)
    val_set = UnifiedDemandDataset(data_path, "val", **dataset_kwargs)
    test_set = UnifiedDemandDataset(data_path, "test", **dataset_kwargs)

    # 날씨 정규화 통계는 train 구간에서만 계산한다(시간 리크 방지) — ir-weather의 train.py와 동일 패턴.
    # 적설처럼 train 내내 값이 고정(분산 0)인 피처가 있어 std에 하한을 둔다(porto 적설이 실제로 그렇다).
    train_weather = train_set.weather[train_set.time_step : train_set.train_end]
    weather_mean = train_weather.mean(axis=0)
    weather_std = train_weather.std(axis=0).clip(min=1e-6)

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
    loss_type = args.loss_type or str(train_cfg.get("loss_type", "combined"))
    model = build_model(
        cfg,
        height=train_set.height,
        width=train_set.width,
        data_path=data_path,
        train_end=train_set.train_end,
        retrieval_scope=retrieval_scope,
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        loss_type=loss_type,
        ablation=args.ablation,
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
                "objective": loss_type,
                "ablation": args.ablation,
                "seed": seed,
                "weather_path": str(weather_path),
                "weather_mean": weather_mean.tolist(),
                "weather_std": weather_std.tolist(),
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

        # 선택 기준은 학습에 쓴 목적함수 자체 — 저장소 공용 하네스의
        # metric_for_best_model="loss"와 같은 규약이다. loss_type=mae면 이전과 동일하게 MAE다.
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
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
        "weather_path": str(weather_path),
        "weather_mean": weather_mean.tolist(),
        "weather_std": weather_std.tolist(),
        "device": str(device),
        "retrieval_scope": retrieval_scope,
        "objective": loss_type,
        "ablation": args.ablation,
        "ablation_mode": ABLATION_MODE,
        "ablation_flags": ABLATIONS[args.ablation],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": test_metrics,
        "history": history,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False))

    output_path = args.output
    if output_path is None:
        # runs/ 자체가 output/ 아래로 옮겨졌다 — output/은 저장소 전체가 이미 gitignore하는
        # 경로라 checkpoint(.pt)처럼 무거운 산출물을 소스 트리 밖에 둘 수 있다.
        output_path = Path(__file__).parents[1] / "output" / "merged_model" / "runs" / f"{args.dataset}_result.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # best_state를 디스크에도 남긴다 — 이게 없으면 나중에 지표를 하나 더 재려고 할 때
    # 전체 재학습이 필요해진다(실제로 MAPE를 추가할 때 그 비용을 치렀다).
    # 재현에 필요한 설정도 함께 저장해 체크포인트만으로 모델을 되살릴 수 있게 한다.
    checkpoint_path = output_path.with_suffix(".pt")
    torch.save(
        {
            "state_dict": best_state,
            "dataset": args.dataset,
            "data_path": str(data_path),
            "weather_path": str(weather_path),
            "weather_mean": weather_mean.tolist(),
            "weather_std": weather_std.tolist(),
            "model_config": cfg.get("model", {}),
            "dataset_kwargs": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in dataset_kwargs.items()
            },
            "retrieval_scope": retrieval_scope,
            "loss_type": loss_type,
            "ablation": args.ablation,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    print(f"result_json={output_path}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
