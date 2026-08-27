from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

from dataset_frame import GridDemandDataset
from models import GridDemandConfig, GridDemandModel, compute_regression_metrics

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Trainer의 기본 콜백은 step/eval 지표를 print()로만 찍어서 Hydra가 관리하는
    로그 파일(${hydra.run.dir}/train.log)에 안 남는다. logging 모듈을 거치도록 감싼다."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is not None:
            logger.info(logs)


class MinEpochEarlyStoppingCallback(EarlyStoppingCallback):
    """최소 학습 epoch 이후부터 patience를 세는 early stopping callback."""

    def __init__(
        self,
        min_epochs: int,
        early_stopping_patience: int,
        early_stopping_threshold: float | None = 0.0,
    ) -> None:
        if min_epochs < 0:
            raise ValueError(f"min_epochs는 0 이상이어야 함: got {min_epochs}")
        if early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience는 1 이상이어야 함: "
                f"got {early_stopping_patience}"
            )
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch is None or state.epoch < self.min_epochs:
            return control
        return super().on_evaluate(args, state, control, metrics, **kwargs)

    def state(self) -> dict:
        callback_state = super().state()
        callback_state["args"]["min_epochs"] = self.min_epochs
        return callback_state


def build_datasets(cfg: DictConfig) -> tuple[GridDemandDataset, GridDemandDataset, GridDemandDataset]:
    npy_path = cfg.dataset.npy_path
    time_step = cfg.dataset.time_step
    weather_csv_path = cfg.dataset.weather_csv_path

    full_ds = GridDemandDataset(npy_path, time_step=time_step, weather_csv_path=weather_csv_path)
    T = full_ds.T
    split1 = int(T * cfg.dataset.train_ratio)
    split2 = int(T * (cfg.dataset.train_ratio + cfg.dataset.val_ratio))

    train_ds = GridDemandDataset(npy_path, time_step=time_step, weather_csv_path=weather_csv_path, t_end=split1)
    val_ds = GridDemandDataset(
        npy_path, time_step=time_step, weather_csv_path=weather_csv_path, t_start=split1, t_end=split2
    )
    test_ds = GridDemandDataset(npy_path, time_step=time_step, weather_csv_path=weather_csv_path, t_start=split2)

    logger.info(
        f"[{cfg.dataset.city}] T={T}, H={full_ds.X}, W={full_ds.Y} | "
        f"train t=[{time_step},{split1}) ({len(train_ds):,} samples), "
        f"val t=[{split1},{split2}) ({len(val_ds):,} samples), "
        f"test t=[{split2},{T}) ({len(test_ds):,} samples)"
    )
    return train_ds, val_ds, test_ds


def compute_metrics(eval_pred) -> dict[str, float]:
    return compute_regression_metrics(eval_pred.predictions, eval_pred.label_ids)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_ds, val_ds, test_ds = build_datasets(cfg)

    # train split에서만 날씨 정규화 통계를 계산(시간 리크 방지) — STResnet의 demand_mean/std
    # 계산과 동일 패턴.
    train_weather = train_ds.weather[train_ds.time_step:train_ds.t_end]
    weather_mean = train_weather.mean(axis=0)
    # 적설처럼 train split 내내 값이 고정(분산 0)인 피처가 있을 수 있음(예: porto는 적설이 항상 0) ->
    # 모델의 (weather - mean) / std 정규화에서 0으로 나누지 않도록 최소값을 둔다.
    weather_std = train_weather.std(axis=0).clip(min=1e-6)
    logger.info(f"weather_mean={weather_mean.tolist()}, weather_std={weather_std.tolist()}")

    # train split 수요만으로 상위 ir_top_pct% 노드를 골라 검색기 앙상블 적용 대상을 정한다
    # (leakage 방지 위해 val/test 구간은 안 봄 — weather_mean/std 계산과 동일한 이유).
    N = train_ds.X * train_ds.Y
    train_grid = train_ds.grid[:train_ds.t_end]  # (t_end, X, Y) — train 샘플이 실제로 보는 구간까지만
    node_avg_demand = train_grid.reshape(train_grid.shape[0], N).mean(axis=0)  # (N,)
    ir_top_pct = cfg.model.ir_top_pct
    if not 0.0 <= ir_top_pct <= 1.0:
        raise ValueError(f"ir_top_pct는 [0,1] 범위여야 함: got {ir_top_pct}")
    n_top = round(ir_top_pct * N)
    top_node_ids = np.argsort(-node_avg_demand)[:n_top]
    ir_node_mask = np.zeros(N, dtype=bool)
    ir_node_mask[top_node_ids] = True
    logger.info(
        f"ir_node_mask: train split 수요 상위 {ir_top_pct * 100:.0f}% = {n_top}/{N}개 노드에만 "
        f"검색기 앙상블 적용"
    )

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_config = GridDemandConfig(
        H=train_ds.X,
        W=train_ds.Y,
        time_step=train_ds.time_step,
        npy_path=str(Path(cfg.dataset.npy_path).resolve()),
        weather_csv_path=str(Path(cfg.dataset.weather_csv_path).resolve()),
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        ir_node_mask=ir_node_mask.tolist(),
        **model_kwargs,
    )
    model = GridDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        **OmegaConf.to_container(cfg.train, resolve=True),
    )

    early_stopping_cfg = cfg.callbacks.early_stopping
    callbacks = [
        LoggingCallback(),
        MinEpochEarlyStoppingCallback(
            min_epochs=early_stopping_cfg.min_epochs,
            early_stopping_patience=early_stopping_cfg.early_stopping_patience,
        ),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
