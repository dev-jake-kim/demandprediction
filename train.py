from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

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

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_config = GridDemandConfig(
        H=train_ds.X,
        W=train_ds.Y,
        weather_csv_path=str(Path(cfg.dataset.weather_csv_path).resolve()),
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        **model_kwargs,
    )
    # Trainer는 __init__에서 seed를 설정하는데 그건 모델이 만들어진 뒤다 — 그대로 두면
    # 초기 가중치가 프로세스마다 달라져(torch 기본 seed가 OS 엔트로피) 같은 seed로도
    # 재현되지 않는다. 실제로 두 프로세스의 node_embed 값이 달랐다. 여기서 먼저 고정한다.
    set_seed(cfg.train.seed)
    model = GridDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    early_stopping_cfg = cfg.callbacks.early_stopping

    def build_trainer(stage_dir: str | None, learning_rate: float) -> Trainer:
        """stage마다 Trainer를 새로 만든다.

        optimizer / LR 스케줄러 / early stopping 상태가 stage 경계에서 초기화돼야 하고,
        stage 1의 optimizer에는 얼어 있는 delta가 들어있지 않기 때문이다. output_dir도
        분리해야 한다 — 같은 디렉터리를 쓰면 save_total_limit이 앞 stage의 체크포인트를 지운다.
        """
        # stage_dir=None은 2-stage를 쓰지 않는 경우 — 체크포인트 경로가 이 변경 이전과
        # 같아야 외부 스크립트(test.py 등)의 탐색이 깨지지 않는다.
        args = TrainingArguments(
            output_dir=output_dir if stage_dir is None else str(Path(output_dir) / stage_dir),
            **{**train_cfg, 'learning_rate': learning_rate},
        )
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[
                LoggingCallback(),
                MinEpochEarlyStoppingCallback(
                    min_epochs=early_stopping_cfg.min_epochs,
                    early_stopping_patience=early_stopping_cfg.early_stopping_patience,
                ),
            ],
        )

    def trainable_count(trainer: Trainer) -> int:
        return sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    # node_adaptive가 켜져 있을 때만 2-stage로 나눈다. 꺼져 있으면 예전과 동일한 단일 stage다.
    deltas = model._node_delta_parameters() if model.node_adaptive else ()

    if deltas:
        # --- stage 1: offset을 0으로 고정 -> baseline-weather와 동일한 학습 ---
        # node_adaptive를 끄면 forward가 nn.LSTM(cuDNN 융합) 경로를 타서 baseline-weather와
        # 비트 단위로 같아지고, 수동 LSTM 루프를 건너뛰어 더 빠르다. delta는 0인 채로 남는다.
        # config.node_adaptive는 True 그대로라 체크포인트에는 정확히 기록된다.
        model.node_adaptive = False
        for param in deltas:
            param.requires_grad_(False)

        stage1 = build_trainer('stage1', train_cfg['learning_rate'])
        logger.info(f"[stage1] offset 고정(baseline-weather 동일), 학습 파라미터 {trainable_count(stage1):,}개")
        stage1.train()
        stage1_metrics = stage1.evaluate(eval_dataset=test_ds, metric_key_prefix="stage1_test")
        logger.info(f"[stage1] Test metrics: {stage1_metrics}")
        # stage 1 Trainer가 optimizer와 모델 wrapper를 계속 붙들고 있지 않게 놓아준다.
        del stage1

        # --- stage 2: offset 제한 해제 후 finetuning ---
        # load_best_model_at_end=true라 model에는 stage 1의 best 가중치가 들어있다.
        model.node_adaptive = True
        for param in deltas:
            param.requires_grad_(True)

        trainer = build_trainer('stage2', float(cfg.stage2.learning_rate))
        logger.info(
            f"[stage2] offset 해제, lr={cfg.stage2.learning_rate}, "
            f"학습 파라미터 {trainable_count(trainer):,}개"
        )
        trainer.train()
    else:
        trainer = build_trainer(None, train_cfg['learning_rate'])
        trainer.train()

    trainer.save_model(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    if deltas:
        # 최종(best) 모델에서 offset이 실제로 학습됐는지. 비식별성 때문에 절대 norm은
        # 의미가 없어서 노드 평균을 뺀 중심화 norm을 함께 남긴다 — 그게 "노드마다 다른
        # 값을 배웠는가"를 말해준다.
        for name, param in zip(
            ('weight_ih', 'weight_hh', 'bias', 'out_weight'), trainer.model._node_delta_parameters()
        ):
            value = param.detach()
            centered = value - value.mean(dim=0, keepdim=True)
            logger.info(
                f"[stage2] node_delta_{name}: norm={value.norm():.4f} "
                f"centered_norm={centered.norm():.4f}"
            )

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
