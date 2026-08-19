from __future__ import annotations

import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainerCallback, TrainingArguments

from dataset_frame import GridDemandDataset
from models import GridDemandConfig, GridDemandModel, compute_regression_metrics

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Trainer의 기본 콜백은 step/eval 지표를 print()로만 찍어서 Hydra가 관리하는
    로그 파일(${hydra.run.dir}/train.log)에 안 남는다. logging 모듈을 거치도록 감싼다."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is not None:
            logger.info(logs)


def build_datasets(cfg: DictConfig) -> tuple[GridDemandDataset, GridDemandDataset, GridDemandDataset]:
    npy_path = cfg.dataset.npy_path
    time_step = cfg.dataset.time_step

    full_ds = GridDemandDataset(npy_path, time_step=time_step)
    T = full_ds.T
    split1 = int(T * cfg.dataset.train_ratio)
    split2 = int(T * (cfg.dataset.train_ratio + cfg.dataset.val_ratio))

    train_ds = GridDemandDataset(npy_path, time_step=time_step, t_end=split1)
    val_ds = GridDemandDataset(npy_path, time_step=time_step, t_start=split1, t_end=split2)
    test_ds = GridDemandDataset(npy_path, time_step=time_step, t_start=split2)

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

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_config = GridDemandConfig(H=train_ds.X, W=train_ds.Y, **model_kwargs)
    model = GridDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        **OmegaConf.to_container(cfg.train, resolve=True),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()],
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
