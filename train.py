from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

from dataset_frame import GridDemandDataset
from models import GridDemandConfig, GridDemandModel, compute_regression_metrics


def build_datasets(cfg: DictConfig) -> tuple[GridDemandDataset, GridDemandDataset]:
    """train/val만 만든다. test는 test.py가 checkpoint_path 하나로 독립 실행하며 담당."""
    npy_path = cfg.dataset.npy_path
    time_step = cfg.dataset.time_step

    full_ds = GridDemandDataset(npy_path, time_step=time_step)
    T = full_ds.T
    split1 = int(T * cfg.dataset.train_ratio)
    split2 = int(T * (cfg.dataset.train_ratio + cfg.dataset.val_ratio))

    train_ds = GridDemandDataset(npy_path, time_step=time_step, t_end=split1)
    val_ds = GridDemandDataset(npy_path, time_step=time_step, t_start=split1, t_end=split2)

    print(
        f"[{cfg.dataset.city}] T={T}, H={full_ds.X}, W={full_ds.Y} | "
        f"train t=[{time_step},{split1}) ({len(train_ds):,} samples), "
        f"val t=[{split1},{split2}) ({len(val_ds):,} samples), "
        f"test t=[{split2},{T}) 는 test.py에서 별도 평가"
    )
    return train_ds, val_ds


def compute_metrics(eval_pred) -> dict[str, float]:
    return compute_regression_metrics(eval_pred.predictions, eval_pred.label_ids)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_ds, val_ds = build_datasets(cfg)

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
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to: {output_dir}")
    print(
        f"Test 평가는 별도로 실행: python test.py {output_dir} "
        f"--npy_path {cfg.dataset.npy_path} --time_step {cfg.dataset.time_step} --t_start <test split 시작>"
    )


if __name__ == "__main__":
    main()
