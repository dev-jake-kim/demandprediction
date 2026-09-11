"""merged 모델(로컬 히스토리 + daily/weekly 주기 + 인과적 검색 + 브랜치 어텐션) 학습 스크립트.

원본 ``merged_model/train.py``의 수동 학습 루프를 이 저장소 공용 관례(Hydra + HF ``Trainer``)로
옮긴 것이다. 학습 하이퍼파라미터(gradient clipping 5.0, 고정 LR, epochs 2000, patience 20,
num_workers 0, best = val loss 최소, torch_compile 없음)는 원본과 1:1로 맞춰 두었다.

도시마다 최적 하이퍼파라미터가 달라 루트 config를 도시별로 분리했다 — 기본값은
``configs/config_ulsan.yaml``, porto는 ``--config-name config_porto``로 명시한다::

    python train.py                                    # ulsan
    python train.py --config-name config_porto          # porto
    python train.py model.use_retrieval=false ablation=no-ir   # ablation (ulsan)

이 브랜치(tmp)에는 다른 모델 구현이 없다(models/config.py, models/modeling.py 부재 —
models/__init__.py 참고) — train.py는 이 모델 하나만 학습하는 단일 진입점이다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandConfig, MergedDemandModel, compute_merged_metrics

logger = logging.getLogger(__name__)

# PyTorch의 memory-efficient SDPA 백엔드는 dropout용 seed/offset을 (배치*시퀀스들을 펼친) 유효
# batch 하나당 인덱스 하나로 추적하는데, 그 인덱스가 65535(uint16)를 넘으면
# "Efficient attention cannot produce valid seed and offset outputs"로 죽는다.
# models/merged/modules/history.py의 LocalHistoryEncoder는 (batch, time_step, H*W) 전체를
# (batch*time_step*H*W, 1+neighbors, d_model)로 펼쳐 넣으므로, 노드 수가 많은 도시(예: porto
# 19*20=380)는 batch_size=8만 돼도 8*24*380=72,960으로 한계를 넘는다(ulsan 14*12=168은
# 32,256이라 안 넘음). 이 백엔드를 끄면 우회는 되지만 fallback(math) 백엔드가 어텐션 행렬을
# 그대로 만들어 메모리를 훨씬 더 써서 d_model이 크면 오히려 OOM이 난다 — 그래서 백엔드를
# 건드리지 않고, 이 한계를 넘지 않도록 배치 크기를 낮추는 쪽을 택한다(아래 main() 참고).
SDPA_BATCH_LIMIT = 65535

# ablation 방식: 모듈을 지우지 않고, 그 모듈이 결과로 이어지는 텐서만 0으로 바꾼다.
# 구조/파라미터 수/텐서 shape가 전부 보존되므로 측정된 차이가 "그 모듈의 정보" 때문임이
# 분리된다. 예외는 no-branch-attn — 어텐션은 출력 텐서가 아니라 선택 메커니즘이라
# 0-치환이 성립하지 않아 균등 가중으로 대체한다.
ABLATION_MODE = 'zero'


class LoggingCallback(TrainerCallback):
    """Trainer의 기본 콜백은 step/eval 지표를 print()로만 찍어서 Hydra가 관리하는
    로그 파일(${hydra.run.dir}/train.log)에 안 남는다. logging 모듈을 거치도록 감싼다."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is not None:
            logger.info(logs)


class MinEpochEarlyStoppingCallback(EarlyStoppingCallback):
    """최소 학습 epoch 이후부터 patience를 세는 early stopping callback.

    ``min_epochs=0``이면 원본 merged_model의 무조건 patience와 동치다.
    """

    def __init__(
        self,
        min_epochs: int,
        early_stopping_patience: int,
        early_stopping_threshold: float | None = 0.0,
    ) -> None:
        if min_epochs < 0:
            raise ValueError(f'min_epochs는 0 이상이어야 함: got {min_epochs}')
        if early_stopping_patience < 1:
            raise ValueError(
                f'early_stopping_patience는 1 이상이어야 함: got {early_stopping_patience}'
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
        callback_state['args']['min_epochs'] = self.min_epochs
        return callback_state


def build_dataset_kwargs(cfg: DictConfig) -> dict:
    """UnifiedDemandDataset 생성 인자. 원본 merged_model/train.py와 같은 값 집합."""

    weather_csv_path = Path(cfg.dataset.weather_csv_path).expanduser()
    if not weather_csv_path.is_absolute():
        weather_csv_path = Path.cwd() / weather_csv_path
    return {
        'weather_csv_path': weather_csv_path.resolve(),
        'time_step': int(cfg.dataset.time_step),
        'daily_period': int(cfg.data.daily_period),
        'daily_lags': int(cfg.data.daily_lags),
        'weekly_period': int(cfg.data.weekly_period),
        'weekly_lags': int(cfg.data.weekly_lags),
        'lag_radius': int(cfg.data.lag_radius),
        'train_ratio': float(cfg.dataset.train_ratio),
        'val_ratio': float(cfg.dataset.val_ratio),
    }


def build_datasets(
    cfg: DictConfig,
) -> tuple[Path, dict, UnifiedDemandDataset, UnifiedDemandDataset, UnifiedDemandDataset]:
    data_path = resolve_dataset_path(cfg.dataset.city, cfg.dataset.npy_path)
    dataset_kwargs = build_dataset_kwargs(cfg)
    train_ds = UnifiedDemandDataset(data_path, 'train', **dataset_kwargs)
    val_ds = UnifiedDemandDataset(data_path, 'val', **dataset_kwargs)
    test_ds = UnifiedDemandDataset(data_path, 'test', **dataset_kwargs)
    logger.info(
        f'[{cfg.dataset.city}] T={train_ds.total_steps}, H={train_ds.height}, W={train_ds.width} | '
        f'train_end={train_ds.train_end}, val_end={train_ds.val_end} | '
        f'train {len(train_ds):,} / val {len(val_ds):,} / test {len(test_ds):,} samples'
    )
    return data_path, dataset_kwargs, train_ds, val_ds, test_ds


def compute_metrics(eval_pred) -> dict[str, float]:
    return compute_merged_metrics(eval_pred.predictions, eval_pred.label_ids)


def _summarize_history(log_history: list[dict]) -> tuple[list[dict], int, float]:
    """Trainer의 log_history를 원본 result JSON의 ``history`` 형태로 압축한다.

    원본은 epoch마다 {"epoch", "train": {...}, "val": {...}}를 남겼다. Trainer는 train 로그와
    eval 로그를 따로 남기므로 epoch을 키로 합친다.
    """

    per_epoch: dict[int, dict] = {}
    for entry in log_history:
        epoch = entry.get('epoch')
        if epoch is None:
            continue
        record = per_epoch.setdefault(int(round(epoch)), {'epoch': int(round(epoch))})
        if 'loss' in entry:
            record['train'] = {'loss': entry['loss']}
        if 'eval_loss' in entry:
            record['val'] = {
                key[len('eval_') :]: value
                for key, value in entry.items()
                if key.startswith('eval_') and isinstance(value, (int, float))
            }
    history = [per_epoch[key] for key in sorted(per_epoch)]

    best_epoch, best_val = 0, float('inf')
    for record in history:
        val = record.get('val', {}).get('loss')
        if val is not None and val < best_val:
            best_val, best_epoch = float(val), record['epoch']
    return history, best_epoch, best_val


@hydra.main(config_path='configs', config_name='config_ulsan', version_base=None)
def main(cfg: DictConfig) -> None:
    data_path, dataset_kwargs, train_ds, val_ds, test_ds = build_datasets(cfg)

    # 날씨 정규화 통계는 train 구간에서만 계산한다(시간 리크 방지).
    # 적설처럼 train 내내 값이 고정(분산 0)인 피처가 있어 std에 하한을 둔다(porto 적설이 실제로 그렇다).
    train_weather = train_ds.weather[train_ds.time_step : train_ds.train_end]
    weather_mean = train_weather.mean(axis=0)
    weather_std = train_weather.std(axis=0).clip(min=1e-6)
    logger.info(f'weather_mean={weather_mean.tolist()}, weather_std={weather_std.tolist()}')

    # smoke 테스트용. null이면 전체를 쓴다(원본 --max-batches에 대응).
    limit = cfg.get('limit_samples')
    train_set, val_set, eval_test_set = train_ds, val_ds, test_ds
    if limit:
        limit = int(limit)
        train_set = Subset(train_ds, range(min(limit, len(train_ds))))
        val_set = Subset(val_ds, range(min(limit, len(val_ds))))
        eval_test_set = Subset(test_ds, range(min(limit, len(test_ds))))
        logger.info(f'[smoke] limit_samples={limit} — 각 split의 앞부분만 사용한다')

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_config = MergedDemandConfig(
        height=train_ds.height,
        width=train_ds.width,
        time_step=train_ds.time_step,
        retrieval_grid_path=str(data_path),
        retrieval_train_end=train_ds.train_end,
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        **model_kwargs,
    )
    # Trainer는 __init__에서 seed를 설정하는데 그건 모델이 만들어진 뒤다 — 그대로 두면
    # 초기 가중치가 프로세스마다 달라져 같은 seed로도 재현되지 않는다. 여기서 먼저 고정한다.
    set_seed(cfg.train.seed)
    model = MergedDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    early_stopping_cfg = cfg.callbacks.early_stopping

    # SDPA_BATCH_LIMIT 주석 참고 — 노드 수가 많은 도시에서 batch_size를 자동으로 낮춘다.
    # train/eval 배치 크기를 같이 낮춰야 train 중간의 eval도 안전하다.
    nodes = train_ds.height * train_ds.width
    per_sample = train_ds.time_step * nodes
    max_batch = max(1, SDPA_BATCH_LIMIT // per_sample)
    for key in ('per_device_train_batch_size', 'per_device_eval_batch_size'):
        if train_cfg[key] > max_batch:
            logger.warning(
                f'{key}={train_cfg[key]}는 time_step({train_ds.time_step}) * nodes({nodes})'
                f'={per_sample}와 곱하면 SDPA_BATCH_LIMIT({SDPA_BATCH_LIMIT})을 넘어'
                f'memory-efficient attention이 죽는다 -> {max_batch}로 낮춘다'
            )
            train_cfg[key] = max_batch

    args = TrainingArguments(output_dir=output_dir, **train_cfg)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks=[
            LoggingCallback(),
            MinEpochEarlyStoppingCallback(
                min_epochs=early_stopping_cfg.min_epochs,
                early_stopping_patience=early_stopping_cfg.early_stopping_patience,
            ),
        ],
    )
    trainer.train()

    trainer.save_model(output_dir)
    logger.info(f'Model saved to: {output_dir}')

    test_metrics = trainer.evaluate(eval_dataset=eval_test_set, metric_key_prefix='test')
    logger.info(f'Test metrics: {test_metrics}')

    history, best_epoch, best_val = _summarize_history(trainer.state.log_history)
    result = {
        'dataset': cfg.dataset.city,
        'data_path': str(data_path),
        'weather_path': str(dataset_kwargs['weather_csv_path']),
        'weather_mean': weather_mean.tolist(),
        'weather_std': weather_std.tolist(),
        'device': str(trainer.args.device),
        'retrieval_scope': cfg.model.retrieval_scope,
        'objective': cfg.model.loss_type,
        'ablation': cfg.ablation,
        'ablation_mode': ABLATION_MODE,
        # 실제로 적용된 스위치. 라벨(cfg.ablation)이 아니라 이 값이 근거다.
        'ablation_flags': {
            key: model_kwargs[key]
            for key in (
                'use_daily',
                'use_weekly',
                'use_retrieval',
                'use_weather',
                'weather_injection',
                'use_calendar',
                'use_branch_attention',
                'use_neighbors',
                'use_softplus',
            )
        },
        'seed': int(cfg.train.seed),
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        # 원본 스키마의 "checkpoint"는 .pt 파일이었다. 이제 save_pretrained 디렉터리다.
        'checkpoint': str(output_dir),
        'test': {
            'loss': test_metrics.get('test_loss'),
            'mae': test_metrics.get('test_mae'),
            'rmse': test_metrics.get('test_rmse'),
            'mape_plus1': test_metrics.get('test_mape_plus1'),
            'mape_excl_zero': test_metrics.get('test_mape_excl_zero'),
        },
        'history': history,
    }

    run_json = cfg.get('run_json')
    if run_json is None:
        run_json = (
            Path('output')
            / cfg.project_name
            / 'runs'
            / f'{cfg.dataset.city}_{cfg.model.loss_type}_{cfg.ablation}_seed{cfg.train.seed}.json'
        )
    run_json = Path(run_json).expanduser()
    if not run_json.is_absolute():
        run_json = Path(hydra.utils.get_original_cwd()) / run_json
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f'result_json={run_json}')
    print(f'result_json={run_json}')
    print(f'checkpoint={output_dir}')


if __name__ == '__main__':
    main()
