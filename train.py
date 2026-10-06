"""MergedDemandModel 학습 스크립트.

사용법과 학습 계약은 ``docs/SPEC.md`` §8을 참고한다.

사용 예::

    python train.py --config-name config_<city> "description='실험 설명'" [seed=...] [model.<key>=...]
    python train.py --config-name config_porto "description='실험 설명'"
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandConfig, MergedDemandModel, compute_merged_metrics

logger = logging.getLogger(__name__)

SDPA_BATCH_LIMIT = 65535

ABLATION_MODE = 'zero'


class LoggingCallback(TrainerCallback):
    """Trainer 로그를 모듈 logger로 전달한다."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is not None:
            logger.info(logs)


class MinEpochEarlyStoppingCallback(EarlyStoppingCallback):
    """설정된 최소 epoch 이후부터 patience를 세는 콜백."""

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


class WarmupCosineAnnealingLR(LRScheduler):
    """epoch 단위 warmup/cosine 스케줄러와 최저 학습률."""

    def __init__(
        self, optimizer, T_max, warmup_t=0, warmup_lr_init=1e-5, eta_min=0,
        last_epoch=-1,
    ) -> None:
        self.T_max = T_max
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_t:
            warmup_lr = (
                self.warmup_lr_init
                + (self.base_lrs[0] - self.warmup_lr_init) * self.last_epoch / self.warmup_t
            )
            return [warmup_lr for _ in self.base_lrs]
        epoch_in_cosine_phase = self.last_epoch - self.warmup_t
        if epoch_in_cosine_phase >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        cosine_lr = (
            self.eta_min + (self.base_lrs[0] - self.eta_min)
            * (1 + math.cos(math.pi * epoch_in_cosine_phase / self.T_max)) / 2
        )
        return [cosine_lr for _ in self.base_lrs]


class PerEpochWarmupCosineAnnealingLR(WarmupCosineAnnealingLR):
    """HF Trainer의 optimizer step을 epoch 단위 scheduler step으로 묶는다."""

    def __init__(self, optimizer, steps_per_epoch: int, **kwargs) -> None:
        if steps_per_epoch < 1:
            raise ValueError(f'steps_per_epoch는 1 이상이어야 함: got {steps_per_epoch}')
        self.steps_per_epoch = steps_per_epoch
        self._pending_steps = 0
        self._initialized = False
        super().__init__(optimizer, **kwargs)
        self._initialized = True

    def step(self, *args, **kwargs):
        if not self._initialized:
            return super().step(*args, **kwargs)
        self._pending_steps += 1
        if self._pending_steps < self.steps_per_epoch:
            return None
        self._pending_steps = 0
        return super().step(*args, **kwargs)


class WarmupCosineTrainer(Trainer):
    """Trainer의 weight-decay 파라미터 그룹은 유지하고 스케줄러만 바꾼다."""

    def __init__(self, *args, schedule: dict, steps_per_epoch: int, **kwargs) -> None:
        self._schedule = schedule
        self._steps_per_epoch = steps_per_epoch
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = PerEpochWarmupCosineAnnealingLR(
                optimizer if optimizer is not None else self.optimizer,
                steps_per_epoch=self._steps_per_epoch,
                T_max=int(self._schedule['cosine_epochs']),
                warmup_t=int(self._schedule['warmup_epochs']),
                warmup_lr_init=float(self._schedule['warmup_lr_init']),
                eta_min=float(self._schedule['eta_min']),
            )
        return self.lr_scheduler


def build_dataset_kwargs(cfg: DictConfig) -> dict:
    """UnifiedDemandDataset 생성 인자를 만든다."""

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
        'context_features': list(cfg.data.get('context_features') or []),
        'start_time': cfg.dataset.get('start_time'),
        'holidays': list(cfg.dataset.get('holidays') or []),
        # 공연·축제 등 이벤트 표(선택). 상대경로는 실행 디렉터리 기준.
        'event_csv_path': (
            str((Path.cwd() / str(cfg.data.event_csv)).resolve()) if cfg.data.get('event_csv') else None
        ),
        'weather_lead': int(cfg.data.get('weather_lead', 0) or 0),
        # 노드·시각별 배차 상태 격자(선택, [T,H,W,C]).
        'aux_grid_path': (
            str((Path.cwd() / str(cfg.data.aux_grid)).resolve()) if cfg.data.get('aux_grid') else None
        ),
    }


def select_functional_neighbors(
    train_ds: UnifiedDemandDataset, k: int, exclude_radius: int, residual: bool = False
) -> list[list[int]]:
    """노드마다 train 구간 수요 시계열 상관이 가장 큰 노드 ``k``개(자기 자신과 격자 거리
    ``exclude_radius`` 이내, 즉 masked Transformer가 이미 보는 노드는 제외). 시간 리크를 막으려고
    ``[time_step, train_end)``만 쓴다. ``residual``이면 노드별 요일×시각 평균을 뺀 잔차의 상관을 써서
    모든 노드가 공유하는 일·주 주기 대신 실제로 함께 움직이는 노드를 고른다."""

    start, end = train_ds.time_step, train_ds.train_end
    series = np.array(train_ds.grid[start:end].reshape(-1, train_ds.num_nodes), dtype=np.float64)
    if residual:
        slot = train_ds.day_of_week_table[start:end] * 24 + train_ds.hour_table[start:end]
        for value in np.unique(slot):
            at = slot == value
            series[at] -= series[at].mean(axis=0)
    centered = series - series.mean(axis=0)
    norms = np.sqrt((centered ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = (centered.T @ centered) / np.outer(norms, norms)
    corr = np.nan_to_num(corr, nan=-np.inf)
    rows, cols = np.divmod(np.arange(train_ds.num_nodes), train_ds.width)
    near = (np.abs(rows[:, None] - rows[None, :]) <= exclude_radius) & (
        np.abs(cols[:, None] - cols[None, :]) <= exclude_radius
    )
    corr[near] = -np.inf
    return np.argsort(-corr, axis=1)[:, :k].tolist()


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


def make_rmse_mape_metrics(rmse_weight: float):
    """전체 validation 예측에서 목적함수를 계산하는 metric callback을 만든다."""

    def compute_stage2_metrics(eval_pred) -> dict[str, float]:
        metrics = compute_merged_metrics(eval_pred.predictions, eval_pred.label_ids)
        metrics['rmse_mape_objective'] = rmse_weight * metrics['rmse'] + metrics['mape_plus1']
        return metrics

    return compute_stage2_metrics


def select_node_adaptive_indices(train_ds: UnifiedDemandDataset, min_demand: float) -> list[int]:
    """train 구간 평균 수요가 ``min_demand``를 넘는 노드를 선택한다.

    선택은 시간 리크를 막기 위해 ``[time_step, train_end)``만 사용한다.
    """

    window = train_ds.grid[train_ds.time_step : train_ds.train_end]
    node_mean = window.reshape(len(window), -1).mean(axis=0)
    indices = [int(node) for node in (node_mean > min_demand).nonzero()[0]]
    if not indices:
        raise ValueError(
            f'node_adaptive_min_demand={min_demand}를 넘는 노드가 없다 — 임계값을 낮춰야 한다 '
            f'(노드 평균 수요 최댓값 {float(node_mean.max()):.3f})'
        )
    return indices


def compute_node_descriptors(train_ds: UnifiedDemandDataset, indices: list[int]) -> list[list[float]]:
    """basis ΔW용 노드 기술자 φ[n] (34차원, 적응 노드 사이 표준화). ``[time_step, train_end)``만 사용.

    시각별(24)·요일별(7, 시작일 기준 상대 요일) 평균 / 노드 평균, log1p 평균, 0 비율, 변동계수.
    """

    start, end = train_ds.time_step, train_ds.train_end
    values = train_ds.grid[start:end].reshape(end - start, -1)[:, indices].astype(np.float64)
    hours = np.asarray(train_ds.hour_table[start:end]).astype(int)
    days = np.asarray(train_ds.day_of_week_table[start:end]).astype(int)
    mean = values.mean(axis=0)
    safe_mean = np.maximum(mean, 1e-6)
    hour_profile = np.stack([values[hours == h].mean(axis=0) for h in range(24)], axis=1) / safe_mean[:, None]
    day_profile = np.stack([values[days == d].mean(axis=0) for d in range(7)], axis=1) / safe_mean[:, None]
    features = np.column_stack([
        hour_profile, day_profile, np.log1p(mean), (values == 0).mean(axis=0), values.std(axis=0) / safe_mean,
    ])
    features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-6)
    return features.tolist()


def compute_node_mix(
    train_ds: UnifiedDemandDataset, lam_low: float, lam_high: float, tau: float, width: float, scaler: bool
) -> tuple[list[float], list[float], list[float], list[float]]:
    """node_mix 손실의 노드별 (λ_n, 1/B_n, 1/(2A_n), r_n) — train 구간 ``[time_step, train_end)``만 쓴다 (docs/SPEC.md §7)."""

    lo, hi = train_ds.time_step, train_ds.train_end
    y = train_ds.grid[lo:hi].reshape(hi - lo, -1).astype(np.float64)
    mu = y.mean(axis=0)
    lam = lam_high + (lam_low - lam_high) / (1.0 + np.exp(-(np.log(tau) - np.log(mu + 0.01)) / width))
    if not scaler:
        ones = np.ones_like(mu)
        return lam.tolist(), ones.tolist(), ones.tolist(), ones.tolist()
    # MAPE 항 기울기 크기 E[1/(y+1)] = B_n, MSE 항 기울기 크기 2·E|y − 기준| = 2A_n (기준: 노드×요일×시각 평균).
    how = (train_ds.day_of_week_table[lo:hi] * 24 + train_ds.hour_table[lo:hi]).astype(np.int64)
    ref = np.zeros_like(y)
    for k in np.unique(how):
        rows = how == k
        ref[rows] = y[rows].mean(axis=0)
    a = np.maximum(np.abs(y - ref).mean(axis=0), 0.05)
    b = (1.0 / (y + 1.0)).mean(axis=0)
    lam_bar = float(lam.mean())
    weight = lam_bar * b + (1.0 - lam_bar) * 2.0 * a
    return lam.tolist(), (1.0 / b).tolist(), (1.0 / (2.0 * a)).tolist(), weight.tolist()


def _summarize_history(
    log_history: list[dict], best_metric_key: str = 'loss'
) -> tuple[list[dict], int, float]:
    """Trainer 로그를 epoch별로 합치고 최적 validation 지표를 반환한다."""

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
        val = record.get('val', {}).get(best_metric_key)
        if val is not None and val < best_val:
            best_val, best_epoch = float(val), record['epoch']
    return history, best_epoch, best_val


def load_stage1_with_new_nodes(init_from: str, config: MergedDemandConfig) -> MergedDemandModel:
    """공유 stage-1 가중치를 불러오고 노드 ΔW를 0으로 초기화한다."""
    source_config = MergedDemandConfig.from_pretrained(init_from)
    # 체크포인트를 다시 지정하기 전에 저장된 목적함수와 구조를 검증한다.
    ignored = {
        'node_adaptive_indices', 'node_adaptive_min_demand', 'node_adaptive', 'node_adaptive_mode',
        'node_delta_rank', 'node_delta_descriptor', 'node_descriptors', 'node_adaptive_scope', 'node_delta_dynamic',
        # stage 2 손실 전용 설정(node_mix 벡터·하이퍼파라미터)은 stage 1 가중치와 무관하다.
        'node_mix_lambda', 'node_mix_mape_scale', 'node_mix_mse_scale', 'node_mix_weight', 'node_mix_lambda_low',
        'node_mix_lambda_high', 'node_mix_tau', 'node_mix_width', 'node_mix_scaler',
        'architectures', 'dtype',
    }
    saved = source_config.to_dict()
    requested = config.to_dict()
    differing = [
        key for key in saved.keys() | requested.keys()
        if key not in ignored and saved.get(key) != requested.get(key)
    ]
    if differing:
        raise ValueError(
            f'stage2.init_from={init_from}은 현재 stage 1 설정과 다름: '
            f'{[(key, saved.get(key), requested.get(key)) for key in sorted(differing)]}'
        )

    source = MergedDemandModel.from_pretrained(init_from)
    nonzero = [
        name for name, param in source.named_parameters()
        if 'node_delta_' in name and torch.count_nonzero(param.detach()).item()
    ]
    if nonzero:
        raise ValueError(
            f'stage2.init_from은 ΔW=0인 stage1 체크포인트여야 함 — 0이 아닌 항목: {nonzero}'
        )

    model = MergedDemandModel(config)
    source_weights = {
        name: value for name, value in source.state_dict().items()
        if 'node_delta_' not in name
    }
    expected = {
        name for name in model.state_dict() if 'node_delta_' in name
    }
    loaded = model.load_state_dict(source_weights, strict=False)
    if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
        raise ValueError(
            f'stage2.init_from 공유 가중치 불일치: '
            f'missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}'
        )
    logger.info(
        f'[stage2] stage1 공유 가중치 {len(source_weights)}개 복원; '
        f'ΔW 노드 {len(source_config.node_adaptive_indices or [])}'
        f' -> {len(config.node_adaptive_indices or [])} (0으로 재초기화)'
    )
    return model


def git_commit() -> str:
    """저장소 HEAD 해시. 추적 파일에 커밋되지 않은 변경이 있으면 ``(dirty)``를 붙인다."""

    root = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return f'{head} (dirty)' if dirty else head


@hydra.main(config_path='configs', config_name='config_ulsan', version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f'[Description] {cfg.description}')
    logger.info(f'[Commit] {git_commit()}')
    data_path, dataset_kwargs, train_ds, val_ds, test_ds = build_datasets(cfg)

    # train 구간 날씨 극값으로 min-max 정규화한다. 강수 최댓값은 snow_scale=1 기준으로 고정한다.
    train_weather = train_ds.weather[train_ds.time_step : train_ds.train_end]
    weather_stats = {
        'temperature_min': float(train_weather[:, 0].min()),
        'temperature_max': float(train_weather[:, 0].max()),
        'precipitation_max': float((train_weather[:, 1] + train_weather[:, 2]).max()),
    }
    logger.info(f'weather_stats={weather_stats}')

    limit = cfg.get('limit_samples')
    train_set, val_set, eval_test_set = train_ds, val_ds, test_ds
    if limit:
        limit = int(limit)
        train_set = Subset(train_ds, range(min(limit, len(train_ds))))
        val_set = Subset(val_ds, range(min(limit, len(val_ds))))
        eval_test_set = Subset(test_ds, range(min(limit, len(test_ds))))
        logger.info(f'[smoke] limit_samples={limit} — 각 split의 앞부분만 사용한다')

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    # train 구간에서만 노드별 ΔW 대상을 고르고, None이면 해당 파라미터를 만들지 않는다.
    node_adaptive = bool(model_kwargs.get('node_adaptive', False))
    node_adaptive_indices = None
    if node_adaptive:
        node_adaptive_indices = select_node_adaptive_indices(
            train_ds, float(model_kwargs['node_adaptive_min_demand'])
        )
        total_nodes = train_ds.height * train_ds.width
        logger.info(
            f'[node_adaptive] train 구간 평균 수요 > {model_kwargs["node_adaptive_min_demand"]}인 '
            f'노드 {len(node_adaptive_indices)}/{total_nodes} '
            f'({len(node_adaptive_indices) / total_nodes * 100:.1f}%)에만 ΔW를 준다'
        )
        if model_kwargs.get('node_delta_descriptor'):
            if model_kwargs.get('node_adaptive_mode') != 'basis':
                raise ValueError('node_delta_descriptor는 node_adaptive_mode=basis에서만 쓴다')
            model_kwargs['node_descriptors'] = compute_node_descriptors(train_ds, node_adaptive_indices)

    # node_mix 손실(stage 1 또는 stage 2)을 쓰면 노드별 λ·정규화·scaler를 train 구간에서 계산한다.
    node_mix = None
    if 'node_mix' in (str(model_kwargs.get('loss_type')), str(cfg.stage2.loss)):
        node_mix = compute_node_mix(
            train_ds,
            float(model_kwargs['node_mix_lambda_low']),
            float(model_kwargs['node_mix_lambda_high']),
            float(model_kwargs['node_mix_tau']),
            float(model_kwargs['node_mix_width']),
            bool(model_kwargs['node_mix_scaler']),
        )
        lam_v, _, _, weight_v = (np.asarray(v) for v in node_mix)
        logger.info(
            f'[node_mix] λ_n {lam_v.min():.2f}~{lam_v.max():.2f} (평균 {lam_v.mean():.2f}), '
            f'scaler r_n {weight_v.min():.2f}~{weight_v.max():.2f}, scaler={model_kwargs["node_mix_scaler"]}'
        )
    # 기능적 이웃: masked Transformer 수용 영역(local_radius × transformer_layers) 밖에서 train 구간
    # 상관이 가장 큰 노드 k개. k=0이면 끔.
    functional_k = int(model_kwargs.get('functional_neighbor_k', 0))
    functional_neighbors = None
    if functional_k > 0:
        exclude = int(model_kwargs['local_radius']) * int(model_kwargs['transformer_layers'])
        residual = model_kwargs.get('functional_neighbor_mode', 'raw') == 'residual'
        functional_neighbors = select_functional_neighbors(train_ds, functional_k, exclude, residual)
        logger.info(
            f'[functional_neighbors] k={functional_k}, 격자 거리 {exclude} 이내 제외, '
            f'상관 기준 {"요일×시각 잔차" if residual else "원 시계열"}'
        )

    model_config = MergedDemandConfig(
        height=train_ds.height,
        width=train_ds.width,
        time_step=train_ds.time_step,
        retrieval_grid_path=str(data_path),
        retrieval_train_end=train_ds.train_end,
        **weather_stats,
        node_adaptive_indices=node_adaptive_indices,
        **model_kwargs,
        context_extra_dim=len(dataset_kwargs['context_features']),
        context_features=dataset_kwargs['context_features'],
        functional_neighbors=functional_neighbors,
        use_event_embedding=dataset_kwargs['event_csv_path'] is not None,
        event_csv_path=dataset_kwargs['event_csv_path'],
        weather_lead=dataset_kwargs['weather_lead'],
        aux_channels=0 if train_ds.aux_table is None else int(train_ds.aux_table.shape[-1]),
        aux_grid_path=dataset_kwargs['aux_grid_path'],
        node_mix_lambda=node_mix[0] if node_mix else None,
        node_mix_mape_scale=node_mix[1] if node_mix else None,
        node_mix_mse_scale=node_mix[2] if node_mix else None,
        node_mix_weight=node_mix[3] if node_mix else None,
    )
    if dataset_kwargs['event_csv_path'] is not None:
        logger.info(
            f'[event] {dataset_kwargs["event_csv_path"]}: train 구간 이벤트 노드·시각 '
            f'{int(train_ds.event_table[: train_ds.train_end].sum())}개, 전체 {int(train_ds.event_table.sum())}개'
        )
    # Trainer 초기화보다 먼저 seed를 고정해 모델 초기화를 재현한다.
    set_seed(cfg.train.seed)

    configured_init_from = cfg.get('stage2', {}).get('init_from')
    init_from = configured_init_from if node_adaptive else None
    if configured_init_from and not node_adaptive:
        logger.warning(
            f'stage2.init_from={configured_init_from}이 주어졌지만 model.node_adaptive=false라 '
            '2-stage 학습을 하지 않는다 — 무시한다'
        )
    if init_from:
        model = load_stage1_with_new_nodes(init_from, model_config)
        logger.info(f'[stage2] stage1 체크포인트에서 이어받음: {init_from}')
    else:
        model = MergedDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    early_stopping_cfg = cfg.callbacks.early_stopping
    schedule_cfg = OmegaConf.to_container(cfg.optimizer_schedule, resolve=True)
    schedule_name = schedule_cfg.get('name')
    if schedule_name not in (None, 'warmup_cosine'):
        raise ValueError(f"알 수 없는 optimizer_schedule.name: {schedule_name!r}")

    # SDPA 제한은 attention 가중치 dropout이 켜진 경우에만 적용된다.
    # 샘플 하나는 이력 시점마다 격자 전체 시퀀스 하나를 attention에 넣는다.
    per_sample = train_ds.time_step
    if model_config.attention_dropout > 0.0:
        max_batch = max(1, SDPA_BATCH_LIMIT // per_sample)
        for key in ('per_device_train_batch_size', 'per_device_eval_batch_size'):
            if train_cfg[key] > max_batch:
                logger.warning(
                    f'{key}={train_cfg[key]}는 time_step({per_sample})와 곱하면 '
                    f'SDPA_BATCH_LIMIT({SDPA_BATCH_LIMIT})을 넘어 memory-efficient attention이 '
                    f'죽는다(attention_dropout={model_config.attention_dropout}>0) -> {max_batch}로 낮춘다'
                )
                train_cfg[key] = max_batch
    else:
        logger.info(
            f'[batch] attention_dropout=0이라 SDPA_BATCH_LIMIT 클램프를 건너뛴다 '
            f'(attention 시퀀스 = batch * {per_sample} = '
            f'{train_cfg["per_device_train_batch_size"] * per_sample:,}).'
        )

    def build_trainer(
        stage_dir: str | None,
        learning_rate: float,
        *,
        metrics_fn=compute_metrics,
        metric_for_best_model: str = 'loss',
    ) -> Trainer:
        """각 학습 stage마다 독립된 Trainer를 만든다.

        optimizer, scheduler, early stopping, checkpoint 상태는 stage 사이에 공유하지 않는다.
        """

        stage_cfg = dict(train_cfg)
        stage_cfg['learning_rate'] = learning_rate
        stage_cfg['metric_for_best_model'] = metric_for_best_model
        stage_args = TrainingArguments(
            output_dir=output_dir if stage_dir is None else str(Path(output_dir) / stage_dir),
            **stage_cfg,
        )
        trainer_kwargs = dict(
            model=model,
            args=stage_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=metrics_fn,
            callbacks=[
                LoggingCallback(),
                MinEpochEarlyStoppingCallback(
                    min_epochs=early_stopping_cfg.min_epochs,
                    early_stopping_patience=early_stopping_cfg.early_stopping_patience,
                ),
            ],
        )
        if schedule_name is None:
            return Trainer(**trainer_kwargs)
        accumulation = int(stage_cfg.get('gradient_accumulation_steps', 1) or 1)
        steps_per_epoch = max(
            1, math.ceil(
                math.ceil(len(train_set) / int(stage_cfg['per_device_train_batch_size']))
                / accumulation
            ),
        )
        logger.info(
            f'[schedule] ADFormer warmup-cosine: warmup {schedule_cfg["warmup_epochs"]} '
            f'epochs ({schedule_cfg["warmup_lr_init"]} -> {learning_rate}), '
            f'cosine {schedule_cfg["cosine_epochs"]} epochs '
            f'({learning_rate} -> {schedule_cfg["eta_min"]}), '
            f'이후 {schedule_cfg["eta_min"]} | steps/epoch={steps_per_epoch}'
        )
        return WarmupCosineTrainer(
            schedule=schedule_cfg, steps_per_epoch=steps_per_epoch, **trainer_kwargs
        )

    def trainable_count(trainer: Trainer) -> int:
        return sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    deltas = model.node_delta_parameters()
    stage_records: dict[str, dict] = {}
    stage1_metrics: dict[str, float] | None = None
    # node_adaptive_schedule: two_stage(기본, stage 1 ΔW 고정 → stage 2 ΔW 해제) | joint(처음부터 ΔW 포함 한 번에 학습).
    schedule = str(cfg.get('node_adaptive_schedule', 'two_stage'))
    if schedule not in ('two_stage', 'joint'):
        raise ValueError(f"node_adaptive_schedule은 'two_stage'|'joint' (받음: {schedule!r})")
    joint = bool(deltas) and schedule == 'joint'
    if joint and init_from:
        raise ValueError('node_adaptive_schedule=joint에서는 stage2.init_from을 쓸 수 없다')

    if not deltas or joint:
        model.begin_node_adaptation()
        for param in deltas:
            param.requires_grad_(True)
        trainer = build_trainer(None, train_cfg['learning_rate'])
        if joint:
            logger.info(
                f'[joint] 처음부터 ΔW 포함 학습, lr={train_cfg["learning_rate"]}, '
                f'학습 파라미터 {trainable_count(trainer):,}개 (ΔW {sum(p.numel() for p in deltas):,}개 포함)'
            )
        trainer.train()
        best_metric_key = 'loss'
    else:
        if init_from:
            # stage 2 전에 weights-only 초기화를 검증한다.
            nonzero = [
                name
                for name, param in model.named_parameters()
                if 'node_delta' in name and float(param.detach().abs().sum()) != 0.0
            ]
            if nonzero:
                raise ValueError(
                    f'stage2.init_from은 ΔW=0인 stage1 체크포인트여야 함 — 0이 아닌 항목: {nonzero}'
                )
            # 같은 목적함수 기준의 stage 1 test 지표를 기록한다.
            model.configure_loss(str(cfg.model.loss_type))
            for param in deltas:
                param.requires_grad_(False)
            probe = build_trainer('stage1_eval', train_cfg['learning_rate'])
            # persistent worker의 eval dataloader 캐시를 피하도록 test는 predict로 평가한다.
            stage1_metrics = probe.predict(eval_test_set, metric_key_prefix='stage1_test').metrics
            logger.info(f'[stage1] Test metrics: {stage1_metrics}')
            del probe
        else:
            model.configure_loss(str(cfg.model.loss_type))
            for param in deltas:
                param.requires_grad_(False)
            stage1 = build_trainer('stage1', train_cfg['learning_rate'])
            logger.info(
                f'[stage1] ΔW 고정(node_adaptive=false와 동일), '
                f'학습 파라미터 {trainable_count(stage1):,}개'
            )
            stage1.train()
            stage1_metrics = stage1.predict(eval_test_set, metric_key_prefix='stage1_test').metrics
            logger.info(f'[stage1] Test metrics: {stage1_metrics}')
            history1, best_epoch1, best_val1 = _summarize_history(stage1.state.log_history)
            stage_records['stage1'] = {
                'history': history1,
                'best_epoch': best_epoch1,
                'best_val_loss': best_val1,
                'best_metric': 'loss',
                'checkpoint': str(Path(output_dir) / 'stage1'),
            }
            # stage 1 Trainer가 optimizer와 모델 wrapper를 계속 붙들고 있지 않게 놓아준다.
            del stage1

        model.begin_node_adaptation()
        for param in deltas:
            param.requires_grad_(True)
        # stage2.freeze: none | shared(ΔW만 학습) | pre_lstm(LSTM 입력부 고정, LSTM 이후만 학습).
        stage2_freeze = str(cfg.stage2.get('freeze', 'none'))
        if stage2_freeze == 'shared':
            for name, param in model.named_parameters():
                if 'node_delta' not in name:
                    param.requires_grad_(False)
        elif stage2_freeze == 'pre_lstm':
            for param in model.pre_lstm_parameters():
                param.requires_grad_(False)
        elif stage2_freeze != 'none':
            raise ValueError(f"stage2.freeze는 'none'|'shared'|'pre_lstm' (받음: {stage2_freeze!r})")
        stage2_loss = str(cfg.stage2.loss)
        model.configure_loss(
            stage2_loss,
            rmse_weight=float(cfg.stage2.rmse_weight),
            mape_weight=float(cfg.stage2.mape_weight),
        )
        if stage2_loss == 'rmse_mape':
            rmse_weight = float(cfg.stage2.rmse_weight)
            metrics_fn = make_rmse_mape_metrics(rmse_weight)
            best_metric_key = 'rmse_mape_objective'
            logger.info(f'[stage2] loss = {rmse_weight} * RMSE + MAPE(+1)')
        else:
            metrics_fn = compute_metrics
            best_metric_key = 'loss'
            logger.info(f'[stage2] loss = {stage2_loss}')

        trainer = build_trainer(
            'stage2',
            float(cfg.stage2.learning_rate),
            metrics_fn=metrics_fn,
            metric_for_best_model=best_metric_key,
        )
        logger.info(
            f'[stage2] ΔW 해제 (freeze={stage2_freeze}), lr={cfg.stage2.learning_rate}, '
            f'학습 파라미터 {trainable_count(trainer):,}개 '
            f'(ΔW {sum(p.numel() for p in deltas):,}개 포함)'
        )
        trainer.train()

    trainer.save_model(output_dir)
    logger.info(f'Model saved to: {output_dir}')

    test_metrics = trainer.predict(eval_test_set, metric_key_prefix='test').metrics
    logger.info(f'Test metrics: {test_metrics}')

    node_gate = model.local_history.node_gate()
    if deltas:
        for name, param in model.named_parameters():
            if 'node_delta' in name:
                logger.info(
                    f'[ΔW] {name}: norm={param.norm():.4f} '
                    f'max_abs={param.abs().max():.4f}'
                )
        if node_gate is not None:
            logger.info(
                f'[ΔW] gate g: mean={node_gate.mean():.4f} min={node_gate.min():.4f} '
                f'max={node_gate.max():.4f}'
            )

    history, best_epoch, best_val = _summarize_history(trainer.state.log_history, best_metric_key)
    result = {
        'dataset': cfg.dataset.city,
        'data_path': str(data_path),
        'weather_path': str(dataset_kwargs['weather_csv_path']),
        'weather_stats': weather_stats,
        'device': str(trainer.args.device),
        'retrieval_scope': cfg.model.retrieval_scope,
        # stage별 목적함수를 별도로 기록한다.
        'objective': cfg.model.loss_type,
        'stage1_loss': cfg.model.loss_type,
        'final_loss': str(cfg.stage2.loss) if deltas and not joint else cfg.model.loss_type,
        'stage2_freeze': str(cfg.stage2.get('freeze', 'none')) if deltas and not joint else None,
        'node_adaptive_schedule': schedule if deltas else None,
        'ablation': cfg.ablation,
        'ablation_mode': ABLATION_MODE,
        # ablation 라벨이 아닌 실제 적용된 스위치를 기록한다.
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
                'use_sink',
                'local_encoder',
                'use_softplus',
            )
        },
        'seed': int(cfg.train.seed),
        # node-adaptive 설정과 stage 메타데이터를 기록한다.
        'node_adaptive': node_adaptive,
        'node_adaptive_min_demand': (
            float(model_kwargs['node_adaptive_min_demand']) if node_adaptive else None
        ),
        'node_adaptive_nodes': len(node_adaptive_indices) if node_adaptive_indices else 0,
        'node_adaptive_indices': node_adaptive_indices,
        'node_delta_params': sum(p.numel() for p in deltas),
        'node_adaptive_mode': model.config.node_adaptive_mode if node_adaptive else None,
        # gated일 때 노드별 g[n](node_adaptive_indices 순서).
        'node_gate': node_gate.detach().cpu().tolist() if node_gate is not None else None,
        'stages': ['joint'] if joint else (['stage1', 'stage2'] if deltas else ['single']),
        'stage2_loss': str(cfg.stage2.loss) if deltas and not joint else None,
        'stage1_init_from': str(init_from) if init_from else None,
        'stage1': stage_records.get('stage1'),
        'stage1_test': stage1_metrics,
        'best_metric': best_metric_key,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
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
        # node-adaptive 실행을 구분하는 접미사를 사용한다.
        suffix = '_nodeadaptive' if node_adaptive else ''
        run_json = (
            Path('output')
            / cfg.project_name
            / 'runs'
            / f'{cfg.dataset.city}_{cfg.model.loss_type}_{cfg.ablation}{suffix}'
            f'_seed{cfg.train.seed}.json'
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
