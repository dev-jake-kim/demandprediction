"""merged 모델(로컬 히스토리 + daily/weekly 주기 + 인과적 검색 + 브랜치 어텐션) 학습 스크립트.

원본 ``merged_model/train.py``의 수동 학습 루프를 이 저장소 공용 관례(Hydra + HF ``Trainer``)로
옮긴 것이다. 학습 하이퍼파라미터(gradient clipping 5.0, 고정 LR, epochs 2000,
num_workers 0, best = val loss 최소, torch_compile 없음)는 원본과 1:1로 맞춰 두었다.
예외는 early stopping으로, 도시별 학습 기록을 시뮬레이션해 원본(min_epochs 0 / patience 20)과
**같은 best_epoch를 내는 선에서** 줄였다 — ulsan은 min_epochs 13 / patience 8, porto는 줄일
실익이 없어 원본 유지. 근거는 각 configs/config_*.yaml의 callbacks 블록 주석에 있다.

도시마다 최적 하이퍼파라미터가 달라 루트 config를 도시별로 분리했다 — 기본값은
``configs/config_ulsan.yaml``, porto는 ``--config-name config_porto``로 명시한다::

    python train.py                                    # ulsan
    python train.py --config-name config_porto          # porto
    python train.py model.use_retrieval=false ablation=no-ir   # ablation (ulsan)

``model.node_adaptive=true``(ulsan 기본값)면 temporal LSTM의 유효 가중치를 적응 노드에서만
``s * W_shared + (1 - s) * ΔW``로 바꾼다. ``s``(스칼라)와 ``ΔW``(노드별)는 ``s=1``/``ΔW=0``에서
출발해 나머지 파라미터와 **같은 단일 stage, 같은 목적함수**로 학습한다 — 별도 stage도
warm start도 없다. ``model.shared_weight_fp8=true``(ulsan 기본값)면 공유 LSTM 가중치 4개를
FP8(e4m3fn) 격자로 fake quantize한다(ΔW/s는 fp32 유지)::

    python train.py                                                          # 채택 설정
    python train.py model.node_adaptive=false model.shared_weight_fp8=false  # 기준선

근거는 ``docs/MERGED_GATED_FP8_RESULTS.md``, 설계/제약은 ``docs/MERGED_ARCHITECTURE.md``.

이 브랜치(tmp)에는 다른 모델 구현이 없다(models/config.py, models/modeling.py 부재 —
models/__init__.py 참고) — train.py는 이 모델 하나만 학습하는 단일 진입점이다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandConfig, MergedDemandModel, compute_merged_metrics

logger = logging.getLogger(__name__)

# PyTorch의 memory-efficient SDPA 백엔드가 거는 한계다. VRAM과는 무관하다 — 실제 코드는
# aten/src/ATen/native/transformers/cuda/attention.cu의 이 검사다:
#
#     constexpr int64_t MAX_BATCH_SIZE = (1LL << 16) - 1;   // = 65535
#     if (batch_size > MAX_BATCH_SIZE) {
#       TORCH_CHECK(dropout_p == 0.0,
#                   "Efficient attention cannot produce valid seed and offset outputs when "
#                   "the batch size exceeds (", MAX_BATCH_SIZE, ").");
#     }
#
# 즉 **dropout>0일 때만** 걸린다. dropout의 RNG 상태(philox seed/offset)를 backward용으로
# 배치마다 기록하는 경로의 제약이고, 근본 원인은 CUDA 그리드 차원 상한이다 — cutlass 커널의
# getBlocksGrid()가 dim3(ceil_div(num_queries, kQueriesPerBlock), num_heads, num_batches)라
# num_batches가 blocks.z로 들어가는데 y/z 차원 상한이 65535다(x만 2^31-1).
#
# models/merged/modeling.py의 LocalViewEncoder는 (batch, time_step, H*W) 전체를
# (batch*time_step*H*W, 1+neighbors, d_model)로 펼쳐 넣으므로 배치가 조금만 커도 한계를
# 넘는다 — ulsan(14*12=168)은 batch 17부터, porto(10*20=200)는 batch 14부터다.
# "어텐션 하나가 크다"가 아니라 "26토큰짜리 작은 어텐션이 너무 많다"가 문제다.
#
# 이 백엔드를 끄는 우회는 쓰지 않는다 — fallback(math) 백엔드가 어텐션 행렬을 그대로 만들어
# 메모리를 훨씬 더 써서 d_model이 크면 오히려 OOM이 난다. 대신 아래 main()에서 어텐션
# dropout이 켜져 있을 때만 배치 크기를 낮춘다. 꺼져 있으면 이 한계가 없으므로 한계는 VRAM이다.
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


def select_node_adaptive_indices(train_ds: UnifiedDemandDataset, min_demand: float) -> list[int]:
    """train 구간 평균 수요가 ``min_demand``를 넘는 노드 id 목록.

    **train 구간에서만 계산해야 한다** — 전체 기간으로 고르면 어떤 노드가 ΔW를 받을지가
    test 구간 정보에 의존하게 되어 시간 리크가 된다. 날씨 정규화 통계와 같은 이유로 같은
    구간(``[time_step, train_end)``)을 쓴다.
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

    # best 선택 기준은 val loss 하나다(학습 목적함수와 같은 값).
    best_epoch, best_val = 0, float('inf')
    for record in history:
        val = record.get('val', {}).get('loss')
        if val is not None and val < best_val:
            best_val, best_epoch = float(val), record['epoch']
    return history, best_epoch, best_val


def select_zero_node_indices(train_ds: UnifiedDemandDataset, max_demand: float) -> list[int]:
    """train 구간 평균 수요가 ``max_demand`` 이하인 노드 id 목록 — 학습에서 제외할 노드.

    ``select_node_adaptive_indices``와 같은 이유로 **train 구간에서만** 계산한다. 전체
    기간으로 고르면 어떤 노드를 뺄지가 test 정보에 의존해 시간 리크가 된다.
    """

    window = train_ds.grid[train_ds.time_step : train_ds.train_end]
    node_mean = window.reshape(len(window), -1).mean(axis=0)
    indices = [int(node) for node in (node_mean <= max_demand).nonzero()[0]]
    if len(indices) >= train_ds.num_nodes:
        raise ValueError(
            f'zero_node_max_demand={max_demand}가 너무 커서 모든 노드가 제외된다 '
            f'(노드 평균 수요 최댓값 {float(node_mean.max()):.3f})'
        )
    return indices


@hydra.main(config_path='configs', config_name='config_ulsan', version_base=None)
def main(cfg: DictConfig) -> None:
    data_path, dataset_kwargs, train_ds, val_ds, test_ds = build_datasets(cfg)

    # 날씨 정규화 통계는 train 구간에서만 계산한다(시간 리크 방지).
    # 적설처럼 train 내내 값이 고정(분산 0)인 피처가 있어 std에 하한을 둔다(porto 적설이 실제로 그렇다).
    train_weather = train_ds.weather[train_ds.time_step : train_ds.train_end]
    weather_mean = train_weather.mean(axis=0)
    weather_std = train_weather.std(axis=0).clip(min=1e-6)
    logger.info(f'weather_mean={weather_mean.tolist()}, weather_std={weather_std.tolist()}')
    # 모델이 실제로 쓰는 것은 이쪽 min-max 기준이다(dataset이 train 구간에서 재 둔다).
    weather_norm = train_ds.weather_norm
    logger.info(f'weather_norm={weather_norm}')

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

    # 노드별 ΔW를 받을 노드를 train 구간에서만 고른다(시간 리크 방지). 꺼져 있으면 None을
    # 넘겨 모델이 delta 파라미터를 아예 만들지 않게 한다 — 그래야 state_dict가 이 기능이
    # 없던 시절과 정확히 같아서 기존 체크포인트/parity 테스트가 유지된다.
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

    # 항상 0에 가까운 노드를 학습에서 뺄지. null이면 이 기능이 없던 때와 동일하게 동작한다.
    zero_node_max_demand = model_kwargs.get('zero_node_max_demand')
    zero_node_indices = None
    if zero_node_max_demand is not None:
        zero_node_indices = select_zero_node_indices(train_ds, float(zero_node_max_demand))
        total_nodes = train_ds.num_nodes
        logger.info(
            f'[zero_node] train 구간 평균 수요 <= {zero_node_max_demand}인 노드 '
            f'{len(zero_node_indices)}/{total_nodes} '
            f'({len(zero_node_indices) / total_nodes * 100:.1f}%)를 학습에서 제외한다 '
            '(예측 0 고정 + 손실 제외). 평가는 전체 노드로 그대로 한다.'
        )

    model_config = MergedDemandConfig(
        zero_node_indices=zero_node_indices,
        height=train_ds.height,
        width=train_ds.width,
        time_step=train_ds.time_step,
        retrieval_grid_path=str(data_path),
        retrieval_train_end=train_ds.train_end,
        weather_mean=weather_mean.tolist(),
        weather_std=weather_std.tolist(),
        temperature_min=weather_norm.temperature_min,
        temperature_max=weather_norm.temperature_max,
        precipitation_max=weather_norm.precipitation_max,
        node_adaptive_indices=node_adaptive_indices,
        **model_kwargs,
    )
    # Trainer는 __init__에서 seed를 설정하는데 그건 모델이 만들어진 뒤다 — 그대로 두면
    # 초기 가중치가 프로세스마다 달라져 같은 seed로도 재현되지 않는다. 여기서 먼저 고정한다.
    set_seed(cfg.train.seed)

    model = MergedDemandModel(model_config)

    output_dir = HydraConfig.get().runtime.output_dir
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    early_stopping_cfg = cfg.callbacks.early_stopping

    # SDPA_BATCH_LIMIT 주석 참고. PyTorch의 검사가 보는 것은 **어텐션 가중치 dropout**이다
    # (attention.cu: `if (batch_size > MAX_BATCH_SIZE) TORCH_CHECK(dropout_p == 0.0, ...)`,
    # 여기 dropout_p는 MultiheadAttention에 넘어간 값이다). nn.TransformerEncoderLayer는
    # dropout 인자 하나를 네 곳(어텐션 가중치 / 어텐션 출력 / FFN 은닉 / FFN 출력)에 쓰지만,
    # 이 한계를 거는 것은 첫 번째 하나뿐이다. 그래서 config.dropout이 아니라
    # config.attention_dropout을 본다 — 둘을 혼동하면 어텐션 dropout을 껐는데도 쓸 수 있는
    # 배치를 근거 없이 버리게 된다(실측: attention_dropout=0이면 batch 70,000도 통과하고,
    # 0.1이면 거부된다). train/eval 배치를 같이 낮춰야 train 중간의 eval도 안전하다.
    nodes = train_ds.height * train_ds.width
    per_sample = train_ds.time_step * nodes
    attention_dropout = float(model_config.attention_dropout)
    if attention_dropout > 0.0:
        max_batch = max(1, SDPA_BATCH_LIMIT // per_sample)
        for key in ('per_device_train_batch_size', 'per_device_eval_batch_size'):
            if train_cfg[key] > max_batch:
                logger.warning(
                    f'{key}={train_cfg[key]}는 time_step({train_ds.time_step}) * nodes({nodes})'
                    f'={per_sample}와 곱하면 SDPA_BATCH_LIMIT({SDPA_BATCH_LIMIT})을 넘어'
                    f'memory-efficient attention이 죽는다'
                    f'(model.attention_dropout={attention_dropout}>0)'
                    f' -> {max_batch}로 낮춘다'
                )
                train_cfg[key] = max_batch
    else:
        logger.info(
            f'[batch] model.attention_dropout=0이라 SDPA_BATCH_LIMIT 클램프를 건너뛴다 '
            f'(rows = batch * {per_sample} = '
            f'{train_cfg["per_device_train_batch_size"] * per_sample:,}). 이제 한계는 VRAM이다. '
            f'model.dropout={cfg.model.dropout}은 나머지 세 자리에 그대로 걸린다.'
        )

    train_args = TrainingArguments(output_dir=output_dir, **train_cfg)
    trainer = Trainer(
        model=model,
        args=train_args,
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    node_delta_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if name.startswith('local_view.node_delta')
    )
    logger.info(
        f'학습 파라미터 {trainable:,}개'
        + (f' (노드별 ΔW {node_delta_params:,}개 포함)' if node_delta_params else '')
    )
    trainer.train()

    trainer.save_model(output_dir)
    logger.info(f'Model saved to: {output_dir}')

    test_metrics = trainer.evaluate(eval_dataset=eval_test_set, metric_key_prefix='test')
    logger.info(f'Test metrics: {test_metrics}')

    if node_delta_params:
        # ΔW와 s가 실제로 얼마나 움직였는지 남긴다 — ΔW가 전부 0에 가깝거나 s가 1에서
        # 안 움직였으면 적응 경로가 아무것도 배우지 못한 것이라 결과 해석이 달라진다.
        for name, param in model.named_parameters():
            if name.startswith('local_view.node_delta') or name == 'local_view.s':
                logger.info(
                    f'[node_adaptive] {name}: norm={param.norm():.4f} '
                    f'max_abs={param.abs().max():.4f}'
                )

    history, best_epoch, best_val = _summarize_history(trainer.state.log_history)
    result = {
        'dataset': cfg.dataset.city,
        'data_path': str(data_path),
        'weather_path': str(dataset_kwargs['weather_csv_path']),
        'weather_mean': weather_mean.tolist(),
        'weather_std': weather_std.tolist(),
        'weather_norm': asdict(weather_norm),
        'zero_node_max_demand': zero_node_max_demand,
        'zero_node_count': len(zero_node_indices) if zero_node_indices is not None else 0,
        'zero_node_indices': zero_node_indices,
        'device': str(trainer.args.device),
        'retrieval_scope': cfg.model.retrieval_scope,
        # 학습 목적함수. 'objective'는 기존 78건 JSON과의 스키마 호환을 위해 남긴 이름이다.
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
        # 적응 노드 경로의 근거. node_adaptive=false면 이 기능 추가 이전과 동일한 학습이다.
        'node_adaptive': node_adaptive,
        'node_adaptive_min_demand': (
            float(model_kwargs['node_adaptive_min_demand']) if node_adaptive else None
        ),
        'node_adaptive_nodes': len(node_adaptive_indices) if node_adaptive_indices else 0,
        'node_adaptive_indices': node_adaptive_indices,
        'node_delta_params': node_delta_params,
        'shared_weight_fp8': bool(model_kwargs.get('shared_weight_fp8', False)),
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
        # node_adaptive 런은 파일 이름을 달리한다 — 안 그러면 같은 (city, loss, ablation, seed)의
        # 기존 단일 stage 결과를 덮어써서 비교 기준 자체가 사라진다.
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
