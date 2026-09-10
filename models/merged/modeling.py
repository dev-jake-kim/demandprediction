"""HuggingFace ``PreTrainedModel`` port of ``merged_model``'s ``UnifiedDemandModel``.

Branch-specific code lives in :mod:`models.merged.modules`; this file intentionally
contains only model construction, data flow, and the final loss calculation.

**계산 그래프는 원본과 동일하다.** 바뀐 것은 껍데기뿐이다:

* 생성자 인자 -> :class:`MergedDemandConfig` (``PretrainedConfig``)
* ``target`` -> ``labels`` (HF ``Trainer``의 ``label_names`` 기본값)
* ``forward``의 반환 dict를 ``{'loss', 'logits'}``로 슬림화. ``Trainer``의 eval loop는
  ``loss``를 제외한 모든 키를 배치마다 gather하므로, 원본이 돌려주던 디버그 텐서
  (``neural_pred``/``ir_out``/``lambda_weight``/``attention_weights``/``h_attn`` 등)를
  그대로 두면 메모리가 폭증하고 shape가 균일하지 않은 키에서 깨진다. 그 텐서들은
  :meth:`MergedDemandModel.forward_debug`로 옮겼다 — 예측/손실 계산 경로는 완전히 동일하다.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from dataset_frame.unified_demand_dataset import NUM_WEATHER_FEATURES

from .config import MergedDemandConfig
from .losses import build_loss
from .modules import (
    BranchAttention,
    CausalRetrieval,
    LocalHistoryEncoder,
    NeuralRetrievalGate,
    PeriodicLSTMEncoder,
)


class MergedDemandModel(PreTrainedModel):
    """One end-to-end model for neural, periodic, and retrieval information."""

    config_class = MergedDemandConfig
    base_model_prefix = 'merged_demand'
    main_input_name = 'demand_history'

    def __init__(self, config: MergedDemandConfig) -> None:
        super().__init__(config)

        height, width = config.height, config.width
        if height <= 0 or width <= 0:
            raise ValueError('height and width must be positive')
        if config.time_step <= 0:
            raise ValueError('time_step must be positive')
        if config.local_radius < 0:
            raise ValueError('local_radius must be non-negative')
        if config.fusion_dim <= 0 or config.retrieval_k <= 0 or config.retrieval_chunk_size <= 0:
            raise ValueError('fusion_dim, retrieval_k, and retrieval_chunk_size must be positive')
        if config.transformer_heads <= 0 or config.d_model % config.transformer_heads != 0:
            raise ValueError('d_model must be divisible by transformer_heads')
        if config.weekday_dim <= 0 or config.hour_dim <= 0:
            raise ValueError('weekday_dim and hour_dim must be positive')
        if config.weather_mean is None or config.weather_std is None:
            raise ValueError(
                'weather_mean/weather_std(각 3개, train split 통계)가 필요함 — '
                '시간 리크를 막으려면 train.py가 train 구간에서만 계산해 넘겨야 한다'
            )
        # 값 검사는 파이썬 리스트에서 한다. from_pretrained가 meta device에서 __init__을
        # 돌기 때문에, 텐서로 만든 뒤 검사하면 meta 텐서에 bool()을 부르게 되어 깨진다.
        weather_mean_list = [float(value) for value in config.weather_mean]
        weather_std_list = [float(value) for value in config.weather_std]
        if len(weather_mean_list) != NUM_WEATHER_FEATURES:
            raise ValueError(f'weather_mean must have {NUM_WEATHER_FEATURES} entries')
        if len(weather_std_list) != NUM_WEATHER_FEATURES:
            raise ValueError(f'weather_std must have {NUM_WEATHER_FEATURES} entries')
        if any(value <= 0 for value in weather_std_list):
            raise ValueError('weather_std must be positive (0으로 나누기 방지)')

        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = config.time_step

        # --- ablation 스위치 ---
        # 각 모듈을 끄고 켜면서 기여도를 재기 위한 것. 전부 True면 기본 모델과 동일하다.
        self.use_daily = config.use_daily
        self.use_weekly = config.use_weekly
        self.use_retrieval = config.use_retrieval
        self.use_weather = config.use_weather
        # 'concat': 정규화 3값을 세 LSTM 입력에 그대로 붙인다(기본).
        # 'cls_add': ir-weather 방식 — d_model로 투영해 history의 CLS 토큰에 더한다.
        #   이 경우 주기 브랜치에는 날씨가 들어가지 않는다(CLS가 거기엔 없다). 즉 주입 지점과
        #   커버리지가 함께 바뀌는 변형이며, 그건 ir-weather 설계의 본질적 성질이다.
        if config.weather_injection not in ('concat', 'cls_add'):
            raise ValueError(
                f"weather_injection은 'concat'|'cls_add' (받음: {config.weather_injection!r})"
            )
        self.weather_injection = config.weather_injection
        self.use_calendar = config.use_calendar
        self.use_branch_attention = config.use_branch_attention
        self.use_neighbors = config.use_neighbors
        self.use_softplus = config.use_softplus

        # 날씨는 임베딩하지 않고 정규화한 3값을 그대로 LSTM 입력에 concat한다. 요일/시간대만
        # 임베딩 테이블을 쓰며, 세 브랜치가 같은 테이블을 공유한다(요일 3은 어느 브랜치에서나 요일 3).
        self.weekday_embedding = nn.Embedding(7, config.weekday_dim)
        # ir-weather는 nn.Embedding(1440, d_model)에 hour*60 인덱스를 넣지만 실제로 학습되는 행은
        # 24개뿐인 분(minute) 해상도 잔재다 — 동일 효과의 24행으로 단순화한다.
        self.hour_embedding = nn.Embedding(24, config.hour_dim)
        # 폭은 끄든 켜든 15로 고정 — 값만 0이 된다. 폭이 바뀌면 LSTM 파라미터 수가 달라져
        # 정보 제거 효과와 용량 감소 효과가 섞인다.
        self.weather_cls_dim = (
            NUM_WEATHER_FEATURES if self.weather_injection == 'cls_add' else 0
        )
        self.extra_dim = (
            (NUM_WEATHER_FEATURES if self.weather_injection == 'concat' else 0)
            + config.weekday_dim
            + config.hour_dim
        )

        self.register_buffer(
            'weather_mean', torch.tensor(weather_mean_list, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            'weather_std', torch.tensor(weather_std_list, dtype=torch.float32), persistent=True
        )

        self.local_history = LocalHistoryEncoder(
            height=height,
            width=width,
            time_step=config.time_step,
            local_radius=config.local_radius,
            d_model=config.d_model,
            num_fourier_bands=config.num_fourier_bands,
            transformer_layers=config.transformer_layers,
            transformer_heads=config.transformer_heads,
            transformer_ffn=config.transformer_ffn,
            history_hidden=config.history_hidden,
            dropout=config.dropout,
            extra_dim=self.extra_dim,
            weather_cls_dim=self.weather_cls_dim,
            use_neighbors=self.use_neighbors,
        )
        self.periodic_hidden = config.periodic_hidden
        # ablation은 "0-치환" 방식이다 — 모듈을 없애지 않고 항상 생성·실행한 뒤, 그 모듈이
        # 결과로 이어지는 텐서만 0으로 바꾼다. 모듈을 지우면 텐서 shape·파라미터 수·attention
        # 후보 개수까지 함께 바뀌어, 측정된 차이가 "그 모듈의 정보" 때문인지 "구조 변화" 때문인지
        # 분리되지 않는다. 0-치환은 그 교란을 없앤다.
        self.daily_branch = PeriodicLSTMEncoder(config.periodic_hidden, extra_dim=self.extra_dim)
        self.weekly_branch = PeriodicLSTMEncoder(config.periodic_hidden, extra_dim=self.extra_dim)
        self.branch_attention = BranchAttention(
            config.history_hidden,
            config.periodic_hidden,
            config.fusion_dim,
            use_attention=self.use_branch_attention,
        )
        # 검색기도 끄든 켜든 항상 만들고 항상 계산한다(느리지만 경로가 동일해진다).
        # use_retrieval=False면 ir_out만 0이 되고, 게이트는 그대로 남아 lambda를 학습한다
        # — 처음부터 재학습하므로 모델이 lambda->1을 배워 보정할 수 있다.
        self.retrieval = CausalRetrieval(
            height=height,
            width=width,
            time_step=config.time_step,
            local_radius=config.local_radius,
            retrieval_grid_path=config.retrieval_grid_path,
            retrieval_k=config.retrieval_k,
            retrieval_chunk_size=config.retrieval_chunk_size,
            retrieval_scope=config.retrieval_scope,
            retrieval_train_end=config.retrieval_train_end,
        )
        self.output_gate = NeuralRetrievalGate(config.fusion_dim, use_softplus=self.use_softplus)
        # 이 저장소의 다른 모델들과 목적함수를 맞추려면 'combined'(CombinedLoss)를 쓴다.
        # 'mae'는 merged_model이 원래 쓰던 raw 스케일 L1이며, 두 경우 모두 reduction='none'
        # 이라 아래 forward에서 mean/sum을 각각 뽑는다.
        self.loss_type = config.loss_type
        self.loss_fn = build_loss(config.loss_type, gamma=config.loss_gamma, eps=config.loss_eps)

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """PyTorch 기본 초기화를 그대로 쓴다(의도적인 no-op).

        ``PreTrainedModel._init_weights``는 Linear/Embedding/LayerNorm을 std=0.02 정규분포로
        다시 초기화한다. 그걸 상속하면 원본 ``UnifiedDemandModel``(순수 ``nn.Module``, 즉
        PyTorch 기본 초기화)과 출발점이 달라져 포팅 자체가 다른 실험이 된다. 이 포팅의 성공
        기준은 "계산이 바뀌지 않았음"이므로 여기서는 아무것도 하지 않는다.
        """

    # Keep the old debugging entry points available while the implementation
    # is organized under named components.
    def _crop_all_nodes(self, demands: Tensor) -> Tensor:
        return self.local_history.crop(demands)

    def _retrieve(self, local_crop: Tensor, sample_idx: Tensor) -> Tensor:
        return self.retrieval(local_crop, sample_idx)

    def _temporal_extra(self, weather: Tensor, hour: Tensor, day_of_week: Tensor) -> Tensor:
        """(정규화 날씨 3) ⊕ (요일 임베딩) ⊕ (시간대 임베딩) -> ``[B, L, extra_dim]``.

        세 브랜치가 같은 방식으로 만든다 — recent는 L=time_step, daily/weekly는 L=lag 개수.
        정규화를 여기 한 곳에서만 하므로 날것의 기온(수십 단위)이 log1p 수요를 압도하는 일이 없다.
        """

        parts: list[Tensor] = []
        if self.weather_injection == 'concat':
            weather_norm = (weather - self.weather_mean) / self.weather_std
            # 정규화 후의 0은 train split 평균에 해당한다 — "정보 없음"의 자연스러운 대체값.
            parts.append(weather_norm if self.use_weather else torch.zeros_like(weather_norm))
        weekday = self.weekday_embedding(day_of_week)
        hour_vec = self.hour_embedding(hour)
        if not self.use_calendar:
            weekday = torch.zeros_like(weekday)
            hour_vec = torch.zeros_like(hour_vec)
        parts.extend([weekday, hour_vec])
        return torch.cat(parts, dim=-1)

    def _compute(
        self,
        *,
        demand_history: Tensor,
        daily_demand: Tensor,
        daily_mask: Tensor,
        weekly_demand: Tensor,
        weekly_mask: Tensor,
        sample_idx: Tensor,
        weather: Tensor,
        hour_of_day: Tensor,
        day_of_week: Tensor,
        daily_weather: Tensor,
        daily_hour: Tensor,
        daily_day_of_week: Tensor,
        weekly_weather: Tensor,
        weekly_hour: Tensor,
        weekly_day_of_week: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """원본 ``UnifiedDemandModel.forward``와 완전히 동일한 계산. 전체 dict를 돌려준다."""

        recent_extra = self._temporal_extra(weather, hour_of_day, day_of_week)
        daily_extra = self._temporal_extra(daily_weather, daily_hour, daily_day_of_week)
        weekly_extra = self._temporal_extra(weekly_weather, weekly_hour, weekly_day_of_week)

        weather_cls = None
        if self.weather_cls_dim:
            weather_cls = (weather - self.weather_mean) / self.weather_std
            if not self.use_weather:
                weather_cls = torch.zeros_like(weather_cls)
        local_crop, h_neural = self.local_history(demand_history, recent_extra, weather_cls)
        # 브랜치는 항상 실행하고, 끈 경우 출력 텐서만 0으로 바꾼다. valid 마스크는 원래 값을
        # 그대로 둬서 attention 후보 개수가 변하지 않게 한다(구조 교란 제거).
        h_daily, daily_valid = self.daily_branch(daily_demand, daily_mask, daily_extra)
        if not self.use_daily:
            h_daily = torch.zeros_like(h_daily)
        h_weekly, weekly_valid = self.weekly_branch(weekly_demand, weekly_mask, weekly_extra)
        if not self.use_weekly:
            h_weekly = torch.zeros_like(h_weekly)
        h_attn, attention_weights = self.branch_attention(
            h_neural, h_daily, h_weekly, daily_valid, weekly_valid
        )

        if self.use_retrieval:
            ir_out = self.retrieval(local_crop, sample_idx)
            neural_pred, lambda_weight, prediction = self.output_gate(h_attn, ir_out)
        else:
            # 검색기를 아예 호출하지 않는다 — 0으로 치환한 ir_out을 게이트에 흘려보내면
            # lambda가 0으로 무너지는 죽음의 함정이 생긴다(modules/fusion.py 참고).
            # bypass_gate=True로 게이트 자체를 건너뛰어 그 함정을 구조적으로 없앤다.
            ir_out = None
            neural_pred, lambda_weight, prediction = self.output_gate(h_attn, None, bypass_gate=True)
        prediction_grid = prediction.reshape(-1, self.height, self.width)

        output: dict[str, Tensor | None] = {
            'logits': prediction_grid,
            'prediction': prediction_grid,
            'prediction_flat': prediction,
            'neural_pred': neural_pred,
            'ir_out': ir_out,
            'lambda_weight': lambda_weight,
            'attention_weights': attention_weights,
            'h_neural': h_neural,
            'h_attn': h_attn,
            'daily_valid': daily_valid,
            'weekly_valid': weekly_valid,
        }
        if labels is not None:
            # reduction='none'으로 한 번만 계산하고 mean(역전파용)과 sum(에폭 집계용)을 함께 낸다.
            # 배치 크기가 균일하지 않아(drop_last=False) 배치 평균의 평균은 원소 평균과 다르다.
            elementwise = self.loss_fn(prediction_grid, labels)
            output['loss'] = elementwise.mean()
            output['loss_sum'] = elementwise.detach().sum()
        return output

    def forward(
        self,
        demand_history: Tensor,
        daily_demand: Tensor,
        daily_mask: Tensor,
        weekly_demand: Tensor,
        weekly_mask: Tensor,
        sample_idx: Tensor,
        weather: Tensor,
        hour_of_day: Tensor,
        day_of_week: Tensor,
        daily_weather: Tensor,
        daily_hour: Tensor,
        daily_day_of_week: Tensor,
        weekly_weather: Tensor,
        weekly_hour: Tensor,
        weekly_day_of_week: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``Trainer``용 슬림 반환: ``{'loss', 'logits'}``.

        ``GridDemandModel.forward``와 정확히 같은 관례다. 디버그 텐서가 필요하면
        :meth:`forward_debug`를 쓴다.
        """

        output = self._compute(
            demand_history=demand_history,
            daily_demand=daily_demand,
            daily_mask=daily_mask,
            weekly_demand=weekly_demand,
            weekly_mask=weekly_mask,
            sample_idx=sample_idx,
            weather=weather,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            daily_weather=daily_weather,
            daily_hour=daily_hour,
            daily_day_of_week=daily_day_of_week,
            weekly_weather=weekly_weather,
            weekly_hour=weekly_hour,
            weekly_day_of_week=weekly_day_of_week,
            labels=labels,
        )
        slim: dict[str, Tensor] = {'logits': output['logits']}
        if 'loss' in output:
            slim['loss'] = output['loss']
        return slim

    def forward_debug(self, **batch) -> dict[str, Tensor | None]:
        """원본이 돌려주던 전체 dict(``neural_pred``/``lambda_weight``/``attention_weights`` 등).

        검증 스크립트(``validate_merged.py``, parity 테스트)만 쓴다. ``Trainer``는 이 dict의
        모든 키를 배치마다 gather하려 들기 때문에 학습 경로에서 쓰면 안 된다.
        """

        return self._compute(**batch)


__all__ = ['MergedDemandModel']
