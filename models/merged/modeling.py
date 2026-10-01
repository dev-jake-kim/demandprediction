"""HuggingFace ``PreTrainedModel`` for merged demand forecasting.

Branch-specific code lives in :mod:`models.merged.modules`; this module assembles the
model data flow and computes training loss.

``forward`` returns the compact ``{'loss', 'logits'}`` interface for ``Trainer``.
Use :meth:`MergedDemandModel.forward_debug` for intermediate outputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .config import MergedDemandConfig
from .losses import SCALAR_LOSS_TYPES, build_loss
from .modules import (
    BranchAttention,
    CausalRetrieval,
    LocalHistoryEncoder,
    PeriodicLSTMEncoder,
    PredictionHead,
    RetrievalFusion,
)

WEATHER_CHANNELS = 2  # 기온, 강수 + g·적설


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
        retrieval_radius = (
            config.local_radius if config.retrieval_local_radius is None
            else config.retrieval_local_radius
        )
        if retrieval_radius < 0:
            raise ValueError('retrieval_local_radius must be non-negative')
        if config.fusion_dim <= 0 or config.retrieval_k <= 0 or config.retrieval_chunk_size <= 0:
            raise ValueError('fusion_dim, retrieval_k, and retrieval_chunk_size must be positive')
        if config.transformer_heads <= 0 or config.d_model % config.transformer_heads != 0:
            raise ValueError('d_model must be divisible by transformer_heads')
        if config.weekday_dim <= 0 or config.hour_dim <= 0:
            raise ValueError('weekday_dim and hour_dim must be positive')
        missing = [
            name for name in ('temperature_min', 'temperature_max', 'precipitation_max')
            if getattr(config, name) is None
        ]
        if missing:
            raise ValueError(
                f'날씨 정규화 통계가 없음: {missing}. train.py가 train 구간에서 계산해 넘겨야 한다'
            )
        # 설정값은 Python 값으로 검사한다. ``from_pretrained``의 meta-device 초기화와 호환된다.
        self.temperature_min = float(config.temperature_min)
        self.temperature_range = float(config.temperature_max) - self.temperature_min
        self.precipitation_max = float(config.precipitation_max)
        if self.temperature_range <= 0:
            raise ValueError('temperature_max는 temperature_min보다 커야 함')
        if self.precipitation_max <= 0:
            raise ValueError('precipitation_max는 양수여야 함')

        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = config.time_step

        self.use_daily = config.use_daily
        self.use_weekly = config.use_weekly
        self.use_retrieval = config.use_retrieval
        self.use_weather = config.use_weather
        if config.weather_injection not in ('concat', 'cls_add'):
            raise ValueError(
                f"weather_injection은 'concat'|'cls_add' (받음: {config.weather_injection!r})"
            )
        self.weather_injection = config.weather_injection
        self.use_calendar = config.use_calendar
        self.use_branch_attention = config.use_branch_attention
        self.use_neighbors = config.use_neighbors
        self.use_softplus = config.use_softplus

        self.node_adaptive = config.node_adaptive
        node_adaptive_indices = None
        if self.node_adaptive:
            if config.node_adaptive_indices is None:
                raise ValueError(
                    'node_adaptive=True면 node_adaptive_indices(대상 노드 id 목록)가 필요함 — '
                    '시간 리크를 막으려면 train.py가 train 구간 평균 수요로 계산해 넘겨야 한다'
                )
            node_adaptive_indices = [int(value) for value in config.node_adaptive_indices]

        self.snow_scale = nn.Parameter(torch.tensor(1.0))
        self.weekday_embedding = nn.Embedding(7, config.weekday_dim)
        self.hour_embedding = nn.Embedding(24, config.hour_dim)
        self.weather_cls_dim = WEATHER_CHANNELS if self.weather_injection == 'cls_add' else 0
        self.extra_dim = (
            (WEATHER_CHANNELS if self.weather_injection == 'concat' else 0)
            + config.weekday_dim
            + config.hour_dim
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
            attention_dropout=config.attention_dropout,
            extra_dim=self.extra_dim,
            weather_cls_dim=self.weather_cls_dim,
            use_neighbors=self.use_neighbors,
            node_adaptive_indices=node_adaptive_indices,
        )
        self.periodic_hidden = config.periodic_hidden
        self.daily_branch = PeriodicLSTMEncoder(config.periodic_hidden, extra_dim=self.extra_dim)
        self.weekly_branch = PeriodicLSTMEncoder(config.periodic_hidden, extra_dim=self.extra_dim)
        self.branch_attention = BranchAttention(
            config.history_hidden,
            config.periodic_hidden,
            config.fusion_dim,
            use_attention=self.use_branch_attention,
        )
        self.retrieval = (
            CausalRetrieval(
                height=height,
                width=width,
                time_step=config.time_step,
                local_radius=retrieval_radius,
                retrieval_grid_path=config.retrieval_grid_path,
                retrieval_k=config.retrieval_k,
                retrieval_chunk_size=config.retrieval_chunk_size,
                retrieval_train_end=config.retrieval_train_end,
                future_mask_hours=config.retrieval_future_mask_hours,
            )
            if self.use_retrieval else None
        )
        if self.use_retrieval and config.retrieval_value_cap is None:
            raise ValueError(
                'use_retrieval=True면 retrieval_value_cap이 필요함 - train.py가 train 구간 상위 0.5% '
                '수요 경계를 계산해 넘겨야 한다'
            )
        self.retrieval_fusion = (
            RetrievalFusion(
                (2 * retrieval_radius + 1) ** 2,
                config.history_hidden,
                config.retrieval_embedding_dim,
                int(config.retrieval_value_cap),
            )
            if self.use_retrieval else None
        )
        # 속성 이름은 기존 체크포인트 키(output_gate.neural_head.*)와 맞추려고 유지한다.
        self.output_gate = PredictionHead(config.fusion_dim, use_softplus=self.use_softplus)
        self.loss_fn: nn.Module
        self.configure_loss(config.loss_type)

        self.post_init()

    def configure_loss(self, loss_type: str, *, rmse_weight: float | None = None) -> None:
        """학습 손실을 교체하고 ``config``의 손실 설정을 동기화한다."""

        if rmse_weight is not None:
            self.config.rmse_weight = float(rmse_weight)
        self.loss_fn = build_loss(
            loss_type,
            gamma=self.config.loss_gamma,
            eps=self.config.loss_eps,
            rmse_weight=self.config.rmse_weight,
            split_threshold=self.config.split_threshold,
            split_high_weight=self.config.split_high_weight,
        )
        self.loss_is_scalar = loss_type in SCALAR_LOSS_TYPES
        # property 금지: PreTrainedModel.__init__이 같은 이름에 대입한다.
        self.loss_type = loss_type
        self.config.loss_type = loss_type

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        """노드별 weight offset 파라미터 전부(node_adaptive가 꺼져 있으면 빈 튜플)."""

        return self.local_history.node_delta_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        """PyTorch 기본 초기화를 유지하고 초기화되지 않은 노드 delta만 0으로 만든다.

        HF가 meta device에서 누락된 체크포인트 키를 실체화할 수 있으므로, 체크포인트에서
        복원된 delta에는 다시 0을 쓰지 않는다.
        """

        if module is not self.local_history:
            return
        for param in module.node_delta_parameters():
            if not getattr(param, '_is_hf_initialized', False):
                param.data.zero_()


    def _weather_features(self, weather: Tensor) -> Tensor:
        """``[B,L,3]`` (기온, 강수, 적설) -> ``[B,L,2]`` (기온, 강수 + g·적설) 정규화."""

        temperature, rainfall, snowfall = weather.unbind(dim=-1)
        temperature_norm = (temperature - self.temperature_min) / self.temperature_range
        precipitation_norm = (rainfall + self.snow_scale * snowfall) / self.precipitation_max
        features = torch.stack([temperature_norm, precipitation_norm], dim=-1)
        return features if self.use_weather else torch.zeros_like(features)

    def _temporal_extra(self, weather: Tensor, hour: Tensor, day_of_week: Tensor) -> Tensor:
        """한 브랜치의 시간 context를 만들고 ``[B, L, extra_dim]``을 반환한다.

        날씨는 ``concat``에서만 포함하고, 요일·시간 임베딩은 브랜치 간 공유한다.
        """

        parts: list[Tensor] = []
        if self.weather_injection == 'concat':
            parts.append(self._weather_features(weather))
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
        """모델 계산을 수행하고 예측·중간 결과·선택적 손실을 반환한다."""

        recent_extra = self._temporal_extra(weather, hour_of_day, day_of_week)
        daily_extra = self._temporal_extra(daily_weather, daily_hour, daily_day_of_week)
        weekly_extra = self._temporal_extra(weekly_weather, weekly_hour, weekly_day_of_week)

        weather_cls = self._weather_features(weather) if self.weather_cls_dim else None
        h_neural = self.local_history(demand_history, recent_extra, weather_cls)
        if self.retrieval is not None:
            retrieval_scores, retrieval_values = self.retrieval(sample_idx)
            h_local = self.retrieval_fusion(h_neural, retrieval_scores, retrieval_values)
        else:
            retrieval_scores = retrieval_values = None
            h_local = h_neural
        # Ablation은 브랜치 출력을 0으로 바꾸되 valid mask와 모듈 shape은 유지한다.
        h_daily, daily_valid = self.daily_branch(daily_demand, daily_mask, daily_extra)
        if not self.use_daily:
            h_daily = torch.zeros_like(h_daily)
        h_weekly, weekly_valid = self.weekly_branch(weekly_demand, weekly_mask, weekly_extra)
        if not self.use_weekly:
            h_weekly = torch.zeros_like(h_weekly)
        h_attn, attention_weights = self.branch_attention(
            h_local, h_daily, h_weekly, daily_valid, weekly_valid
        )

        prediction = self.output_gate(h_attn)
        prediction_grid = prediction.reshape(-1, self.height, self.width)

        output: dict[str, Tensor | None] = {
            'logits': prediction_grid,
            'prediction': prediction_grid,
            'prediction_flat': prediction,
            'attention_weights': attention_weights,
            'h_neural': h_neural,
            'h_local': h_local,
            'retrieval_scores': retrieval_scores,
            'retrieval_values': retrieval_values,
            'h_attn': h_attn,
            'daily_valid': daily_valid,
            'weekly_valid': weekly_valid,
        }
        if labels is not None:
            if self.loss_is_scalar:
                # 스칼라 손실은 원소별 loss_sum을 정의할 수 없다.
                output['loss'] = self.loss_fn(prediction_grid, labels)
            else:
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
        """``Trainer``용으로 ``{'loss', 'logits'}``만 반환한다.

        중간 결과가 필요하면 ``forward_debug``를 사용한다.
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
        """검증과 분석을 위한 전체 예측·중간 결과·선택적 손실을 반환한다."""

        return self._compute(**batch)


__all__ = ['MergedDemandModel']
