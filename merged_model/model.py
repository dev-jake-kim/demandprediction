"""High-level orchestration for the unified demand model.

Branch-specific code lives in :mod:`components`; this file intentionally
contains only model construction, data flow, and the final loss calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from .data import NUM_WEATHER_FEATURES
from .losses import build_loss
from .modules import (
    BranchAttention,
    CausalRetrieval,
    LocalHistoryEncoder,
    NeuralRetrievalGate,
    PeriodicLSTMEncoder,
)


class UnifiedDemandModel(nn.Module):
    """One end-to-end model for neural, periodic, and retrieval information."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        time_step: int = 24,
        local_radius: int = 2,
        d_model: int = 64,
        num_fourier_bands: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
        history_hidden: int = 64,
        periodic_hidden: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        retrieval_grid_path: str | Path | None = None,
        retrieval_k: int = 20,
        retrieval_chunk_size: int = 256,
        retrieval_scope: Literal["observed_past", "train_prefix"] = "observed_past",
        retrieval_train_end: int | None = None,
        weather_mean: Sequence[float] | None = None,
        weather_std: Sequence[float] | None = None,
        weekday_dim: int = 7,
        hour_dim: int = 5,
        use_daily: bool = True,
        use_weekly: bool = True,
        use_retrieval: bool = True,
        use_weather: bool = True,
        weather_injection: str = "concat",
        use_calendar: bool = True,
        use_branch_attention: bool = True,
        loss_type: str = "combined",
        loss_gamma: float = 1.0,
        loss_eps: float = 0.5,
    ) -> None:
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        if local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if fusion_dim <= 0 or retrieval_k <= 0 or retrieval_chunk_size <= 0:
            raise ValueError("fusion_dim, retrieval_k, and retrieval_chunk_size must be positive")
        if transformer_heads <= 0 or d_model % transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads")
        if weekday_dim <= 0 or hour_dim <= 0:
            raise ValueError("weekday_dim and hour_dim must be positive")
        if weather_mean is None or weather_std is None:
            raise ValueError(
                "weather_mean/weather_std(각 3개, train split 통계)가 필요함 — "
                "시간 리크를 막으려면 train.py가 train 구간에서만 계산해 넘겨야 한다"
            )

        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = time_step

        # --- ablation 스위치 ---
        # 각 모듈을 끄고 켜면서 기여도를 재기 위한 것. 전부 True면 기본 모델과 동일하다.
        self.use_daily = use_daily
        self.use_weekly = use_weekly
        self.use_retrieval = use_retrieval
        self.use_weather = use_weather
        # 'concat': 정규화 3값을 세 LSTM 입력에 그대로 붙인다(기본).
        # 'cls_add': ir-weather 방식 — d_model로 투영해 history의 CLS 토큰에 더한다.
        #   이 경우 주기 브랜치에는 날씨가 들어가지 않는다(CLS가 거기엔 없다). 즉 주입 지점과
        #   커버리지가 함께 바뀌는 변형이며, 그건 ir-weather 설계의 본질적 성질이다.
        if weather_injection not in ("concat", "cls_add"):
            raise ValueError(f"weather_injection은 'concat'|'cls_add' (받음: {weather_injection!r})")
        self.weather_injection = weather_injection
        self.use_calendar = use_calendar
        self.use_branch_attention = use_branch_attention

        # 날씨는 임베딩하지 않고 정규화한 3값을 그대로 LSTM 입력에 concat한다. 요일/시간대만
        # 임베딩 테이블을 쓰며, 세 브랜치가 같은 테이블을 공유한다(요일 3은 어느 브랜치에서나 요일 3).
        self.weekday_embedding = nn.Embedding(7, weekday_dim)
        # ir-weather는 nn.Embedding(1440, d_model)에 hour*60 인덱스를 넣지만 실제로 학습되는 행은
        # 24개뿐인 분(minute) 해상도 잔재다 — 동일 효과의 24행으로 단순화한다.
        self.hour_embedding = nn.Embedding(24, hour_dim)
        # 폭은 끄든 켜든 15로 고정 — 값만 0이 된다. 폭이 바뀌면 LSTM 파라미터 수가 달라져
        # 정보 제거 효과와 용량 감소 효과가 섞인다.
        self.weather_cls_dim = NUM_WEATHER_FEATURES if weather_injection == "cls_add" else 0
        self.extra_dim = (
            (NUM_WEATHER_FEATURES if weather_injection == "concat" else 0) + weekday_dim + hour_dim
        )

        weather_mean_tensor = torch.tensor(weather_mean, dtype=torch.float32)
        weather_std_tensor = torch.tensor(weather_std, dtype=torch.float32)
        if weather_mean_tensor.shape != (NUM_WEATHER_FEATURES,):
            raise ValueError(f"weather_mean must have {NUM_WEATHER_FEATURES} entries")
        if weather_std_tensor.shape != (NUM_WEATHER_FEATURES,):
            raise ValueError(f"weather_std must have {NUM_WEATHER_FEATURES} entries")
        if bool((weather_std_tensor <= 0).any()):
            raise ValueError("weather_std must be positive (0으로 나누기 방지)")
        self.register_buffer("weather_mean", weather_mean_tensor, persistent=True)
        self.register_buffer("weather_std", weather_std_tensor, persistent=True)

        self.local_history = LocalHistoryEncoder(
            height=height,
            width=width,
            time_step=time_step,
            local_radius=local_radius,
            d_model=d_model,
            num_fourier_bands=num_fourier_bands,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn=transformer_ffn,
            history_hidden=history_hidden,
            dropout=dropout,
            extra_dim=self.extra_dim,
            weather_cls_dim=self.weather_cls_dim,
        )
        self.periodic_hidden = periodic_hidden
        # ablation은 "0-치환" 방식이다 — 모듈을 없애지 않고 항상 생성·실행한 뒤, 그 모듈이
        # 결과로 이어지는 텐서만 0으로 바꾼다. 모듈을 지우면 텐서 shape·파라미터 수·attention
        # 후보 개수까지 함께 바뀌어, 측정된 차이가 "그 모듈의 정보" 때문인지 "구조 변화" 때문인지
        # 분리되지 않는다. 0-치환은 그 교란을 없앤다.
        self.daily_branch = PeriodicLSTMEncoder(periodic_hidden, extra_dim=self.extra_dim)
        self.weekly_branch = PeriodicLSTMEncoder(periodic_hidden, extra_dim=self.extra_dim)
        self.branch_attention = BranchAttention(
            history_hidden, periodic_hidden, fusion_dim, use_attention=use_branch_attention
        )
        # 검색기도 끄든 켜든 항상 만들고 항상 계산한다(느리지만 경로가 동일해진다).
        # use_retrieval=False면 ir_out만 0이 되고, 게이트는 그대로 남아 lambda를 학습한다
        # — 처음부터 재학습하므로 모델이 lambda->1을 배워 보정할 수 있다.
        self.retrieval = CausalRetrieval(
            height=height,
            width=width,
            time_step=time_step,
            local_radius=local_radius,
            retrieval_grid_path=retrieval_grid_path,
            retrieval_k=retrieval_k,
            retrieval_chunk_size=retrieval_chunk_size,
            retrieval_scope=retrieval_scope,
            retrieval_train_end=retrieval_train_end,
        )
        self.output_gate = NeuralRetrievalGate(fusion_dim)
        # 이 저장소의 다른 모델들과 목적함수를 맞추려면 'combined'(CombinedLoss)를 쓴다.
        # 'mae'는 merged_model이 원래 쓰던 raw 스케일 L1이며, 두 경우 모두 reduction='none'
        # 이라 아래 forward에서 mean/sum을 각각 뽑는다.
        self.loss_type = loss_type
        self.loss_fn = build_loss(loss_type, gamma=loss_gamma, eps=loss_eps)

    # Keep the old debugging entry points available while the implementation
    # is organized under named components.
    def _crop_all_nodes(self, demands: Tensor) -> Tensor:
        return self.local_history.crop(demands)

    def _retrieve(self, local_crop: Tensor, sample_idx: Tensor) -> Tensor:
        return self.retrieval(local_crop, sample_idx)

    def _temporal_extra(
        self, weather: Tensor, hour: Tensor, day_of_week: Tensor
    ) -> Tensor:
        """(정규화 날씨 3) ⊕ (요일 임베딩) ⊕ (시간대 임베딩) -> ``[B, L, extra_dim]``.

        세 브랜치가 같은 방식으로 만든다 — recent는 L=time_step, daily/weekly는 L=lag 개수.
        정규화를 여기 한 곳에서만 하므로 날것의 기온(수십 단위)이 log1p 수요를 압도하는 일이 없다.
        """

        parts: list[Tensor] = []
        if self.weather_injection == "concat":
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

    def forward(
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
        target: Tensor | None = None,
    ) -> dict[str, Tensor]:
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

        ir_out = self.retrieval(local_crop, sample_idx)
        if not self.use_retrieval:
            ir_out = torch.zeros_like(ir_out)
        neural_pred, lambda_weight, prediction = self.output_gate(h_attn, ir_out)
        prediction_grid = prediction.reshape(-1, self.height, self.width)

        output = {
            "prediction": prediction_grid,
            "prediction_flat": prediction,
            "neural_pred": neural_pred,
            "ir_out": ir_out,
            "lambda_weight": lambda_weight,
            "attention_weights": attention_weights,
            "h_neural": h_neural,
            "h_attn": h_attn,
            "daily_valid": daily_valid,
            "weekly_valid": weekly_valid,
        }
        if target is not None:
            # reduction='none'으로 한 번만 계산하고 mean(역전파용)과 sum(에폭 집계용)을 함께 낸다.
            # 배치 크기가 균일하지 않아(drop_last=False) 배치 평균의 평균은 원소 평균과 다르다.
            elementwise = self.loss_fn(prediction_grid, target)
            output["loss"] = elementwise.mean()
            output["loss_sum"] = elementwise.detach().sum()
        return output


__all__ = ["UnifiedDemandModel"]
