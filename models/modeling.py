from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import GridDemandConfig
from .embeddings import FourierScalarEmbedding
from .losses import CombinedLoss

CLS_TOKEN_ID = 0
EDGE_TOKEN_ID = 1


class GridDemandModel(PreTrainedModel):
    config_class = GridDemandConfig
    base_model_prefix = 'grid_demand'

    def __init__(self, config: GridDemandConfig) -> None:
        super().__init__(config)

        self.a = config.a
        self.H = config.H
        self.W = config.W
        self.n_side = 2 * config.a + 1
        self.n_neighbors = self.n_side * self.n_side
        self.seq_len = self.n_neighbors + 1  # + CLS

        self.scalar_embed = FourierScalarEmbedding(config.d_model)
        self.special_embed = nn.Embedding(2, config.d_model)  # 0=CLS, 1=EDGE(격자 밖)
        self.node_embed = nn.Embedding(self.H * self.W, config.d_model)  # 노드 고유 임베딩 (node_id로 조회)
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(config.lstm_hidden, 1)

        # 날씨(Linear 투영, Lambda-F 스타일) + 캘린더(ADFormer의 DataEmbedding과 동일한 임베딩
        # 테이블 방식) — 둘 다 CLS 토큰에 더해져서 주입된다(forward 참고).
        self.weather_proj = nn.Linear(3, config.d_model)
        self.daytime_embedding = nn.Embedding(1440, config.d_model)
        self.weekday_embedding = nn.Embedding(7, config.d_model)

        # daily/weekly 주기 브랜치: "정확히 같은 시각"의 과거 수요를 recent와는 별개의 LSTM으로
        # 인코딩한다. 공간 인코딩(_spatial_encode: scalar_embed/special_embed/node_embed/pos_embed/
        # encoder)은 recent/daily/weekly가 전부 공유하지만, 이 LSTM들은 브랜치마다 독립된 가중치다.
        self.daily_lstm = nn.LSTM(config.d_model, config.lstm_hidden, batch_first=True)
        self.weekly_lstm = nn.LSTM(config.d_model, config.lstm_hidden, batch_first=True)
        # 0-init: 학습 시작 시점엔 daily/weekly의 기여가 정확히 0이라 순수 ir-weather와 동일하게
        # 시작하고, 필요한 만큼만 학습되며 서서히 섞여 들어간다("중립 보정" 초기화).
        # [ablation: assemble-random-init] "중립 보정"(0-init) 트릭을 안 쓰고 전부 랜덤 초기화 그대로
        # 둔다 — daily_projection/weekly_projection은 post_init()이 적용하는 기본
        # normal_(0,0.02)/bias=0을 그대로 유지(zeros_ 호출 삭제).
        self.daily_projection = nn.Linear(config.lstm_hidden, config.lstm_hidden)
        self.weekly_projection = nn.Linear(config.lstm_hidden, config.lstm_hidden)

        # recent를 Query, daily/weekly를 Key/Value 후보로 하는 attention 융합. 학습 가능한
        # "null" key를 하나 더 둬서 attention이 "daily도 weekly도 안 쓰겠다"를 선택할 자유를 준다.
        self.periodic_attn_query = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        self.periodic_attn_key = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        # [ablation: assemble-random-init] null key도 0이 아니라 다른 임베딩류(pos_embed 등)와
        # 동일한 스케일의 랜덤값으로 초기화.
        self.periodic_null_key = nn.Parameter(torch.randn(config.lstm_hidden) * 0.02)

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        if config.weather_mean is None or config.weather_std is None:
            raise ValueError("weather_mean/weather_std(3개씩, train split 통계)가 필요함")
        # 작은 buffer(3개짜리)라 idx_table과 같은 범주 — persistent=True로 충분, 검색 DB처럼
        # from_pretrained 이후 수동 재로드가 필요 없다.
        self.register_buffer(
            'weather_mean', torch.tensor(config.weather_mean, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            'weather_std', torch.tensor(config.weather_std, dtype=torch.float32), persistent=True
        )

        # node_id -> (padded grid 기준 이웃 (2a+1)^2개의 flat index, 격자 안 여부)는
        # H,W,a로만 정해지는 순수 함수라 가능한 모든 node_id(H*W개)에 대해 미리 계산해둔다.
        # persistent=True로 저장해야 함: persistent=False 버퍼는 transformers 5.0의
        # from_pretrained가 meta-device에서 fast-init할 때 체크포인트에 없는 값이라
        # 복원되지 않고 값이 깨지는 문제가 있음(models/modeling.py 개발 중 확인함).
        idx_table, mask_table = self._build_neighbor_tables()
        self.register_buffer('idx_table', idx_table, persistent=True)  # (H*W, n_neighbors)
        self.register_buffer('mask_table', mask_table, persistent=True)  # (H*W, n_neighbors)

        self.post_init()
        # [ablation: assemble-random-init] "중립 보정" 0-init을 여기서 하지 않음 — daily_projection/
        # weekly_projection/periodic_null_key 전부 랜덤 초기화 그대로 학습을 시작한다(assemble의
        # 0-init 버전과 비교하기 위한 실험).

    def _build_neighbor_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        H, W, a = self.H, self.W, self.a
        Wp = W + 2 * a

        offsets = torch.arange(-a, a + 1)
        grid_di, grid_dj = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_di = grid_di.reshape(-1)  # (n_neighbors,)
        offset_dj = grid_dj.reshape(-1)

        node_ids = torch.arange(H * W)
        h = torch.div(node_ids, W, rounding_mode='floor')  # (H*W,)
        w = node_ids % W

        rows = h.unsqueeze(-1) + a + offset_di.unsqueeze(0)  # (H*W, n_neighbors)
        cols = w.unsqueeze(-1) + a + offset_dj.unsqueeze(0)  # (H*W, n_neighbors)
        idx = rows * Wp + cols  # padded 좌표계 기준이라 항상 유효 범위

        valid = torch.zeros(H + 2 * a, W + 2 * a, dtype=torch.bool)
        valid[a:a + H, a:a + W] = True
        mask = valid.reshape(-1)[idx]  # (H*W, n_neighbors)

        return idx, mask

    # NOTE: _init_weights를 오버라이드하지 않음 — transformers 5.0의 from_pretrained에서
    # 커스텀 _init_weights가 건드리는 nn.Linear/nn.Embedding 모듈만 체크포인트 로드 후에
    # 다시 랜덤 초기화되는 버그가 확인됨 (models/modeling.py 개발 중 재현/검증함).
    # PyTorch 기본 초기화(Linear: kaiming_uniform, Embedding: normal)를 그대로 사용한다.

    def _crop_all_nodes(self, demands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """demands: (B, k, H, W) -> values (B,k,N,n_neighbors), mask (N,n_neighbors).

        N=H*W. idx_table/mask_table은 이미 모든 node_id(0..N-1)에 대해 미리 계산돼 있으므로,
        (예전처럼 일부 행만 고르지 않고) 전체를 그대로 사용해 N개 노드 전부의 이웃을 한 번에 뽑는다.
        노드별 계산 자체(각 노드가 자기 이웃만 보는 것)는 예전과 동일 — 배치 차원만 늘어난다.
        """
        B, k, H, W = demands.shape
        a = self.a
        N = H * W

        padded = F.pad(demands, (a, a, a, a), mode='constant', value=0.0)  # (B,k,H+2a,W+2a)
        padded_flat = padded.reshape(B, k, -1)

        flat_idx = self.idx_table.reshape(-1)  # (N*n_neighbors,)
        values = padded_flat[:, :, flat_idx].reshape(B, k, N, self.n_neighbors)

        return values, self.mask_table  # mask: (N, n_neighbors), 배치/시간에 안 붙어도 브로드캐스트됨

    def _spatial_encode(
        self,
        demand_window: torch.Tensor,
        weather: torch.Tensor | None = None,
        hour_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """demand_window: (B,k,H,W) -> (B,k,N,d_model).

        recent/daily/weekly 세 브랜치가 전부 이 메서드를 그대로 호출한다 -> scalar_embed/
        special_embed/node_embed/pos_embed/encoder(로컬 (2a+1)^2 공간 인코딩)를 공유한다. k(시퀀스
        길이)는 브랜치마다 다를 수 있다(recent=time_step, daily=daily_lag_count 등) — 이 메서드
        자체는 k에 무관하게 동작한다.

        weather/hour_of_day/day_of_week는 recent 브랜치에서만 넘어온다(daily/weekly는 CLS에
        날씨/캘린더를 더하지 않음 — main model도 주기 브랜치엔 캘린더/날씨를 안 넣는 것과 동일한
        설계, GridDemandModel.forward 참고).
        """
        B, k, H, W = demand_window.shape
        N = H * W
        d_model = self.config.d_model

        values, mask = self._crop_all_nodes(demand_window)  # (B,k,N,n) / (N,n)
        log_values = torch.log1p(values.clamp(min=0))
        neighbor_emb = self.scalar_embed(log_values)  # (B,k,N,n,d_model)

        edge_emb = self.special_embed.weight[EDGE_TOKEN_ID]  # (d_model,)
        mask_expanded = mask.view(1, 1, N, self.n_neighbors, 1).expand(B, k, N, self.n_neighbors, d_model)
        neighbor_emb = torch.where(mask_expanded, neighbor_emb, edge_emb)

        cls_base = self.special_embed.weight[CLS_TOKEN_ID]  # (d_model,)
        node_vecs = self.node_embed.weight  # (N, d_model)
        cls_emb = (cls_base + node_vecs).view(1, 1, N, 1, d_model).expand(B, k, N, 1, d_model)
        if weather is not None:
            weather_norm = (weather - self.weather_mean) / self.weather_std  # (B,k,3)
            weather_emb = self.weather_proj(weather_norm)  # (B,k,d_model)
            daytime_idx = (hour_of_day.float() / 24.0 * 1440).round().long().clamp(0, 1439)
            temporal_extra = (
                weather_emb + self.daytime_embedding(daytime_idx) + self.weekday_embedding(day_of_week)
            )  # (B,k,d_model)
            cls_emb = cls_emb + temporal_extra.view(B, k, 1, 1, d_model)

        seq = torch.cat([cls_emb, neighbor_emb], dim=3)  # (B,k,N,seq_len,d_model)
        seq = seq + self.pos_embed  # (seq_len,d_model) 브로드캐스트

        seq = seq.reshape(B * k * N, self.seq_len, d_model)
        encoded = self.encoder(seq)  # (B*k*N, seq_len, d_model)
        return encoded[:, 0, :].reshape(B, k, N, d_model)

    def forward(
        self,
        demands: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
        hour_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
        daily_demands: torch.Tensor | None = None,
        weekly_demands: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if weather is None or hour_of_day is None or day_of_week is None:
            raise ValueError("이 모델은 날씨/캘린더 피처가 필수라 weather/hour_of_day/day_of_week가 필요함")
        if daily_demands is None or weekly_demands is None:
            raise ValueError("이 모델은 daily/weekly 주기 브랜치가 필수라 daily_demands/weekly_demands가 필요함")

        B, k, H, W = demands.shape
        N = H * W
        d_model = self.config.d_model
        lstm_hidden = self.config.lstm_hidden

        # recent 공간 인코딩 (날씨/캘린더 CLS에 포함) -> LSTM(recent 전용 가중치).
        cls_out = self._spatial_encode(demands, weather, hour_of_day, day_of_week)  # (B,k,N,d_model)
        cls_out = cls_out.permute(0, 2, 1, 3).reshape(B * N, k, d_model)
        lstm_out, _ = self.lstm(cls_out)  # (B*N,k,lstm_hidden)
        last = lstm_out[:, -1, :].reshape(B, N, -1)  # (B,N,lstm_hidden)

        # daily/weekly 공간 인코딩은 recent와 동일한 _spatial_encode(=동일 가중치)를 쓰지만
        # 날씨/캘린더는 안 넣는다. 각자 독립된 LSTM(daily_lstm/weekly_lstm)으로 취합.
        daily_k = daily_demands.shape[1]
        weekly_k = weekly_demands.shape[1]
        daily_cls = self._spatial_encode(daily_demands)  # (B,daily_k,N,d_model)
        weekly_cls = self._spatial_encode(weekly_demands)  # (B,weekly_k,N,d_model)
        daily_cls = daily_cls.permute(0, 2, 1, 3).reshape(B * N, daily_k, d_model)
        weekly_cls = weekly_cls.permute(0, 2, 1, 3).reshape(B * N, weekly_k, d_model)
        _, (_, daily_h) = self.daily_lstm(daily_cls)
        _, (_, weekly_h) = self.weekly_lstm(weekly_cls)
        daily_last = self.daily_projection(daily_h[-1]).reshape(B, N, lstm_hidden)  # 랜덤 초기화 투영
        weekly_last = self.weekly_projection(weekly_h[-1]).reshape(B, N, lstm_hidden)

        # recent(Query) <-> {daily, weekly, null}(Key) attention 융합. null 옵션 덕분에 attention이
        # 주기 브랜치를 아예 안 쓰는 것도 선택할 수 있다.
        periodic = torch.stack([daily_last, weekly_last], dim=2)  # (B,N,2,lstm_hidden)
        query = self.periodic_attn_query(last).unsqueeze(2)  # (B,N,1,lstm_hidden)
        keys = self.periodic_attn_key(periodic)  # (B,N,2,lstm_hidden)
        branch_scores = (query * keys).sum(-1) / (lstm_hidden ** 0.5)  # (B,N,2)
        null_score = (query * self.periodic_null_key.view(1, 1, 1, -1)).sum(-1)  # (B,N,1)
        weights = torch.softmax(torch.cat([branch_scores, null_score], dim=-1), dim=-1)  # (B,N,3)
        periodic_correction = weights[..., 0:1] * daily_last + weights[..., 1:2] * weekly_last  # (B,N,lstm_hidden), vec_2

        # 스케일 보정: vec_1(last)은 원래 매번 풀스케일로 더해지고 vec_2(periodic_correction)만
        # 샘플마다(null-option이 얼마나 이겼는지에 따라) 크기가 흔들려서, 단순히 last + gate*vec_2로
        # 더하면 fused의 노름이 "주기 정보를 얼마나 섞었는지"에 따라 들쭉날쭉해진다. vec_2가 gate만큼
        # 관여하는 크기를 vec_1의 노름 대비 비율로 계산해 vec_1을 그만큼 깎아준다 -> null이 이겨서
        # periodic_correction≈0이면 correction≈1(vec_1 그대로, null-option의 "순수 recent" 의미 유지),
        # daily/weekly가 강하게 관여할수록 vec_1의 기여가 줄어 fused의 스케일이 안정된다(vec_1/vec_2가
        # 평행하지 않으므로 완벽한 노름 보존은 아니지만, vec_1이 항상 풀스케일로 고정되던 문제는 완화됨).
        eps = 1e-6
        vec1_norm = last.norm(dim=-1, keepdim=True)  # (B,N,1)
        vec2_norm = periodic_correction.norm(dim=-1, keepdim=True)  # (B,N,1)
        scale_correction = (1.0 - self.config.residual_scale * vec2_norm / (vec1_norm + eps)).clamp(min=0.0)
        fused = scale_correction * last + self.config.residual_scale * periodic_correction  # (B,N,lstm_hidden)

        pred = F.softplus(self.output_proj(fused)).squeeze(-1)  # (B,N)

        logits = pred.reshape(B, H, W)  # labels(B,H,W)와 동일 shape, node_id=row*W+col 순서와 일치

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
