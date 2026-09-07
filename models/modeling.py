from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import GridDemandConfig
from .embeddings import FourierScalarEmbedding
from .losses import CombinedLoss, RmseMapeLoss

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

        # 노드별 weight offset. self.lstm / self.output_proj는 모듈 그대로 두고(체크포인트 키
        # 호환 + PyTorch 기본 초기화 유지) 여기의 delta만 더해서 쓴다. forward는 호출 대신
        # 파라미터를 직접 읽는다(_node_adaptive_lstm 참고).
        # delta는 0으로 시작한다 — 학습 시작 시점의 모델이 baseline-weather와 완전히 동일해야
        # "baseline에 노드별 offset만 추가"라는 이 실험의 전제가 성립한다.
        self.node_adaptive = config.node_adaptive
        if self.node_adaptive:
            if config.lstm_layers != 1:
                raise ValueError(
                    f'node_adaptive는 lstm_layers=1만 지원함 (받음: {config.lstm_layers}) — '
                    '다층 노드별 LSTM은 이번 실험 범위 밖이다'
                )
            n_nodes = self.H * self.W
            n_gates = 4 * config.lstm_hidden
            self.node_delta_weight_ih = nn.Parameter(torch.zeros(n_nodes, n_gates, config.d_model))
            self.node_delta_weight_hh = nn.Parameter(torch.zeros(n_nodes, n_gates, config.lstm_hidden))
            # PyTorch LSTM은 bias_ih/bias_hh 두 개를 갖지만 계산에는 둘의 합만 들어간다
            # (cuDNN 호환용 중복). 노드별로 둘을 따로 두면 파라미터만 2배가 되고 표현력은
            # 그대로이므로 합에 더하는 하나만 둔다.
            self.node_delta_bias = nn.Parameter(torch.zeros(n_nodes, n_gates))
            self.node_delta_out_weight = nn.Parameter(torch.zeros(n_nodes, config.lstm_hidden))

        # 날씨(Linear 투영, Lambda-F 스타일) + 캘린더(ADFormer의 DataEmbedding과 동일한 임베딩
        # 테이블 방식) — 둘 다 CLS 토큰에 더해져서 주입된다(forward 참고).
        self.weather_proj = nn.Linear(3, config.d_model)
        self.daytime_embedding = nn.Embedding(1440, config.d_model)
        self.weekday_embedding = nn.Embedding(7, config.d_model)

        self.loss_fn: nn.Module
        self.configure_loss(
            config.loss_type,
            rmse_weight=config.rmse_weight,
        )

        if config.weather_mean is None or config.weather_std is None:
            raise ValueError("weather_mean/weather_std(3개씩, train split 통계)가 필요함")
        # 작은 buffer(3개짜리)라 idx_table과 같은 범주 — persistent=True로 충분, ir의 검색 DB처럼
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

    def configure_loss(
        self,
        loss_type: str,
        *,
        rmse_weight: float | None = None,
    ) -> None:
        """학습 loss를 교체하고 선택값을 HF config에도 동기화한다.

        config를 함께 갱신해야 stage 2 체크포인트를 ``from_pretrained``로 다시 열었을 때
        런타임에 선택했던 loss가 CombinedLoss로 조용히 되돌아가지 않는다.
        """
        if loss_type == 'combined':
            self.loss_fn = CombinedLoss(gamma=self.config.loss_gamma, eps=self.config.loss_eps)
        elif loss_type == 'rmse_mape':
            resolved_rmse_weight = self.config.rmse_weight if rmse_weight is None else rmse_weight
            self.loss_fn = RmseMapeLoss(
                rmse_weight=float(resolved_rmse_weight),
            )
            self.config.rmse_weight = float(resolved_rmse_weight)
        else:
            raise ValueError(f"loss_type은 'combined'|'rmse_mape' (받음: {loss_type!r})")
        self.config.loss_type = loss_type

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

    def _node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        """노드별 weight offset 파라미터 전부. 초기화/검증에서 한 곳으로 모아 쓴다."""
        return (
            self.node_delta_weight_ih,
            self.node_delta_weight_hh,
            self.node_delta_bias,
            self.node_delta_out_weight,
        )

    def _init_weights(self, module: nn.Module) -> None:
        """상속받은 초기화를 그대로 유지한 채, 노드별 delta만 추가로 0으로 만든다.

        **`super()._init_weights(module)` 호출을 빼면 안 된다.** 이 저장소는 원래 이 메서드를
        재정의하지 않아 `PreTrainedModel._init_weights`(Linear/Embedding을 std=0.02로 초기화)를
        상속해 쓰고 있었다. 여기서 조기 반환하면 모델 전체의 초기 분포가 바뀌어(node_embed의
        std가 0.02 -> 1.06으로 관측됨) baseline-weather와 같은 출발점이 아니게 된다.

        delta는 여기서 명시적으로 0을 넣어야 한다. transformers 5.0은 meta device에서 모델을
        만든 뒤 체크포인트에 없는 키를 torch.empty로 실체화하므로, 생성자의 torch.zeros(...)가
        delta 키가 없는 체크포인트(baseline-weather)를 로드할 때는 적용되지 않는다 — 실제로
        NaN이 들어오는 것을 확인했다. 체크포인트에서 값을 받은 파라미터는 _is_hf_initialized가
        붙으므로 건너뛴다(학습된 adaptive delta를 0으로 덮어쓰면 안 된다).
        """
        super()._init_weights(module)
        if module is not self or not self.node_adaptive:
            return
        for param in self._node_delta_parameters():
            if not getattr(param, '_is_hf_initialized', False):
                param.data.zero_()

    def _node_adaptive_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,N,k,d_model) -> 마지막 시점의 hidden (B,N,lstm_hidden).

        nn.LSTM은 샘플(노드)별 가중치를 받을 수 없어서 셀을 직접 돈다. 가중치는 self.lstm이
        그대로 들고 있고 여기서는 노드별 delta만 더해 쓴다. delta가 전부 0이면 이 경로의 출력은
        nn.LSTM과 수치적으로 동일하다(그래야 측정된 차이가 delta 때문임이 분리된다).

        노드 인덱싱에 gather가 필요 없다 — 이 모델은 매 forward마다 N개 노드를 전부 예측하므로
        delta의 노드 축이 einsum에서 그대로 맞물린다.
        """
        B, N, k, _ = x.shape
        weight_ih = self.lstm.weight_ih_l0 + self.node_delta_weight_ih  # (N, 4h, d_model)
        weight_hh = self.lstm.weight_hh_l0 + self.node_delta_weight_hh  # (N, 4h, lstm_hidden)
        bias = self.lstm.bias_ih_l0 + self.lstm.bias_hh_l0 + self.node_delta_bias  # (N, 4h)

        # 입력 투영은 h에 의존하지 않으므로 k개 시점을 한 번에 계산해 순차 구간을 절반으로 줄인다.
        gates_x = torch.einsum('bnkd,nfd->bnkf', x, weight_ih) + bias.unsqueeze(1)  # (B,N,k,4h)

        h = x.new_zeros(B, N, self.config.lstm_hidden)
        c = x.new_zeros(B, N, self.config.lstm_hidden)
        for t in range(k):
            gates = gates_x[:, :, t] + torch.einsum('bnh,nfh->bnf', h, weight_hh)
            i, f, g, o = gates.chunk(4, dim=-1)  # PyTorch LSTM 게이트 순서: i, f, g, o
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
        return h

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

    def forward(
        self,
        demands: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
        hour_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if weather is None or hour_of_day is None or day_of_week is None:
            raise ValueError("이 모델은 날씨/캘린더 피처가 필수라 weather/hour_of_day/day_of_week가 필요함")

        B, k, H, W = demands.shape
        N = H * W
        d_model = self.config.d_model

        values, mask = self._crop_all_nodes(demands)  # (B,k,N,n) / (N,n)

        log_values = torch.log1p(values.clamp(min=0))
        neighbor_emb = self.scalar_embed(log_values)  # (B,k,N,n,d_model)

        edge_emb = self.special_embed.weight[EDGE_TOKEN_ID]  # (d_model,)
        mask_expanded = mask.view(1, 1, N, self.n_neighbors, 1).expand(B, k, N, self.n_neighbors, d_model)
        neighbor_emb = torch.where(mask_expanded, neighbor_emb, edge_emb)

        # 날씨(예보값 가정, GridDemandDataset docstring 참고)+캘린더 -> 시점별(B,k) 임베딩 하나로
        # 합쳐서 CLS에 더한다(노드 축엔 무관하게 브로드캐스트).
        weather_norm = (weather - self.weather_mean) / self.weather_std  # (B,k,3)
        weather_emb = self.weather_proj(weather_norm)  # (B,k,d_model)
        daytime_idx = (hour_of_day.float() / 24.0 * 1440).round().long().clamp(0, 1439)  # ADFormer와 동일 변환
        temporal_extra = (
            weather_emb + self.daytime_embedding(daytime_idx) + self.weekday_embedding(day_of_week)
        )  # (B,k,d_model)

        # CLS = "CLS 마커" + "이 노드의 고유 임베딩" + "이 시점의 날씨/캘린더" -> attention이 노드
        # 정체성 + 외부 요인을 함께 알고 이웃을 취합.
        # node_id로 한 행만 고르던 걸 전체 N행(node_embed.weight)으로 바꿔 N개 노드 전부의 CLS를 구성.
        cls_base = self.special_embed.weight[CLS_TOKEN_ID]  # (d_model,)
        node_vecs = self.node_embed.weight  # (N, d_model)
        cls_emb = (cls_base + node_vecs).view(1, 1, N, 1, d_model).expand(B, k, N, 1, d_model)
        cls_emb = cls_emb + temporal_extra.view(B, k, 1, 1, d_model)  # 노드 축으로 브로드캐스트
        seq = torch.cat([cls_emb, neighbor_emb], dim=3)  # (B,k,N,seq_len,d_model)
        seq = seq + self.pos_embed  # (seq_len,d_model) 브로드캐스트

        # (B,k,N)을 하나의 배치로 합쳐 encoder에 넣음 -> 각 (b,t,node)는 서로 완전히 독립적으로 처리됨
        # (예전에 (B,k)만 합치던 것과 동일한 원리, node 차원만 추가로 합친 것뿐 -> 노드별 연산은 그대로).
        seq = seq.reshape(B * k * N, self.seq_len, d_model)
        encoded = self.encoder(seq)  # (B*k*N, seq_len, d_model)
        cls_out = encoded[:, 0, :].reshape(B, k, N, d_model)

        # 노드별로 독립적인 시계열이므로 노드 축을 배치로 돌리고 k를 시퀀스 축으로 둔다.
        cls_out = cls_out.permute(0, 2, 1, 3)  # (B,N,k,d_model)

        if self.node_adaptive:
            last = self._node_adaptive_lstm(cls_out)  # (B,N,lstm_hidden)
            # 출력 헤드도 노드마다 다른 weight/bias를 쓴다. weight가 (1,lstm_hidden)이라
            # squeeze 후 노드 축으로 브로드캐스트하면 노드별 내적이 된다.
            # bias는 노드별로 두지 않는다 — 노드마다 상수를 더하는 자리라 "이 셀의 평균 수요"를
            # 외우는 지름길이 되기 쉽고, 그러면 weight offset이 무엇을 배웠는지 해석이 흐려진다.
            out_weight = self.output_proj.weight.squeeze(0) + self.node_delta_out_weight  # (N,h)
            pred = F.softplus((last * out_weight).sum(-1) + self.output_proj.bias)  # (B,N)
        else:
            lstm_out, _ = self.lstm(cls_out.reshape(B * N, k, d_model))  # (B*N,k,lstm_hidden)
            last = lstm_out[:, -1, :].reshape(B, N, -1)  # (B,N,lstm_hidden)
            pred = F.softplus(self.output_proj(last)).squeeze(-1)  # (B,N)
        logits = pred.reshape(B, H, W)  # labels(B,H,W)와 동일 shape, node_id=row*W+col 순서와 일치

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
