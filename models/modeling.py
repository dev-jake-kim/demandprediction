"""DMVST-Net(Yao, Wu, Ke, Tang, Jia, Lu, Gong, Ye, Li. "Deep Multi-View Spatial-Temporal Network for
Taxi Demand Prediction", AAAI 2018) 모델 포팅. 3-view(Spatial/Temporal/Semantic) 구조는 논문 Figure 1,
식(1)-(8)을 따르되, 사용자가 예전에 직접 구현한 `/home/jinsu/PycharmProjects/DMVST/models/DMVSTModel.py`
(`LocalCNN`, `DMVST`)의 모델 코드를 그대로 가져와 all-node 배치 처리에 맞게 shape만 확장했다.
`docs/DMVST_PLAN.md`에 논문/기존 구현 대비 우리가 내린 결정을 정리해뒀다.

Spatial view의 이웃 crop(`idx_table`/`_crop_all_nodes`)은 `master`의 `GridDemandModel`이 쓰던
메커니즘("H*W 노드 전체의 (2a+1)^2 로컬 윈도우를 한 번의 배치 연산으로 추출")을 그대로 재사용한다
(mask_table은 재사용하지 않음 — DMVST-Net은 논문 자체가 zero-padding을 그대로 이미지 픽셀로 쓰므로,
master의 Transformer처럼 경계를 별도 EDGE 토큰으로 치환할 필요가 없음).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import DMVSTConfig
from .losses import CombinedLoss

DAY_OF_WEEK_DIM = 7
EXT_DIM = 1 + DAY_OF_WEEK_DIM  # time_in_day(1) + day_of_week one-hot(7)


class LocalCNN(nn.Module):
    """Lambda-F `models/DMVSTModel.py`의 `LocalCNN`을 그대로 포팅 — 입력 마지막 두 차원만
    (S,S) 이미지로 보고 나머지 leading 차원(B,k,N 등)은 그대로 유지한 채 배치 연산한다."""

    def __init__(self, num_filters: int, num_cnn_layers: int, kernel_size: int, patch_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        channels = 1
        padding = kernel_size // 2
        for _ in range(num_cnn_layers):
            out_channels = num_filters * channels
            self.convs.append(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding))
            self.convs.append(nn.ReLU())
            channels = out_channels
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(patch_size * patch_size * channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, s1, s2 = x.shape
        x = x.reshape(-1, 1, s1, s2)
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = F.relu(self.embedding_layer(x))  # 식(2): ŝ_t^i = f(W_fc*s_t^i + b_fc), f=ReLU
        return x.reshape(*lead, -1)


class DMVSTModel(PreTrainedModel):
    config_class = DMVSTConfig
    base_model_prefix = 'dmvst'

    def __init__(self, config: DMVSTConfig) -> None:
        super().__init__(config)
        H, W = config.H, config.W
        self.H, self.W = H, W
        self.a = (config.patch_size - 1) // 2
        self.patch_size = config.patch_size

        # master의 GridDemandModel과 동일한 이웃 테이블 메커니즘 재사용(mask는 불필요 — 위 모듈
        # docstring 참고). H,W,a로만 정해지는 순수 함수라 미리 계산해 buffer로 저장한다.
        idx_table = self._build_neighbor_table()
        self.register_buffer('idx_table', idx_table, persistent=True)  # (H*W, patch_size^2)

        self.local_cnn = LocalCNN(
            num_filters=config.num_filters,
            num_cnn_layers=config.num_cnn_layers,
            kernel_size=config.kernel_size,
            patch_size=config.patch_size,
            embedding_dim=config.demand_embedding_dim,
        )
        self.temporal_layer = nn.Linear(EXT_DIM, config.temporal_embedding_dim)

        # Semantic view: preprocessing/build_dmvst_graph.py + build_dmvst_line_embeddings.py(Torch env,
        # cogdl)가 미리 계산해둔 LINE 임베딩을 그대로 불러온다. __init__ 안에서 이 numpy 배열과 새로
        # 만든 torch 텐서를 직접 연산하지 않고(연산 없이 그대로 wrap) persistent buffer로만 등록 —
        # transformers 5.0 from_pretrained의 meta-device 버그(STResnet/ADFormer에서 확인)를 피한다.
        line_embeddings = np.load(config.line_embeddings_path).astype(np.float32)  # (N, line_dim)
        self.register_buffer('line_embeddings', torch.from_numpy(line_embeddings), persistent=True)
        self.context_embedding_layer = nn.Linear(config.line_dim, config.context_embedding_dim)

        self.lstm = nn.LSTM(
            input_size=config.demand_embedding_dim + config.temporal_embedding_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.final_fc = nn.Linear(config.lstm_hidden_size + config.context_embedding_dim, 1)

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        self.post_init()

    def _build_neighbor_table(self) -> torch.Tensor:
        """master의 `_build_neighbor_tables`에서 idx 계산 부분만 재사용(mask는 DMVST에서 불필요)."""
        H, W, a = self.H, self.W, self.a
        Wp = W + 2 * a

        offsets = torch.arange(-a, a + 1)
        grid_di, grid_dj = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_di = grid_di.reshape(-1)  # (patch_size^2,)
        offset_dj = grid_dj.reshape(-1)

        node_ids = torch.arange(H * W)
        h = torch.div(node_ids, W, rounding_mode='floor')  # (H*W,)
        w = node_ids % W

        rows = h.unsqueeze(-1) + a + offset_di.unsqueeze(0)  # (H*W, patch_size^2)
        cols = w.unsqueeze(-1) + a + offset_dj.unsqueeze(0)
        idx = rows * Wp + cols  # padded 좌표계 기준이라 항상 유효 범위

        return idx

    # NOTE: _init_weights를 오버라이드하지 않음 — transformers 5.0의 from_pretrained에서 커스텀
    # _init_weights가 건드리는 모듈만 체크포인트 로드 후 다시 랜덤 초기화되는 버그가 확인됨
    # (master의 GridDemandModel 개발 중 재현/검증, STResnet/ADFormer에도 동일 원칙 적용).

    def _crop_all_nodes(self, demands: torch.Tensor) -> torch.Tensor:
        """demands: (B, k, H, W) -> (B, k, N, patch_size, patch_size). 경계는 zero-padding
        (논문 Figure 1(a) 설명 그대로 — "we use zero padding for location at boundaries")."""
        B, k, H, W = demands.shape
        a = self.a
        N = H * W

        padded = F.pad(demands, (a, a, a, a), mode='constant', value=0.0)  # (B,k,H+2a,W+2a)
        padded_flat = padded.reshape(B, k, -1)

        flat_idx = self.idx_table.reshape(-1)  # (N*patch_size^2,)
        values = padded_flat[:, :, flat_idx].reshape(B, k, N, self.patch_size, self.patch_size)
        return values

    def forward(
        self,
        demands: torch.Tensor,  # (B, k, H, W)
        hour_of_day: torch.Tensor,  # (B, k)
        day_of_week: torch.Tensor,  # (B, k)
        labels: torch.Tensor | None = None,  # (B, H, W)
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        H, W, N = self.H, self.W, self.H * self.W
        B, k = demands.shape[0], demands.shape[1]

        demand_min, demand_max = self.config.demand_min, self.config.demand_max
        denom = max(demand_max - demand_min, 1e-6)

        # Spatial view: 논문 "We normalized the demand values ... to [0, 1] by using Max-Min
        # normalization on the training set" — local crop을 CNN에 넣기 전에 정규화.
        patches = self._crop_all_nodes(demands)  # (B,k,N,S,S)
        patches_norm = (patches - demand_min) / denom
        demand_features = self.local_cnn(patches_norm)  # (B,k,N,demand_embedding_dim)

        # Temporal view: 식(4)의 g_t^i = s_t^i (+) e_t^i에서 e_t^i(외부 요인)을 ADFormer와 동일하게
        # (time_in_day, day_of_week one-hot) 산술 계산으로 구성 — 모든 노드가 같은 시간대 컨텍스트를
        # 공유하므로 N 차원으로 broadcast한다.
        time_in_day = (hour_of_day.float() / 24.0).unsqueeze(-1)  # (B,k,1)
        day_onehot = F.one_hot(day_of_week, num_classes=DAY_OF_WEEK_DIM).float()  # (B,k,7)
        ext_feat = torch.cat([time_in_day, day_onehot], dim=-1)  # (B,k,EXT_DIM)
        temporal_emb = self.temporal_layer(ext_feat)  # (B,k,temporal_embedding_dim)
        temporal_emb = temporal_emb.unsqueeze(2).expand(B, k, N, -1)  # (B,k,N,temporal_embedding_dim)

        lstm_input = torch.cat([demand_features, temporal_emb], dim=-1)  # (B,k,N,D1+D2)
        lstm_input = lstm_input.permute(0, 2, 1, 3).reshape(B * N, k, -1)  # 노드별 독립 시계열
        lstm_out, _ = self.lstm(lstm_input)  # (B*N,k,lstm_hidden_size)
        h_last = lstm_out[:, -1, :].reshape(B, N, -1)  # (B,N,lstm_hidden_size), 식(7)의 h_t^i

        # Semantic view: 식(6) m_hat^i = f(W_fe * m^i + b_fe), f=ReLU, 모든 노드에 한 번에 적용.
        context_emb = F.relu(self.context_embedding_layer(self.line_embeddings))  # (N,context_embedding_dim)
        context_emb = context_emb.unsqueeze(0).expand(B, N, -1)  # (B,N,context_embedding_dim)

        # 식(7)-(8): q_t^i = h_t^i (+) m_hat^i -> Sigmoid(W_ff*q_t^i + b_ff), [0,1] 정규화 공간.
        q = torch.cat([h_last, context_emb], dim=-1)  # (B,N,lstm_hidden_size+context_embedding_dim)
        pred_norm = torch.sigmoid(self.final_fc(q)).squeeze(-1)  # (B,N)

        logits = (pred_norm * denom + demand_min).reshape(B, H, W)  # 실제 수요 단위로 역정규화

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
