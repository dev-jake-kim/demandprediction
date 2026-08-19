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

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        self.post_init()

    # NOTE: _init_weights를 오버라이드하지 않음 — transformers 5.0의 from_pretrained에서
    # 커스텀 _init_weights가 건드리는 nn.Linear/nn.Embedding 모듈만 체크포인트 로드 후에
    # 다시 랜덤 초기화되는 버그가 확인됨 (models/modeling.py 개발 중 재현/검증함).
    # PyTorch 기본 초기화(Linear: kaiming_uniform, Embedding: normal)를 그대로 사용한다.

    def _crop_neighbors(self, demands: torch.Tensor, node_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """demands: (B, k, H, W), node_id: (B,) -> values (B,k,n_neighbors), mask (B,n_neighbors).

        mask는 padding으로 생긴 "격자 밖" 여부를 좌표로 직접 계산한 것이라, 실제 수요값이
        0인 경우(mask=True, 정상 데이터)와 격자 밖이라 값이 없는 경우(mask=False)가 섞이지 않는다.
        """
        B, k, H, W = demands.shape
        a = self.a
        Wp = W + 2 * a
        device = demands.device

        padded = F.pad(demands, (a, a, a, a), mode='constant', value=0.0)  # (B,k,H+2a,W+2a)
        padded_flat = padded.reshape(B, k, -1)

        h = torch.div(node_id, W, rounding_mode='floor')  # (B,)
        w = node_id % W

        # (2a+1)^2 이웃 상대 오프셋. config(a)로만 정해지는 작은 텐서라 매 forward마다
        # 입력과 같은 device에서 다시 계산한다 (persistent=False 버퍼는 fast-init/meta-device
        # 로딩 경로에서 값이 복원되지 않는 문제가 있어 버퍼 대신 이 방식을 씀).
        offsets = torch.arange(-a, a + 1, device=device)
        grid_di, grid_dj = torch.meshgrid(offsets, offsets, indexing='ij')
        offset_di = grid_di.reshape(-1)  # (n_neighbors,)
        offset_dj = grid_dj.reshape(-1)

        rows = h.unsqueeze(-1) + a + offset_di.unsqueeze(0)  # (B, n_neighbors)
        cols = w.unsqueeze(-1) + a + offset_dj.unsqueeze(0)  # (B, n_neighbors)
        idx = rows * Wp + cols  # (B, n_neighbors), padded 좌표계 기준이라 항상 유효 범위

        idx_expanded = idx.unsqueeze(1).expand(B, k, self.n_neighbors)
        values = torch.gather(padded_flat, dim=2, index=idx_expanded)  # (B,k,n_neighbors)

        valid = torch.zeros(H + 2 * a, W + 2 * a, dtype=torch.bool, device=device)
        valid[a:a + H, a:a + W] = True
        mask = valid.reshape(-1)[idx]  # (B, n_neighbors)

        return values, mask

    def forward(
        self,
        demands: torch.Tensor,
        node_id: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        B, k, H, W = demands.shape
        d_model = self.config.d_model

        values, mask = self._crop_neighbors(demands, node_id)  # (B,k,n) / (B,n)

        log_values = torch.log1p(values.clamp(min=0))
        neighbor_emb = self.scalar_embed(log_values)  # (B,k,n,d_model)

        edge_emb = self.special_embed.weight[EDGE_TOKEN_ID]  # (d_model,)
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(B, k, self.n_neighbors, d_model)
        neighbor_emb = torch.where(mask_expanded, neighbor_emb, edge_emb)

        # CLS = "CLS 마커" + "이 노드의 고유 임베딩" -> attention이 노드 정체성을 알고 이웃을 취합
        cls_base = self.special_embed.weight[CLS_TOKEN_ID]  # (d_model,)
        node_vec = self.node_embed(node_id)  # (B, d_model)
        cls_emb = (cls_base + node_vec).unsqueeze(1).unsqueeze(1).expand(B, k, 1, d_model)  # (B,k,1,d_model)
        seq = torch.cat([cls_emb, neighbor_emb], dim=2)  # (B,k,seq_len,d_model)
        seq = seq + self.pos_embed  # (seq_len,d_model) 브로드캐스트

        seq = seq.reshape(B * k, self.seq_len, d_model)
        encoded = self.encoder(seq)  # (B*k, seq_len, d_model)
        cls_out = encoded[:, 0, :].reshape(B, k, d_model)

        lstm_out, _ = self.lstm(cls_out)  # (B,k,lstm_hidden)
        last = lstm_out[:, -1, :]  # (B, lstm_hidden)

        pred = F.softplus(self.output_proj(last)).squeeze(-1)  # (B,)

        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels.to(pred.dtype))

        return {'loss': loss, 'logits': pred}
