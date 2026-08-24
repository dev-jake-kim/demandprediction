from __future__ import annotations

import numpy as np
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
        # 검색(retrieval) 브랜치를 cross-attention으로 융합하는 Q/K/V 프로젝션 + 최종 예측 레이어.
        # Q(쿼리 자신)와 K(검색된 후보를 쿼리와 동일한 가중치로 재인코딩한 결과)는 서로 다른
        # 가중치로 투영해야 하므로 별도 Linear로 둔다. V는 검색된 후보의 실제 수요값을 푸리에
        # 임베딩한 뒤 투영한 것. single-head로 충분하다고 판단(commute-attention과 동일한 이유).
        self.retrieval_q_proj = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        self.retrieval_k_proj = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        self.retrieval_v_proj = nn.Linear(config.d_model, config.lstm_hidden, bias=False)
        # pred = softplus(retrieval_fc(Q + cross_attn(Q,K,V))) — residual 연결.
        self.retrieval_fc = nn.Linear(config.lstm_hidden, 1)

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        # node_id -> (padded grid 기준 이웃 (2a+1)^2개의 flat index, 격자 안 여부)는
        # H,W,a로만 정해지는 순수 함수라 가능한 모든 node_id(H*W개)에 대해 미리 계산해둔다.
        # persistent=True로 저장해야 함: persistent=False 버퍼는 transformers 5.0의
        # from_pretrained가 meta-device에서 fast-init할 때 체크포인트에 없는 값이라
        # 복원되지 않고 값이 깨지는 문제가 있음(models/modeling.py 개발 중 확인함).
        idx_table, mask_table = self._build_neighbor_tables()
        self.register_buffer('idx_table', idx_table, persistent=True)  # (H*W, n_neighbors)
        self.register_buffer('mask_table', mask_table, persistent=True)  # (H*W, n_neighbors)

        # 검색 DB(retrieval_keys/values/norms)는 반대로 persistent=False로 등록한다 — config.npy_path의
        # 순수 함수라 idx_table과 마찬가지로 결정론적이지만, 크기가 커서(ulsan ~1.7GB, porto ~4GB, fp32)
        # 체크포인트에 통째로 중복 저장하는 게 낭비이기 때문. 그 대가로 from_pretrained 직후에는 이
        # 버퍼가 깨져 있으므로(meta-device fast-init이 persistent=False 버퍼를 복원 안 함), 호출자가
        # build_retrieval_db()를 명시적으로 다시 호출해야 한다 (test.py 참고).
        self.build_retrieval_db()

        self.post_init()

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

    def _encode_from_values(self, values: torch.Tensor) -> torch.Tensor:
        """values: (Batch,k,N,n_neighbors)(이미 크롭된 로컬 패치 값) -> (Batch,N,lstm_hidden).

        공간 Fourier 임베딩 + CLS/노드 임베딩 + 위치 임베딩 + Transformer 인코더 + LSTM까지의
        전체 인코딩 로직. 쿼리 자신의 입력(`forward`)과 검색된 후보 시점의 재인코딩
        (`_encode_retrieval_candidates`) 양쪽에서 완전히 동일한 가중치로 호출된다 — 검색된 후보가
        "그 시점이 쿼리였다면" 나왔을 임베딩을 만들기 위함.
        """
        Batch, k, N, _ = values.shape
        d_model = self.config.d_model

        log_values = torch.log1p(values.clamp(min=0))
        neighbor_emb = self.scalar_embed(log_values)  # (Batch,k,N,n,d_model)

        edge_emb = self.special_embed.weight[EDGE_TOKEN_ID]  # (d_model,)
        mask_expanded = self.mask_table.view(1, 1, N, self.n_neighbors, 1).expand(
            Batch, k, N, self.n_neighbors, d_model
        )
        neighbor_emb = torch.where(mask_expanded, neighbor_emb, edge_emb)

        cls_base = self.special_embed.weight[CLS_TOKEN_ID]  # (d_model,)
        node_vecs = self.node_embed.weight  # (N, d_model)
        cls_emb = (cls_base + node_vecs).view(1, 1, N, 1, d_model).expand(Batch, k, N, 1, d_model)
        seq = torch.cat([cls_emb, neighbor_emb], dim=3)  # (Batch,k,N,seq_len,d_model)
        seq = seq + self.pos_embed

        seq = seq.reshape(Batch * k * N, self.seq_len, d_model)
        encoded = self.encoder(seq)  # (Batch*k*N, seq_len, d_model)
        cls_out = encoded[:, 0, :].reshape(Batch, k, N, d_model)

        cls_out = cls_out.permute(0, 2, 1, 3).reshape(Batch * N, k, d_model)
        lstm_out, _ = self.lstm(cls_out)  # (Batch*N,k,lstm_hidden)
        return lstm_out[:, -1, :].reshape(Batch, N, -1)  # (Batch,N,lstm_hidden)

    def _encode_retrieval_candidates(self, cand_keys_flat: torch.Tensor) -> torch.Tensor:
        """cand_keys_flat: (N,z,flat_dim) -> (N,z,lstm_hidden).

        flat_dim = n_neighbors*time_step는 build_retrieval_db와 동일한 (n_neighbors,time_step)
        순서로 flatten돼 있다 — 이를 복원해 _encode_from_values로 재인코딩한다(쿼리와 동일한
        가중치 재사용).

        _encode_from_values 내부의 self.encoder 호출은 (Batch*k*N)을 하나의 배치로 합쳐서 SDPA에
        넘기는데, PyTorch의 memory-efficient attention 커널은 배치 크기가 65535를 넘으면
        seed/offset을 만들지 못해 런타임에 크래시한다(z*time_step*N이 이 한도를 넘기 쉬움 — 예:
        ulsan에서 retrieval_k=20이면 20*24*168=80,640). 결과에 영향 없이(각 (z,t,node)는 서로
        완전히 독립적으로 계산됨) z 축을 청크로 나눠 여러 번 호출하는 것으로 회피한다.
        """
        N, z, _ = cand_keys_flat.shape
        time_step = self.config.time_step
        values = cand_keys_flat.reshape(N, z, self.n_neighbors, time_step)
        values = values.permute(1, 3, 0, 2)  # (z,time_step,N,n_neighbors)

        max_sdpa_batch = 65535
        chunk_size = max(1, max_sdpa_batch // (time_step * N))
        if chunk_size >= z:
            encoded = self._encode_from_values(values)  # (z,N,lstm_hidden)
        else:
            chunks = [self._encode_from_values(values[start:start + chunk_size]) for start in range(0, z, chunk_size)]
            encoded = torch.cat(chunks, dim=0)  # (z,N,lstm_hidden)

        return encoded.permute(1, 0, 2)  # (N,z,lstm_hidden)

    def build_retrieval_db(self) -> None:
        """config.npy_path의 원본 grid 전체 시계열로부터 검색(retrieval) DB를 만든다.

        DB는 절대 시간 인덱스 t(0..T-1)로 인덱싱된다: retrieval_keys[:, t]는 t시점을 예측하는
        입력 윈도우(grid[t-time_step:t])와 동일한 (2a+1)^2 로컬 패치 히스토리를 _crop_all_nodes로
        뽑아 flatten한 것, retrieval_values[:, t]는 t시점의 실제 수요값이다. t < time_step인
        구간은 유효한 윈도우가 없어 0으로 남겨두고(forward에서 후보 구간 계산 시 자동으로 제외됨).

        __init__에서 자동 호출되지만(fresh construction 시 그걸로 충분), persistent=False 버퍼라
        from_pretrained 이후에는 반드시 호출자가 다시 호출해야 한다(test.py 참고).
        """
        if self.config.npy_path is None:
            raise ValueError("검색 DB를 만들려면 config.npy_path(원본 grid npy 절대경로)가 필요함")

        # 재구성(예: from_pretrained 이후 재호출) 시 이전 buffer를 새로 만들기 전에 먼저 참조를
        # 끊어야 한다 — 안 그러면 "이전 buffer + 재구성 중간 텐서(keys_by_window) + 새 buffer"가
        # 동시에 메모리에 존재해서 porto 기준 host RAM에서 최대 ~12GB까지 순간적으로 잡아먹음.
        for name in ('retrieval_keys', 'retrieval_values', 'retrieval_norms'):
            if name in self._buffers:
                self._buffers[name] = None

        device = next(self.parameters()).device
        time_step = self.config.time_step
        N = self.H * self.W

        grid = np.load(self.config.npy_path).astype(np.float32)  # (T, H, W)
        if grid.shape[1:] != (self.H, self.W):
            raise ValueError(
                f"grid의 공간 shape({grid.shape[1:]})이 모델 config의 (H,W)=({self.H},{self.W})와 다름 — "
                f"검색 DB가 잘못된 이웃 구조를 쓰게 되므로(총 셀 개수가 같아도 위험) 진행 불가"
            )
        T = grid.shape[0]
        if T <= time_step:
            raise ValueError(
                f"grid 길이(T={T})가 time_step({time_step})보다 커야 검색 DB를 만들 수 있음"
            )

        grid_t = torch.from_numpy(grid).to(device)
        with torch.no_grad():
            values, _ = self._crop_all_nodes(grid_t.unsqueeze(0))  # (1,T,N,n_neighbors)
        values = values.squeeze(0).permute(1, 0, 2)  # (N,T,n_neighbors)

        # (N,T,n_neighbors) -> unfold(dim=1, time_step) -> (N,T-time_step+1,n_neighbors,time_step).
        # 이 (...,n_neighbors,time_step) 순서가 flatten의 정본(canonical) 순서다 — forward()의 query
        # 생성도 반드시 동일한 순서로 permute한 뒤 flatten해야 코사인 유사도가 의미를 가짐.
        windows = values.unfold(1, time_step, 1)
        windows = windows[:, :T - time_step, :, :]  # 마지막 윈도우는 라벨(t=T)이 없어 제외
        flat_dim = self.n_neighbors * time_step
        keys_by_window = windows.reshape(N, T - time_step, flat_dim)  # window s -> 예측 대상 t=s+time_step

        retrieval_keys = torch.zeros(N, T, flat_dim, device=device)
        retrieval_keys[:, time_step:T, :] = keys_by_window

        retrieval_norms = torch.zeros(N, T, device=device)
        retrieval_norms[:, time_step:T] = keys_by_window.norm(dim=-1)

        retrieval_values = grid_t.reshape(T, N).transpose(0, 1).contiguous()  # (N,T)

        self.register_buffer('retrieval_keys', retrieval_keys, persistent=False)
        self.register_buffer('retrieval_values', retrieval_values, persistent=False)
        self.register_buffer('retrieval_norms', retrieval_norms, persistent=False)

    def _retrieve(
        self, query: torch.Tensor, sample_idx: torch.Tensor, q_proj_all: torch.Tensor
    ) -> torch.Tensor:
        """query: (B,N,flat_dim), sample_idx: (B,) 절대 시간 인덱스 t, q_proj_all: (B,N,lstm_hidden)
        -> context (B,N,lstm_hidden).

        후보 "선택"(인과적 슬라이싱 + 코사인 유사도로 top-z, z=config.retrieval_k)은 기존과 동일.
        선택된 z개 후보는 각각 (1) 자기 자신의 입력 윈도우(key seq)를 쿼리와 완전히 동일한 가중치로
        재인코딩한 것을 K로, (2) 그 시점의 실제 수요값을 푸리에 임베딩+투영한 것을 V로 만들어,
        Q(=q_proj_all)로 cross-attention한다. 배치 원소 b는 gir 샘플 하나가 이미 전체 N개 노드를
        담고 있으므로, 인과적 후보 구간이 노드에 상관없이 b 하나당 하나로 통일된다 — 그래서 배치
        원소 b에 대해서만 루프를 돈다. 후보 구간을 항상 t_b(자기 자신) 미만으로 슬라이스하므로
        미래 시점을 절대 참조하지 않는다.
        """
        B, N, flat_dim = query.shape
        time_step = self.config.time_step
        z_cfg = self.config.retrieval_k
        hidden = self.config.lstm_hidden

        context = torch.zeros(B, N, hidden, device=query.device, dtype=query.dtype)
        for b in range(B):
            t_b = int(sample_idx[b].item())
            num_candidates = t_b - time_step
            if num_candidates <= 0:
                continue  # 유효 후보 없음 (학습 극초반 샘플) -> context=0, residual이 Q만으로 처리

            cand_keys = self.retrieval_keys[:, time_step:t_b, :].to(device=query.device, dtype=query.dtype)
            cand_values = self.retrieval_values[:, time_step:t_b].to(device=query.device, dtype=query.dtype)
            cand_norms = self.retrieval_norms[:, time_step:t_b].to(device=query.device, dtype=query.dtype)

            q_sim = query[b]  # (N,flat_dim)
            q_norm = q_sim.norm(dim=-1).clamp_min(1e-8)  # (N,)
            sim = torch.einsum('nd,ncd->nc', q_sim, cand_keys)
            sim = sim / (q_norm.unsqueeze(1) * cand_norms.clamp_min(1e-8))  # (N,num_candidates)

            z = min(z_cfg, num_candidates)
            _, top_idx = torch.topk(sim, k=z, dim=1)  # (N,z) — 유사도 값 자체는 후보 "선택"에만 씀

            gathered_keys_flat = torch.gather(
                cand_keys, 1, top_idx.unsqueeze(-1).expand(-1, -1, flat_dim)
            )  # (N,z,flat_dim)
            gathered_values = torch.gather(cand_values, 1, top_idx)  # (N,z)

            key_lstm_out = self._encode_retrieval_candidates(gathered_keys_flat)  # (N,z,hidden)
            k_proj = self.retrieval_k_proj(key_lstm_out)  # (N,z,hidden)

            value_emb = self.scalar_embed(torch.log1p(gathered_values.clamp(min=0)))  # (N,z,d_model)
            v_proj = self.retrieval_v_proj(value_emb)  # (N,z,hidden)

            q_b = q_proj_all[b]  # (N,hidden)
            attn_logits = (q_b.unsqueeze(1) * k_proj).sum(-1) / (hidden ** 0.5)  # (N,z)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            context[b] = (attn_weights.unsqueeze(-1) * v_proj).sum(dim=1)  # (N,hidden)

        return context

    def forward(
        self,
        demands: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if sample_idx is None:
            raise ValueError("이 모델은 검색(retrieval) 브랜치가 필수라 sample_idx가 반드시 필요함")

        B, _, H, W = demands.shape
        N = H * W

        values, _ = self._crop_all_nodes(demands)  # (B,k,N,n)
        # 검색 query: DB(build_retrieval_db)와 동일한 (...,n_neighbors,time_step) 순서로 flatten.
        retrieval_query = values.permute(0, 2, 3, 1).reshape(B, N, -1)  # (B,N,n_neighbors*k)

        last = self._encode_from_values(values)  # (B,N,lstm_hidden)

        q_proj_all = self.retrieval_q_proj(last)  # (B,N,lstm_hidden)
        context = self._retrieve(retrieval_query, sample_idx, q_proj_all)  # (B,N,lstm_hidden)

        # pred = softplus(fc(Q + cross_attn(Q,K,V))) — residual 연결. 후보가 없으면 context=0이라
        # Q(쿼리 자신의 인코딩)만으로 pred가 자연스럽게 산출됨.
        pred = F.softplus(self.retrieval_fc(q_proj_all + context)).squeeze(-1)  # (B,N)

        logits = pred.reshape(B, H, W)  # labels(B,H,W)와 동일 shape, node_id=row*W+col 순서와 일치

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
