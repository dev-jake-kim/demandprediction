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
        self.output_proj = nn.Linear(config.lstm_hidden, 1)
        # 검색(retrieval) 브랜치와 뉴럴 브랜치를 게이트로 섞는 레이어. 입력은
        # [LSTM 마지막 hidden state, 검색 예측 스칼라] concat (최종 레이어 직전 concat).
        self.lambda_layer = nn.Linear(config.lstm_hidden + 1, 1)

        self.loss_fn = CombinedLoss(gamma=config.loss_gamma, eps=config.loss_eps)

        # node_id -> (padded grid 기준 이웃 (2a+1)^2개의 flat index, 격자 안 여부)는
        # H,W,a로만 정해지는 순수 함수라 가능한 모든 node_id(H*W개)에 대해 미리 계산해둔다.
        # persistent=True로 저장해야 함: persistent=False 버퍼는 transformers 5.0의
        # from_pretrained가 meta-device에서 fast-init할 때 체크포인트에 없는 값이라
        # 복원되지 않고 값이 깨지는 문제가 있음(models/modeling.py 개발 중 확인함).
        idx_table, mask_table = self._build_neighbor_tables()
        self.register_buffer('idx_table', idx_table, persistent=True)  # (H*W, n_neighbors)
        self.register_buffer('mask_table', mask_table, persistent=True)  # (H*W, n_neighbors)

        # commute-attention 전용: 노드마다 (2a+1)^2 로컬 윈도우 밖에서 DTW 기준 가장 비슷한 n개
        # 참조 노드(preprocessing/build_commute_map.py가 미리 계산). idx_table/mask_table과 같은
        # 이유로 persistent=True(크기가 N*n로 작아서 체크포인트에 통째로 저장해도 무방 —
        # 검색 DB(retrieval_keys 등, 수백MB~GB)와는 다른 케이스).
        commute_idx, commute_sim = self._load_commute_map(config)
        self.register_buffer('commute_idx', commute_idx, persistent=True)  # (H*W, n)
        self.register_buffer('commute_sim', commute_sim, persistent=True)  # (H*W, n)

        self.commute_q_proj = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        self.commute_k_proj = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        self.commute_v_proj = nn.Linear(config.lstm_hidden, config.lstm_hidden, bias=False)
        # DTW 유사도(스칼라)를 attention logit에 더해지는 학습 가능한 scale+bias로만 씀 — DTW
        # 자체를 최종 가중치로 쓰지 않는 이유는 docs/MODEL_PLAN.md §1-6 참고.
        self.commute_dtw_bias = nn.Linear(1, 1)

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

    def _load_commute_map(self, config: GridDemandConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """config.commute_map_path(preprocessing/build_commute_map.py가 만든 npz)를 로드하고,
        저장된 모든 참조가 실제로 노드의 체비쇼프 a 윈도우 밖인지 재검증한다 — 전처리 스크립트의
        --a가 모델 config.a와 어긋나면(사용자가 맞춰야 하는 값이라 실수 가능) 조용히 틀린 채로
        넘어가지 않고 여기서 바로 에러가 나게 하기 위함.

        검증은 전부 순수 numpy로 한다(torch 팩토리 함수 X) — from_pretrained의 meta-device
        fast-init 중에는 __init__ 안에서 만든 torch.arange 등이 meta tensor가 되어 .any()/.item()
        같은 즉시 값 평가가 불가능해지기 때문(models/modeling.py 개발 중 재현/확인함). numpy 배열은
        이 문제에서 자유로우므로, 검증을 numpy로 끝내고 최종 결과만 한 번 torch.from_numpy로 감싼다.
        """
        if config.commute_map_path is None:
            raise ValueError("commute_map_path(원본 commute map npz 절대경로)가 필요함")

        data = np.load(config.commute_map_path)
        commute_idx_np = data['commute_idx']  # (N,n) int
        commute_sim_np = data['commute_sim']  # (N,n) float

        H, W, a = self.H, self.W, self.a
        N = H * W

        if commute_idx_np.ndim != 2 or commute_idx_np.shape[0] == 0 or commute_idx_np.shape[1] == 0:
            raise ValueError(f"commute_idx shape={commute_idx_np.shape}가 비어있지 않은 (N,n) 2D 배열이어야 함")
        if commute_idx_np.shape[0] != N:
            raise ValueError(
                f"commute_idx의 노드 수({commute_idx_np.shape[0]})가 모델의 N=H*W={N}와 다름"
            )
        if commute_sim_np.shape != commute_idx_np.shape:
            raise ValueError(
                f"commute_sim shape={commute_sim_np.shape}가 commute_idx shape={commute_idx_np.shape}와 다름"
            )
        if not np.issubdtype(commute_idx_np.dtype, np.integer):
            raise ValueError(f"commute_idx dtype={commute_idx_np.dtype}가 정수형이 아님")
        if (commute_idx_np < 0).any() or (commute_idx_np >= N).any():
            raise ValueError(f"commute_idx에 [0,{N}) 범위 밖 인덱스가 있음")
        if not np.isfinite(commute_sim_np).all():
            raise ValueError("commute_sim에 NaN/Inf가 있음")

        node_ids = np.arange(N)
        rows, cols = node_ids // W, node_ids % W
        ref_rows, ref_cols = rows[commute_idx_np], cols[commute_idx_np]  # (N,n)
        chebyshev = np.maximum(np.abs(rows[:, None] - ref_rows), np.abs(cols[:, None] - ref_cols))
        if (chebyshev <= a).any():
            raise ValueError(
                f"commute_map_path의 일부 참조가 모델의 로컬 윈도우(a={a}) 안에 있음 — "
                f"preprocessing/build_commute_map.py를 실행할 때 --a가 이 모델의 a와 다르게 지정된 것으로 보임"
            )

        commute_idx = torch.from_numpy(commute_idx_np).long()
        commute_sim = torch.from_numpy(commute_sim_np).float()
        return commute_idx, commute_sim

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

    def _retrieve(self, query: torch.Tensor, sample_idx: torch.Tensor) -> torch.Tensor:
        """query: (B,N,flat_dim), sample_idx: (B,) 절대 시간 인덱스 t -> ir_out (B,N).

        각 배치 원소 b는 gir 샘플 하나가 이미 전체 N개 노드를 담고 있으므로, 인과적 후보 구간
        (retrieval_keys[:, time_step:t_b])이 노드에 상관없이 b 하나당 하나로 통일된다 — 그래서
        원본(IRModule)의 "배치 안 node_id별 group" 루프 대신 배치 원소 b에 대해서만 루프를 돈다.
        후보 구간을 항상 t_b(자기 자신) 미만으로 슬라이스하므로 미래 시점을 절대 참조하지 않는다.
        """
        B, N, _ = query.shape
        time_step = self.config.time_step
        top_k = self.config.retrieval_k

        ir_out = torch.zeros(B, N, device=query.device, dtype=query.dtype)
        for b in range(B):
            t_b = int(sample_idx[b].item())
            num_candidates = t_b - time_step
            if num_candidates <= 0:
                continue  # 유효 후보 없음 (학습 극초반 샘플) -> ir_out=0, lambda 게이트가 알아서 처리

            cand_keys = self.retrieval_keys[:, time_step:t_b, :].to(device=query.device, dtype=query.dtype)
            cand_values = self.retrieval_values[:, time_step:t_b].to(device=query.device, dtype=query.dtype)
            cand_norms = self.retrieval_norms[:, time_step:t_b].to(device=query.device, dtype=query.dtype)

            q = query[b]  # (N,flat_dim)
            q_norm = q.norm(dim=-1).clamp_min(1e-8)  # (N,)
            sim = torch.einsum('nd,ncd->nc', q, cand_keys)
            sim = sim / (q_norm.unsqueeze(1) * cand_norms.clamp_min(1e-8))  # (N,num_candidates)

            k = min(top_k, num_candidates)
            top_vals, top_idx = torch.topk(sim, k=k, dim=1)  # (N,k)
            weights = torch.softmax(top_vals, dim=1)  # (N,k)
            gathered = torch.gather(cand_values, 1, top_idx)  # (N,k)
            ir_out[b] = (gathered * weights).sum(dim=1)

        return ir_out

    def forward(
        self,
        demands: torch.Tensor,
        labels: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if sample_idx is None:
            raise ValueError("이 모델은 검색(retrieval) 브랜치가 필수라 sample_idx가 반드시 필요함")

        B, k, H, W = demands.shape
        N = H * W
        d_model = self.config.d_model

        values, mask = self._crop_all_nodes(demands)  # (B,k,N,n) / (N,n)
        # 검색 query: DB(build_retrieval_db)와 동일한 (...,n_neighbors,time_step) 순서로 flatten.
        retrieval_query = values.permute(0, 2, 3, 1).reshape(B, N, -1)  # (B,N,n_neighbors*k)

        log_values = torch.log1p(values.clamp(min=0))
        neighbor_emb = self.scalar_embed(log_values)  # (B,k,N,n,d_model)

        edge_emb = self.special_embed.weight[EDGE_TOKEN_ID]  # (d_model,)
        mask_expanded = mask.view(1, 1, N, self.n_neighbors, 1).expand(B, k, N, self.n_neighbors, d_model)
        neighbor_emb = torch.where(mask_expanded, neighbor_emb, edge_emb)

        # CLS = "CLS 마커" + "이 노드의 고유 임베딩" -> attention이 노드 정체성을 알고 이웃을 취합.
        # node_id로 한 행만 고르던 걸 전체 N행(node_embed.weight)으로 바꿔 N개 노드 전부의 CLS를 구성.
        cls_base = self.special_embed.weight[CLS_TOKEN_ID]  # (d_model,)
        node_vecs = self.node_embed.weight  # (N, d_model)
        cls_emb = (cls_base + node_vecs).view(1, 1, N, 1, d_model).expand(B, k, N, 1, d_model)
        seq = torch.cat([cls_emb, neighbor_emb], dim=3)  # (B,k,N,seq_len,d_model)
        seq = seq + self.pos_embed  # (seq_len,d_model) 브로드캐스트

        # (B,k,N)을 하나의 배치로 합쳐 encoder에 넣음 -> 각 (b,t,node)는 서로 완전히 독립적으로 처리됨
        # (예전에 (B,k)만 합치던 것과 동일한 원리, node 차원만 추가로 합친 것뿐 -> 노드별 연산은 그대로).
        seq = seq.reshape(B * k * N, self.seq_len, d_model)
        encoded = self.encoder(seq)  # (B*k*N, seq_len, d_model)
        cls_out = encoded[:, 0, :].reshape(B, k, N, d_model)

        # 노드별로 독립적인 시계열이므로 (B,N)을 LSTM 배치로 합치고 k를 시퀀스 축으로 둔다.
        cls_out = cls_out.permute(0, 2, 1, 3).reshape(B * N, k, d_model)
        lstm_out, _ = self.lstm(cls_out)  # (B*N,k,lstm_hidden)
        last = lstm_out[:, -1, :].reshape(B, N, -1)  # (B,N,lstm_hidden)

        # commute-attention: 로컬 윈도우 밖, DTW로 미리 골라둔 n개 노드의 정보를 주입한다.
        # "얼마나 반영할지"는 DTW 스칼라로 직접 정하지 않고(노드마다 DTW 분포가 달라 스칼라 하나로
        # 비교 가능한 신뢰도를 못 만듦) 학습 가능한 cross-attention이 결정하며, DTW 유사도는 그
        # attention logit에 더해지는 learnable scale+bias(사전지식)로만 참여한다.
        ref_emb = last[:, self.commute_idx, :]  # (B,N,n,lstm_hidden) — idx_table과 동일한 gather 트릭
        q = self.commute_q_proj(last).unsqueeze(2)  # (B,N,1,lstm_hidden)
        ref_k = self.commute_k_proj(ref_emb)  # (B,N,n,lstm_hidden)
        ref_v = self.commute_v_proj(ref_emb)  # (B,N,n,lstm_hidden)
        commute_logits = (q * ref_k).sum(-1) / (self.config.lstm_hidden ** 0.5)  # (B,N,n)
        commute_bias = self.commute_dtw_bias(self.commute_sim.unsqueeze(-1)).squeeze(-1)  # (N,n)
        commute_weights = torch.softmax(commute_logits + commute_bias, dim=-1)  # (B,N,n)
        commute_context = (commute_weights.unsqueeze(-1) * ref_v).sum(dim=2)  # (B,N,lstm_hidden)
        last = last + commute_context  # residual

        neural_pred = F.softplus(self.output_proj(last)).squeeze(-1)  # (B,N)

        ir_out = self._retrieve(retrieval_query, sample_idx)  # (B,N), 이미 실제 수요값의 가중평균이라 비음수

        lambda_input = torch.cat([last, ir_out.unsqueeze(-1)], dim=-1)  # (B,N,lstm_hidden+1)
        lambda_weight = torch.sigmoid(self.lambda_layer(lambda_input)).squeeze(-1)  # (B,N)
        pred = lambda_weight * neural_pred + (1 - lambda_weight) * ir_out  # (B,N)

        logits = pred.reshape(B, H, W)  # labels(B,H,W)와 동일 shape, node_id=row*W+col 순서와 일치

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
