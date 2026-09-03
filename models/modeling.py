"""ADFormer(arXiv:2506.02576) 모델 포팅.

공식 구현(https://github.com/decisionintelligence/ADFormer, model/module.py + model/ADFormer.py)을
그대로 이식하되, `flash_attn_func`(CUDA 전용 패키지) 대신 표준 PyTorch
`F.scaled_dot_product_attention`으로 Differential Attention을 구현한다(수학적으로 동일).
그 외 세부 사항(DataEmbedding, Spatial Differential/Cluster Attention, Temporal Self/Aggregation
Attention, skip-connection 출력 헤드)은 공식 구현 그대로다. `docs/ADFORMER_PLAN.md`에 논문/공식
구현 대비 우리가 내린 결정을 정리해뒀다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import ADFormerConfig
from .losses import build_loss

EXT_DIM = 8  # time_in_day(1) + day_of_week one-hot(7)


def lambda_init_fn(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).type_as(x)


class PositionalEncoding(nn.Module):
    """표준 sinusoidal PE (고정, 학습 안 함)."""

    def __init__(self, embed_dim: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=True)  # (max_len, embed_dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len].unsqueeze(0).unsqueeze(2)  # (1,T,1,d)


class SpatialPE(nn.Module):
    def __init__(self, se_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(se_dim, embed_dim)

    def forward(self, spa_mx: torch.Tensor) -> torch.Tensor:
        return self.proj(spa_mx).unsqueeze(0).unsqueeze(0)  # (1,1,n_regions,d)


class DataEmbedding(nn.Module):
    """지역(또는 클러스터) 개수에 무관하게 재사용 가능 — main level(N)과 클러스터 level(M_i) 둘 다
    이 클래스를 별도 인스턴스로 사용한다(공식 구현과 동일)."""

    def __init__(self, feature_dim: int, embed_dim: int, se_dim: int) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.daytime_embedding = nn.Embedding(1440, embed_dim)
        self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.spatial_embedding = SpatialPE(se_dim, embed_dim)

    def forward(
        self,
        x_raw: torch.Tensor,  # (B,T,R,feature_dim)
        hour_of_day: torch.Tensor,  # (B,T) int [0,24)
        day_of_week: torch.Tensor,  # (B,T) int [0,7)
        spa_mx: torch.Tensor,  # (R, se_dim)
    ) -> torch.Tensor:
        x = self.value_embedding(x_raw)
        x = x + self.position_encoding(x_raw.size(1)).to(x.dtype)

        minute_idx = (hour_of_day.float() / 24.0 * 1440).round().long().clamp(0, 1439)
        x = x + self.daytime_embedding(minute_idx).unsqueeze(2)  # (B,T,1,d) 브로드캐스트
        x = x + self.weekday_embedding(day_of_week).unsqueeze(2)
        x = x + self.spatial_embedding(spa_mx)
        return x


class SpatialDiffAttn(nn.Module):
    """Spatial Differential Attention (SDA). 매 timestep마다 N개 지역 전체에 대한 dense attention."""

    def __init__(self, layer_idx: int, in_dim: int, hid_dim: int, heads: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hid_dim // heads // 2
        self.q_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, hid_dim, bias=False)

        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.subln = RMSNorm(2 * self.head_dim)
        self.attn_drop = attn_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B * T, N, self.heads, 2 * self.head_dim)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        v1, v2 = v.chunk(2, dim=-1)

        def to_sdpa(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 2, 1, 3)  # (B*T,heads,N,head_dim)

        drop_p = self.attn_drop if self.training else 0.0
        attn11 = F.scaled_dot_product_attention(to_sdpa(q1), to_sdpa(k1), to_sdpa(v1), dropout_p=drop_p)
        attn12 = F.scaled_dot_product_attention(to_sdpa(q1), to_sdpa(k1), to_sdpa(v2), dropout_p=drop_p)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = F.scaled_dot_product_attention(to_sdpa(q2), to_sdpa(k2), to_sdpa(v1), dropout_p=drop_p)
        attn22 = F.scaled_dot_product_attention(to_sdpa(q2), to_sdpa(k2), to_sdpa(v2), dropout_p=drop_p)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        lambda2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        lambda_full = lambda1 - lambda2 + self.lambda_init

        spa_attn = attn1 - lambda_full * attn2  # (B*T,heads,N,2*head_dim)
        spa_attn = self.subln(spa_attn) * (1 - self.lambda_init)
        spa_attn = spa_attn.permute(0, 2, 1, 3).reshape(B, T, N, self.heads * 2 * self.head_dim)
        return spa_attn


class SpatialAttn(nn.Module):
    """Spatial Cluster Attention(SCA) 한 계층. 클러스터 레벨 self-attention을 학습 가능한
    라우팅 행렬(M_sep^S)로 지역(N) 단위 출력으로 되돌린다."""

    def __init__(self, in_dim: int, hid_dim: int, heads: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hid_dim // heads
        self.q_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cluster: torch.Tensor, cls_map: torch.Tensor) -> torch.Tensor:
        # x_cluster: (B,T,M,d) 클러스터 레벨 임베딩. cls_map: (M,N) 학습 가능 라우팅(M_sep^S).
        B, T, M, _ = x_cluster.shape
        N = cls_map.shape[1]

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, T, M, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        q = split_heads(self.q_proj(x_cluster))
        k = split_heads(self.k_proj(x_cluster))
        v = split_heads(self.v_proj(x_cluster))

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B,T,heads,M,M)

        route = cls_map.t().reshape(1, 1, 1, N, M).expand(B, T, self.heads, N, M)
        attn = route @ attn  # (B,T,heads,N,M)
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = attn @ v  # (B,T,heads,N,head_dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N, self.heads * self.head_dim)
        return out


class TemporalAttn(nn.Module):
    """agg=False: Temporal Self Attention(TSA). agg=True: Temporal Aggregation Attention(TAA)
    — 학습 가능한 P개 쿼리로 시간축을 집약한 뒤, day/hour 기반 게이트(tmp_gate)로 다시 T개
    시점으로 복원한다(둘 다 최종 출력 길이는 T로 맞춰져서 STAttention에서 concat 가능)."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        heads: int,
        agg: bool = False,
        num_reg: int | None = None,
        seg_num: int | None = None,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hid_dim // heads
        self.agg = agg
        self.seg_num = seg_num
        if agg:
            assert num_reg is not None and seg_num is not None
            self.query = nn.Parameter(torch.randn(num_reg, seg_num, hid_dim) * 0.02)
        else:
            self.q_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, hid_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor, tmp_gate: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,T,N,d). tmp_gate(TAA 전용): (B,N,T,P) 복원 행렬.
        B, T, N, _ = x.shape
        x_t = x.permute(0, 2, 1, 3)  # (B,N,T,d)

        k = self.k_proj(x_t).reshape(B, N, T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x_t).reshape(B, N, T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        if self.agg:
            q = self.query.unsqueeze(0).expand(B, -1, -1, -1)
            q = q.reshape(B, N, self.seg_num, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)  # (B,N,heads,P,hd)
        else:
            q = self.q_proj(x_t).reshape(B, N, T, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # agg: (B,N,heads,P,T) / else: (B,N,heads,T,T)

        if tmp_gate is not None:
            gate = tmp_gate.unsqueeze(2).expand(-1, -1, self.heads, -1, -1)  # (B,N,heads,T,P)
            attn = gate @ attn  # (B,N,heads,T,T)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v  # (B,N,heads,T,hd)
        out_len = out.shape[-2]
        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, out_len, self.heads * self.head_dim)
        return out.permute(0, 2, 1, 3)  # (B,out_len,N,d)


class STAttention(nn.Module):
    def __init__(self, config: ADFormerConfig, layer_idx: int) -> None:
        super().__init__()
        dim = config.embed_dim
        total_heads = config.s_heads + config.sa_heads + config.t_heads + config.ta_heads
        avg_dim = dim // total_heads
        self.s_dim = config.s_heads * avg_dim
        self.sa_dim = config.sa_heads * avg_dim
        self.t_dim = config.t_heads * avg_dim
        self.ta_dim = config.ta_heads * avg_dim

        self.spa_attn = SpatialDiffAttn(layer_idx, dim, self.s_dim, config.s_heads, config.attn_drop)
        self.dtw_agg_attn = nn.ModuleList([
            SpatialAttn(dim, self.sa_dim, config.sa_heads, config.agg_drop)
            for _ in config.cluster_reg_nums
        ])
        self.spa_agg_linear = nn.Linear(self.sa_dim, self.sa_dim)

        self.tmp_attn = TemporalAttn(dim, self.t_dim, config.t_heads, attn_drop=config.attn_drop)
        self.tmp_agg_attn = TemporalAttn(
            dim, self.ta_dim, config.ta_heads, agg=True,
            num_reg=config.H * config.W, seg_num=config.cluster_seg_num, attn_drop=config.agg_drop,
        )
        self.out_proj = nn.Linear(self.s_dim + self.sa_dim + self.t_dim + self.ta_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        dtw_agg_x: list[torch.Tensor],
        tmp_gate: torch.Tensor,
        cls_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        s_x = self.spa_attn(x)

        sg_x = 0.0
        for i, attn in enumerate(self.dtw_agg_attn):
            sg_x = sg_x + attn(dtw_agg_x[i], cls_maps[i])
        sg_x = self.spa_agg_linear(sg_x)

        t_x = self.tmp_attn(x)
        tg_x = self.tmp_agg_attn(x, tmp_gate)

        out = torch.cat([s_x, sg_x, t_x, tg_x], dim=-1)
        return self.out_proj(out)


class STEncoder(nn.Module):
    def __init__(self, config: ADFormerConfig, layer_idx: int, drop_path_rate: float) -> None:
        super().__init__()
        dim = config.embed_dim
        self.tmp_map_linear = nn.Linear(EXT_DIM, config.cluster_seg_num)
        self.norm1 = nn.LayerNorm(dim)
        self.st_attn = STAttention(config, layer_idx)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * config.mlp_ratio, drop=config.attn_drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        dtw_agg_x: list[torch.Tensor],
        ext_feat: torch.Tensor,
        cls_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        B, T, N, _ = x.shape
        tmp_gate = self.tmp_map_linear(ext_feat).unsqueeze(1).expand(B, N, T, -1)  # (B,N,T,P)

        sh = self.st_attn(x, dtw_agg_x, tmp_gate, cls_maps)
        h = self.norm1(self.drop_path(sh) + x)
        y = self.norm2(self.drop_path(self.mlp(h)) + h)
        return y


class ADFormerModel(PreTrainedModel):
    config_class = ADFormerConfig
    base_model_prefix = 'adformer'

    def __init__(self, config: ADFormerConfig) -> None:
        super().__init__(config)
        H, W = config.H, config.W
        N = H * W
        d = config.embed_dim
        self.H, self.W, self.N = H, W, N

        cluster_maps = self._load_cluster_maps(config)
        self.n_levels = len(cluster_maps)
        for i, m in enumerate(cluster_maps):
            self.register_buffer(f'cluster_map_{i}', torch.from_numpy(m).float(), persistent=True)

        # M_sep^S = randn_like(M_cls) * M_cls: 초기화는 클러스터 마스크를 따르지만 자유롭게 학습됨.
        # numpy로만 계산해서 마지막에 한 번만 torch로 감싼다 — torch.randn(...)*torch.from_numpy(...)처럼
        # __init__ 안에서 새로 만든 텐서와 외부 실데이터 텐서를 직접 연산하면, from_pretrained의
        # meta-device fast-init 경로에서 "meta 텐서 x 실제 텐서" device mismatch 에러가 남
        # (cluster_map 버퍼는 register_buffer라 안전하지만, 이 연산은 그 경로를 안 타서 문제됨).
        self.cls_sep = nn.ParameterList([
            nn.Parameter(torch.from_numpy(np.random.randn(*m.shape).astype(np.float32) * m))
            for m in cluster_maps
        ])

        self.spatial_emb = nn.Parameter(torch.randn(N, config.SE_dim) * 0.02)
        self.data_embedding = DataEmbedding(1, d, config.SE_dim)

        self.cls_data_embedding = nn.ModuleList([DataEmbedding(1, d, config.SE_dim) for _ in cluster_maps])
        self.spa_cls_emb = nn.ParameterList([
            nn.Parameter(torch.randn(m.shape[0], config.SE_dim) * 0.02) for m in cluster_maps
        ])

        drop_rates = np.linspace(0, config.drop_path, config.depth).tolist()
        self.enc_blocks = nn.ModuleList([
            STEncoder(config, layer_idx=i + 1, drop_path_rate=drop_rates[i])
            for i in range(config.depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(d, config.skip_dim, kernel_size=1) for _ in range(config.depth)
        ])
        self.end_conv1 = nn.Conv2d(config.time_step, 1, kernel_size=1, bias=True)  # window -> horizon(1)
        self.end_conv2 = nn.Conv2d(config.skip_dim, 1, kernel_size=1, bias=True)  # skip_dim -> output_dim(1)

        self.loss_fn = build_loss(config)

        self.post_init()

    @staticmethod
    def _load_cluster_maps(config: ADFormerConfig) -> list[np.ndarray]:
        data = np.load(config.cluster_map_path)
        return [data[f'level_{i}'] for i in range(len(data.files))]

    def forward(
        self,
        demands: torch.Tensor,  # (B,T,H,W)
        hour_of_day: torch.Tensor,  # (B,T)
        day_of_week: torch.Tensor,  # (B,T)
        labels: torch.Tensor | None = None,  # (B,H,W)
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        B, T, H, W = demands.shape
        N = H * W

        x_raw = demands.reshape(B, T, N, 1)
        x_raw_norm = (x_raw - self.config.demand_mean) / self.config.demand_std

        time_in_day = (hour_of_day.float() / 24.0).unsqueeze(-1)  # (B,T,1)
        day_onehot = F.one_hot(day_of_week, num_classes=7).float()  # (B,T,7)
        ext_feat = torch.cat([time_in_day, day_onehot], dim=-1)  # (B,T,8)

        x = self.data_embedding(x_raw_norm, hour_of_day, day_of_week, self.spatial_emb)  # (B,T,N,d)

        dtw_agg_x = []
        for i in range(self.n_levels):
            cluster_map = getattr(self, f'cluster_map_{i}')  # (M_i,N)
            agg_raw = torch.einsum('mn,btnd->btmd', cluster_map, x_raw_norm)  # (B,T,M_i,1)
            dtw_agg_x.append(self.cls_data_embedding[i](agg_raw, hour_of_day, day_of_week, self.spa_cls_emb[i]))

        cls_maps = list(self.cls_sep)

        skip = None
        for i, block in enumerate(self.enc_blocks):
            x = block(x, dtw_agg_x, ext_feat, cls_maps)
            skip_i = self.skip_convs[i](x.permute(0, 3, 2, 1))  # (B,skip_dim,N,T)
            skip = skip_i if skip is None else skip + skip_i

        out = self.end_conv1(F.relu(skip).permute(0, 3, 2, 1))  # (B,T,N,skip_dim) -> (B,1,N,skip_dim)
        out = self.end_conv2(F.relu(out).permute(0, 3, 2, 1))  # (B,skip_dim,N,1) -> (B,1,N,1)
        out = out.permute(0, 3, 2, 1)  # (B,1,N,1)

        pred_norm = out.squeeze(1).squeeze(-1)  # (B,N)
        pred = pred_norm * self.config.demand_std + self.config.demand_mean  # 실제 수요 단위로 역정규화
        logits = pred.reshape(B, H, W)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
