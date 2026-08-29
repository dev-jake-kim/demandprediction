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

        # 검색 DB(retrieval_keys/values/norms)는 반대로 persistent=False로 등록한다 — config.npy_path의
        # 순수 함수라 idx_table과 마찬가지로 결정론적이지만, 크기가 커서(ulsan ~1.7GB, porto ~4GB, fp32)
        # 체크포인트에 통째로 중복 저장하는 게 낭비이기 때문. 그 대가로 from_pretrained 직후에는 이
        # 버퍼가 깨져 있으므로(meta-device fast-init이 persistent=False 버퍼를 복원 안 함), 호출자가
        # build_retrieval_db()를 명시적으로 다시 호출해야 한다 (test.py 참고).
        self.build_retrieval_db()

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
        weather: torch.Tensor | None = None,
        hour_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
        daily_demands: torch.Tensor | None = None,
        weekly_demands: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if sample_idx is None:
            raise ValueError("이 모델은 검색(retrieval) 브랜치가 필수라 sample_idx가 반드시 필요함")
        if weather is None or hour_of_day is None or day_of_week is None:
            raise ValueError("이 모델은 날씨/캘린더 피처가 필수라 weather/hour_of_day/day_of_week가 필요함")
        if daily_demands is None or weekly_demands is None:
            raise ValueError("이 모델은 daily/weekly 주기 브랜치가 필수라 daily_demands/weekly_demands가 필요함")

        B, k, H, W = demands.shape
        N = H * W
        d_model = self.config.d_model
        lstm_hidden = self.config.lstm_hidden

        # _spatial_encode가 내부에서 다시 crop하지만(재사용성을 위한 트레이드오프, cheap한 인덱싱
        # 연산이라 비용 무시 가능), 검색 query는 여기서 한 번 더 뽑아야 한다 — mask는 여기선 불필요.
        values, _ = self._crop_all_nodes(demands)  # (B,k,N,n)
        # 검색 query: DB(build_retrieval_db)와 동일한 (...,n_neighbors,time_step) 순서로 flatten.
        # daily/weekly와는 무관 — 검색기 앙상블은 recent 윈도우만 사용(스코프 밖).
        retrieval_query = values.permute(0, 2, 3, 1).reshape(B, N, -1)  # (B,N,n_neighbors*k)

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

        neural_pred = F.softplus(self.output_proj(fused)).squeeze(-1)  # (B,N)

        ir_out = self._retrieve(retrieval_query, sample_idx)  # (B,N), 이미 실제 수요값의 가중평균이라 비음수

        lambda_input = torch.cat([fused, ir_out.unsqueeze(-1)], dim=-1)  # (B,N,lstm_hidden+1)
        lambda_weight = torch.sigmoid(self.lambda_layer(lambda_input)).squeeze(-1)  # (B,N)
        pred = lambda_weight * neural_pred + (1 - lambda_weight) * ir_out  # (B,N)

        logits = pred.reshape(B, H, W)  # labels(B,H,W)와 동일 shape, node_id=row*W+col 순서와 일치

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return {'loss': loss, 'logits': logits}
