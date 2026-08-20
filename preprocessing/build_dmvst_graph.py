"""DMVST-Net의 Semantic View(LINE 그래프 임베딩)가 쓰는 지역 간 유사도 그래프를 미리 계산한다.

ADFormer의 `preprocessing/build_cluster_map.py`에 있는 `compute_dtw_matrix`(train split만 사용,
시간 리크 방지)는 그대로 재사용하되, DTW 입력 프로파일은 논문이 요구하는 "average weekly demand
time series"(주간 패턴, 요일별 차이 보존)에 맞춰 `build_weekly_profile`로 새로 만들었다(ADFormer의
`build_daily_profile`(24,N)은 하루 단위라 요일 차이가 사라져 그대로 못 씀). 논문 식(5)
`ω_ij = exp(-α·DTW(i,j))`(α=1)로 DTW 거리를 유사도로 변환해 완전연결 그래프의 엣지 가중치를 만들고
`{u,v,w}` CSV로 저장한다.

사용자가 예전에 이 논문을 직접 구현한 `/home/jinsu/PycharmProjects/DMVST/dataset_struct/dmvst_dataset.py`
의 `make_graph()`는 `w`에 raw DTW distance를 그대로 넣는데, 그러면 LINE 임베딩이 "멀수록 강하게
연결됨"으로 학습돼 논문 식(5)와 반대 의미가 된다 — 여기서는 그 버그를 고쳐 exp(-DTW)를 쓴다.

출력 CSV는 `preprocessing/build_dmvst_line_embeddings.py`(별도 `Torch` conda env, cogdl 필요)의
입력으로 쓰인다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw

REPO_ROOT = Path(__file__).parent.parent


def build_weekly_profile(train_grid: np.ndarray) -> np.ndarray:
    """train_grid: (T_train, H, W) -> (168, N) 지역별 평균 "일주일" 패턴(요일별 차이 보존).

    논문 "Semantic View" 절: "We use the average weekly demand time series as the demand
    patterns" — 하루(24) 단위로 평균 내면 평일/주말 차이(논문이 예로 든 주거지 아침 수요 vs
    상업지 주말 수요)가 사라져 semantic view의 목적 자체가 무의미해진다. ADFormer의
    `build_daily_profile`(24,N)을 그대로 재사용하려 했으나 DMVST-Net은 요구사항이 달라
    168시간(일주일) 단위로 바꿨다.
    """
    t_train, h, w = train_grid.shape
    n_weeks = t_train // (24 * 7)
    if n_weeks == 0:
        raise ValueError(f"train 구간이 일주일(168시간)도 안 됨: T_train={t_train}")

    flat = train_grid[:n_weeks * 24 * 7].reshape(n_weeks, 24 * 7, h * w)
    return flat.mean(axis=0)  # (168, N)


def compute_dtw_matrix(profile: np.ndarray, radius: int = 6) -> np.ndarray:
    """profile: (24, N) -> (N, N) DTW 거리 행렬 (대칭)."""
    n = profile.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d, _ = fastdtw(profile[:, i], profile[:, j], radius=radius)
            dist[i, j] = dist[j, i] = d
    return dist


def build_edge_table(dtw_dist: np.ndarray, alpha: float = 1.0) -> pd.DataFrame:
    """dtw_dist: (N,N) -> 완전연결 양방향 엣지 테이블(u,v,w=exp(-alpha*dist)), 식(5)."""
    n = dtw_dist.shape[0]
    similarity = np.exp(-alpha * dtw_dist)

    us, vs, ws = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            us.append(i); vs.append(j); ws.append(similarity[i, j])
            us.append(j); vs.append(i); ws.append(similarity[i, j])
    return pd.DataFrame({'u': us, 'v': vs, 'w': ws})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=['ulsan', 'porto'])
    parser.add_argument('--npy_path', type=str, default=None, help='기본값: data/raw/{city}_temporal_grid.npy')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--dtw_radius', type=int, default=6)
    parser.add_argument('--alpha', type=float, default=1.0, help='식(5) ω_ij = exp(-alpha*DTW(i,j))')
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    npy_path = Path(args.npy_path) if args.npy_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_temporal_grid.npy'
    output_path = Path(args.output_path) if args.output_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_dmvst_graph_edges.csv'

    grid = np.load(npy_path)
    t_total, h, w = grid.shape
    n = h * w
    t_train = int(t_total * args.train_ratio)
    train_grid = grid[:t_train]
    print(f"[{args.city}] grid={grid.shape}, N={n}, train t=[0,{t_train})")

    # 논문 "We normalized the demand values ... to [0, 1] by using Max-Min normalization on the
    # training set" — DTW를 raw demand로 계산하면(특히 168시간 weekly profile로 늘리면서) 거리
    # 스케일이 너무 커져 exp(-DTW)가 float32에서 대량 언더플로(0)돼 일부 노드가 고립됨(Codex 재검증
    # 라운드에서 Porto 11개 노드가 양의 가중치 이웃 없이 고립되는 걸로 확인). train split 기준
    # Min-Max [0,1] 정규화 후 프로파일/DTW를 계산해 거리 스케일을 논문과 동일하게 맞춘다.
    demand_min = float(train_grid.min())
    demand_max = float(train_grid.max())
    denom = max(demand_max - demand_min, 1e-6)
    train_grid_norm = (train_grid - demand_min) / denom
    print(f"  demand_min={demand_min:.4f}, demand_max={demand_max:.4f} (DTW 입력 정규화용)")

    profile = build_weekly_profile(train_grid_norm)
    print(f"  weekly profile shape: {profile.shape}")

    print("  computing DTW distance matrix...")
    dtw_dist = compute_dtw_matrix(profile, radius=args.dtw_radius)
    print(f"  dtw_dist shape: {dtw_dist.shape}, mean={dtw_dist.mean():.3f}")

    edges = build_edge_table(dtw_dist, alpha=args.alpha)
    print(f"  edges: {len(edges)} rows (완전연결, 양방향), w(similarity) mean={edges['w'].mean():.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_path, index=False)
    print(f"  saved to: {output_path}")


if __name__ == '__main__':
    main()
