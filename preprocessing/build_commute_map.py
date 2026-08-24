"""ir-commute의 commute-attention이 쓰는 DTW 기반 노드별 참조 목록을 미리 계산한다.

ADFormer(`preprocessing/build_cluster_map.py`)의 `build_daily_profile`/DTW 거리행렬 계산과
동일한 방식(train split의 지역별 "평균 하루 패턴"끼리 DTW 거리를 구함, radius=6 fastdtw)을
재사용하되, 클러스터링 대신 노드마다 "로컬 윈도우((2a+1)^2) 밖에서 DTW 기준 가장 비슷한 n개
노드"를 직접 골라 data/raw/{city}_commute_map.npz 로 저장한다. 로컬 윈도우는 이미 공간
Transformer 인코더가 보므로, commute 참조는 그 밖의(공간은 멀지만 패턴이 비슷한) 노드를
향하게 하는 게 이 기능의 핵심이다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from fastdtw import fastdtw

REPO_ROOT = Path(__file__).parent.parent


def build_daily_profile(train_grid: np.ndarray) -> np.ndarray:
    """train_grid: (T_train, H, W) -> (24, N) 지역별 평균 하루 패턴. ADFormer와 동일 로직."""
    t_train, h, w = train_grid.shape
    n_days = t_train // 24
    if n_days == 0:
        raise ValueError(f"train 구간이 24시간(하루)도 안 됨: T_train={t_train}")

    flat = train_grid[:n_days * 24].reshape(n_days, 24, h * w)
    return flat.mean(axis=0)  # (24, N)


def compute_dtw_matrix(profile: np.ndarray, radius: int = 6) -> np.ndarray:
    """profile: (24, N) -> (N, N) DTW 거리 행렬 (대칭). ADFormer와 동일 로직."""
    n = profile.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d, _ = fastdtw(profile[:, i], profile[:, j], radius=radius)
            dist[i, j] = dist[j, i] = d
    return dist


def build_commute_map(dist: np.ndarray, h: int, w: int, a: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """dist: (N,N) DTW 거리행렬 -> commute_idx (N,n) int64, commute_sim (N,n) float32.

    노드 i마다 체비쇼프 거리(max(|Δrow|,|Δcol|)) <= a인 노드(로컬 윈도우, 자기 자신 포함)를
    후보에서 제외하고, 남은 노드 중 DTW 거리가 가장 작은 n개를 고른다. commute_sim은 거리행렬
    전체를 표준화(평균 0, 분산 1)한 뒤 부호를 뒤집은 값(클수록 유사) — 학습 가능한 bias가
    잘 스케일된 입력을 받도록 하기 위함.
    """
    num_regions = h * w
    if dist.shape != (num_regions, num_regions):
        raise ValueError(f"dist shape={dist.shape}가 (N,N)=({num_regions},{num_regions})와 다름")

    node_ids = np.arange(num_regions)
    rows = node_ids // w
    cols = node_ids % w
    row_diff = np.abs(rows[:, None] - rows[None, :])
    col_diff = np.abs(cols[:, None] - cols[None, :])
    chebyshev = np.maximum(row_diff, col_diff)  # (N,N)
    local_window = chebyshev <= a  # 자기 자신(diff=0)도 포함되어 자동으로 제외됨

    mean, std = dist.mean(), dist.std()
    std = max(std, 1e-8)
    sim_full = -(dist - mean) / std  # 클수록 유사, (N,N)

    masked_sim = np.where(local_window, -np.inf, sim_full)

    commute_idx = np.zeros((num_regions, n), dtype=np.int64)
    commute_sim = np.zeros((num_regions, n), dtype=np.float32)
    for i in range(num_regions):
        valid_count = int(np.isfinite(masked_sim[i]).sum())
        if valid_count < n:
            raise ValueError(
                f"노드 {i}: 로컬 윈도우(a={a}) 밖 유효 후보가 {valid_count}개뿐이라 n={n}개를 고를 수 없음 — "
                f"a를 줄이거나 n을 줄여야 함"
            )
        top_n = np.argpartition(-masked_sim[i], n)[:n]
        top_n = top_n[np.argsort(-masked_sim[i][top_n])]  # 유사도 내림차순 정렬
        commute_idx[i] = top_n
        commute_sim[i] = sim_full[i, top_n]

    return commute_idx, commute_sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=['ulsan', 'porto'])
    parser.add_argument('--npy_path', type=str, default=None, help='기본값: data/raw/{city}_temporal_grid.npy')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--a', type=int, default=2, help='로컬 윈도우 반경 — 모델 config의 a와 반드시 같아야 함')
    parser.add_argument('--n', type=int, default=3, help='commute 참조 개수')
    parser.add_argument('--dtw_radius', type=int, default=6)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    npy_path = Path(args.npy_path) if args.npy_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_temporal_grid.npy'
    output_path = Path(args.output_path) if args.output_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_commute_map.npz'

    print(f"[{args.city}] a={args.a} <- 반드시 configs/model/baseline.yaml의 `a`와 같은 값이어야 함!")
    print(f"  n={args.n}, dtw_radius={args.dtw_radius}")

    grid = np.load(npy_path)
    t_total, h, w = grid.shape
    n_regions = h * w
    t_train = int(t_total * args.train_ratio)
    train_grid = grid[:t_train]
    print(f"  grid={grid.shape}, N={n_regions}, train t=[0,{t_train})")

    profile = build_daily_profile(train_grid)
    print(f"  daily profile shape: {profile.shape}")

    print("  computing DTW distance matrix...")
    dtw_dist = compute_dtw_matrix(profile, radius=args.dtw_radius)
    print(f"  dtw_dist shape: {dtw_dist.shape}, mean={dtw_dist.mean():.3f}")

    commute_idx, commute_sim = build_commute_map(dtw_dist, h, w, args.a, args.n)

    raw_dists = np.take_along_axis(dtw_dist, commute_idx, axis=1)
    print(
        f"  선택된 참조의 원래 DTW 거리: min={raw_dists.min():.3f} max={raw_dists.max():.3f} "
        f"mean={raw_dists.mean():.3f}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, commute_idx=commute_idx, commute_sim=commute_sim)
    print(f"  saved to: {output_path}")


if __name__ == '__main__':
    main()
