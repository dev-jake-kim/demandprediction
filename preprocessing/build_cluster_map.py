"""ADFormer의 Spatial Cluster Attention이 쓰는 DTW 기반 지역 클러스터 맵을 미리 계산한다.

공식 구현(https://github.com/decisionintelligence/ADFormer)의
utils/ADFormer_dataset.py:get_dtw/get_cluster, model/module.py:hierarchical_clustering을
그대로 이식 — train split의 지역별 "평균 하루 패턴"끼리 DTW 거리를 구하고, 계층적
클러스터링으로 여러 레벨의 (M_i, N) 이진 지역->클러스터 배정 행렬을 만들어
data/raw/{city}_cluster_maps.npz 로 저장한다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.cluster.hierarchy import fcluster, linkage

REPO_ROOT = Path(__file__).parent.parent

# N=168(ulsan)/200(porto)에 맞춰 공식 기본값([64,16], N=263 기준)을 축소
DEFAULT_CLUSTER_LEVELS = {
    'ulsan': [40, 10],
    'porto': [48, 12],
}


def build_daily_profile(train_grid: np.ndarray) -> np.ndarray:
    """train_grid: (T_train, H, W) -> (24, N) 지역별 평균 하루 패턴."""
    t_train, h, w = train_grid.shape
    n_days = t_train // 24
    if n_days == 0:
        raise ValueError(f"train 구간이 24시간(하루)도 안 됨: T_train={t_train}")

    flat = train_grid[:n_days * 24].reshape(n_days, 24, h * w)
    return flat.mean(axis=0)  # (24, N)


def compute_dtw_matrix(profile: np.ndarray, radius: int = 6) -> np.ndarray:
    """profile: (24, N) -> (N, N) DTW 거리 행렬 (대칭)."""
    n = profile.shape[1]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d, _ = fastdtw(profile[:, i], profile[:, j], radius=radius)
            dist[i, j] = dist[j, i] = d
    return dist


def cluster_regions(distance_matrix: np.ndarray, target_clusters: int, balance: bool, tolerance: float) -> list[list[int]]:
    """공식 구현 model/module.py:cluster_regions 이식."""
    condensed = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    linkage_matrix = linkage(condensed, method='average')
    labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)

    if not balance:
        return list(clusters.values())

    avg_size = len(distance_matrix) // target_clusters
    max_size = max(1, int(avg_size * (1 + tolerance)))

    for label in list(clusters.keys()):
        while len(clusters[label]) > max_size:
            point = clusters[label].pop()
            best_label, best_dist = None, float('inf')
            for other_label, other_points in clusters.items():
                if other_label != label and len(other_points) < max_size:
                    avg_dist = float(np.mean([distance_matrix[point, p] for p in other_points]))
                    if avg_dist < best_dist:
                        best_dist, best_label = avg_dist, other_label
            if best_label is not None:
                clusters[best_label].append(point)
            else:
                clusters[label].append(point)
                break

    return list(clusters.values())


def hierarchical_clustering(
    distance_matrix: np.ndarray, cluster_targets: list[int], balance: bool, tolerance: float
) -> list[list[list[int]]]:
    """공식 구현 model/module.py:hierarchical_clustering 이식."""
    results = []
    current_matrix = distance_matrix
    for target in cluster_targets:
        clusters = cluster_regions(current_matrix, target, balance, tolerance)
        results.append(clusters)

        new_matrix = np.zeros((target, target))
        for i, group_i in enumerate(clusters):
            for j, group_j in enumerate(clusters):
                distances = [
                    distance_matrix[p_i, p_j] for p_i in group_i for p_j in group_j
                ]
                new_matrix[i, j] = float(np.mean(distances))
        current_matrix = new_matrix
    return results


def build_level_maps(clusters_per_level: list[list[list[int]]], n: int, cluster_targets: list[int]) -> list[np.ndarray]:
    """레벨별 (M_i, N) 이진 지역->클러스터 배정 행렬. 상위 레벨은 하위 레벨과 합성해
    항상 '원본 N지역 -> 이 레벨의 클러스터' 직접 매핑으로 만든다."""
    maps = []
    cur_map = None
    for i, clusters in enumerate(clusters_per_level):
        rows = cluster_targets[i]
        cols = n if i == 0 else cluster_targets[i - 1]
        level_map = np.zeros((rows, cols), dtype=np.float32)
        for row_idx, members in enumerate(clusters):
            for member_idx in members:
                level_map[row_idx, member_idx] = 1.0

        cur_map = level_map if i == 0 else level_map @ cur_map
        maps.append(cur_map)
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=['ulsan', 'porto'])
    parser.add_argument('--npy_path', type=str, default=None, help='기본값: data/raw/{city}_temporal_grid.npy')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--cluster_levels', type=str, default=None, help='예: "40,10" (기본값: 도시별 사전 정의)')
    parser.add_argument('--balance', action='store_true', default=True)
    parser.add_argument('--tolerance', type=float, default=0.7)
    parser.add_argument('--dtw_radius', type=int, default=6)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    npy_path = Path(args.npy_path) if args.npy_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_temporal_grid.npy'
    output_path = Path(args.output_path) if args.output_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_cluster_maps.npz'
    cluster_levels = (
        [int(x) for x in args.cluster_levels.split(',')]
        if args.cluster_levels
        else DEFAULT_CLUSTER_LEVELS[args.city]
    )

    grid = np.load(npy_path)
    t_total, h, w = grid.shape
    n = h * w
    t_train = int(t_total * args.train_ratio)
    train_grid = grid[:t_train]
    print(f"[{args.city}] grid={grid.shape}, N={n}, train t=[0,{t_train}), cluster_levels={cluster_levels}")

    profile = build_daily_profile(train_grid)
    print(f"  daily profile shape: {profile.shape}")

    print("  computing DTW distance matrix...")
    dtw_dist = compute_dtw_matrix(profile, radius=args.dtw_radius)
    print(f"  dtw_dist shape: {dtw_dist.shape}, mean={dtw_dist.mean():.3f}")

    clusters_per_level = hierarchical_clustering(dtw_dist, cluster_levels, args.balance, args.tolerance)
    level_maps = build_level_maps(clusters_per_level, n, cluster_levels)

    save_kwargs = {}
    for i, level_map in enumerate(level_maps):
        row_sums = level_map.sum(axis=0)
        assert np.all(row_sums == 1), f"level {i}: 일부 지역이 정확히 하나의 클러스터에 속하지 않음"
        sizes = level_map.sum(axis=1)
        print(
            f"  level {i}: shape={level_map.shape}, cluster size min={int(sizes.min())} "
            f"max={int(sizes.max())} avg={sizes.mean():.1f}"
        )
        save_kwargs[f'level_{i}'] = level_map

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_kwargs)
    print(f"  saved to: {output_path}")


if __name__ == '__main__':
    main()
