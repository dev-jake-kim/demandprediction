from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pulp
from shapely.geometry import Point
from tqdm import tqdm

from .config import GraphBuildConfig


@dataclass
class GraphBuildArtifacts:
    bounds: tuple[float, float, float, float]
    temporal_grid: np.ndarray
    unique_hours: list[pd.Timestamp]
    sum_grid: np.ndarray
    patches: list[list[dict]]
    near_patches: list[list[dict]]
    total_score: int
    landuse_grid: np.ndarray
    node_compositions: list[dict]
    nodes: list[dict]
    demands: np.ndarray
    near_demands: list[np.ndarray]
    coord_to_node: dict[tuple[int, int], int]
    od_flows: list[list[dict]]
    temporal_features: list[dict]


def calculate_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    return x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()


def create_temporal_grid(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    print("\n" + "=" * 60)
    print("Creating temporal demand grid")
    print("=" * 60)

    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    n_cols = int(np.ceil(width / config.grid_size))
    n_rows = int(np.ceil(height / config.grid_size))

    df_with_time = df.copy()
    df_with_time['hour'] = df_with_time['call_date'].dt.floor('h')

    start_hour = df_with_time['hour'].min()
    end_hour = df_with_time['hour'].max()
    unique_hours = list(pd.date_range(start=start_hour, end=end_hour, freq='h'))
    n_timesteps = len(unique_hours)

    print(f"\nTemporal grid configuration:")
    print(f"  Spatial: {n_rows} rows x {n_cols} columns")
    print(f"  Temporal: {n_timesteps} time steps (hours)")
    print(f"  Time range: {unique_hours[0]} to {unique_hours[-1]}")
    print(f"  Total shape: ({n_timesteps}, {n_rows}, {n_cols})")

    temporal_grid = np.zeros((n_timesteps, n_rows, n_cols), dtype=np.int32)

    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    cols = ((x_coords - min_x) / config.grid_size).astype(int)
    rows = ((max_y - y_coords) / config.grid_size).astype(int)

    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}
    df_with_time['time_idx'] = df_with_time['hour'].map(hour_to_idx)

    print("\nMapping demands to temporal grid...")
    for idx in tqdm(range(len(df_with_time)), desc="Processing records"):
        time_idx = df_with_time.iloc[idx]['time_idx']
        row = rows[idx]
        col = cols[idx]
        if 0 <= row < n_rows and 0 <= col < n_cols and pd.notna(time_idx):
            temporal_grid[int(time_idx), row, col] += 1

    print(f"\nTemporal grid statistics:")
    print(f"  Total demands: {temporal_grid.sum():,}")
    print(f"  Non-zero cells (space-time): {np.count_nonzero(temporal_grid):,}")
    return temporal_grid, unique_hours


def get_patch(
    grid: np.ndarray,
    a: int,
    b: int,
    x: int,
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
    padding_size: int = 0,
) -> tuple[list[list[dict]], list[list[dict]], int]:
    n_rows, n_cols = grid.shape

    cumulative = np.zeros((n_rows + 1, n_cols + 1), dtype=np.int64)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            cumulative[row_idx + 1, col_idx + 1] = (
                grid[row_idx, col_idx]
                + cumulative[row_idx, col_idx + 1]
                + cumulative[row_idx + 1, col_idx]
                - cumulative[row_idx, col_idx]
            )

    def get_sum(row_idx: int, col_idx: int) -> int:
        return int(
            cumulative[row_idx + a, col_idx + b]
            - cumulative[row_idx, col_idx + b]
            - cumulative[row_idx + a, col_idx]
            + cumulative[row_idx, col_idx]
        )

    problem = pulp.LpProblem("Patch_Optimization", pulp.LpMaximize)
    choices = pulp.LpVariable.dicts("Choice", (range(n_rows - a + 1), range(n_cols - b + 1)), cat='Binary')

    problem += pulp.lpSum(
        choices[row_idx][col_idx] * get_sum(row_idx, col_idx)
        for row_idx in range(n_rows - a + 1)
        for col_idx in range(n_cols - b + 1)
    )

    problem += pulp.lpSum(
        choices[row_idx][col_idx]
        for row_idx in range(n_rows - a + 1)
        for col_idx in range(n_cols - b + 1)
    ) == x

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            applicable_patches = [
                choices[patch_row][patch_col]
                for patch_row in range(max(0, row_idx - a + 1), min(row_idx + 1, n_rows - a + 1))
                for patch_col in range(max(0, col_idx - b + 1), min(col_idx + 1, n_cols - b + 1))
            ]
            if applicable_patches:
                problem += pulp.lpSum(applicable_patches) <= 1

    solver = pulp.COIN_CMD(msg=0, path=config.cbc_path) if config.cbc_path else pulp.PULP_CBC_CMD(msg=0)
    problem.solve(solver)

    best_patches_tl: list[tuple[int, int]] = []
    for row_idx in range(n_rows - a + 1):
        for col_idx in range(n_cols - b + 1):
            if pulp.value(choices[row_idx][col_idx]) == 1:
                best_patches_tl.append((row_idx, col_idx))

    total_score = int(pulp.value(problem.objective))

    print("  Generating patch and near-cell details...")
    patches = [[] for _ in range(len(best_patches_tl))]
    near_patches = [[] for _ in range(len(best_patches_tl))]

    min_x, max_x, min_y, max_y = bounds
    all_cells: list[tuple[int, int, int, bool]] = []

    for patch_idx, (row_start, col_start) in enumerate(best_patches_tl):
        inner_row_start, inner_col_start = row_start, col_start
        inner_row_end, inner_col_end = row_start + a, col_start + b

        for row_idx in range(inner_row_start, min(inner_row_end, n_rows)):
            for col_idx in range(inner_col_start, min(inner_col_end, n_cols)):
                all_cells.append((row_idx, col_idx, patch_idx, False))

        if padding_size > 0:
            outer_row_start = max(0, inner_row_start - padding_size)
            outer_row_end = min(n_rows, inner_row_end + padding_size)
            outer_col_start = max(0, inner_col_start - padding_size)
            outer_col_end = min(n_cols, inner_col_end + padding_size)

            for row_idx in range(outer_row_start, outer_row_end):
                for col_idx in range(outer_col_start, outer_col_end):
                    if inner_row_start <= row_idx < inner_row_end and inner_col_start <= col_idx < inner_col_end:
                        continue
                    all_cells.append((row_idx, col_idx, patch_idx, True))

    if not all_cells:
        return [], [], total_score

    rows = np.array([item[0] for item in all_cells])
    cols = np.array([item[1] for item in all_cells])
    indices = [item[2] for item in all_cells]
    near_flags = [item[3] for item in all_cells]

    xs = min_x + (cols + 0.5) * config.grid_size
    ys = max_y - (rows + 0.5) * config.grid_size

    points_df = gpd.GeoDataFrame(
        {'geometry': gpd.points_from_xy(xs, ys)},
        crs=config.target_crs
    ).to_crs('EPSG:4326')

    lons = points_df.geometry.x.values
    lats = points_df.geometry.y.values

    for idx, patch_idx in enumerate(indices):
        cell_info = {
            'x': int(cols[idx]),
            'y': int(rows[idx]),
            'lat': float(lats[idx]),
            'lon': float(lons[idx]),
        }
        if near_flags[idx]:
            near_patches[patch_idx].append(cell_info)
        else:
            patches[patch_idx].append(cell_info)

    return patches, near_patches, total_score


def map_landuse_to_grid(
    gdf: gpd.GeoDataFrame,
    grid_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
) -> np.ndarray:
    print("  Mapping land use to grid...")

    min_x, max_x, min_y, max_y = bounds
    print(f"  Grid bounds: x({min_x}, {max_x}), y({min_y}, {max_y})")
    n_rows, n_cols = grid_shape

    landuse_grid = np.zeros((n_rows, n_cols), dtype=np.int8)

    def atrb_to_num(atrb: str) -> int:
        if atrb.startswith('UQA'):
            try:
                return int(atrb[3])
            except (ValueError, IndexError):
                print(f"Warning: Unexpected ATRB_SE format: {atrb}")
                return 0
        return 0

    sindex = gdf.sindex

    for row_idx in tqdm(range(n_rows), desc="  Mapping rows"):
        for col_idx in range(n_cols):
            center_x = min_x + (col_idx + 0.5) * config.grid_size
            center_y = max_y - (row_idx + 0.5) * config.grid_size
            point = Point(center_x, center_y)

            possible_matches_idx = list(sindex.intersection(point.bounds))
            if possible_matches_idx:
                possible_matches = gdf.iloc[possible_matches_idx]
                containing = possible_matches[possible_matches.contains(point)]
                if len(containing) > 0:
                    atrb = containing.iloc[0]['ATRB_SE']
                    landuse_grid[row_idx, col_idx] = atrb_to_num(atrb)

    print(f"  Classified: {np.count_nonzero(landuse_grid)}/{landuse_grid.size} cells")
    return landuse_grid


def compute_node_composition(
    landuse_grid: np.ndarray,
    patches: list[list[dict]],
    poi_gdf: gpd.GeoDataFrame | None,
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
) -> list[dict]:
    print("\n" + "=" * 60)
    print("Computing node composition (Land Use + POI)")
    print("=" * 60)

    poi_grid_map: defaultdict[tuple[int, int], list[object]] = defaultdict(list)
    if poi_gdf is not None:
        print("  Mapping POIs to grid...")
        min_x, max_x, min_y, max_y = bounds
        poi_x = poi_gdf.geometry.x.values
        poi_y = poi_gdf.geometry.y.values
        categories = poi_gdf['개방서비스아이디'].values

        poi_cols = ((poi_x - min_x) / config.grid_size).astype(int)
        poi_rows = ((max_y - poi_y) / config.grid_size).astype(int)

        n_rows, n_cols = landuse_grid.shape
        valid_mask = (
            (poi_rows >= 0) & (poi_rows < n_rows) &
            (poi_cols >= 0) & (poi_cols < n_cols)
        )

        valid_rows = poi_rows[valid_mask]
        valid_cols = poi_cols[valid_mask]
        valid_categories = categories[valid_mask]

        for row_idx, col_idx, category in zip(valid_rows, valid_cols, valid_categories):
            poi_grid_map[(row_idx, col_idx)].append(category)

        print(f"  Mapped {len(valid_rows):,} POIs to grid")

    node_compositions: list[dict] = []
    for patch_cells in tqdm(patches, desc="Computing compositions"):
        composition = Counter()
        poi = Counter()

        for cell in patch_cells:
            row_idx = cell['y']
            col_idx = cell['x']

            try:
                value = landuse_grid[row_idx, col_idx]
                if value > 0:
                    composition[f'UQA{value}'] += 1
                elif value == 0:
                    composition['Unclassified'] += 1
            except IndexError:
                pass

            if (row_idx, col_idx) in poi_grid_map:
                for category in poi_grid_map[(row_idx, col_idx)]:
                    if pd.notna(category):
                        poi[category] += 1

        node_compositions.append({
            'land_use': dict(composition),
            'poi': dict(poi),
        })

    return node_compositions


def create_integrated_nodes(
    patches: list[list[dict]],
    near_patches: list[list[dict]],
    node_compositions: list[dict],
    config: GraphBuildConfig,
) -> list[dict]:
    print("\n" + "=" * 60)
    print("Creating integrated node information")
    print("=" * 60)

    nodes: list[dict] = []
    for node_idx, patch_cells in enumerate(tqdm(patches, desc="Creating nodes")):
        if not patch_cells:
            continue

        lats = [cell['lat'] for cell in patch_cells]
        lons = [cell['lon'] for cell in patch_cells]
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)

        nodes.append({
            'node_id': node_idx,
            'lat': float(avg_lat),
            'lon': float(avg_lon),
            'composition': node_compositions[node_idx],
            'cells': patch_cells,
            'near_cells': near_patches[node_idx] if node_idx < len(near_patches) else [],
            'size': len(patch_cells) * (config.grid_size / 1000) ** 2,
        })

    return nodes


def create_coord_to_node_mapping(patches: list[list[dict]]) -> dict[tuple[int, int], int]:
    coord_to_node: dict[tuple[int, int], int] = {}
    for node_idx, patch_cells in enumerate(patches):
        for cell in patch_cells:
            coord_to_node[(cell['y'], cell['x'])] = node_idx
    return coord_to_node


def extract_node_demands(temporal_grid: np.ndarray, patches: list[list[dict]]) -> np.ndarray:
    print("\n" + "=" * 60)
    print("Extracting node demands")
    print("=" * 60)

    n_timesteps = temporal_grid.shape[0]
    n_nodes = len(patches)
    demands = np.zeros((n_timesteps, n_nodes), dtype=np.int32)

    for node_idx, patch_cells in enumerate(tqdm(patches, desc="Extracting demands")):
        if not patch_cells:
            continue

        rows = [cell['y'] for cell in patch_cells]
        cols = [cell['x'] for cell in patch_cells]
        demands[:, node_idx] = temporal_grid[:, rows, cols].sum(axis=1)

    print(f"\nDemand extraction complete:")
    print(f"  Shape: {demands.shape} (T={n_timesteps}, N={n_nodes})")
    print(f"  Total demand: {demands.sum():,}")
    print(f"  Average demand per node per hour: {demands.mean():.2f}")
    return demands


def extract_near_cell_demands(temporal_grid: np.ndarray, near_patches: list[list[dict]]) -> list[np.ndarray]:
    """노드별 near cell을 합산하지 않고 셀 단위 해상도로 유지한 수요 시계열을 반환.

    노드마다 near cell 개수가 다를 수 있어(경계 클리핑) 노드별 (T, n_near_cells) 배열의
    리스트로 반환한다. 각 배열의 열 순서는 near_patches[node_idx]의 셀 순서와 동일하다.
    """
    print("\n" + "=" * 60)
    print("Extracting near-cell demands (per-cell resolution)")
    print("=" * 60)

    n_timesteps = temporal_grid.shape[0]
    near_cell_demands: list[np.ndarray] = []

    for near_cells in tqdm(near_patches, desc="Extracting near-cell demands"):
        if not near_cells:
            near_cell_demands.append(np.zeros((n_timesteps, 0), dtype=np.int32))
            continue

        rows = [cell['y'] for cell in near_cells]
        cols = [cell['x'] for cell in near_cells]
        near_cell_demands.append(temporal_grid[:, rows, cols])

    cell_counts = [arr.shape[1] for arr in near_cell_demands]
    total_near_demand = sum(int(arr.sum()) for arr in near_cell_demands)

    print(f"\nNear-cell demand extraction complete:")
    print(f"  Nodes: {len(near_cell_demands)}")
    if cell_counts:
        print(f"  Near cells per node: min={min(cell_counts)}, max={max(cell_counts)}, avg={np.mean(cell_counts):.2f}")
    print(f"  Total near-cell demand: {total_near_demand:,}")
    return near_cell_demands


def extract_od_flows(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    unique_hours: list[pd.Timestamp],
    coord_to_node: dict[tuple[int, int], int],
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
) -> list[list[dict]]:
    print("\n" + "=" * 60)
    print("Extracting OD flows")
    print("=" * 60)

    min_x, max_x, min_y, max_y = bounds
    if 'dest_xpos' not in df.columns or 'dest_ypos' not in df.columns:
        print("  No destination columns found, skipping OD extraction")
        return [[] for _ in range(len(unique_hours))]

    df_with_dest = df.dropna(subset=['dest_xpos', 'dest_ypos']).copy()
    print(f"  Records with valid destinations: {len(df_with_dest):,} / {len(df):,}")

    if len(df_with_dest) == 0:
        return [[] for _ in range(len(unique_hours))]

    df_with_dest['hour'] = df_with_dest['call_date'].dt.floor('h')
    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}
    df_with_dest['time_idx'] = df_with_dest['hour'].map(hour_to_idx)

    origin_coords = gdf.loc[df_with_dest.index]
    origin_x = origin_coords.geometry.x.values
    origin_y = origin_coords.geometry.y.values

    df_with_dest['dest_lon'] = df_with_dest['dest_xpos'] / 1_000_000
    df_with_dest['dest_lat'] = df_with_dest['dest_ypos'] / 1_000_000

    gdf_dest = gpd.GeoDataFrame(
        df_with_dest[['dest_lon', 'dest_lat']],
        geometry=gpd.points_from_xy(df_with_dest['dest_lon'], df_with_dest['dest_lat']),
        crs='EPSG:4326'
    )
    gdf_dest = gdf_dest.to_crs(config.target_crs)

    dest_x = gdf_dest.geometry.x.values
    dest_y = gdf_dest.geometry.y.values

    origin_cols = ((origin_x - min_x) / config.grid_size).astype(int)
    origin_rows = ((max_y - origin_y) / config.grid_size).astype(int)
    dest_cols = ((dest_x - min_x) / config.grid_size).astype(int)
    dest_rows = ((max_y - dest_y) / config.grid_size).astype(int)

    df_with_dest['origin_node'] = -1
    df_with_dest['dest_node'] = -1

    for idx in range(len(df_with_dest)):
        origin_coord = (origin_rows[idx], origin_cols[idx])
        dest_coord = (dest_rows[idx], dest_cols[idx])
        df_with_dest.iloc[idx, df_with_dest.columns.get_loc('origin_node')] = coord_to_node.get(origin_coord, -1)
        df_with_dest.iloc[idx, df_with_dest.columns.get_loc('dest_node')] = coord_to_node.get(dest_coord, -1)

    valid_od = df_with_dest[(df_with_dest['origin_node'] >= 0) & (df_with_dest['dest_node'] >= 0)]
    print(f"  Valid OD pairs (both in selected nodes): {len(valid_od):,}")

    od_flows = [[] for _ in range(len(unique_hours))]
    if len(valid_od) > 0:
        grouped = valid_od.groupby(['time_idx', 'origin_node', 'dest_node']).size().reset_index(name='cnt')

        print("  Computing OD flows by timestep...")
        for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="  Processing OD pairs"):
            time_idx = int(row['time_idx'])
            origin_node = int(row['origin_node'])
            dest_node = int(row['dest_node'])
            count = int(row['cnt'])

            if pd.notna(time_idx) and 0 <= time_idx < len(unique_hours):
                od_flows[time_idx].append({'u': origin_node, 'v': dest_node, 'cnt': count})

    total_od_records = sum(len(od_list) for od_list in od_flows)
    total_od_count = sum(od['cnt'] for od_list in od_flows for od in od_list)
    non_empty_timesteps = sum(1 for od_list in od_flows if len(od_list) > 0)

    print(f"\nOD flow statistics:")
    print(f"  Timesteps with OD data: {non_empty_timesteps} / {len(unique_hours)}")
    print(f"  Unique OD pairs (across all time): {total_od_records:,}")
    print(f"  Total OD trips: {total_od_count:,}")
    return od_flows


def create_temporal_features(unique_hours: list[pd.Timestamp], config: GraphBuildConfig) -> list[dict]:
    print("\n" + "=" * 60)
    print("Creating temporal features")
    print("=" * 60)

    holiday_set = set(config.korean_holidays)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    temporal_features: list[dict] = []

    for hour in tqdm(unique_hours, desc="Processing timestamps"):
        date_str = hour.strftime('%Y-%m-%d')
        temporal_features.append({
            'day': day_names[hour.weekday()],
            'time': hour.hour,
            'holiday': date_str in holiday_set,
        })

    n_holidays = sum(1 for feature in temporal_features if feature['holiday'])
    print(f"\nTemporal features statistics:")
    print(f"  Total timesteps: {len(temporal_features)}")
    print(f"  Holiday timesteps: {n_holidays} ({n_holidays / len(temporal_features) * 100:.2f}%)")

    day_counts = {day: sum(1 for feature in temporal_features if feature['day'] == day) for day in day_names}
    print("  Day distribution:")
    for day, count in day_counts.items():
        print(f"    {day}: {count} timesteps")

    return temporal_features


def load_landuse_geodataframe(shp_path: Path, config: GraphBuildConfig) -> gpd.GeoDataFrame:
    print("\nLoading shapefile and creating land use grid...")
    print("  Creating landuse_grid...")
    gdf = gpd.read_file(shp_path)
    if gdf.crs != config.target_crs:
        gdf = gdf.to_crs(config.target_crs)
    return gdf


def load_poi_geodataframe(poi_csv_path: Path) -> gpd.GeoDataFrame:
    print("\nLoading POI data...")
    poi_df = pd.read_csv(poi_csv_path, encoding='utf-8')
    poi_df = poi_df.dropna(subset=['좌표정보x(epsg5174)', '좌표정보y(epsg5174)'])
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df['좌표정보x(epsg5174)'], poi_df['좌표정보y(epsg5174)']),
        crs='EPSG:5174'
    )
    print(f"  Loaded {len(poi_gdf):,} POIs")
    return poi_gdf


def build_graph_data(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    shp_path: Path,
    poi_csv_path: Path,
    config: GraphBuildConfig,
) -> GraphBuildArtifacts:
    bounds = calculate_bounds(gdf)
    temporal_grid, unique_hours = create_temporal_grid(df, gdf, bounds, config)
    sum_grid = temporal_grid.sum(axis=0)

    print("\n" + "=" * 60)
    print("Selecting optimal patches")
    print("=" * 60)
    patches, near_patches, total_score = get_patch(
        sum_grid,
        config.patch_size,
        config.patch_size,
        config.n_patches,
        bounds,
        config=config,
        padding_size=config.padding_size,
    )
    print(f'patches sample: {patches[0][:3] if patches and patches[0] else "No patches"}')
    print(f'near_patches sample: {near_patches[0][:3] if near_patches and near_patches[0] else "No near cells"}')
    print(f'len near_patches: {len(near_patches)}, len(near_patches[0]): {len(near_patches[0]) if near_patches else "N/A"}')

    total_cells = sum(len(patch) for patch in patches)
    coverage = total_cells / (sum_grid.shape[0] * sum_grid.shape[1])

    print(f"\nPatch selection results:")
    print(f"  Selected patches: {len(patches)}")
    print(f"  Total cells in patches: {total_cells}")
    print(f"  Coverage: {coverage * 100:.2f}%")
    print(f"  Total score: {total_score:,} / {sum_grid.sum():,} ({total_score / sum_grid.sum() * 100:.2f}%)")

    expected_near_cells = (config.patch_size + 2 * config.padding_size) ** 2 - (config.patch_size ** 2)
    near_cells_per_node = [len(cells) for cells in near_patches]
    if near_cells_per_node:
        print(f"  Expected near cells per node (no boundary clipping): {expected_near_cells}")
        print(
            "  Actual near cells per node: "
            f"min={min(near_cells_per_node)}, max={max(near_cells_per_node)}, avg={np.mean(near_cells_per_node):.2f}"
        )

    landuse_gdf = load_landuse_geodataframe(shp_path, config)
    landuse_grid = map_landuse_to_grid(landuse_gdf, sum_grid.shape, bounds, config)
    print(f'  landuse_grid shape: {landuse_grid.shape}')

    poi_gdf = load_poi_geodataframe(poi_csv_path)
    node_compositions = compute_node_composition(landuse_grid, patches, poi_gdf, bounds, config)
    nodes = create_integrated_nodes(patches, near_patches, node_compositions, config)

    demands = extract_node_demands(temporal_grid, patches)
    near_demands = extract_near_cell_demands(temporal_grid, near_patches)
    coord_to_node = create_coord_to_node_mapping(patches)

    print("\n" + "=" * 60)
    print("Creating coordinate-to-node mapping")
    print("=" * 60)
    print(f"  Mapped grid cells: {len(coord_to_node):,}")

    od_flows = extract_od_flows(df, gdf, unique_hours, coord_to_node, bounds, config)
    temporal_features = create_temporal_features(unique_hours, config)

    return GraphBuildArtifacts(
        bounds=bounds,
        temporal_grid=temporal_grid,
        unique_hours=unique_hours,
        sum_grid=sum_grid,
        patches=patches,
        near_patches=near_patches,
        total_score=total_score,
        landuse_grid=landuse_grid,
        node_compositions=node_compositions,
        nodes=nodes,
        demands=demands,
        near_demands=near_demands,
        coord_to_node=coord_to_node,
        od_flows=od_flows,
        temporal_features=temporal_features,
    )
