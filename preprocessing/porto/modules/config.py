from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphBuildConfig:
    # Porto 도심 bbox, lon/lat * 1e6 (ulsan의 xpos/ypos 스케일과 동일한 표현)
    # x: (min_lon, max_lon), y: (max_lat, min_lat) - filters.remove_other_region 관례와 동일
    porto_range_x: tuple[int, int] = (-8_750000, -8_500000)
    porto_range_y: tuple[int, int] = (41_250000, 41_050000)
    # target_crs(EPSG:3763, 미터) 투영 후 적용하는 하드캡. 둘 다 (min, max) 순서.
    # Porto 데이터가 해안을 따라 동서로 길게 뻗어있어 정사각형이 아니므로,
    # crop_filter의 반복적 이상치 제거 대신 이 사각 범위로 한 번에 잘라낸다.
    porto_projected_x_range: tuple[float, float] = (-48_000.0, -34_000.0)
    porto_projected_y_range: tuple[float, float] = (163_000.0, 170_000.0)
    grid_size: int = 700
    target_crs: str = 'EPSG:3763'  # ETRS89 / Portugal TM06
    n_patches: int = 50
    patch_size: int = 7
    padding_size: int = 3
    cbc_path: str | None = None
    train_csv_name: str = 'train.csv'
    output_graph_json_name: str = 'porto_data.json'
    output_step1_points_name: str = 'step1_cropped_pickup_points.png'
    output_patch_near_demands_name: str = 'patch_near_demands.png'
    output_node_density_name: str = 'node_demand_density_curves.png'
    output_landuse_grid_name: str = 'landuse_grid.npy'
    output_temporal_grid_name: str = 'porto_temporal_grid.npy'
    # 데이터 기간(2013-07-01 ~ 2014-06-30) 내 포르투갈 공휴일.
    # 2013~2015년은 긴축정책으로 Corpus Christi/10.5/11.1/12.1이 임시 폐지되어 제외했고,
    # 06-24(상 주앙 두 포르투)는 포르투 시 지역 공휴일이라 함께 포함.
    portuguese_holidays: tuple[str, ...] = field(default_factory=lambda: (
        '2013-08-15',
        '2013-12-08',
        '2013-12-25',
        '2014-01-01',
        '2014-04-18',
        '2014-04-20',
        '2014-04-25',
        '2014-05-01',
        '2014-06-10',
        '2014-06-24',
    ))
