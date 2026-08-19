from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphBuildConfig:
    us_range_x: tuple[int, int] = (128_950739, 129_500739)
    us_range_y: tuple[int, int] = (35_756099, 35_305729)
    grid_size: int = 700
    target_crs: str = 'EPSG:5174'
    n_patches: int = 50
    patch_size: int = 7
    padding_size: int = 3
    cbc_path: str | None = None
    origin_csv_name: str = 'origin_data.csv'
    shapefile_name: str = 'UPIS_C_UQ111.shp'
    poi_csv_name: str = 'poi_data.csv'
    output_graph_json_name: str = 'ulsan_data.json'
    output_step1_points_name: str = 'step1_cropped_demand_points.png'
    output_patch_near_demands_name: str = 'patch_near_demands.png'
    output_node_density_name: str = 'node_demand_density_curves.png'
    output_landuse_grid_name: str = 'landuse_grid.npy'
    output_temporal_grid_name: str = 'ulsan_temporal_grid.npy'
    korean_holidays: tuple[str, ...] = field(default_factory=lambda: (
        '2024-01-01',
        '2024-02-09',
        '2024-02-10',
        '2024-02-11',
        '2024-02-12',
        '2024-03-01',
        '2024-04-10',
        '2024-05-05',
        '2024-05-06',
        '2024-05-15',
        '2024-06-06',
        '2024-08-15',
        '2024-09-16',
        '2024-09-17',
        '2024-09-18',
        '2024-10-03',
        '2024-10-09',
        '2024-12-25',
        '2025-01-01',
        '2025-01-28',
        '2025-01-29',
        '2025-01-30',
        '2025-03-01',
        '2025-03-03',
        '2025-05-05',
        '2025-05-06',
        '2025-06-06',
        '2025-08-15',
        '2025-10-05',
        '2025-10-06',
        '2025-10-07',
        '2025-10-08',
        '2025-10-03',
        '2025-10-09',
        '2025-12-25',
    ))
