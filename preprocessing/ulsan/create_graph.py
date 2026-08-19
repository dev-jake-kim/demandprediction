from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.build import build_graph_data
from modules.config import GraphBuildConfig
from modules.export import save_graph_json
from modules.preprocess import load_and_preprocess_data
from modules.visualization import plot_density_curves, plot_patch_near_demand_map, plot_point_map


def main() -> None:
    config = GraphBuildConfig()
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'raw'
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    origin_csv = data_dir / config.origin_csv_name
    shapefile_path = data_dir / config.shapefile_name
    poi_csv_path = data_dir / config.poi_csv_name

    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "GRAPH GENERATION FROM GRID" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝\n")

    df, gdf = load_and_preprocess_data(origin_csv, config)
    plot_point_map(
        gdf=gdf,
        output_path=output_dir / config.output_step1_points_name,
        title='Cropped Demand Points After Step 1',
        config=config,
    )

    artifacts = build_graph_data(df, gdf, shapefile_path, poi_csv_path, config)

    plot_patch_near_demand_map(
        sum_grid=artifacts.sum_grid,
        patches=artifacts.patches,
        near_patches=artifacts.near_patches,
        output_path=output_dir / config.output_patch_near_demands_name,
        bounds=artifacts.bounds,
        config=config,
    )

    plot_density_curves(
        demands=artifacts.demands,
        output_path=output_dir / config.output_node_density_name,
    )

    landuse_grid_path = output_dir / config.output_landuse_grid_name
    np.save(landuse_grid_path, artifacts.landuse_grid)
    print(f"  Saved landuse_grid to {landuse_grid_path}")

    temporal_grid_path = output_dir / config.output_temporal_grid_name
    np.save(temporal_grid_path, artifacts.temporal_grid)
    print(f"  Saved temporal_grid to {temporal_grid_path}")

    save_graph_json(
        output_path=output_dir / config.output_graph_json_name,
        nodes=artifacts.nodes,
        demands=artifacts.demands,
        near_demands=artifacts.near_demands,
        temporal_features=artifacts.temporal_features,
        od_flows=artifacts.od_flows,
    )

    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 18 + "GRAPH GENERATION COMPLETE!" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝\n")
    print(f"Output saved to: {output_dir / config.output_graph_json_name}")


if __name__ == '__main__':
    main()
