from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from .config import GraphBuildConfig

try:
    import contextily as ctx
except ImportError:
    ctx = None


def _pad_bounds(bounds: tuple[float, float, float, float], grid_size: int, pad_ratio: float = 0.03) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    pad_x = max(width * pad_ratio, grid_size)
    pad_y = max(height * pad_ratio, grid_size)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _bounds_from_gdf(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = gdf.total_bounds
    return min_x, max_x, min_y, max_y


def add_basemap(
    ax,
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
    pad_ratio: float = 0.03,
) -> None:
    min_x, max_x, min_y, max_y = _pad_bounds(bounds, config.grid_size, pad_ratio=pad_ratio)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                crs=config.target_crs,
                source=ctx.providers.CartoDB.Positron,
                attribution=False,
            )
        except Exception as exc:  # pragma: no cover - network/provider dependent
            print(f"  Failed to load basemap: {exc}")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_xlabel(f'X ({config.target_crs})')
    ax.set_ylabel(f'Y ({config.target_crs})')


def plot_point_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    title: str,
    config: GraphBuildConfig,
    color: str = '#d7301f',
    alpha: float = 0.12,
    markersize: float = 0.5,
) -> None:
    if gdf.empty:
        print("  GeoDataFrame is empty, skipping point visualization")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    add_basemap(ax, _bounds_from_gdf(gdf), config)
    gdf.plot(ax=ax, markersize=markersize, color=color, alpha=alpha)
    ax.set_title(title)
    fig.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved point map to: {output_path}")
    if ctx is None:
        print("  contextily is not installed, basemap was skipped")


def plot_polygon_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    title: str,
    config: GraphBuildConfig,
    column: str,
    cmap: Colormap | str,
    linewidth: float = 0.15,
    edgecolor: str = 'black',
    alpha: float = 0.55,
    legend: bool = True,
) -> None:
    if gdf.empty:
        print("  GeoDataFrame is empty, skipping polygon visualization")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    add_basemap(ax, _bounds_from_gdf(gdf), config)
    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        linewidth=linewidth,
        edgecolor=edgecolor,
        alpha=alpha,
        legend=legend,
    )
    ax.set_title(title)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved polygon map to: {output_path}")
    if ctx is None:
        print("  contextily is not installed, basemap was skipped")


def plot_grid_overlay(
    grid: np.ndarray,
    bounds: tuple[float, float, float, float],
    output_path: Path,
    title: str,
    config: GraphBuildConfig,
    cmap: Colormap | str = 'hot',
    colorbar_label: str = 'Value',
    alpha: float | np.ndarray = 0.75,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_zeros: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_x, max_x, min_y, max_y = bounds
    extent = (min_x, max_x, min_y, max_y)
    masked_grid = np.ma.masked_where(grid == 0, grid) if mask_zeros else grid

    fig, ax = plt.subplots(figsize=(12, 10))
    add_basemap(ax, bounds, config)
    image = ax.imshow(
        masked_grid,
        cmap=cmap,
        interpolation='nearest',
        origin='upper',
        extent=extent,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    plt.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved grid overlay to: {output_path}")
    if ctx is None:
        print("  contextily is not installed, basemap was skipped")


def plot_patch_near_demand_map(
    sum_grid: np.ndarray,
    patches: list[list[dict]],
    near_patches: list[list[dict]],
    output_path: Path,
    bounds: tuple[float, float, float, float],
    config: GraphBuildConfig,
) -> None:
    print("\n" + "=" * 60)
    print("Visualizing sum grid, patches, and near patches")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_x, max_x, min_y, max_y = bounds
    extent = (min_x, max_x, min_y, max_y)
    max_demand = float(sum_grid.max()) if sum_grid.size > 0 else 0.0

    if max_demand > 0:
        normalized_demand = sum_grid.astype(float) / max_demand
        demand_alpha = np.where(sum_grid > 0, normalized_demand ** 0.8, 0.0)
        masked_sum_grid = np.ma.masked_where(sum_grid <= 0, sum_grid)
    else:
        demand_alpha = np.zeros_like(sum_grid, dtype=float)
        masked_sum_grid = np.ma.masked_where(np.ones_like(sum_grid, dtype=bool), sum_grid)

    patch_mask = np.zeros_like(sum_grid, dtype=np.uint8)
    near_mask = np.zeros_like(sum_grid, dtype=np.uint8)

    for patch_cells in patches:
        for cell in patch_cells:
            patch_mask[cell['y'], cell['x']] = 1

    for near_cells in near_patches:
        for cell in near_cells:
            near_mask[cell['y'], cell['x']] = 1

    near_only_mask = (near_mask == 1) & (patch_mask == 0)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    def setup_axis(ax) -> None:
        add_basemap(ax, bounds, config, pad_ratio=0.0)

    setup_axis(axes[0])
    im0 = axes[0].imshow(
        masked_sum_grid,
        cmap='Reds',
        origin='upper',
        extent=extent,
        alpha=demand_alpha,
        vmin=0,
        vmax=max_demand if max_demand > 0 else 1,
    )
    axes[0].set_title('Sum Grid Demand (All Cells)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Demand')

    setup_axis(axes[1])
    im1 = axes[1].imshow(
        masked_sum_grid,
        cmap='Reds',
        origin='upper',
        extent=extent,
        alpha=demand_alpha,
        vmin=0,
        vmax=max_demand if max_demand > 0 else 1,
    )
    axes[1].imshow(
        np.ma.masked_where(~near_only_mask, near_only_mask),
        cmap=ListedColormap(['#5DA5DA']),
        alpha=0.35,
        origin='upper',
        extent=extent,
    )
    axes[1].imshow(
        np.ma.masked_where(patch_mask == 0, patch_mask),
        cmap=ListedColormap(['#F15854']),
        alpha=0.60,
        origin='upper',
        extent=extent,
    )
    axes[1].set_title('Demand + Near Cells(Blue) + Patch Cells(Red)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Demand')

    selected_coords = set()
    for patch_cells in patches:
        selected_coords.update((cell['y'], cell['x']) for cell in patch_cells)
    for near_cells in near_patches:
        selected_coords.update((cell['y'], cell['x']) for cell in near_cells)

    if len(selected_coords) <= 1200:
        for row_idx, col_idx in selected_coords:
            x_center = min_x + (col_idx + 0.5) * config.grid_size
            y_center = max_y - (row_idx + 0.5) * config.grid_size
            axes[1].text(
                x_center,
                y_center,
                str(int(sum_grid[row_idx, col_idx])),
                fontsize=5,
                ha='center',
                va='center',
                color='black',
            )
    else:
        print(f"  Skip cell text labels: too many selected cells ({len(selected_coords):,})")

    category = np.zeros_like(sum_grid, dtype=np.uint8)
    category[near_only_mask] = 1
    category[patch_mask == 1] = 2

    setup_axis(axes[2])
    axes[2].imshow(masked_sum_grid, cmap='Greys', alpha=demand_alpha * 0.35, origin='upper', extent=extent)
    axes[2].imshow(
        np.ma.masked_where(category == 0, category),
        cmap=ListedColormap(['#5DA5DA', '#F15854']),
        vmin=1,
        vmax=2,
        alpha=0.95,
        origin='upper',
        extent=extent,
    )
    axes[2].set_title('Patch/Near Layout')
    axes[2].legend(
        handles=[
            Patch(facecolor='#F15854', label='Patch cells'),
            Patch(facecolor='#5DA5DA', label='Near cells (excluding patch)'),
        ],
        loc='upper right',
    )

    fig.suptitle('Patch and Near-Patch Visualization on Sum Grid', fontsize=14)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)

    print(f"  Visualization saved to: {output_path}")
    print(f"  Patch cells: {int((patch_mask == 1).sum()):,}")
    print(f"  Near cells (excluding patch): {int(near_only_mask.sum()):,}")
    if ctx is None:
        print("  contextily is not installed, basemap was skipped")


def plot_density_curves(demands: np.ndarray, output_path: Path) -> None:
    print("\n" + "=" * 60)
    print("Plotting node demand density curves")
    print("=" * 60)

    if demands.size == 0 or demands.shape[1] == 0:
        print("  No node demand data available, skipping density plot")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_mean_demands = demands.mean(axis=0).astype(float)
    node_sigma_demands = demands.std(axis=0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    plot_specs = [
        (axes[0], node_mean_demands, 'Node Mean Demand Density', 'Mean demand', '#D95F02'),
        (axes[1], node_sigma_demands, 'Node Sigma Density', 'Sigma (std)', '#1B9E77'),
    ]

    for ax, values, title, xlabel, color in plot_specs:
        values = np.asarray(values, dtype=float)
        if len(values) == 1 or np.allclose(values, values[0]):
            x_grid = np.linspace(values[0] - 1, values[0] + 1, 200)
            y_grid = np.zeros_like(x_grid)
            y_grid[len(y_grid) // 2] = 1.0
        else:
            kde = gaussian_kde(values)
            x_min = values.min()
            x_max = values.max()
            padding = max((x_max - x_min) * 0.15, 1e-6)
            x_grid = np.linspace(x_min - padding, x_max + padding, 400)
            y_grid = kde(x_grid)

        ax.plot(x_grid, y_grid, color=color, linewidth=2)
        ax.fill_between(x_grid, y_grid, color=color, alpha=0.25)
        ax.axvline(values.mean(), color=color, linestyle='--', linewidth=1.5, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.grid(alpha=0.25)

    fig.suptitle('Node Demand Statistics Density Curves', fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"  Density plot saved to: {output_path}")
    print(f"  Mean demand stats: min={node_mean_demands.min():.3f}, max={node_mean_demands.max():.3f}, avg={node_mean_demands.mean():.3f}")
    print(f"  Sigma stats: min={node_sigma_demands.min():.3f}, max={node_sigma_demands.max():.3f}, avg={node_sigma_demands.mean():.3f}")
