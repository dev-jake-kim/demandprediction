from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'data' / 'raw'
CITIES = {
    'ulsan': DATA_DIR / 'ulsan_temporal_grid.npy',
    'porto': DATA_DIR / 'porto_temporal_grid.npy',
}
CAP = 20  # 이 값보다 큰 수요는 하나의 막대(CAP+)로 묶어서 표시


def plot_demand_distribution(city: str, npy_path: Path) -> Path:
    grid = np.load(npy_path)
    flat = grid.reshape(-1)

    capped = np.minimum(flat, CAP)
    counts = np.bincount(capped, minlength=CAP + 1)
    labels = [str(v) for v in range(CAP)] + [f'{CAP}+']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, counts, color='#3182bd')
    ax.set_yscale('log')
    ax.set_xlabel('Demand per grid cell per hour')
    ax.set_ylabel('Count (log scale)')
    ax.set_title(
        f'{city.upper()} Demand Distribution '
        f'(n={flat.size:,} grid-hour cells, mean={flat.mean():.3f}, max={int(flat.max())})'
    )
    ax.grid(axis='y', alpha=0.3)

    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=7, rotation=90)

    output_dir = BASE_DIR / city / 'analyze'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'demand_distribution.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f'[{city}] saved to {output_path}')
    return output_path


def plot_daily_avg_heatmap(city: str, npy_path: Path) -> Path:
    grid = np.load(npy_path)
    n_days = grid.shape[0] / 24
    daily_avg = grid.sum(axis=0) / n_days  # (H, W)

    fig, ax = plt.subplots(figsize=(grid.shape[2] * 0.5 + 2, grid.shape[1] * 0.5 + 2))
    image = ax.imshow(daily_avg, cmap='hot', interpolation='nearest')
    ax.set_title(f'{city.upper()} Daily Avg Demand per Grid Cell ({n_days:.0f} days)')
    ax.set_xlabel('grid col')
    ax.set_ylabel('grid row')
    plt.colorbar(image, ax=ax, label='Daily avg demand')

    output_dir = BASE_DIR / city / 'analyze'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'daily_avg_heatmap.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f'[{city}] saved to {output_path}')
    return output_path


def main() -> None:
    for city, npy_path in CITIES.items():
        plot_demand_distribution(city, npy_path)
        plot_daily_avg_heatmap(city, npy_path)


if __name__ == '__main__':
    main()
