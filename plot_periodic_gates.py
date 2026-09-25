"""Plot the effective node-wise daily/weekly gates from a trained MA/EMA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open


def plot_gates(checkpoint: Path, output: Path) -> None:
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if config['periodic_mode'] not in ('ma', 'ema'):
        raise ValueError('노드별 게이트는 periodic_mode=ma/ema 체크포인트에만 있다')
    height, width = int(config['height']), int(config['width'])
    with safe_open(checkpoint / 'model.safetensors', framework='np') as weights:
        daily_logits = weights.get_tensor('fusion.daily_gate')
        weekly_logits = weights.get_tensor('fusion.weekly_gate')
    if daily_logits.shape != (height * width,) or weekly_logits.shape != (height * width,):
        raise ValueError('게이트 길이와 체크포인트의 격자 크기가 일치하지 않는다')

    # 실제 예측에 곱해지는 값은 raw parameter가 아니라 sigmoid(parameter)다.
    daily = np.exp(-np.logaddexp(0.0, -daily_logits)).reshape(height, width)
    weekly = np.exp(-np.logaddexp(0.0, -weekly_logits)).reshape(height, width)
    difference = daily - weekly
    low = max(0.0, float(min(daily.min(), weekly.min())) - 0.02)
    high = min(1.0, float(max(daily.max(), weekly.max())) + 0.02)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, values, label in zip(axes[0], (daily, weekly), ('Daily', 'Weekly')):
        image = ax.imshow(values, origin='upper', cmap='viridis', vmin=low, vmax=high)
        ax.set_title(f'{label} gate: min {values.min():.3f}, median {np.median(values):.3f}, max {values.max():.3f}')
        ax.set_xlabel('Grid column')
        ax.set_ylabel('Grid row')
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
    fig.colorbar(image, ax=list(axes[0]), label='sigmoid(gate) (shared scale)', shrink=0.8)

    ax = axes[1, 0]
    span = float(np.abs(difference).max())
    image = ax.imshow(difference, origin='upper', cmap='RdBu_r', vmin=-span, vmax=span)
    ax.set_title('Daily - weekly gate')
    ax.set_xlabel('Grid column')
    ax.set_ylabel('Grid row')
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    fig.colorbar(image, ax=ax, label='Gate difference', shrink=0.8)

    ax = axes[1, 1]
    bins = np.linspace(low, high, 25)
    ax.hist(daily.ravel(), bins=bins, alpha=0.55, label='Daily')
    ax.hist(weekly.ravel(), bins=bins, alpha=0.55, label='Weekly')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Initial gate = 0.5')
    ax.set(xlabel='sigmoid(gate)', ylabel='Number of nodes', title='Distribution across nodes')
    ax.legend()

    fig.suptitle(f'{config["periodic_mode"].upper()} periodic gates ({height} x {width} grid)')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f'Saved {output} ({height * width} nodes); daily > weekly: {int((difference > 0).sum())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot_gates(args.checkpoint, args.output)
