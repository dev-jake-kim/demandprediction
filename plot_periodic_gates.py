"""Plot effective node-wise periodic gates from historical or current MA/EMA checkpoints."""

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
    if config['periodic_mode'] not in ('ma', 'ma_no_local', 'ema'):
        raise ValueError('노드별 게이트는 periodic_mode=ma/ma_no_local/ema 체크포인트에만 있다')
    height, width = int(config['height']), int(config['width'])
    with safe_open(checkpoint / 'model.safetensors', framework='np') as weights:
        mixture = 'fusion.daily_mix_logit' in weights.keys()
        daily_logits = weights.get_tensor(
            'fusion.daily_mix_logit' if mixture else 'fusion.daily_gate'
        )
        weekly_logits = weights.get_tensor(
            'fusion.weekly_mix_logit' if mixture else 'fusion.weekly_gate'
        )
    if daily_logits.shape != (height * width,) or weekly_logits.shape != (height * width,):
        raise ValueError('게이트 길이와 체크포인트의 격자 크기가 일치하지 않는다')

    if mixture:
        # Local has fixed reference logit 0; normalize all three branches.
        shift = np.maximum(0.0, np.maximum(daily_logits, weekly_logits))
        local_weight = np.exp(-shift)
        daily_weight = np.exp(daily_logits - shift)
        weekly_weight = np.exp(weekly_logits - shift)
        total = local_weight + daily_weight + weekly_weight
        local_gate = (local_weight / total).reshape(height, width)
        daily = (daily_weight / total).reshape(height, width)
        weekly = (weekly_weight / total).reshape(height, width)
    else:
        # Historical MA and current EMA use independent sigmoid gates.
        daily = np.exp(-np.logaddexp(0.0, -daily_logits)).reshape(height, width)
        weekly = np.exp(-np.logaddexp(0.0, -weekly_logits)).reshape(height, width)
    difference = daily - weekly
    displayed = (local_gate, daily, weekly) if mixture else (daily, weekly)
    low = max(0.0, float(min(values.min() for values in displayed)) - 0.02)
    high = min(1.0, float(max(values.max() for values in displayed)) + 0.02)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, values, label in zip(axes[0], (daily, weekly), ('Daily', 'Weekly')):
        image = ax.imshow(values, origin='upper', cmap='viridis', vmin=low, vmax=high)
        ax.set_title(f'{label} gate: min {values.min():.3f}, median {np.median(values):.3f}, max {values.max():.3f}')
        ax.set_xlabel('Grid column')
        ax.set_ylabel('Grid row')
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
    fig.colorbar(image, ax=list(axes[0]), label='Effective gate (shared scale)', shrink=0.8)

    ax = axes[1, 0]
    if mixture:
        image = ax.imshow(local_gate, origin='upper', cmap='viridis', vmin=low, vmax=high)
        ax.set_title(f'Local gate: min {local_gate.min():.3f}, median {np.median(local_gate):.3f}, max {local_gate.max():.3f}')
    else:
        span = float(np.abs(difference).max())
        image = ax.imshow(difference, origin='upper', cmap='RdBu_r', vmin=-span, vmax=span)
        ax.set_title('Daily - weekly gate')
    ax.set_xlabel('Grid column')
    ax.set_ylabel('Grid row')
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    fig.colorbar(image, ax=ax, label='Local gate' if mixture else 'Gate difference', shrink=0.8)

    ax = axes[1, 1]
    bins = np.linspace(low, high, 25)
    ax.hist(daily.ravel(), bins=bins, alpha=0.55, label='Daily')
    ax.hist(weekly.ravel(), bins=bins, alpha=0.55, label='Weekly')
    if mixture:
        ax.hist(local_gate.ravel(), bins=bins, alpha=0.55, label='Local')
        for value in (0.7, 0.2, 0.1):
            ax.axvline(value, color='black', linestyle='--', linewidth=1)
    else:
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Initial gate = 0.5')
    ax.set(xlabel='Effective gate', ylabel='Number of nodes', title='Distribution across nodes')
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
