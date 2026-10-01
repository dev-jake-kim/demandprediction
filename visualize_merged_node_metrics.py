#!/usr/bin/env python3
"""Plot per-node test metrics for merged-model checkpoints.

Rows contain RMSE, MAE, and MAPE(+1); columns contain the baseline, variant,
and their difference.

    python visualize_merged_node_metrics.py \\
        --baseline output/past/merged/2026-09-12/00-48-02 \\
        --variant output/past/merged/2026-09-12/16-18-35 \\
        --variant-label "stage2 (combined)" \\
        --out output/merged_node_metrics/ulsan_lossC.png

    CUDA_VISIBLE_DEVICES=1 python visualize_merged_node_metrics.py \\
        --city porto --adformer-dir output/ADFormer/window24/regional_metrics \\
        --variant output/past/experiments/checkpoints/tmp_node01_mae_porto_seed245 \\
        --baseline-label "ADFormer (5-seed mean)" --variant-label "stage2 (seed=245)" \\
        --out output/node_metric_comparison/porto_seed245_vs_adformer.png

    CUDA_VISIBLE_DEVICES=1 python visualize_merged_node_metrics.py \\
        --city porto --adformer-dir output/ADFormer/window24/regional_metrics \\
        --variant output/past/experiments/checkpoints/tmp_node01_mae_porto_seed245 \\
                  output/past/experiments/checkpoints/tmp_node01_mae_porto_seed6835 \\
                  output/past/experiments/checkpoints/tmp_node01_mae_porto_seed851 \\
        --baseline-label "ADFormer (5-seed mean)" \\
        --variant-label "stage2 (3-seed mean)" \\
        --out output/node_metric_comparison/porto_stage2_mean_vs_adformer.png

Multiple variants are averaged per node and metric after each variant's metrics
are computed; predictions are not averaged first. ADFormer maps are aggregate
node metrics, not values from a matched seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import DataLoader

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.merged import MergedDemandModel
from train import build_dataset_kwargs

ROOT = Path(__file__).resolve().parent
METRICS = ('rmse', 'mae', 'mape_plus1')
METRIC_LABELS = {'rmse': 'RMSE', 'mae': 'MAE', 'mape_plus1': 'MAPE(+1) [%]'}


def predict(checkpoint: Path, dataset: UnifiedDemandDataset, device: torch.device,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if device.type == 'cuda':
        torch.cuda.init()  # cuDNN LSTM 가중치를 GPU로 옮기기 전에 CUDA device를 확정한다.
    model = MergedDemandModel.from_pretrained(str(checkpoint)).to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            target = batch.pop('labels')
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.append(model(**batch)['logits'].cpu().numpy())
            labels.append(target.numpy())
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(labels)


def node_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """[T,H,W] -> 노드별(H,W) 지표. 시간 축으로만 집계한다."""
    error = labels.astype(np.float64) - predictions.astype(np.float64)
    return {
        'rmse': np.sqrt(np.mean(error ** 2, axis=0)),
        'mae': np.mean(np.abs(error), axis=0),
        'mape_plus1': np.mean(np.abs(error) / (np.abs(labels) + 1.0), axis=0) * 100.0,
    }


def load_adformer_node_metrics(directory: Path, city: str,
                               shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Load the supplied ADFormer node maps; its ``mape`` uses the +1 denominator."""
    prefix = city.upper()
    filenames = {'rmse': 'rmse', 'mae': 'mae', 'mape_plus1': 'mape'}
    metrics = {}
    for name, suffix in filenames.items():
        path = directory / f'{prefix}_{suffix}.npy'
        values = np.load(path, allow_pickle=False)
        if values.shape != shape or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f'{path}: expected numeric grid {shape}, got {values.shape} {values.dtype}')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{path}: node metrics must be finite and nonnegative')
        metrics[name] = values
    return metrics


def annotate_delta(axis: plt.Axes, values: np.ndarray) -> None:
    fontsize = 4.3 if values.size >= 190 else 4.8
    threshold = np.nanmax(np.abs(values)) * 0.55
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        color = 'white' if abs(value) > threshold else 'black'
        axis.text(col, row, f'{value:+.2f}', ha='center', va='center',
                  fontsize=fontsize, color=color)


def plot(city: str, baseline: dict, variant: dict, demand: np.ndarray,
         base_label: str, var_label: str, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(23, 15), constrained_layout=True)
    for row_idx, metric in enumerate(METRICS):
        base, var = baseline[metric], variant[metric]
        delta = var - base
        lo = float(min(base.min(), var.min()))
        hi = float(max(base.max(), var.max()))
        if lo == hi:
            hi = lo + 1e-12
        for col, (values, title) in enumerate(((base, base_label), (var, var_label))):
            axis = axes[row_idx, col]
            image = axis.imshow(values, cmap='viridis', vmin=lo, vmax=hi, origin='upper')
            axis.set_title(f'{title} — {METRIC_LABELS[metric]}')
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        limit = float(np.max(np.abs(delta))) or 1e-12
        delta_axis = axes[row_idx, 2]
        delta_image = delta_axis.imshow(
            delta, cmap='RdYlGn_r',
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), origin='upper')
        improved = int(np.count_nonzero(delta < 0))
        degraded = int(np.count_nonzero(delta > 0))
        high = demand > np.median(demand)
        high_delta = float(delta[high].mean())
        delta_axis.set_title(
            f'{var_label} - {base_label} — {METRIC_LABELS[metric]}\n'
            f'green/lower: {improved}, red/higher: {degraded}; '
            f'high-demand mean Δ: {high_delta:+.3f}')
        annotate_delta(delta_axis, delta)
        fig.colorbar(delta_image, ax=delta_axis, fraction=0.046, pad=0.04)

        for axis in axes[row_idx]:
            axis.set_xlabel('grid column')
            axis.set_ylabel('grid row')
            axis.set_xticks(np.arange(base.shape[1]))
            axis.set_yticks(np.arange(base.shape[0]))
            axis.tick_params(labelsize=7)

    fig.suptitle(f'Per-node test metrics: {city}  —  {base_label} vs {var_label}', fontsize=18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f'saved: {output}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='ulsan', choices=('ulsan', 'porto'))
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline', help='기준 체크포인트 디렉터리')
    baseline.add_argument('--adformer-dir', help='도시별 *_rmse/mae/mape.npy가 있는 디렉터리')
    parser.add_argument('--variant', required=True, nargs='+',
                        help='변형 체크포인트 디렉터리(여러 개면 노드별 지표 평균)')
    parser.add_argument('--baseline-label', default='stage1')
    parser.add_argument('--variant-label', default='stage2')
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name=f'config_{args.city}')
    dataset = UnifiedDemandDataset(
        resolve_dataset_path(args.city, cfg.dataset.npy_path), 'test',
        **build_dataset_kwargs(cfg))

    if args.adformer_dir:
        base_metrics = load_adformer_node_metrics(
            Path(args.adformer_dir), args.city, (dataset.height, dataset.width))
    else:
        base_pred, base_labels = predict(Path(args.baseline), dataset, device, args.batch_size)
        base_metrics = node_metrics(base_pred, base_labels)

    variant_maps = []
    variant_global = []
    for checkpoint in args.variant:
        prediction, labels = predict(Path(checkpoint), dataset, device, args.batch_size)
        variant_maps.append(node_metrics(prediction, labels))
        error = labels - prediction
        variant_global.append((
            Path(checkpoint).name,
            float(np.sqrt((error ** 2).mean())),
            float(np.abs(error).mean()),
            float((np.abs(error) / (np.abs(labels) + 1)).mean() * 100),
        ))

    # Average per-node metrics, not predictions.
    var_metrics = (
        variant_maps[0] if len(variant_maps) == 1 else
        {name: np.mean([maps[name] for maps in variant_maps], axis=0) for name in METRICS}
    )
    demand = labels.astype(np.float64).mean(axis=0)
    plot(args.city, base_metrics, var_metrics,
         demand, args.baseline_label, args.variant_label, Path(args.out))

    if args.adformer_dir:
        print(f'  {args.baseline_label:24s} RMSE={np.sqrt(np.mean(base_metrics["rmse"] ** 2)):.5f} '
              f'MAE={base_metrics["mae"].mean():.5f} '
              f'MAPE+1={base_metrics["mape_plus1"].mean():.3f} '
              '(aggregated node maps, not a matched seed)')
    else:
        error = base_labels - base_pred
        print(f'  {args.baseline_label:24s} RMSE={np.sqrt((error ** 2).mean()):.5f} '
              f'MAE={np.abs(error).mean():.5f} '
              f'MAPE+1={(np.abs(error)/(np.abs(base_labels)+1)).mean()*100:.3f}')
    for checkpoint, rmse, mae, mape in variant_global:
        label = args.variant_label if len(variant_global) == 1 else checkpoint
        print(f'  {label:24s} RMSE={rmse:.5f} MAE={mae:.5f} MAPE+1={mape:.3f}')
    if len(variant_global) > 1:
        print(f'  {args.variant_label:24s} mean of node RMSE={var_metrics["rmse"].mean():.5f} '
              f'MAE={var_metrics["mae"].mean():.5f} '
              f'MAPE+1={var_metrics["mape_plus1"].mean():.3f}')


if __name__ == '__main__':
    main()
