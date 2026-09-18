#!/usr/bin/env python3
"""merged 모델 체크포인트들의 노드별 test 지표를 격자 위에 그린다.

lora 브랜치의 ``visualize_node_metrics.py``와 같은 형식이다 — 지표 3개(RMSE / MAE / MAPE(+1))를
행으로, [기준 / 변형 / 차이]를 열로 놓는다. 차이 열은 0을 중심으로 한 발산 컬러맵이라
초록이 개선(감소), 빨강이 악화(증가)다.

전체 지표 하나(예: "RMSE +2%")로는 모델이 **어느 노드에서** 이득을 보고 잃었는지 알 수 없다.
수요 분포가 심하게 치우친 격자(ulsan은 노드 중앙값 0.141)에서는 그 구분이 결정적이다.

    python visualize_merged_node_metrics.py \\
        --baseline output/merged/2026-09-12/00-48-02 \\
        --variant  output/merged/2026-09-12/16-18-35 \\
        --variant-label "stage2 (combined)" \\
        --out docs/merged_node_metrics/ulsan_lossC.png
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
    model = MergedDemandModel.from_pretrained(str(checkpoint)).to(device).eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            target = batch.pop('labels')
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.append(model(**batch)['logits'].cpu().numpy())
            labels.append(target.numpy())
    del model
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
        # 수요가 높은 노드에서의 변화가 전체 지표를 좌우하므로 따로 센다.
        high = demand > np.median(demand)
        high_delta = float(delta[high].mean())
        delta_axis.set_title(
            f'{var_label} - {base_label} — {METRIC_LABELS[metric]}\n'
            f'green: decreased ({improved}), red: increased ({degraded})  |  '
            f'수요 상위 절반 노드 평균 Δ = {high_delta:+.3f}')
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
    parser.add_argument('--baseline', required=True, help='기준 체크포인트 디렉터리')
    parser.add_argument('--variant', required=True, help='비교할 체크포인트 디렉터리')
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

    base_pred, labels = predict(Path(args.baseline), dataset, device, args.batch_size)
    var_pred, _ = predict(Path(args.variant), dataset, device, args.batch_size)
    demand = labels.astype(np.float64).mean(axis=0)   # 노드별 평균 실제 수요

    plot(args.city, node_metrics(base_pred, labels), node_metrics(var_pred, labels),
         demand, args.baseline_label, args.variant_label, Path(args.out))

    # 전체 지표도 같이 찍어 그림과 숫자가 어긋나지 않게 한다.
    for label, pred in ((args.baseline_label, base_pred), (args.variant_label, var_pred)):
        err = labels - pred
        print(f'  {label:24s} RMSE={np.sqrt((err**2).mean()):.5f} '
              f'MAE={np.abs(err).mean():.5f} '
              f'MAPE+1={(np.abs(err)/(np.abs(labels)+1)).mean()*100:.3f}')


if __name__ == '__main__':
    main()
