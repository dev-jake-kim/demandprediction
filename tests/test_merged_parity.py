"""Check forward/loss, checkpoint, and short-training parity against legacy code.

Legacy files are restored with ``git show`` from a pinned commit: a branch tip can
advance and stop identifying the reference files.

    conda run -n DA python tests/test_merged_parity.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path  # noqa: E402
from models.merged import MergedDemandConfig, MergedDemandModel  # noqa: E402

LEGACY_FILES = [
    '__init__.py',
    'data.py',
    'losses.py',
    'model.py',
    'modules/__init__.py',
    'modules/attention.py',
    'modules/embeddings.py',
    'modules/fusion.py',
    'modules/history.py',
    'modules/periodic.py',
    'modules/retrieval.py',
]

# Ablation cases covered by the parity check.
ABLATION_CASES = {
    'full': {},
    'no-ir': {'use_retrieval': False},
    'no-softplus': {'use_softplus': False},
    'no-neighbors': {'use_neighbors': False},
    # Includes a branch-shape-changing weather injection case.
    'weather-cls-add': {'weather_injection': 'cls_add'},
}

MODEL_KWARGS = dict(
    local_radius=2,
    d_model=64,
    num_fourier_bands=8,
    transformer_layers=2,
    transformer_heads=4,
    transformer_ffn=128,
    history_hidden=64,
    periodic_hidden=64,
    fusion_dim=128,
    dropout=0.10,
    retrieval_k=20,
    retrieval_chunk_size=256,
    retrieval_scope='observed_past',
    weekday_dim=7,
    hour_dim=5,
)
DATA_KWARGS = dict(
    time_step=24,
    daily_period=24,
    daily_lags=6,
    weekly_period=168,
    weekly_lags=4,
    lag_radius=0,
    train_ratio=0.80,
    val_ratio=0.10,
)


LEGACY_REF = '81472a5'


def restore_legacy_package(destination: Path) -> None:
    """Restore legacy package files from the pinned reference commit."""

    package = destination / 'merged_legacy'
    (package / 'modules').mkdir(parents=True, exist_ok=True)
    for relative in LEGACY_FILES:
        blob = subprocess.run(
            ['git', 'show', f'{LEGACY_REF}:merged_model/{relative}'],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        (package / relative).write_bytes(blob)


def build_batch(device: torch.device, batch_size: int, start: int):
    """실제 ulsan 배치 하나 + train split 날씨 통계."""

    data_path = resolve_dataset_path('ulsan', 'data/raw/ulsan_temporal_grid.npy')
    dataset = UnifiedDemandDataset(
        data_path,
        'train',
        weather_csv_path=REPO_ROOT / 'data/raw/ulsan_meteorological_data.csv',
        **DATA_KWARGS,
    )
    train_weather = dataset.weather[dataset.time_step : dataset.train_end]
    weather_mean = train_weather.mean(axis=0).tolist()
    weather_std = train_weather.std(axis=0).clip(min=1e-6).tolist()

    # Use a time with retrieval candidates; the first target has none.
    subset = torch.utils.data.Subset(dataset, range(start, start + batch_size))
    batch = next(iter(DataLoader(subset, batch_size=batch_size, shuffle=False)))
    batch = {key: value.to(device) for key, value in batch.items()}
    return dataset, batch, weather_mean, weather_std


def make_models(legacy_module, dataset, weather_mean, weather_std, flags, seed=1234):
    data_path = str(dataset.data_path)
    torch.manual_seed(seed)
    old = legacy_module.UnifiedDemandModel(
        height=dataset.height,
        width=dataset.width,
        time_step=dataset.time_step,
        retrieval_grid_path=data_path,
        retrieval_train_end=dataset.train_end,
        weather_mean=weather_mean,
        weather_std=weather_std,
        loss_type='mae',
        loss_gamma=1.0,
        loss_eps=0.5,
        **MODEL_KWARGS,
        **flags,
    )
    torch.manual_seed(seed)
    new = MergedDemandModel(
        MergedDemandConfig(
            height=dataset.height,
            width=dataset.width,
            time_step=dataset.time_step,
            retrieval_grid_path=data_path,
            retrieval_train_end=dataset.train_end,
            weather_mean=weather_mean,
            weather_std=weather_std,
            loss_type='mae',
            loss_gamma=1.0,
            loss_eps=0.5,
            **MODEL_KWARGS,
            **flags,
        )
    )
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    # Persistent buffer required by meta-device ``from_pretrained``; legacy state_dict lacks it.
    assert list(unexpected) == [], f'unexpected keys: {unexpected}'
    assert list(missing) == ['local_history.neighbor_valid'], f'missing keys: {missing}'
    return old, new


def check_parity(legacy_module, device: torch.device, batch_size: int, start: int) -> None:
    dataset, batch, weather_mean, weather_std = build_batch(device, batch_size, start)
    old_batch = {key: value for key, value in batch.items() if key != 'labels'}
    old_batch['target'] = batch['labels']

    print(f'[parity] ulsan H={dataset.height} W={dataset.width} '
          f'batch={batch_size} sample_idx={batch["sample_idx"].tolist()} device={device}')
    for name, flags in ABLATION_CASES.items():
        old, new = make_models(legacy_module, dataset, weather_mean, weather_std, flags)
        old = old.to(device).eval()
        new = new.to(device).eval()
        with torch.no_grad():
            old_out = old(**old_batch)
            new_out = new.forward_debug(**batch)

        pred_diff = (old_out['prediction'] - new_out['logits']).abs().max().item()
        loss_diff = (old_out['loss'] - new_out['loss']).abs().max().item()
        ok_pred = torch.allclose(old_out['prediction'], new_out['logits'], atol=1e-6)
        ok_loss = torch.allclose(old_out['loss'], new_out['loss'], atol=1e-6)
        print(
            f'  {name:<12} max|Δprediction|={pred_diff:.3e} max|Δloss|={loss_diff:.3e} '
            f'-> {"PASS" if ok_pred and ok_loss else "FAIL"}'
        )
        assert ok_pred, f'{name}: prediction mismatch ({pred_diff:.3e})'
        assert ok_loss, f'{name}: loss mismatch ({loss_diff:.3e})'

        with torch.no_grad():
            slim = new(**batch)
        assert torch.equal(slim['logits'], new_out['logits'])
        assert torch.equal(slim['loss'], new_out['loss'])
        assert set(slim) == {'loss', 'logits'}, f'slim forward returned {sorted(slim)}'


def check_roundtrip(device: torch.device) -> None:
    dataset, batch, weather_mean, weather_std = build_batch(device, 1, 3000)
    config = MergedDemandConfig(
        height=dataset.height,
        width=dataset.width,
        time_step=dataset.time_step,
        retrieval_grid_path=str(dataset.data_path),
        retrieval_train_end=dataset.train_end,
        weather_mean=weather_mean,
        weather_std=weather_std,
        loss_type='mae',
        **MODEL_KWARGS,
    )
    torch.manual_seed(7)
    model = MergedDemandModel(config).eval()
    with tempfile.TemporaryDirectory(prefix='merged_roundtrip_') as temp_dir:
        model.save_pretrained(temp_dir)
        restored = MergedDemandModel.from_pretrained(temp_dir).eval()

    before = dict(model.state_dict())
    after = dict(restored.state_dict())
    assert set(before) == set(after), f'state_dict keys differ: {set(before) ^ set(after)}'
    worst = 0.0
    for key, value in before.items():
        other = after[key].to(value.device)
        assert value.dtype == other.dtype, f'{key}: dtype {value.dtype} != {other.dtype}'
        assert torch.equal(value, other), f'{key}: value changed on round trip'
        if value.is_floating_point():
            worst = max(worst, (value - other).abs().max().item())
    for key in ('weather_mean', 'weather_std', 'local_history.neighbor_valid'):
        assert key in after, f'{key} missing from restored checkpoint'
    for key in (
        'height',
        'width',
        'time_step',
        'loss_type',
        'use_retrieval',
        'use_softplus',
        'weather_injection',
        'retrieval_grid_path',
        'retrieval_train_end',
    ):
        assert getattr(model.config, key) == getattr(restored.config, key), f'config.{key} changed'
    with torch.no_grad():
        a = model.to(device)(**batch)
        b = restored.to(device)(**batch)
    output_diff = (a['logits'] - b['logits']).abs().max().item()
    print(
        f'[roundtrip] {len(before)} tensors identical (max|Δ|={worst:.3e}), '
        f'forward max|Δlogits|={output_diff:.3e} -> PASS'
    )
    assert torch.equal(a['logits'], b['logits'])


def _manual_train(model, batches, *, is_new: bool, lr: float, weight_decay: float, epochs: int,
                  seed: int) -> list[float]:
    """Run the training step used by the parity smoke check."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    torch.manual_seed(seed)
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            if is_new:
                output = model(**batch)
            else:
                legacy_batch = {k: v for k, v in batch.items() if k != 'labels'}
                legacy_batch['target'] = batch['labels']
                output = model(**legacy_batch)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item()
            count += 1
        losses.append(total / count)
    return losses


def check_smoke(legacy_module, device: torch.device, epochs: int, steps: int, start: int) -> None:
    """Compare short loss trajectories with identical seeds and sequential batches.

    Fixed batch order ensures both implementations receive the same inputs.
    """

    dataset, _, weather_mean, weather_std = build_batch(device, 1, start)
    batch_size = 8
    batches = []
    for index in range(steps):
        subset = torch.utils.data.Subset(
            dataset, range(start + index * batch_size, start + (index + 1) * batch_size)
        )
        batch = next(iter(DataLoader(subset, batch_size=batch_size, shuffle=False)))
        batches.append({key: value.to(device) for key, value in batch.items()})

    old, new = make_models(legacy_module, dataset, weather_mean, weather_std, {})
    old_losses = _manual_train(
        old.to(device), batches, is_new=False, lr=1e-3, weight_decay=1e-4, epochs=epochs, seed=99
    )
    new_losses = _manual_train(
        new.to(device), batches, is_new=True, lr=1e-3, weight_decay=1e-4, epochs=epochs, seed=99
    )
    print(f'[smoke] {epochs} epochs x {steps} batches (batch_size={batch_size}), seed=99')
    worst = 0.0
    for index, (old_loss, new_loss) in enumerate(zip(old_losses, new_losses), start=1):
        delta = abs(old_loss - new_loss)
        worst = max(worst, delta)
        print(f'  epoch {index}: old={old_loss:.8f} new={new_loss:.8f} |Δ|={delta:.3e}')
    print(f'[smoke] max|Δloss|={worst:.3e} -> {"PASS" if worst < 1e-5 else "CHECK"}')
    assert worst < 1e-5, f'smoke loss trajectories diverged: max|Δ|={worst:.3e}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--start', type=int, default=3000, help='train split 내 시작 인덱스')
    parser.add_argument('--smoke-epochs', type=int, default=3)
    parser.add_argument('--smoke-steps', type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)

    with tempfile.TemporaryDirectory(prefix='merged_legacy_') as temp_dir:
        restore_legacy_package(Path(temp_dir))
        sys.path.insert(0, temp_dir)
        legacy_module = importlib.import_module('merged_legacy.model')
        check_parity(legacy_module, device, args.batch_size, args.start)
        check_roundtrip(device)
        check_smoke(legacy_module, device, args.smoke_epochs, args.smoke_steps, args.start)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
