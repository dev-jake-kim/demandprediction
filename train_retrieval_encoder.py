"""검색 encoder 단독 학습 (docs/RETRIEVAL_ENCODER.md).

사용 예::

    python train_retrieval_encoder.py "description='실험 설명'" [dataset=porto] [seed=...]

출력(``hydra.run.dir``): ``encoder.pt``(가중치+config), ``metrics.json``(epoch 기록, val·test 평가와
raw cosine 기준), 학습 로그.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open

from dataset_frame import UnifiedDemandDataset, resolve_dataset_path
from models.retrieval_encoder import (
    RetrievalEncoder,
    balanced_sample_weights,
    build_pairs,
    cosine_similarity,
    demand_bucket_bounds,
    encode_all,
    gather_windows,
    gaussian_kl,
    load_retrieval_encoder,
    raw_keys,
    retrieval_metrics,
    save_retrieval_encoder,
    select_value_cap,
    weighted_contrastive_loss,
    window_table,
)
from train import build_dataset_kwargs, git_commit

logger = logging.getLogger(__name__)


def load_node_embedding(checkpoint_dir: str) -> torch.Tensor:
    path = Path(hydra.utils.to_absolute_path(checkpoint_dir)) / 'model.safetensors'
    with safe_open(str(path), 'pt') as handle:
        return handle.get_tensor('local_history.node_embedding').float()


@hydra.main(config_path='configs', config_name='retrieval_encoder', version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f'[Description] {cfg.description}')
    logger.info(f'[Commit] {git_commit()}')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # 본 모델과 같은 분할 경계.
    data_path = resolve_dataset_path(cfg.dataset.city, cfg.dataset.npy_path)
    train_ds = UnifiedDemandDataset(data_path, 'train', **build_dataset_kwargs(cfg))
    time_step, train_end, val_end = train_ds.time_step, train_ds.train_end, train_ds.val_end
    grid = torch.from_numpy(np.asarray(train_ds.grid, dtype=np.float32)).to(device)  # [T, H, W]
    total = grid.shape[0]
    grid_flat = grid.reshape(total, -1)
    crops = window_table(grid, int(cfg.local_radius))  # [T, N, P]

    value_cap = select_value_cap(train_ds.grid[:train_end], float(cfg.train.value_cap_top_fraction))
    bounds_list = demand_bucket_bounds(value_cap)
    bounds = torch.tensor(bounds_list, dtype=torch.float32, device=device)
    train_pairs = build_pairs(crops, grid_flat, time_step, train_end, time_step, bounds)
    val_pairs = build_pairs(crops, grid_flat, train_end, val_end, time_step, bounds)
    test_pairs = build_pairs(crops, grid_flat, val_end, total, time_step, bounds)
    counts = torch.bincount(train_pairs.buckets, minlength=len(bounds_list)).tolist()
    logger.info(
        f'[{cfg.dataset.city}] T={total}, N={grid_flat.shape[1]}, train_end={train_end}, val_end={val_end} | '
        f'value_cap={value_cap}, buckets={bounds_list}, train bucket counts={counts} | '
        f'pairs (입력 0 제외) train {len(train_pairs.times):,} / val {len(val_pairs.times):,} / '
        f'test {len(test_pairs.times):,}'
    )

    node_embedding = load_node_embedding(cfg.node_embedding_checkpoint)
    if node_embedding.shape[0] != grid_flat.shape[1]:
        raise ValueError(f'node_embedding 노드 수 {node_embedding.shape[0]} != 격자 노드 수 {grid_flat.shape[1]}')
    encoder = RetrievalEncoder(
        node_embedding,
        time_step=time_step,
        num_neighbors=crops.shape[-1],
        hidden_dim=int(cfg.encoder.hidden_dim),
        latent_dim=int(cfg.encoder.latent_dim),
        input_noise_std=float(cfg.encoder.input_noise_std),
        temperature=float(cfg.encoder.temperature),
    ).to(device)
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f'encoder: node_dim={node_embedding.shape[1]} (frozen), 학습 파라미터 {trainable:,}')
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=float(cfg.train.learning_rate), weight_decay=float(cfg.train.weight_decay)
    )

    sample_weights = balanced_sample_weights(
        train_pairs.buckets, len(bounds_list), float(cfg.train.balance_power)
    )
    batch_size = int(cfg.train.batch_size)
    steps_per_epoch = max(1, len(train_pairs.times) // batch_size)
    beta_max, warmup = float(cfg.train.kl_beta), float(cfg.train.kl_warmup_epochs)
    top_k = int(cfg.eval.top_k)

    def evaluate(pairs) -> dict[str, float]:
        keys = encode_all(encoder, crops, time_step)
        return retrieval_metrics(keys, encoder.similarity, pairs, grid_flat, bounds, time_step, top_k)

    history: list[dict] = []
    best_mae, best_epoch, stale = math.inf, 0, 0
    checkpoint = output_dir / 'encoder.pt'
    for epoch in range(1, int(cfg.train.max_epochs) + 1):
        encoder.train()
        sums = {'loss': 0.0, 'con': 0.0, 'kl': 0.0, 'sigma': 0.0, 'anchors': 0.0}
        for step in range(steps_per_epoch):
            beta = beta_max * min(1.0, (epoch - 1 + step / steps_per_epoch) / warmup) if warmup > 0 else beta_max
            index = torch.multinomial(sample_weights, batch_size, replacement=True)
            times, nodes = train_pairs.times[index], train_pairs.nodes[index]
            mu, logvar = encoder(gather_windows(crops, times, nodes, time_step), nodes)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            con, anchors = weighted_contrastive_loss(
                z, train_pairs.labels[index], train_pairs.buckets[index], encoder.temperature
            )
            kl = gaussian_kl(mu, logvar)
            loss = con + beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums['loss'] += float(loss)
            sums['con'] += float(con)
            sums['kl'] += float(kl)
            sums['sigma'] += float(torch.exp(0.5 * logvar).mean())
            sums['anchors'] += anchors
        record = {'epoch': epoch, 'beta': beta, **{k: v / steps_per_epoch for k, v in sums.items()}}

        encoder.eval()
        with torch.no_grad():
            val_mu = encode_all(encoder, crops, time_step)[val_pairs.times, val_pairs.nodes]
        record['active_units'] = int((val_mu.var(dim=0) > 0.01).sum())
        record['val'] = evaluate(val_pairs)
        history.append(record)
        logger.info(
            f"epoch {epoch}: loss {record['loss']:.4f} (con {record['con']:.4f}, kl {record['kl']:.3f}, "
            f"β {beta:.3f}), σ {record['sigma']:.3f}, anchors {record['anchors']:.1f}/{batch_size}, "
            f"active {record['active_units']}/{encoder.latent_dim} | val MAE {record['val']['mae']:.4f}, "
            f"RMSE {record['val']['rmse']:.4f}, same-bucket {record['val']['same_bucket_ratio']:.3f}"
        )
        if record['val']['mae'] < best_mae:
            best_mae, best_epoch, stale = record['val']['mae'], epoch, 0
            save_retrieval_encoder(encoder, checkpoint, value_cap=value_cap, bucket_bounds=bounds_list,
                                   city=cfg.dataset.city, epoch=epoch)
        else:
            stale += 1
            if stale >= int(cfg.train.patience):
                logger.info(f'early stopping: val MAE가 {stale} epoch 동안 개선되지 않음 (best epoch {best_epoch})')
                break

    best = load_retrieval_encoder(checkpoint, device)
    best_keys = encode_all(best, crops, time_step)
    cosine_keys = raw_keys(crops, time_step)
    result = {
        'description': cfg.description,
        'commit': git_commit(),
        'city': cfg.dataset.city,
        'seed': int(cfg.seed),
        'value_cap': value_cap,
        'bucket_bounds': bounds_list,
        'best_epoch': best_epoch,
        'encoder_path': str(checkpoint.resolve()),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'history': history,
    }
    for split, pairs in (('val', val_pairs), ('test', test_pairs)):
        result[split] = {
            'encoder': retrieval_metrics(best_keys, best.similarity, pairs, grid_flat, bounds, time_step, top_k),
            'raw_cosine': retrieval_metrics(cosine_keys, cosine_similarity, pairs, grid_flat, bounds, time_step, top_k),
        }
        for name in ('encoder', 'raw_cosine'):
            m = result[split][name]
            logger.info(f"[{split}] {name}: MAE {m['mae']:.4f}, RMSE {m['rmse']:.4f}, same-bucket {m['same_bucket_ratio']:.3f}")
    (output_dir / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'encoder={checkpoint.resolve()}')


if __name__ == '__main__':
    main()
