# Unified another_model + main model

This directory contains the single end-to-end model specified in
[`MODEL_PLAN.md`](./MODEL_PLAN.md).

For code navigation, start with [`ARCHITECTURE.md`](./ARCHITECTURE.md). The
high-level data flow is in `model.py`; branch implementations are isolated in
the `modules/` directory; `data.py` and `train.py` do not contain model
architecture. `components.py` remains only as a backwards-compatible import
facade.

The implementation uses only free local dependencies: PyTorch, NumPy, and
PyYAML. It does not call a paid API or hosted inference service. The neural
backbone follows the architecture described by the public `ir` branch of
[`dev-jake-kim/demandprediction`](https://github.com/dev-jake-kim/demandprediction/tree/ir),
but the code here is a standalone PyTorch implementation so that the new
daily/weekly fusion and MAE objective are explicit.

## What is implemented

- raw temporal-grid history with local `5 x 5` crops and EDGE tokens;
- another-style Transformer-per-history-step followed by an LSTM;
- raw daily and weekly lag branches with chronological valid-lag compaction;
- branch attention over `[daily, weekly, h_neural]`, with `h_neural` as query and
  also as a key/value candidate;
- raw causal cosine retrieval with `tau < target_time`, computed in CPU chunks;
- another-style node-wise gate between neural and retrieval predictions;
- one forward pass, one optimizer, and raw-scale MAE;
- no main recent branch and no weather input.

## Run

From `/home/jinu/lab` (GPU 0 only):

```bash
source /home/jinu/miniconda3/etc/profile.d/conda.sh
conda activate Torch
CUDA_VISIBLE_DEVICES=0 python -m comparison_models.merged_model.train \
  --dataset ulsan --device cuda:0
```

The configured defaults are `epochs=2000`, `patience=20`, `batch_size=8`, and
`retrieval_scope=observed_past`. Override them explicitly when reproducing an
experiment.

For a short smoke test:

```bash
source /home/jinu/miniconda3/etc/profile.d/conda.sh
conda activate Torch
CUDA_VISIBLE_DEVICES=0 python -m comparison_models.merged_model.train \
  --dataset ulsan --device cpu --epochs 1 --max-batches 1 \
  --output comparison_models/merged_model/runs/ulsan_smoke.json
```

Use `--retrieval-scope train_prefix` when validation/test retrieval must not
use observed labels after the training boundary. The default
`observed_past` mode uses only absolute times `[time_step, target_time)`.

The loader first looks in `comparison_models/data/...` for the temporal-grid
files. Until those files are copied there, it uses the already-existing
temporal-grid files under `research_700x700/comparison_models/data/...`; it
never falls back to legacy `demand.npy`.
