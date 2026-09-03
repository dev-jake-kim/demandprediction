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
daily/weekly fusion and the selectable objective are explicit.

## What is implemented

- raw temporal-grid history with local `5 x 5` crops and EDGE tokens;
- another-style Transformer-per-history-step followed by an LSTM;
- raw daily and weekly lag branches with chronological valid-lag compaction;
- branch attention over `[daily, weekly, h_neural]`, with `h_neural` as query and
  also as a key/value candidate;
- raw causal cosine retrieval with `tau < target_time`, computed in CPU chunks;
- another-style node-wise gate between neural and retrieval predictions;
- one forward pass, one optimizer, and one raw-scale objective
  (`combined` or `mae` — see below);
- weather and calendar features concatenated onto **all three** LSTM inputs;
- no main recent branch.

### Objective (`training.loss_type`)

The model optimizes exactly one loss, chosen by `training.loss_type` in
`config.yaml` or `--loss-type` on the command line:

| Value | Loss | Why |
| --- | --- | --- |
| `combined` (default) | `CombinedLoss(gamma=1.0, eps=0.5)` = `(y-ŷ)² + gamma·((y-ŷ)/(y+eps))²` | The objective every other branch in this repository trains with, so merged_model numbers are directly comparable to the `docs/PROJECT_SUMMARY.md` table |
| `mae` | raw-scale `L1` | The objective this model was originally specified and first trained with (`MODEL_PLAN.md`) |

`merged_model/losses.py` is a port of the repository's shared
`models/losses.py`; the two were checked to produce bit-identical values.
Both losses are built with `reduction='none'` so the epoch aggregate is a true
element mean rather than a mean of batch means (batches are uneven because
`drop_last=False`).

Early stopping and best-checkpoint selection use the **validation value of
whichever loss is active**, matching the shared harness's
`metric_for_best_model: loss`. Under `loss_type=mae` that value is identical to
validation MAE, so runs recorded before this option existed reproduce exactly.

Reported metrics (MAE / RMSE / MAPE(+1) / MAPE(0-excluded)) do not depend on
`loss_type` — only the thing being minimized does.

### Weather and calendar features

Each of the three LSTMs receives a shared 15-dimensional context vector
concatenated to its per-step input:

| Feature | Dimensions | How it is encoded |
| --- | --- | --- |
| weather (temperature, precipitation, snow) | 3 | normalized `(x - mean) / std`, **no embedding layer** |
| day of week | 7 | `nn.Embedding(7, 7)` |
| hour of day | 5 | `nn.Embedding(24, 5)` |

This changes the LSTM input widths to `64 + 15 = 79` for the history branch and
`1 + 15 = 16` for the daily and weekly branches. The embedding tables are owned
by `UnifiedDemandModel` and shared across branches.

Normalization statistics come from the **training split only**
(`weather[time_step:train_end]`), computed in `train.py` and stored as model
buffers, so validation and test windows never leak into them. `std` is clamped
at `1e-6` because some features are constant within the training split (Porto's
snow column is always zero).

The recent branch reads weather over `[t-k+1, t+1)` — shifted by one step so it
includes the target hour's weather. This mirrors the `ir-weather` branch's
deliberate "short-range weather forecasts are already known" assumption; it is
not the same as knowing future demand. Calendar features need no forecast and
use the same `[t-k, t)` window as demand.

Weekly lags are multiples of 168 hours, so every weekly lag shares the target's
weekday and hour — the calendar input is constant within a weekly sequence
(weather still varies). This is structural, not a bug.

The weather CSV must be cp949-encoded and contain `기온(°C)`, `강수량(mm)`, and
`적설(cm)`; its row count must equal the temporal grid length. Paths are set per
dataset under `datasets.<name>.weather_path` in `config.yaml`.

## Run

From `/home/jinu/lab` (GPU 0 only):

```bash
source /home/jinu/miniconda3/etc/profile.d/conda.sh
conda activate Torch
CUDA_VISIBLE_DEVICES=0 python -m comparison_models.merged_model.train \
  --dataset ulsan --device cuda:0
```

The configured defaults are `epochs=2000`, `patience=20`, `batch_size=8`,
`retrieval_scope=observed_past`, and `loss_type=combined`. Override them
explicitly when reproducing an experiment.

For a multi-seed sweep (the repository's shared five seeds), use the driver
script — it skips runs whose result JSON already exists:

```bash
CUDA_VISIBLE_DEVICES=1 ./merged_model/run_seeds.sh ulsan combined
```

`train.py` deliberately accepts only `--device cuda:0`; select a physical GPU
with `CUDA_VISIBLE_DEVICES`.

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
