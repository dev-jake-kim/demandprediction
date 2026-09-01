# Unified model code map

`UnifiedDemandModel` is one model and one training graph. The code is split by
responsibility so that changes to one branch do not require edits to the
dataset or training loop.

## Start here

1. Read `MODEL_PLAN.md` for the design contract and Mermaid diagrams.
2. Read `model.py` for the end-to-end tensor flow.
3. Read the relevant file under `modules/` for a branch-level change.
4. Run `validate.py` before starting an expensive training run.

| Path | Responsibility | Typical changes |
| --- | --- | --- |
| `data.py` | temporal-grid loading, split, lag tables, invalid masks, weather/calendar tables | time split, lag count, `lag_radius`, weather source |
| `model.py` | branch wiring, weather/calendar context assembly, raw-scale MAE contract | tensor flow or output keys |
| `modules/embeddings.py` | scalar/Fourier token embedding | scalar feature encoding |
| `modules/history.py` | local crop, EDGE/CLS tokens, Transformer, history LSTM (context concatenated) | another neural branch |
| `modules/periodic.py` | daily/weekly LSTM, invalid-lag compaction, context concatenation | periodic encoders |
| `modules/attention.py` | node-wise daily/weekly/neural attention | branch selection/fusion |
| `modules/retrieval.py` | raw causal cosine retrieval and cache | retrieval scope/top-k/chunking |
| `modules/fusion.py` | neural/retrieval output gate | final prediction fusion |
| `components.py` | compatibility re-exports only | keep imports stable; no new logic |
| `train.py` | optimizer, early stopping, metrics, result JSON | training schedule or CLI |
| `validate.py` | structural, gradient, mask, and causal checks | regression checks |
| `config.yaml` | experiment defaults | dimensions, lags, retrieval policy, `weather_path`, `weekday_dim`, `hour_dim` |
| `README.md` | run instructions and behavior summary | user-facing usage |

## Directory layout

```text
merged_model/
├── data.py                 # temporal-grid Dataset
├── model.py                # one UnifiedDemandModel graph
├── train.py                # one MAE training entry point
├── validate.py             # fast pre-training checks
├── modules/                # independently maintainable model blocks
│   ├── embeddings.py
│   ├── history.py
│   ├── periodic.py
│   ├── attention.py
│   ├── retrieval.py
│   ├── fusion.py
│   └── __init__.py
├── components.py           # backwards-compatible exports
├── config.yaml
├── ARCHITECTURE.md         # code map and maintenance rules
├── MODEL_PLAN.md           # design contract and Mermaid diagrams
└── README.md               # run instructions
```

## Forward data flow

```text
context (per branch, 15 dims) = normalized weather (3) ⊕ weekday_emb (7) ⊕ hour_emb (5)

raw history
  ├─ LocalHistoryEncoder: crop → log1p/Fourier → Transformer → cls ⊕ context → LSTM(79) → h_neural
  ├─ PeriodicLSTMEncoder: daily raw lags → log1p ⊕ context → LSTM(16) → h_daily
  └─ PeriodicLSTMEncoder: weekly raw lags → log1p ⊕ context → LSTM(16) → h_weekly

h_neural → query
[h_daily, h_weekly, h_neural] → key/value candidates → branch attention → h_attn

raw local crop + absolute sample_idx → CausalRetrieval → ir_out
h_attn + ir_out → NeuralRetrievalGate → prediction → MAE(target)
```

The daily/weekly masks are applied twice: valid lags are compacted before the
LSTM, and invalid branch tokens are excluded from attention. `h_neural` is
always a valid candidate, so all-invalid periodic samples remain well-defined.

Weather and calendar are node-independent, so the context is broadcast across
nodes before concatenation. Weather is normalized with training-split statistics
held as model buffers; it has no embedding layer. Invalid periodic lags carry
zeroed context and are dropped by compaction, so the widened feature dimension
must flow through the `gather` in `periodic.py` — `validate.py` pins this with a
reference-implementation comparison.

The retrieval module uses raw values and only candidate times
`[time_step, target_time)`. Its CPU cache is not part of the checkpoint; it is
reconstructed from the temporal grid when a model is created.

## Maintenance rules

- Keep `UnifiedDemandModel.forward()` as the single public model contract.
- Add branch-specific code under `modules/`; do not put new architecture into
  `train.py` or `data.py`.
- Preserve output keys consumed by training and validation unless the contract
  and documentation are updated together.
- Run `python -m comparison_models.merged_model.validate --device cpu` after
  changes to data flow, masks, shapes, or retrieval boundaries.
