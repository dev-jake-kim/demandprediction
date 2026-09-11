# Merged model code map

`MergedDemandModel` is one model and one training graph. The code is split by
responsibility so that changes to one branch do not require edits to the
dataset or training loop.

이 문서는 원래 `merged_model/ARCHITECTURE.md`였다. 그 자기 완결형 패키지를 저장소 공용
관례(HuggingFace `PreTrainedModel` + Hydra + `Trainer`)로 포팅하면서 경로만 갱신했다 —
계산 그래프는 그대로다(`UnifiedDemandModel` → `MergedDemandModel`).

## Start here

1. Read `docs/MODEL_PLAN.md` for the design contract and Mermaid diagrams.
2. Read `models/merged/modeling.py` for the end-to-end tensor flow.
3. Read the relevant file under `models/merged/modules/` for a branch-level change.
4. Run `validate_merged.py` before starting an expensive training run.

| Path | Responsibility | Typical changes |
| --- | --- | --- |
| `dataset_frame/unified_demand_dataset.py` | temporal-grid loading, split, lag tables, invalid masks, weather/calendar tables | time split, lag count, `lag_radius`, weather source |
| `models/merged/modeling.py` | branch wiring, weather/calendar context assembly, raw-scale loss contract | tensor flow or output keys |
| `models/merged/config.py` | `MergedDemandConfig` — 생성자 인자 전부와 9개 ablation 스위치 | 새 하이퍼파라미터 |
| `models/merged/losses.py` | the two selectable objectives (`combined`, `mae`); `combined`은 공용 `models/losses.py`를 재사용 | the objective itself |
| `models/merged/metrics.py` | RMSE/MAE/MAPE(+1)/MAPE(0제외) | 보고 지표 |
| `models/merged/modules/embeddings.py` | scalar/Fourier token embedding | scalar feature encoding |
| `models/merged/modules/history.py` | local crop, EDGE/CLS tokens, Transformer, history LSTM (context concatenated) | another neural branch |
| `models/merged/modules/periodic.py` | daily/weekly LSTM, invalid-lag compaction, context concatenation | periodic encoders |
| `models/merged/modules/attention.py` | node-wise daily/weekly/neural attention | branch selection/fusion |
| `models/merged/modules/retrieval.py` | raw causal cosine retrieval and cache | retrieval scope/top-k/chunking |
| `models/merged/modules/fusion.py` | neural/retrieval output gate | final prediction fusion |
| `train.py` | Hydra + `Trainer` 학습 진입점, early stopping, 결과 JSON | training schedule or CLI |
| `test.py` | 체크포인트 단독 평가 | 평가 지표/스플릿 |
| `validate_merged.py` | structural, gradient, mask, and causal checks | regression checks |
| `configs/config_ulsan.yaml` / `configs/config_porto.yaml` | 학습 하이퍼파라미터(원본 `training:` 블록과 1:1) — 도시별 루트 config | 학습 스케줄, `--config-name` 선택 |
| `configs/model/merged_ulsan.yaml` / `configs/model/merged_porto.yaml` | 모델 하이퍼파라미터 + ablation 스위치 기본값 — 도시별 최적값(`docs/MERGED_TUNING_RESULTS.md`) | dimensions, lags, retrieval policy, `weekday_dim`, `hour_dim` |
| `run_ablation.sh` / `run_seeds.sh` | 큐/다중 시드 러너 (ablation 이름 → Hydra 오버라이드) | 실행 조합 |

## Directory layout

```text
dataset_frame/unified_demand_dataset.py   # temporal-grid Dataset
models/merged/
├── config.py               # MergedDemandConfig(PretrainedConfig)
├── modeling.py             # MergedDemandModel(PreTrainedModel) — one graph
├── losses.py               # combined | mae objectives
├── metrics.py              # RMSE / MAE / MAPE(+1) / MAPE(0제외)
└── modules/                # independently maintainable model blocks
    ├── embeddings.py
    ├── history.py
    ├── periodic.py
    ├── attention.py
    ├── retrieval.py
    ├── fusion.py
    └── __init__.py
configs/config_ulsan.yaml        # Hydra 루트 설정(학습 하이퍼파라미터) — ulsan, 기본값
configs/config_porto.yaml        # Hydra 루트 설정 — porto, --config-name config_porto
configs/model/merged_ulsan.yaml  # 모델 하이퍼파라미터 + ablation 스위치 — ulsan 최적값
configs/model/merged_porto.yaml  # 모델 하이퍼파라미터 + ablation 스위치 — porto 최적값
train.py                   # Hydra + Trainer 학습 진입점
test.py                    # 체크포인트 단독 평가
validate_merged.py         # fast pre-training checks
run_ablation.sh            # ablation 큐 러너
run_seeds.sh                # 다중 시드 러너
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
h_attn + ir_out → NeuralRetrievalGate → prediction → loss(labels)   # combined | mae
```

The daily/weekly masks are applied twice: valid lags are compacted before the
LSTM, and invalid branch tokens are excluded from attention. `h_neural` is
always a valid candidate, so all-invalid periodic samples remain well-defined.

Weather and calendar are node-independent, so the context is broadcast across
nodes before concatenation. Weather is normalized with training-split statistics
held as model buffers; it has no embedding layer. Invalid periodic lags carry
zeroed context and are dropped by compaction, so the widened feature dimension
must flow through the `gather` in `periodic.py` — `validate_merged.py` pins this with a
reference-implementation comparison.

The retrieval module uses raw values and only candidate times
`[time_step, target_time)`. Its CPU cache is not part of the checkpoint; it is
reconstructed from the temporal grid when a model is created.

## Maintenance rules

- Keep `MergedDemandModel.forward()` as the single public model contract. It
  returns `{'loss', 'logits'}` — the slim dict the HF `Trainer` eval loop needs.
  `forward_debug()` returns the full tensor dict (`neural_pred`, `ir_out`,
  `lambda_weight`, `attention_weights`, ...) for validation scripts only.
- Add branch-specific code under `models/merged/modules/`; do not put new
  architecture into `train.py` or `dataset_frame/`.
- Preserve output keys consumed by training and validation unless the contract
  and documentation are updated together.
- Run `python validate_merged.py --device cpu` after changes to data flow,
  masks, shapes, or retrieval boundaries.
