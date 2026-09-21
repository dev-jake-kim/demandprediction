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
| `models/merged/losses.py` | selectable objectives (`combined`, `mae`, `rmse_mape`); `combined`/`rmse_mape`은 공용 `models/losses.py`를 재사용 | the objective itself |
| `models/merged/metrics.py` | RMSE/MAE/MAPE(+1)/MAPE(0제외) | 보고 지표 |
| `models/merged/modules/embeddings.py` | scalar/Fourier token embedding | scalar feature encoding |
| `models/merged/modules/history.py` | local crop, EDGE/CLS tokens, Transformer, history LSTM (context concatenated), 노드별 ΔW | another neural branch |
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
h_attn + ir_out → NeuralRetrievalGate → prediction → loss(labels)   # combined | mae | rmse_mape
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

## 게이티드 노드별 LSTM weight offset + shared-weight FP8 (`node_adaptive`)

lora 브랜치의 node_adaptive(W + ΔW)를 이 모델의 **`LocalViewEncoder`의 temporal LSTM
하나에만** 옮긴 뒤, 공유 가중치와 노드별 보정의 비율을 학습하는 구조다.
`model.node_adaptive=true`일 때만
켜지며, Ulsan 모델 config의 기본값은 `node_adaptive=true`와
`shared_weight_fp8=true`이고, Porto는 근거가 없어 `node_adaptive=false`다.

적응 노드 `a`의 네 shared LSTM tensor(`weight_ih_l0`, `weight_hh_l0`,
`bias_ih_l0`, `bias_hh_l0`)는 다음처럼 합성한다:

```text
W_eff,a = s * W_shared + (1 - s) * ΔW_a

s: 학습 가능한 fp32 scalar, init 1.0
ΔW_a: 노드별 fp32 tensor, init 0
```

`s=1`, `ΔW=0`이면 학습 시작점에서 `W_eff`가 shared LSTM과 정확히 같은
function-level copy다. `s`와 ΔW는 single-stage 학습에서 일반 파라미터로 함께 갱신한다.

`A`는 전체 노드가 아니라 **train 구간 평균 수요가
`node_adaptive_min_demand`(기본 0.8)를 넘는 노드** 수다. 채택한 Ulsan 설정은
168개 중 44개 노드(44/168)만 적응 경로를 사용한다. 선택 기준은 시간 리크를 막기 위해
train 구간에서만 계산하며(`train.py: select_node_adaptive_indices`), 결과 노드 목록은
`config.node_adaptive_indices`에 저장돼 `from_pretrained`가 같은 마스크를 복원한다.

설계상 지켜야 할 것들:

- **선택되지 않은 노드와 적응 노드의 계산 경로가 다르다.** 비적응 노드는 외부 weight를
  넘기는 `torch._VF.lstm`으로 shared 가중치를 실행한다. 적응 노드는 `i,f,g,o` 게이트
  순서의 수동 cell loop에서 `W_eff`를 다시 계산한 뒤 `index_copy`로 결과를 합친다.
- **ΔW는 0, s는 1로 시작한다.** 따라서 게이트를 켜도 학습 첫 순간의 함수는 shared LSTM과
  동일하며, 이후 학습된 s가 shared weight와 ΔW의 비율을 정한다.
- **노드 목록을 버퍼로 저장하면 안 된다.** transformers 5.0의 `from_pretrained`는 meta
  device에서 모델을 만든 뒤 체크포인트에 있는 키만 실체화한다. config의 파이썬 리스트에서
  forward 시점에 텐서를 만들어 device별로 캐시한다.
- **ΔW(0)와 s(1.0)는 생성자에서 직접 초기화한다.** `_init_weights`는 의도적으로 아무것도
  하지 않는다 — `PreTrainedModel._init_weights`를 부르면 나머지 블록이 전제하는 PyTorch
  기본 초기화가 std=0.02 정규분포로 조용히 덮인다.
- daily/weekly 주기 브랜치는 대상이 아니다 — `pack_padded_sequence`가 노드 축을 흐트러뜨려
  노드별 가중치를 붙이려면 packing 자체를 걷어내야 한다.
- `node_adaptive`와 `zero_node_max_demand`/`zero_node_indices`는 함께 쓸 수 없다. 두
  옵션의 노드 축이 서로 달라 `modeling.py`가 `ValueError`를 발생시킨다.

### shared-weight FP8 fake quantization

`shared_weight_fp8=true`이면 네 shared LSTM tensor를 absmax per-tensor 방식으로
`torch.float8_e4m3fn`에 fake quantize하고 straight-through gradient를 쓴다. ΔW와 s는
fp32를 유지하며 master weight와 모든 수학 연산도 fp32다. 이는 **정확도 QAT 동작**이지
FP8 메모리 절약이나 속도 향상이 아니다. 외부 weight를 전달하는 경로 때문에 cuDNN에서
non-contiguous RNN weight warning이 발생할 수 있다.

### 단일 stage 학습과 결과 JSON

`node_adaptive=true`여도 `train.py`는 단일 stage만 학습하며 `combined` loss를 사용한다.
예전처럼 stage를 나누거나 ΔW를 별도 stage에서 해제하지 않고, s와 ΔW를 일반 파라미터로
동시에 학습한다. 채택한 Ulsan 결과와 재현 절차는
[`MERGED_GATED_FP8_RESULTS.md`](MERGED_GATED_FP8_RESULTS.md)에 기록한다.

결과 JSON은 현재 다음 node-adaptive 필드를 보존한다:

- `node_adaptive`, `node_adaptive_min_demand`, `node_adaptive_nodes`,
  `node_adaptive_indices`
- `node_delta_params` — 실제 적응 노드의 ΔW 파라미터 수(s는 세지 않는다)
- `shared_weight_fp8` — 공유 weight FP8 fake quantization 사용 여부

stage별 결과를 나타내는 키는 더 이상 없다. 파일 이름은 `node_adaptive`가 켜진 런에
계속 `_nodeadaptive` suffix를 붙인다.

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
