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
| `configs/config_ulsan.yaml` / `configs/config_porto.yaml` | 도시별 `optimizer_schedule`, `stage2:` 블록, 시간순 70/15/15 분할 | LR 스케줄, 배치, `--config-name` |
| `configs/model/merged_ulsan.yaml` / `configs/model/merged_porto.yaml` | 모델 폭·dropout·ablation 스위치 — 종전 d_model 튜닝 기록은 `docs/MERGED_TUNING_RESULTS.md` | dimensions, retrieval policy, `weekday_dim`, `hour_dim` |
| `run_ablation.sh` / `run_seeds.sh` | 큐/다중 시드 러너 (ablation 이름 → Hydra 오버라이드) | 실행 조합 |

## Directory layout

```text
dataset_frame/unified_demand_dataset.py   # temporal-grid Dataset
models/merged/
├── config.py               # MergedDemandConfig(PretrainedConfig)
├── modeling.py             # MergedDemandModel(PreTrainedModel) — one graph
├── losses.py               # combined | mae | rmse_mape objectives
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

검색 켬: raw local crop + absolute sample_idx → CausalRetrieval → ir_out
         h_attn + ir_out → NeuralRetrievalGate → prediction → loss(labels)
검색 pass(기본값): h_attn → neural_head → prediction → loss(labels)
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

`model.use_retrieval=false`이면 tmp-extracted의 미사용 검색 관점처럼
검색기를 만들거나 raw grid·검색 후보 crop/cache를 적재하지 않는다.
게이트의 `lambda_layer`는 건너뛰며 `neural_head`만 예측에 사용한다
(기존 검색 끔 체크포인트의 신경망 가중치와 호환). 검색 켬에서는 raw
수요를 사용하며 후보 시간은 `[time_step, target_time)`로 제한한다.
검색 켬의 CPU cache는 체크포인트에 저장되지 않고 모델 생성 시 재구성된다.

## 노드별 LSTM weight offset (`node_adaptive`)

lora 브랜치의 node_adaptive(W + ΔW)를 이 모델의 **history LSTM 하나에만** 옮긴 것이다.
`model.node_adaptive=true`일 때만 켜지며, 현재 도시별 모델 설정은 `true`다.
꺼져 있으면 delta 파라미터가 아예 만들어지지 않아 `state_dict`가 기능 추가 이전과
같고 기존 체크포인트 및 `tests/test_merged_parity.py`와 호환된다.

```text
weight_ih = history_lstm.weight_ih_l0 + node_delta_weight_ih   # (A, 4h, d_model+extra_dim)
weight_hh = history_lstm.weight_hh_l0 + node_delta_weight_hh   # (A, 4h, history_hidden)
bias      = bias_ih_l0 + bias_hh_l0   + node_delta_bias        # (A, 4h)
```

`A`는 전체 노드가 아니라 **train 구간 평균 수요가 `node_adaptive_min_demand`(기본 0.8)를
넘는 노드** 수다. 현재 격자는 Ulsan 14×12=168, Porto 10×20=200노드이고
선택 노드 수는 새 70% 학습 구간에서 다시 계산한다. 노드당 delta 파라미터는 모델 폭에
따라 달라지므로 결과 JSON의 `node_delta_params`를 확인한다. 선택 기준은 시간 리크를
막기 위해 train 구간에서만 계산하며(`train.py: select_node_adaptive_indices`),
결과 노드 목록은 `config.node_adaptive_indices`에 저장돼 같은 마스크를 복원한다.

설계상 지켜야 할 것들:

- **선택되지 않은 노드는 `nn.LSTM`(cuDNN fused) 결과를 그대로 쓴다.** 선택된 노드만 수동 셀
  루프로 다시 계산해 `index_copy`로 덮어쓴다. 그래서 대다수 노드는 속도 손해가 없다.
- **ΔW는 0으로 시작한다.** 그래야 학습 시작 시점이 `node_adaptive=false`와 동일해서
  측정된 차이가 ΔW 때문임이 분리된다. `validate_merged.py`의
  `_check_node_adaptive_identity`가 ΔW=0 ≡ `nn.LSTM`을 대조로 고정한다.
- **노드 목록을 버퍼로 저장하면 안 된다.** transformers 5.0의 `from_pretrained`는 meta device에서
  모델을 만든 뒤 체크포인트에 있는 키만 실체화한다. stage 2가 이어받는 stage 1 체크포인트에는
  그 키가 없어서, 버퍼로 두면 `torch.empty`(쓰레기값)로 남아 엉뚱한 노드를 고른다.
  config의 파이썬 리스트에서 forward 시점에 텐서를 만들어 device별로 캐시한다.
- **`_init_weights`에서 `super()._init_weights(module)`를 부르면 안 된다.** lora 브랜치의 같은
  이름 메서드는 정반대 규칙이라 그쪽 코드를 복사해 오면 모델 전체의 초기 분포가 바뀌어
  기존 ablation/튜닝 결과와 출발점이 달라진다. delta만 0으로 채우고 나머지는 건드리지 않는다.
- daily/weekly 주기 브랜치는 대상이 아니다 — `pack_padded_sequence`가 노드 축을 흐트러뜨려
  노드별 가중치를 붙이려면 packing 자체를 걷어내야 한다.

### 2-stage 학습

`node_adaptive=true`면 `train.py`가 자동으로 2-stage로 나눈다(꺼져 있으면 지금까지와 동일한
단일 stage다).

| | stage 1 | stage 2 |
|---|---|---|
| ΔW | `requires_grad_(False)` (0 고정) | `requires_grad_(True)` |
| loss | `model.loss_type` (Ulsan 기본 `mae`, Porto `combined`) | `stage2.loss` (기본 `rmse_mape`) |
| lr | `train.learning_rate` | `stage2.learning_rate` (기본 1e-4) |
| best 기준 | `eval_loss` | `rmse_mape_objective` |
| output_dir | `<run>/stage1` | `<run>/stage2` |

- stage마다 `Trainer`를 새로 만든다 — optimizer / LR 스케줄러 / early stopping 상태가 경계에서
  초기화돼야 하고, `save_total_limit=1`이라 같은 디렉터리를 쓰면 stage 2가 stage 1 체크포인트를
  지운다.
- `rmse_mape`는 `rmse_weight * RMSE + MAPE(+1)`이고 **요소별로 분해되지 않는 스칼라**다
  (RMSE가 전체 원소에 걸친 하나의 `sqrt(mean(...))`). `modeling.py`의 `_compute`가
  `SCALAR_LOSS_TYPES`로 분기해 `loss_sum` 집계를 건너뛴다. best checkpoint 선택은 학습 중의
  미니배치 surrogate가 아니라 `train.py`가 전체 validation 예측으로 다시 계산한
  `rmse_mape_objective`를 쓴다.
- `stage2.init_from`에 체크포인트 경로를 주면 stage 1을 건너뛰고 그 가중치에서 시작한다.
  optimizer/step을 복구하는 resume이 아니라 weights-only warm start이며, `node_adaptive`를 끄고
  학습한 단일 stage 체크포인트를 그대로 쓸 수 있다(ΔW 키가 없으면 `_init_weights`가 0으로 채움).
  학습된 stage 2 체크포인트를 실수로 넣지 않도록 `train.py`가 ΔW=0인지 검사한다.
- 결과 JSON은 `node_adaptive`/`node_adaptive_nodes`/`node_delta_params`/`stages`/`stage1`/
  `stage1_test`/`best_metric` 필드를 추가로 남기고, 파일 이름에 `_nodeadaptive`가 붙어
  기존 단일 stage 결과를 덮어쓰지 않는다.

### 설정 이식 실험 (단일-stage 기록과 신규 2-stage 구분)

`tmp-extracted`에서 검증한 학습 설정을 이식했다. Ulsan은 `d_model=16`,
batch 24, stage 1 MAE 손실이고 Porto는 `d_model=64`, batch 8,
stage 1 combined 손실이다. 두 도시 모두 AdamW lr=0.001·weight decay=0.05,
warmup 5 epoch (1e-6→stage LR), cosine 60 epoch (stage LR→0.0001),
이후 0.0001, 각 stage 최대 120 epoch를 사용한다. stage 2는 lr=0.0001,
`10 * RMSE + MAPE(+1)` 손실이며 그 검증 지표로 best를 선택한다.
출력/FFN dropout 0.1을 유지하고 **어텐션 가중치 dropout만 0**으로 설정한다.
기존 체크포인트에 `attention_dropout`이 없으면 종전처럼 `dropout`을 따른다.
이전 `output/experiments/tmp_settings_3seed_*`는 `node_adaptive=false`,
80/10/10인 단일-stage 실험이다. 현재 도시별 설정은 `node_adaptive=true`,
70/15/15로 2-stage를 실행한다. split 및 stage가 모두 바뀌었으므로 이전 테스트
수치와 직접 비교하지 않는다. 자세한 기록은 `docs/MERGED_TUNING_RESULTS.md`를 참고한다.

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
