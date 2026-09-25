# Merged 모델의 현재 실행 경로

`models/merged/modeling.py`가 local 창 인코더, 주기 인코더, 융합/예측 및
손실을 구현한다. `dataset_frame/unified_demand_dataset.py`가 제공하는
daily/weekly lag는 오래된 순서에서 가까운 순서로 정렬되어 있고, mask의 `True`는
데이터 시작 이전이라 유효하지 않은 시점이다. Retrieval 관점은 현재 구현되지 않았다.

## 주기 관점 실험 (`model.periodic_mode`)

Ulsan 기본값은 `ma`, `d_model=16`, `local_encoder=transformer`다. 이전 local-only
모델은 `model.periodic_mode=none`으로 실행하며 Porto 기본값도 기존 `none`이다.

| mode | PeriodicViewEncoder의 daily/weekly 출력 | ViewFusion / PredictionHead |
|---|---|---|
| `none` | 호출하지 않음 | 기존 `PredictionHead(local)` |
| `lstm` | 유효 lag만 시간순으로 압축하고 `log1p(수요) ⊕ 정규화 날씨/캘린더`를 별도 LSTM에 입력, 각각 `[B,N,D]` | `Linear(concat(local,daily,weekly)) → PredictionHead` |
| `ma` | 유효한 raw 수요 lag의 산술평균, 각각 `[B,N]` | `Softplus(Linear(local) + sigmoid(g_daily[node])·daily + sigmoid(g_weekly[node])·weekly)`; 별도의 PredictionHead 없음 |
| `ema` | 가장 가까운 lag의 가중치가 높은 지수평균. `α=2/(L+1)`, 유효 lag의 가중치만 재정규화 | `ma`와 같은 직접 예측 경로 |

mask가 전부 무효인 샘플의 주기 출력은 정확히 0이며 융합 시에도 그 관점은 0으로
취급한다. `none`에서는 새 파라미터가 생성되지 않아 기존 local-only 체크포인트와
학습 초기화가 유지된다. MA/EMA에서는 노드별 게이트가 `[N]`의 학습 가능한
스칼라 파라미터이며 0에서 초기화해 sigmoid 값 0.5로 출발한다. `zero_node_indices`
사용 시 local에서 제외된 노드에 맞춰 주기 입력과 게이트도 같은 노드 축으로 자른다.

Ulsan 기본 실행은 `python train.py --config-name config_ulsan`이다. 이전
local-only 및 다른 변종은 `model.periodic_mode=none|lstm|ema`로 선택한다.
MA의 MAPE(+1)가 단일 seed 기준보다 0.5% 이상 개선돼 EMA도 시험했지만,
EMA는 MA보다 RMSE와 MAPE(+1)이 모두 악화했다. 수치는
[`ADFORMER_REFERENCE_RESULTS.md`](ADFORMER_REFERENCE_RESULTS.md)에 있다.

## 게이티드 노드별 LSTM weight offset + shared-weight FP8 (`node_adaptive`)

lora 브랜치의 node_adaptive(W + ΔW)를 이 모델의 **`LocalViewEncoder`의 temporal LSTM
하나에만** 옮긴 뒤, 공유 가중치와 노드별 보정의 비율을 학습하는 구조다.
`model.node_adaptive=true`일 때만
켜지며, Ulsan/Porto 모델 config 모두 기본값이 `node_adaptive=true`와
`shared_weight_fp8=true`다(Porto는 아직 자체 대조 근거가 없다).

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
`node_adaptive_min_demand`(기본 0.8)를 넘는 노드** 수다. 현재 Ulsan 0.70/0.15
분할에서는 168개 중 45개 노드를 고른다. 선택 기준은 시간 리크를 막기 위해 train
구간에서만 계산하며(`train.py: select_node_adaptive_indices`), 결과 노드 목록은
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

`node_adaptive=true`여도 `train.py`는 단일 stage만 학습하며, 목적함수는
`model.loss_type`(현재 Ulsan 기본값 `mae`)을 따른다. 과거 다른 분할/손실에서의 결과는
[`MERGED_GATED_FP8_RESULTS.md`](MERGED_GATED_FP8_RESULTS.md)에 별도 기록돼 있다.

결과 JSON은 현재 다음 node-adaptive 필드를 보존한다:

- `node_adaptive`, `node_adaptive_min_demand`, `node_adaptive_nodes`,
  `node_adaptive_indices`
- `node_delta_params` — 실제 적응 노드의 ΔW 파라미터 수(s는 세지 않는다)
- `shared_weight_fp8` — 공유 weight FP8 fake quantization 사용 여부

stage별 결과를 나타내는 키는 더 이상 없다. 자동 생성하는 결과 JSON 이름은
기존 `none`의 이름을 유지하고, 다른 변종은 `_ma`/`_lstm`/`_ema`를 붙인 뒤
`node_adaptive`가 켜지면 `_nodeadaptive`를 이어 붙인다.

## 유지보수

- `MergedDemandModel.forward()`는 학습용 `{'loss', 'logits'}`만 돌려준다.
  관점별 출력은 분석용 `forward_views()`에서 확인한다.
- mask/시간순 압축, MA/EMA 평균, 게이트 미분 및 체크포인트 복원 계약은
  `tests/test_periodic_variants.py`에서 검사한다.
