# Merged 모델의 현재 실행 경로

`models/merged/modeling.py`가 local 창 인코더, 주기 인코더, 융합/예측 및
손실을 구현한다. `dataset_frame/unified_demand_dataset.py`가 제공하는
daily/weekly lag는 오래된 순서에서 가까운 순서로 정렬되어 있고, mask의 `True`는
데이터 시작 이전이라 유효하지 않은 시점이다. Retrieval 관점은 현재 구현되지 않았다.

## 패치 이후 노드 간 Transformer 실험

두 도시의 기본 `model.use_inter_node_transformer=false`는 기존 채택 `ma`의
파라미터·출력을 유지한다. `true`면 각 시각의 5×5 패치를 기존 공간
Transformer로 인코딩해 얻은 CLS `[B,k,N,D]`를 `[B*k,N,D]`로 바꿔,
**N개 노드를 토큰으로 하는 1층 TransformerEncoder**에 통과시킨다.
기존 CLS에는 노드 고유 임베딩이 포함된다. 출력은 `[B,k,N,D]`로 되돌려
기존 시간 LSTM, 자기노드 MA·daily·weekly 융합 및 손실에 그대로 넘긴다.
헤드 수 4, FFN 폭 128, dropout/attention dropout은 도시별 기존 설정을
재사용하고 Ulsan/Porto의 D는 각각 16/64다. 추가 층의 초기화는 별도
RNG 컨텍스트에서 실행해 같은 seed의 기존 층 초기화를 보존한다.
이는 `tmp`나 `ir-weather`에 존재했던 층을 복원한 것이 아니라 **새 실험**이다.

Ulsan/Porto seed `245/6835/851`의 채택 MA 대비 전체 재학습 결과는
[`ADFORMER_REFERENCE_RESULTS.md`](ADFORMER_REFERENCE_RESULTS.md) 표 12,
채택 MA에서 노드별 ΔW만 끈 실험은
[`MERGED_ABLATION_RESULTS.md`](MERGED_ABLATION_RESULTS.md) 5절에 있다.

## 주기 관점 실험 (`model.periodic_mode`)

Ulsan 기본값은 `ma`, `d_model=16`, `local_encoder=transformer`,
`history_weights=null`, `periodic_window_weights=null`, `lag_radius=0`이다.
이전 local-only 모델은 `model.periodic_mode=none`으로 실행한다.
Porto의 과거 기본값 `none`도 3-seed MA 대조 이후 `ma`로 교체했다.

| mode | PeriodicViewEncoder의 daily/weekly 출력 | ViewFusion / PredictionHead |
|---|---|---|
| `none` | 호출하지 않음 | 기존 `PredictionHead(local)` |
| `lstm` | 유효 lag만 시간순으로 압축하고 `log1p(수요) ⊕ 정규화 날씨/캘린더`를 별도 LSTM에 입력, 각각 `[B,N,D]` | `Linear(concat(local,daily,weekly)) → PredictionHead` |
| `ma` | 유효한 raw 수요 lag의 산술평균, 각각 `[B,N]` | `Softplus(Linear(local_view) + g_local·mean(자기노드 raw history) + g_daily·daily + g_weekly·weekly)`; local/daily/weekly 합 1 |
| `ma_no_local` | `ma`와 같은 유효 raw daily/weekly lag 산술평균 | 이전 구조를 재실행하는 대조군: `Softplus(Linear(local_view) + sigmoid(g_daily)·daily + sigmoid(g_weekly)·weekly)`; local 원수요 평균 없음 |
| `ema` | 가장 가까운 lag의 가중치가 높은 지수평균. `α=2/(L+1)`, 유효 lag의 가중치만 재정규화 | 독립 sigmoid daily/weekly 게이트 + local view 선형항으로 직접 예측 |
| `lag_lstm` | 5시간 창을 한 lag로 접은 뒤 유효 lag만 오래된 순서대로 `LSTM(input=1,hidden=4) → Linear(4,1)`에 넣어 각각 `[B,N]` | `ema`와 같은 독립 sigmoid 게이트·Softplus 직접 예측 |

mask가 전부 무효인 샘플의 주기 출력은 정확히 0이며 융합 시에도 그 관점은 0으로
취급한다. `none`에서는 새 파라미터가 생성되지 않아 기존 local-only 체크포인트와
학습 초기화가 유지된다. `ma`의 자기노드 평균은 target 시점 `t` 기준
`demand_history[t-k:t]`의 **raw 중앙 격자 노드**에서 시간축으로 계산한다.
patch 주변 노드를 평균내거나 날씨를 섞지 않는다. `history_weights`가 있어도
MA용 평균은 원본 k시점에서 계산한다. 노드마다 daily/weekly **상대 logit**
`log(0.2/0.7)`과 `log(0.1/0.7)`을 학습하며 local logit은 0으로 고정한다.
3-way softmax가 모든 노드에서 처음 `(g_local,g_daily,g_weekly)=(0.7,0.2,0.1)`을
만들고 학습 중에도 양수·합 1을 보장한다. 유효한 daily/weekly lag가 없는 샘플은
해당 출력만 0으로 두고 나머지 게이트를 재정규화하지 않는다. `ma_no_local`은
과거 MA를 재실행하기 위한 대조군으로 노드별 독립 sigmoid 게이트를 0.5에서
시작하고 local 원수요 평균은 쓰지 않는다. `ema`/`lag_lstm`도 동일한
독립 sigmoid 게이트를 사용한다. `zero_node_indices` 사용 시 raw 평균과
주기 입력·게이트를 같은 노드 축으로 자른다.

Ulsan 기본 실행은 `python train.py --config-name config_ulsan`이다.
local-only는 `model.periodic_mode=none`, 평활화 실험은
`model.history_weights=[0.1,0.2,0.7]`로 선택한다.
기존 독립 sigmoid MA의 MAPE(+1)가 단일 seed 기준보다 0.5% 이상 개선돼
EMA도 시험했지만, EMA는 그 MA보다 RMSE와 MAPE(+1)이 모두 악화했다.
이전 구조의 수치는 [`ADFORMER_REFERENCE_RESULTS.md`](ADFORMER_REFERENCE_RESULTS.md)
표 6에 별도 기록돼 있으며 현재 `ma`의 재학습 성능을 뜻하지 않는다.

새 3-way MA의 Ulsan seed 245 결과는 같은 분할의 이전 sigmoid MA보다
RMSE/MAPE(0 제외)는 개선됐으나 MAE/MAPE(+1)는 악화했다.
3-seed 결과에서도 RMSE 평균은 MA가 0.189% 낮지만 MAE와 MAPE(+1)는
각각 0.088%와 0.832% 높다. RMSE 기준으로 기존 Ulsan 채택값을 유지한다.
Porto의 3-seed 비교에서는 MA가 네 지표 모두 각 seed에서 개선되어
`configs/model/merged_porto.yaml`의 기본값으로 채택했다.
자세한 결과는 [`ADFORMER_REFERENCE_RESULTS.md`](ADFORMER_REFERENCE_RESULTS.md)
표 11에 있다.

양 도시 모두 기본 `ma`로 실행한다. 이전 독립 sigmoid MA 대조는
`model.periodic_mode=ma_no_local`, 과거 Porto local-only는
`model.periodic_mode=none`으로 선택한다. 두 MA 모드는 local 평균뿐 아니라
**게이트 정규화도 다르므로** 두 모드의 차이를 local 평균만의 인과적 효과로
해석하지 않는다. local MA만 제거하려면 채택 `ma`에서
`model.use_local_mean=false`를 사용해 같은 3-way gate를 유지한다.

### 채택 MA 모델의 모듈별 ablation

두 도시 모두 seed 245의 채택 `ma` 전체 학습과 동일한 분할·손실·학습 설정을
기준으로, 한 번에 아래 모듈 하나만 비활성화해 **처음부터 재학습**한다.
`model.use_local_view=false`는 공간 Transformer+temporal LSTM의 전체
출력을 0으로 대체한다. `model.use_local_mean=false`는 같은 3-way gate를
유지하면서 자기노드 raw 평균만 제거하므로 앞의 `ma_no_local` 비교와 달리
gate 방식은 고정된다. `model.use_daily=false`와
`model.use_weekly=false`는 해당 주기 신호를 0으로, `model.use_weather=false`와
`model.use_calendar=false`는 temporal LSTM에 연결되는 해당 context 채널을
0으로 만든다. `model.use_neighbors=false`는 중앙 노드 이외의 demand만
0으로 바꾼다(격자 경계와 노드 위치는 보존). 각 비활성 설정은 체크포인트
config에도 남는다.

`model.use_softplus=false`는 비음수 출력을 강제하는 활성을 제거해 음수
예측도 허용한다. `model.node_adaptive=false`는 노드별 ΔW 파라미터 자체를
제거하므로 동일 파라미터 수의 정보 0-치환 실험이 아니다.
`model.shared_weight_fp8=false`는 공유 LSTM의 fake FP8만 끄며
fp32 master weight·연산은 원래도 동일하다. 이 셋은 정보 삭제와
다른 종류의 스위치로 구분한다. `use_retrieval`,
`use_branch_attention`, `weather_injection`은 현재 예측 경로에
영향이 없어 ablation 결과로 보고하지 않는다.

## 주기 기준 시점 주변 ±2시간 (`model.periodic_window_weights`)

기존 `lag_radius=0`에서는 daily/weekly 관점에 정확히 `t-24*i`,
`t-168*i`의 수요만 들어간다. 실험은 `data.lag_radius=2`로 각 기준 시점
주변의 과거 순서 `[t-p*i-2, t-p*i-1, t-p*i, t-p*i+1, t-p*i+2]`를
가져오고, PeriodicViewEncoder 안에서 **bias 없는 5→1 선형 가중합**으로
한 주기 lag로 접는다. daily와 weekly는 각각 독립적으로 가중치를
학습하며 모두 `[0.05, 0.1, 0.7, 0.1, 0.05]`에서 시작한다. 그 뒤
MA/EMA, 기존 D벡터 `lstm`, 또는 `lag_lstm`이 주기 lag를 요약한다.
기존 `lstm`의 context는 기준 시점 한 시점의 날씨·캘린더를 사용한다.

데이터 시작 전 시각이 끼어 있는 5개 묶음은 전체를 무효 처리한다.
따라서 `t=24`에서 `t-24`가 존재해도 daily 묶음은 아직 불완전해
사용하지 않으며 `t=26`부터 첫 묶음이 유효하다. 가장 가까운
`t-24+2`도 예측 시각 `t`보다 이전이라 미래 수요를 사용하지 않는다.
local 관점과 local history 길이는 변경되지 않는다. 재현 옵션은
`data.lag_radius=2 model.periodic_window_weights=[0.05,0.1,0.7,0.1,0.05]`이며
두 설정이 동시에 필요하다. 기본값에서는 창 모듈과 추가 창 파라미터가 없지만,
이전 sigmoid MA 체크포인트의 게이트는 **새 3-way MA와 호환되지 않는다**.
결과 JSON은 초기값과 daily/weekly 학습 가중치를 각각 기록하고 기본 결과 이름에
`_window5`를 붙인다.

표 8의 Ulsan seed 245 수치는 **이전 독립 sigmoid MA 구조**의 실행 결과다.
당시 MAE는 개선되고 RMSE/MAPE(+1)는 악화했으며 weekly 창은 음수 계수도
학습했다. 새 MA 구조의 결과로 해석하지 않는다.

`model.periodic_mode=lag_lstm`은 **창 5→1은 그대로 두고** 이후 주기 lag 간
MA만 raw lag 수요의 `LSTM(1,4) → Linear(4,1)`로 교체한다. daily/weekly마다
별도 LSTM을 학습하며, 없거나 불완전한 lag는 시퀀스에서 빼고
유효한 lag가 없으면 출력 스칼라는 정확히 0이다. 이 모드는 이전과 동일하게
**독립 sigmoid 게이트**로 두 주기 출력을 더하며 local 원수요 평균은
넣지 않는다.

표 9에서 lag LSTM이 모든 지표에서 악화한 비교 대상 역시 이전 sigmoid MA다.
이전 결과를 새 3-way MA의 성능 비교로 사용하지 않는다.

## Local 입력의 시간 평활화 (`model.history_weights`)

Ulsan 실험에서는 `LocalViewEncoder`가 격자 셀마다 최근 24시간의 **raw 수요**에
공유하는 3탭 valid temporal kernel을 먼저 적용했다:
`y_i=0.1*x_i+0.2*x_{i+1}+0.7*x_{i+2}` (실험 초기값).
`softmax(history_logits)`로 가중치를 학습하므로 합이 1이고 양수이며,
padding은 없다. 따라서 local의 공간 창 crop → log1p → Fourier
scalar embedding → Transformer → LSTM 경로가 받는 길이는 24→22로 줄어든다.
LSTM 입력의 날씨·시간·요일 context도 `context[:, 2:]`로 맞춰 각 가중합의
**마지막 수요 시점**에 대응한다(날씨 자체의 기존 한 시점 앞 정렬은 유지).
daily/weekly 관점의 lag 수요는 평활화하지 않는다. `history_weights=null`이면
해당 파라미터와 연산이 없어져 기존 체크포인트 경로로 돌아간다.
결과 JSON에는 초기/학습 가중치를 기록하고, 자동 결과 파일명에는 `_history3`을
붙여 이전 실험을 덮어쓰지 않는다.

Ulsan seed 245의 기존 MA와 비교하면 test MAE/RMSE/MAPE(+1)가 모두 악화했다.
결과는 [`ADFORMER_REFERENCE_RESULTS.md`](ADFORMER_REFERENCE_RESULTS.md)의
표 7에 있으며 Ulsan 기본 설정에는 적용하지 않았다.

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
기존 `none`의 이름을 유지하고, 주기 변종이면 `_ma_localmix`/`_lstm`/
`_ema`/`_lag_lstm`, local 평활화하면 `_history3`, 주기창이면 `_window5`,
`node_adaptive`가 켜지면 `_nodeadaptive`를 차례로 붙인다.

## 유지보수

- `MergedDemandModel.forward()`는 학습용 `{'loss', 'logits'}`만 돌려준다.
  관점별 출력은 분석용 `forward_views()`에서 확인한다.
- mask/시간순 압축, MA/EMA 평균, 게이트 미분 및 체크포인트 복원 계약은
  `tests/test_periodic_variants.py`에서 검사한다.
- local 수요 평활화의 valid kernel·context 정렬·학습 경사는
  `tests/test_history_smoothing.py`에서 검사한다.
- 주기창의 시점 정렬·결측 lag 제외·daily/weekly 가중치 학습 및 체크포인트는
  `tests/test_periodic_window.py`에서 검사한다.
- 주기 lag를 스칼라 LSTM으로 압축하는 시퀀스·결측·체크포인트는
  `tests/test_periodic_lag_lstm.py`에서 검사한다.
