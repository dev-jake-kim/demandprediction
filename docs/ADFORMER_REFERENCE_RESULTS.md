# ADFormer 기준 성능 (seed 245 / 6835 / 851 / 5123 / 535)

## 표 1. 도시별 평균 ± 표준편차

| 도시 | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| Ulsan | 0.3265 ± 0.0011 | 0.7284 ± 0.0029 | 15.7116 ± 0.1842 |
| Porto | 0.4913 ± 0.0022 | 1.6753 ± 0.0305 | 15.4257 ± 0.1298 |

## 표 2. Ulsan seed별 성능

| Seed | MAE | RMSE | MAPE (%) |
|---:|---:|---:|---:|
| 245 | 0.3286 | 0.7309 | 15.9353 |
| 6835 | 0.3253 | 0.7278 | 15.6674 |
| 851 | 0.3261 | 0.7287 | 15.5390 |
| 5123 | 0.3264 | 0.7312 | 15.4987 |
| 535 | 0.3262 | 0.7233 | 15.9177 |

## 표 3. Porto seed별 성능

| Seed | MAE | RMSE | MAPE (%) |
|---:|---:|---:|---:|
| 245 | 0.4943 | 1.6749 | 15.4567 |
| 6835 | 0.4898 | 1.6766 | 15.2616 |
| 851 | 0.4917 | 1.6826 | 15.5951 |
| 5123 | 0.4928 | 1.7191 | 15.2914 |
| 535 | 0.4880 | 1.6235 | 15.5237 |

## 표 4. 이전 채택 모델(d_model=64 Transformer) Ulsan seed별 성능

### commithash: 7677ab3

| Seed | MAE | RMSE | MAPE (%) |
|---:|---:|---:|---:|
| 245 | 0.3210(-2.322%) | 0.7311(+0.029%) | 15.1114(-5.170%) |

## 표 4-1. 현재 채택 모델(d_model=16 Transformer) Ulsan seed 245

기존 d_model=64와 나머지 설정이 같은 단일 seed 비교다. 다른 seed에서도 재현되는지는
검증하지 않았다. 원자료: `output/experiments/var_d{64,16}_transformer_seed245.json`.

| 설정 | MAE | RMSE | MAPE(+1) (%) |
|---|---:|---:|---:|
| d_model=64 Transformer | 0.320971 | 0.731112 | 15.111426 |
| d_model=16 Transformer | 0.320882 | 0.724449 | 15.505391 |

## 표 5. merged 모델 Porto seed별 성능

### commithash: 7677ab3 + early_stopping_patience 10 (미커밋)
### model.node_adaptive=false / model.shared_weight_fp8=false

| Seed | MAE | RMSE | MAPE (%) |
|---:|---:|---:|---:|
| 245 | 0.4872(-1.433%) | 1.7891(+6.819%) | 15.5593(+0.664%) |
| 6835 | 0.4851(-0.955%) | 1.7880(+6.644%) | 15.1863(-0.493%) |
| 851 | 0.4858(-1.205%) | 1.7648(+4.887%) | 15.2208(-2.400%) |
| 5123 | 0.4856(-1.452%) | 1.7925(+4.270%) | 15.1881(-0.676%) |
| 535 | 0.4879(-0.028%) | 1.8267(+12.519%) | 14.9161(-3.914%) |
| 평균 | 0.4863(-1.017%) | 1.7922(+6.977%) | 15.2141(-1.372%) |

## 표 6. PeriodicViewEncoder 변종 비교 — Ulsan seed 245

Ulsan 0.70/0.15 분할, d_model=16 Transformer local encoder, MAE 손실,
`model.history_weights=null`, 동일 학습 설정. 기준 `none`은 앞서 완료한
`output/experiments/var_d16_transformer_seed245.json`이며 주기 입력을 쓰지 않는다.
아래 `ma`는 **과거의 독립 sigmoid daily/weekly 게이트** 구조로, 현재의
자기노드 local MA + 3-way softmax 구조와 다르다. 물리 GPU 1에서 실행했고
test 구간 전체로 평가했다. **단일 seed 결과이며 Porto에는 적용해 보지 않았다.**

| periodic_mode | 학습 파라미터 | best / 학습 epoch | MAE | RMSE | MAPE(+1) (%) | MAPE(0 제외) (%) |
|---|---:|---:|---:|---:|---:|---:|
| none (기준) | 153,108 | 34 / 54 | 0.320882 | 0.724449 | 15.505391 | 58.159538 |
| lstm (D벡터 → concat+linear → 기존 head) | 158,116 | 31 / 51 | 0.318011 | 0.725838 | 15.231358 | 56.352214 |
| ma (노드별 gate + local 선형항) | 153,172 | 33 / 53 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| ema (ma와 같은 gate, 최근 lag 가중) | 153,172 | 35 / 55 | 0.323335 | 0.725314 | 15.319414 | 60.517180 |

기준 대비 lstm은 RMSE **+0.192%**, MAE **-0.895%**, MAPE(+1) **-1.767%**;
ma는 RMSE **-0.269%**, MAE **+0.487%**, MAPE(+1) **-1.252%**다.
ma의 RMSE 개선은 0.5% 미만이지만 MAPE(+1)이 **1.252%** 개선돼,
약속한 “둘 중 하나가 0.5% 이상 개선” 조건으로 ema도 실행했다.
ema는 ma 대비 RMSE **+0.389%**, MAPE(+1) **+0.054%**로 둘 다 악화했다.
당시 Ulsan 기본 설정으로 `periodic_mode=ma`를 채택했다. 그 과거 구조는 단일
seed에서 RMSE/MAPE(+1)는 개선됐지만 MAE(+0.487%)와 MAPE(0 제외)(+3.552%)는
악화했다. EMA 실행 여부를 test 지표로 결정한 탐색 실험이므로 이 비교를
독립 test 검증이나 새 3-way MA의 성능으로 해석해서는 안 된다.

원자료/체크포인트:
`output/experiments/periodic_{lstm,ma,ema}_d16_ulsan_seed245.json`,
`output/experiments/checkpoints/periodic_{lstm,ma,ema}_d16_ulsan_seed245/`.
현재 `ma`는 다른 융합식이다. 이전 구조는 `model.periodic_mode=ma_no_local`로 재학습할 수 있지만 옛 `periodic_mode=ma` 체크포인트를 새 `ma`로 불러 직접 비교하면 안 된다.

### 과거 MA의 노드별 게이트 시각화

**이전 체크포인트**에서 실제 곱해지는 값은 `fusion.daily_gate` /
`fusion.weekly_gate` 원 파라미터에 sigmoid를 씌운 값이다. 새 MA는
`daily_mix_logit`/`weekly_mix_logit`을 local 고정 logit 0과 함께 softmax한다.
옛 MA 게이트는 독립 계수라 합이 1인 attention 비율이 아니었다.
행·열은 Ulsan 14×12 격자의 row-major 노드 순서다. 히트맵과 분포를 다시 그리려면:

```bash
conda run -n DA python plot_periodic_gates.py \
  output/experiments/checkpoints/periodic_ma_d16_ulsan_seed245 \
  output/experiments/periodic_ma_gate_ulsan_seed245.png
```

## 표 7. Local history 3탭 평활화 — Ulsan seed 245

표 6의 MA 모델과 같은 분할·학습 설정으로 처음부터 학습했다.
local 입력의 최근 24시간 raw 수요에만 valid 3탭 시간 평균을 적용해
22시점으로 줄이고, 커널을 `[0.1, 0.2, 0.7]`에서 학습했다.
daily/weekly 입력은 변경하지 않았다. 물리 GPU 1, 전체 test 구간 평가.

| 설정 | 파라미터 | best / 학습 epoch | MAE | RMSE | MAPE(+1) (%) | MAPE(0 제외) (%) |
|---|---:|---:|---:|---:|---:|---:|
| MA + raw history (기준) | 153,172 | 33 / 53 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| MA + 3탭 평활화 | 153,175 | 29 / 49 | 0.323620 | 0.724047 | 15.402957 | 60.864867 |

기준 대비 MAE **+0.365%**, RMSE **+0.214%**, MAPE(+1) **+0.599%**,
MAPE(0 제외) **+1.062%**로 모두 악화했다. 학습된 커널은
`[0.098200, 0.197440, 0.704360]`이다. 당시 Ulsan 기본값을
`model.history_weights=null`로 유지했다. **이전 sigmoid MA의 단일 seed 탐색
실험**이므로 현재 3-way MA 또는 다른 seed·도시에서의 영향은 확인되지 않았다.

원자료: `output/experiments/history3_ma_d16_ulsan_seed245.json`;
체크포인트: `output/experiments/checkpoints/history3_ma_d16_ulsan_seed245/`.
과거 sigmoid MA 전용 결과다. 같은 구조의 재학습에는 `model.periodic_mode=ma_no_local`과 위 3탭 커널 옵션을 함께 사용한다.

## 표 8. PeriodicView 기준 시점 ±2시간 선형 결합 — Ulsan seed 245

표 6의 MA 모델과 같은 분할·loss·학습 설정, local history는 raw 그대로다.
`lag_radius=2`로 기준 daily/weekly lag 주변 5시간을 만들고 각 관점에서
별도의 bias 없는 `Linear(5,1)`로 합쳤다. 두 가중치 모두
`[0.05,0.1,0.7,0.1,0.05]`에서 시작하며 이후 제약 없이 학습된다.
데이터 시작 전 시각을 포함한 묶음은 사용하지 않는다. 물리 GPU 1,
전체 test 구간 평가.

| 설정 | 파라미터 | best / 학습 epoch | MAE | RMSE | MAPE(+1) (%) | MAPE(0 제외) (%) |
|---|---:|---:|---:|---:|---:|---:|
| MA + 정확한 주기 lag (기준) | 153,172 | 33 / 53 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| MA + lag ±2시간 선형 결합 | 153,182 | 35 / 55 | 0.321764 | 0.722896 | 15.321746 | 59.438770 |

기준 대비 MAE **-0.211%**, MAPE(0 제외) **-1.306%**로 개선했지만
RMSE **+0.055%**, MAPE(+1) **+0.069%**로 악화했다. 단일 seed의
혼합 결과라 당시 `lag_radius=0`, `periodic_window_weights=null`을 유지했다.
학습된 가중치는 daily
`[0.033572,0.103709,0.417061,0.130545,0.053947]`, weekly
`[-0.068115,-0.012872,0.236715,0.014568,-0.084153]`이다.
weekly의 일부가 음수로 바뀌므로 학습 후 연산은 양수 평균으로 제한된
평활화가 아니라 **bias 없는 선형 필터**다. 다른 seed·도시는 검증하지 않았다.

원자료: `output/experiments/periodic_window5_ma_d16_ulsan_seed245.json`;
체크포인트: `output/experiments/checkpoints/periodic_window5_ma_d16_ulsan_seed245/`.
과거 sigmoid MA 전용 결과다. 같은 구조의 재학습에는 `model.periodic_mode=ma_no_local`과 `data.lag_radius=2` 및 위 5시간 창 옵션을 함께 사용한다.

## 표 9. PeriodicView lag MA → 스칼라 LSTM — Ulsan seed 245

표 8의 ±2시간 5→1 선형창, raw local history, 분할, loss, 노드별 게이트와
직접 예측 경로는 유지했다. 유효 daily/weekly lag들의 산술평균만
각각 독립적인 `LSTM(input_size=1, hidden_size=4) → Linear(4,1)`로
교체했다. 오래된 lag부터 입력하며 결측 묶음은 빼고, 유효 lag가 없으면
출력은 0이다. 물리 GPU 1에서 전체 test 구간 656개 샘플을 평가했다.

| 설정 | 파라미터 | best / 학습 epoch | MAE | RMSE | MAPE(+1) (%) | MAPE(0 제외) (%) |
|---|---:|---:|---:|---:|---:|---:|
| MA + 정확한 주기 lag | 153,172 | 33 / 53 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| MA + lag ±2시간 선형 결합 | 153,182 | 35 / 55 | 0.321764 | 0.722896 | 15.321746 | 59.438770 |
| lag LSTM(1,4) + lag ±2시간 선형 결합 | 153,416 | 40 / 60 | 0.322219 | 0.725134 | 15.420036 | 59.977191 |

같은 선형창 MA 대비 MAE **+0.141%**, RMSE **+0.310%**,
MAPE(+1) **+0.642%**, MAPE(0 제외) **+0.906%**로 모두 악화했다.
정확한 주기 lag MA 대비로는 MAE **-0.070%**, MAPE(0 제외)
**-0.412%**이나 RMSE **+0.364%**, MAPE(+1) **+0.711%**다.
이 비교의 MA 기준선은 **과거 독립 sigmoid 게이트**이며 현재 3-way MA의
성능과 비교한 것이 아니다. 다른 seed·도시에서의 효과는 확인되지 않았다.

초기 선형창 계수는 두 관점 모두 `[0.05,0.1,0.7,0.1,0.05]`였고,
학습 후 daily는 `[0.009189,0.091166,0.639520,0.120073,-0.010827]`,
weekly는 `[-0.180265,0.016038,0.684534,0.100762,-0.165783]`다.
체크포인트를 별도 `test.py`로 다시 읽어 평가한 지표도 위 값과 일치했다.

원자료: `output/experiments/periodic_window5_lag_lstm_d16_ulsan_seed245.json`;
체크포인트: `output/experiments/checkpoints/periodic_window5_lag_lstm_d16_ulsan_seed245/`.
재현:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n DA python train.py seed=245 \
  data.lag_radius=2 model.periodic_mode=lag_lstm \
  'model.periodic_window_weights=[0.05,0.1,0.7,0.1,0.05]' \
  hydra.run.dir=output/experiments/checkpoints/periodic_window5_lag_lstm_d16_ulsan_seed245 \
  run_json=output/experiments/periodic_window5_lag_lstm_d16_ulsan_seed245.json
```

## 표 10. 자기노드 local MA + 3-way gate — Ulsan seed 245

표 6의 **과거 독립 sigmoid MA**와 비교한다. 최근 24시간의 raw local patch
중앙 노드 수요를 시간 평균해 local 스칼라를 추가하고, 노드마다 고정 local
logit 0과 학습 가능한 daily/weekly 상대 logit의 3-way softmax를 사용한다.
초기 local/daily/weekly 계수는 모든 노드에서 `0.7/0.2/0.1`이며
항상 양수·합 1이다. 유효한 주기 lag가 없는 샘플은 해당 출력만 0으로 두고
계수는 재정규화하지 않는다. 기존 `Linear(local_view)`와 Softplus는 유지했다.
동일한 Ulsan 0.70/0.15 분할, MAE 손실, seed 245, GPU 1의 전체 test 656개 샘플.

| 구조 | 파라미터 | best / 학습 epoch | MAE | RMSE | MAPE(+1) (%) | MAPE(0 제외) (%) |
|---|---:|---:|---:|---:|---:|---:|
| 이전 MA: daily/weekly 독립 sigmoid, local 평균 없음 | 153,172 | 33 / 53 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| 현재 MA: local/daily/weekly 3-way 혼합 | 153,172 | 32 / 52 | 0.322653 | 0.720627 | 15.530688 | 59.594836 |

이전 MA 대비 RMSE **-0.259%**, MAPE(0 제외) **-1.047%**지만
MAE **+0.065%**, MAPE(+1) **+1.434%**다. 단일 seed의 혼합 결과라
일반적인 성능 개선으로 해석하지 않는다. 체크포인트를 별도 `test.py`로
복원해 동일한 지표를 확인했다. 학습 후 168개 노드의 평균 유효 계수는
local **0.631140**, daily **0.235402**, weekly **0.133457**이다.

원자료: `output/experiments/periodic_ma_localmix_d16_ulsan_seed245.json`;
체크포인트: `output/experiments/checkpoints/periodic_ma_localmix_d16_ulsan_seed245/`.
재현:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n DA python train.py seed=245 \
  hydra.run.dir=output/experiments/checkpoints/periodic_ma_localmix_d16_ulsan_seed245 \
  run_json=output/experiments/periodic_ma_localmix_d16_ulsan_seed245.json
```

현재 체크포인트의 노드별 세 gate는 다음 명령으로 확인한다. 과거 체크포인트는
같은 스크립트가 독립 sigmoid 값을 표시한다.

```bash
conda run -n DA python plot_periodic_gates.py \
  output/experiments/checkpoints/periodic_ma_localmix_d16_ulsan_seed245 \
  output/experiments/periodic_ma_localmix_gate_ulsan_seed245.png
```

## 표 11. Local 자기노드 MA 적용/미적용 3-seed 비교 — Ulsan / Porto

Ulsan과 Porto에서 seed `245/6835/851`, 시간순 train/val/test `0.70/0.15/0.15`,
MAE 손실, 120 epoch 한도와 각 도시의 기존 모델·학습 설정을 유지한
**새 전체 학습**을 비교한다. `ma`는 raw local 24시간 평균을 추가해
local/daily/weekly 3-way softmax를 `(0.7,0.2,0.1)`로 시작한다.
`ma_no_local`은 local 평균 없이 독립 sigmoid daily/weekly 게이트를
각각 0.5에서 시작하는 이전 구조다. 즉 **local 평균과 게이트 방식이 동시에
달라져서** 차이를 local 평균만의 인과적 효과로 볼 수 없다.
ADFormer의 3개 seed 원자료는 이 문서 표 2·3에서 가져왔으며, 표 1의
5-seed 평균을 섞지 않는다. ADFormer `MAPE`의 원 계산식은 이 저장소에서
확인하지 못했으므로, 해당 열은 수치를 함께 싣되 `MAPE(+1)`과의
엄밀한 동등 비교에는 사용하지 않는다.

### Ulsan (14×12 노드, d_model=16, batch=24)

| seed | 설정 | MAE | RMSE | MAPE(+1) 또는 ADFormer MAPE (%) | MAPE(0 제외) (%) |
|---:|---|---:|---:|---:|---:|
| 245 | local MA 적용 | 0.322653 | 0.720627 | 15.530688 | 59.594836 |
| 245 | local MA 미적용 | 0.322443 | 0.722502 | 15.311199 | 60.225509 |
| 245 | ADFormer (표 2) | 0.328600 | 0.730900 | 15.935300 | — |
| 6835 | local MA 적용 | 0.323046 | 0.724833 | 15.253790 | 60.468116 |
| 6835 | local MA 미적용 | 0.322984 | 0.724793 | 15.292531 | 61.081096 |
| 6835 | ADFormer (표 2) | 0.325300 | 0.727800 | 15.667400 | — |
| 851 | local MA 적용 | 0.323649 | 0.722001 | 15.518695 | 60.668484 |
| 851 | local MA 미적용 | 0.323066 | 0.724266 | 15.317317 | 61.244167 |
| 851 | ADFormer (표 2) | 0.326100 | 0.728700 | 15.539000 | — |
| **3-seed 평균 ± 표본 sd** | **local MA 적용** | **0.323116 ± 0.000502** | **0.722487 ± 0.002145** | **15.434391 ± 0.156520** | **60.243812 ± 0.570889** |
|  | **local MA 미적용** | **0.322831 ± 0.000338** | **0.723854 ± 0.001200** | **15.307016 ± 0.012912** | **60.850257 ± 0.547157** |
|  | **ADFormer (3-seed만)** | **0.326667 ± 0.001721** | **0.729133 ± 0.001595** | **15.713900 ± 0.202201** | **—** |

적용 대비 미적용의 seed별 paired 평균 변화율은 MAE **+0.088%** (개선 0/3),
RMSE **-0.189%** (2/3), MAPE(+1) **+0.832%** (1/3),
MAPE(0 제외) **-0.997%** (3/3)다. 두 모델 모두 같은 3개 seed의 ADFormer보다
MAE/RMSE가 낮았지만, MAPE는 원 정의 확인 전 수치상 병렬 표시만 한다.
같은 세 seed의 평균 기준 local MA 적용 모델은 ADFormer 대비
MAE **-1.087%**, RMSE **-0.912%**다.
적용 모델의 학습 후 세 계수의 노드별 평균은 seed 순으로
`(0.631140,0.235402,0.133457)`,
`(0.634040,0.234244,0.131716)`,
`(0.635811,0.233637,0.130553)` (local/daily/weekly)이다.

각 seed의 원자료는
`output/experiments/ma_localmix_3seed_ulsan_seed{245,6835,851}.json`과
`output/experiments/ma_no_local_3seed_ulsan_seed{245,6835,851}.json`이며,
체크포인트는 `output/experiments/checkpoints/` 아래의 같은 이름에 있다.
적용 seed 245 체크포인트의 별도 `test.py` 복원·전체 test 평가도 저장된 지표와 일치했다.

### Porto (10×20 노드, d_model=64, batch=8)

| seed | 설정 | MAE | RMSE | MAPE(+1) 또는 ADFormer MAPE (%) | MAPE(0 제외) (%) |
|---:|---|---:|---:|---:|---:|
| 245 | local MA 적용 | 0.489788 | 1.784967 | 15.520057 | 60.650364 |
| 245 | local MA 미적용 | 0.498197 | 1.842360 | 15.986217 | 61.840861 |
| 245 | ADFormer (표 3) | 0.494300 | 1.674900 | 15.456700 | — |
| 6835 | local MA 적용 | 0.488421 | 1.750618 | 15.468143 | 59.894078 |
| 6835 | local MA 미적용 | 0.494894 | 1.807432 | 15.639955 | 61.054026 |
| 6835 | ADFormer (표 3) | 0.489800 | 1.676600 | 15.261600 | — |
| 851 | local MA 적용 | 0.489957 | 1.781163 | 15.618069 | 60.615568 |
| 851 | local MA 미적용 | 0.496060 | 1.801839 | 16.025793 | 61.695785 |
| 851 | ADFormer (표 3) | 0.491700 | 1.682600 | 15.595100 | — |
| **3-seed 평균 ± 표본 sd** | **local MA 적용** | **0.489389 ± 0.000843** | **1.772249 ± 0.018829** | **15.535423 ± 0.076135** | **60.386670 ± 0.426952** |
|  | **local MA 미적용** | **0.496384 ± 0.001675** | **1.817210 ± 0.021959** | **15.883988 ± 0.212263** | **61.530224 ± 0.418731** |
|  | **ADFormer (3-seed만)** | **0.491933 ± 0.002259** | **1.678033 ± 0.004045** | **15.437800 ± 0.167551** | **—** |

적용 대비 미적용의 paired 평균 변화율은 MAE **-1.409%**, RMSE **-2.469%**,
MAPE(+1) **-2.186%**, MAPE(0 제외) **-1.859%**이며 모두 **개선 3/3 seed**다.
Porto에서 local MA 적용 모델은 ADFormer보다 MAE가 각 seed에서 낮지만
RMSE는 각 seed에서 높다. ADFormer MAPE의 계산식을 확인하지 못했으므로
그 열의 모델 간 우열을 확정하지 않는다.
같은 세 seed의 평균 기준 local MA 적용 모델은 ADFormer 대비
MAE **-0.517%**지만 RMSE **+5.615%**다.

원자료는 `output/experiments/ma_localmix_3seed_porto_seed{245,6835,851}.json`과
`output/experiments/ma_no_local_3seed_porto_seed{245,6835,851}.json`,
체크포인트는 `output/experiments/checkpoints/`의 같은 이름이다.
적용 seed 245 체크포인트를 별도 `test.py`에서 복원해 전체 test
1,314개 샘플로 평가했고 저장된 네 지표가 일치했다.
모든 신규 학습은 `CUDA_VISIBLE_DEVICES=1`(물리 GPU 1)에서 수행했다.
결과 JSON의 `device=cuda:0`은 가시 GPU 재번호다. GPU 0에서 중단한
Porto 작업의 결과는 포함하지 않았다.

채택 `ma` 모델에서 **gate 방식은 유지한 채 local 평균만 끈**
seed 245 포함, 실제 모듈 10종을 각각 한 번씩 끄고 재학습한 결과는
[`MERGED_ABLATION_RESULTS.md`](MERGED_ABLATION_RESULTS.md) 표 4에 있다.
1-seed ablation의 우열을 이 표의 3-seed 평균과 동일한 신뢰도로
해석하지 않는다.

## 표 12. 채택 MA + 패치 뒤 노드 간 1층 Transformer — 동일 seed 재학습

각 시각의 `(2a+1)²` 지역 패치 인코딩 이후 `[B*k,N,D]`의 **전체 노드**를
토큰으로 하는 1층 `TransformerEncoder`를 추가했다. 인코딩 결과는 원래의
시간 LSTM에 넘기며 local 24시간 MA, daily/weekly, 3-way gate, 손실은
그대로다. `model.use_inter_node_transformer=true`만 채택 `ma`에서 바꾼다.
seed `245/6835/851`, 시간순 `0.70/0.15/0.15` 분할, MAE 손실,
Ulsan batch 24 / Porto batch 8, 최대 120 epoch와 기존 val MAE 선택·
early stopping을 그대로 사용했다. 모든 신규 학습은
`CUDA_VISIBLE_DEVICES=1`(물리 GPU 1)로 실행했다. 아래 `MAPE(+1)`은
`|오차|/(|실측|+1)`의 평균 ×100이고, ADFormer 원 MAPE와 동일한
정의라고 가정하지 않는다.

### Ulsan (14×12, D=16, seed별 paired 비교)

| seed | 설정 | MAE | RMSE | MAPE(+1) % | MAPE(0 제외) % |
|---:|---|---:|---:|---:|---:|
| 245 | 채택 MA | 0.322653 | 0.720627 | 15.530688 | 59.594836 |
| 245 | + 노드 간 1층 | 0.323000 | 0.723242 | 15.315415 | 60.216592 |
| 6835 | 채택 MA | 0.323046 | 0.724833 | 15.253790 | 60.468116 |
| 6835 | + 노드 간 1층 | 0.323152 | 0.725279 | 15.187903 | 60.280474 |
| 851 | 채택 MA | 0.323649 | 0.722001 | 15.518695 | 60.668484 |
| 851 | + 노드 간 1층 | 0.323136 | 0.722956 | 15.457522 | 60.592623 |
| **평균 ± 표본 sd** | **채택 MA** | **0.323116 ± 0.000502** | **0.722487 ± 0.002145** | **15.434391 ± 0.156520** | **60.243812 ± 0.570889** |
|  | **+ 노드 간 1층** | **0.323096 ± 0.000083** | **0.723826 ± 0.001267** | **15.320280 ± 0.134875** | **60.363230 ± 0.201212** |

seed별 `(변형/채택−1)×100%`의 평균: MAE **-0.006%** (개선 1/3),
RMSE **+0.186%** (개선 0/3), MAPE(+1) **-0.737%** (개선 3/3),
MAPE(0 제외) **+0.203%** (개선 2/3). 채택 지표인 RMSE가 모든
seed에서 악화하므로 Ulsan 기본값은 변경하지 않는다.
원자료: `output/experiments/inter_node_3seed_ulsan_seed{245,6835,851}.json`;
기준선: `output/experiments/ma_localmix_3seed_ulsan_seed{245,6835,851}.json`.
체크포인트는 `output/experiments/checkpoints/` 아래 JSON과 동명 디렉터리.

### Porto (10×20, D=64, seed별 paired 비교)

| seed | 설정 | MAE | RMSE | MAPE(+1) % | MAPE(0 제외) % |
|---:|---|---:|---:|---:|---:|
| 245 | 채택 MA | 0.489788 | 1.784967 | 15.520057 | 60.650364 |
| 245 | + 노드 간 1층 | 0.489433 | 1.770382 | 15.769723 | 60.050087 |
| 6835 | 채택 MA | 0.488421 | 1.750618 | 15.468143 | 59.894078 |
| 6835 | + 노드 간 1층 | 0.487151 | 1.774237 | 15.390452 | 60.343225 |
| 851 | 채택 MA | 0.489957 | 1.781163 | 15.618069 | 60.615568 |
| 851 | + 노드 간 1층 | 0.487663 | 1.779707 | 15.417640 | 61.243656 |
| **평균 ± 표본 sd** | **채택 MA** | **0.489389 ± 0.000843** | **1.772249 ± 0.018829** | **15.535423 ± 0.076135** | **60.386670 ± 0.426952** |
|  | **+ 노드 간 1층** | **0.488082 ± 0.001198** | **1.774775 ± 0.004685** | **15.525938 ± 0.211561** | **60.545656 ± 0.622001** |

paired 변화율 평균: MAE **-0.267%** (개선 3/3), RMSE **+0.150%**
(개선 2/3), MAPE(+1) **-0.059%** (개선 2/3), MAPE(0 제외)
**+0.265%** (개선 1/3). MAE는 3개 seed에서 개선됐지만 RMSE
평균과 MAPE(0 제외) 평균은 악화했다. 두 도시 모두 seed 3개만으로
통계적 유의성은 주장하지 않으며 채택 기본 설정을 유지한다.
원자료: `output/experiments/inter_node_3seed_porto_seed{245,6835,851}.json`;
기준선: `output/experiments/ma_localmix_3seed_porto_seed{245,6835,851}.json`.
체크포인트는 `output/experiments/checkpoints/` 아래 JSON과 동명 디렉터리.

6개 변형 체크포인트의 유효 모델 설정을 같은 seed의 채택 MA와
대조했다. 이전 체크포인트에 없던 `use_local_view=true`,
`use_local_mean=true`, `use_inter_node_transformer=false` 기본값을
적용하면 **새 1층 스위치 외에는 동일**하다. Ulsan seed 245의 변형
체크포인트를 별도 `test.py`로 복원해 test 656개 샘플에서 기록된
RMSE 0.7232, MAE 0.3230, MAPE(+1) 15.3154를 확인했다.
재현 예시(다른 도시·seed는 경로와 `--config-name`만 교체):

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n DA python train.py --config-name config_ulsan \
  seed=245 model.use_inter_node_transformer=true \
  hydra.run.dir=output/experiments/checkpoints/inter_node_3seed_ulsan_seed245 \
  run_json=output/experiments/inter_node_3seed_ulsan_seed245.json
```
