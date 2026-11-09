# PROJECT SUMMARY — 비교 모델 실험

`docs/crn.pdf`의 비교 모델(CRN 제외) STID, AGCRN, MTGNN, BigST, BiST, PDFormer, ADFormer를 우리 데이터·분할·지표로 다시 잰 결과다.
ADFormer는 기존 실험 기록(`git show tmp:docs/ADFORMER_REFERENCE.md`, 8시간 입력)을 그대로 옮겼고, 나머지 6개는 upstream 코드를 모델별 브랜치에 복사해 새로 학습했다.
MTGNN·BigST·BiST는 upstream 학습 코드가 수요 0을 결측으로 보고 loss에서 빼므로, 그 mask만 제거한 변형(‡, `<M>-nomask` 브랜치)을 주 결과로 쓴다(§4).
`OURS_*` 행은 채택 전 개선·축소 후보다(§7).

## 1. 공통 조건

| 항목 | 값 |
|---|---|
| 데이터 | `data/raw/{city}_temporal_grid.npy` 시간당 승차 수. 울산 T=4368, 14×12=168 셀 / Porto T=8760, 10×20=200 셀. 수요가 항상 0인 셀도 포함 |
| 분할 | 시간순 train/val/test = 0.70/0.15/0.15 (`train_end=int(0.7T)`, `val_end=int(0.85T)`) |
| 입력/예측 | 대상 시각 t의 입력은 [t−8, t), 1시간 뒤 1스텝 예측 |
| test 대상 | 울산 t=3712..4367 (656개), Porto t=7446..8759 (1314개) |
| 시드 | 245, 6835, 851, 5123, 535 (모델·도시마다 5회) |
| GPU | 물리 GPU 1만 사용 (`CUDA_VISIBLE_DEVICES=1`) |
| 평가 | 모든 모델의 원 단위 test 예측을 `gir-baselines/_common/gir_common.py`의 `evaluate_test` 하나로 평가. 반올림·클리핑 없음. repo가 출력한 지표는 교차 확인에만 사용 |

지표(모든 셀·모든 test 시각 평균, 낮을수록 좋음):

- `RMSE`, `MAE`: 전체 셀.
- `MAPE+1` = mean(|y−ŷ| / (|y|+1)) × 100. 우리 모델의 기본 MAPE와 같은 정의.
- `MAPE(y>0)` = y≠0 셀만의 mean(|y−ŷ| / y) × 100.
- `mMAE`, `mRMSE`: y>0 셀만의 MAE/RMSE. 참고용.

test 셀 중 0의 비율은 울산 75.0%, Porto 73.1%다. 그래서 학습 손실이 0을 빼는지에 따라 결과가 크게 달라진다(§4).
±는 5시드 표본 표준편차(ddof=1)다.

## 2. 결과 — 우리 test 전체 (울산 656, Porto 1314 시각)

학습 시간은 run당 평균 초이며, 모델마다 재는 방식이 다르다(§5). OURS의 학습 시간은 기록이 없어 `-`로 둔다.

- **굵은 값**: 열마다 OURS와 비교 모델 중 가장 좋은(낮은) 값이다. OURS_dW(이전 버전)와 `OURS_*` 후보(§7)는 순위에서 뺐다.
- OURS·후보 옆 괄호: 비교 모델 중 가장 좋은 값 대비 % = (값/최고 비교 모델−1)×100. 음수면 그만큼 낮고(좋고), 양수면 그만큼 높다(나쁘다).

### 울산

| 모델 | RMSE | MAE | MAPE+1 | MAPE(y>0) | mMAE | mRMSE | params | 학습 s |
|---|---|---|---|---|---|---|---:|---:|
| **OURS** | **0.7141 ± 0.0018** (−0.96%) | **0.3193 ± 0.0005** (−0.09%) | **15.21 ± 0.17** (−1.85%) | 58.96 ± 0.74 (+14.28%) | 1.0149 (+8.78%) | 1.3099 (+4.12%) | 171,063 | - |
| OURS_dW | 0.7124 ± 0.0017 | 0.3186 ± 0.0004 | 15.18 ± 0.07 | 58.34 ± 0.19 | 1.0103 | 1.3081 | 6,111,673 | - |
| OURS_SE (S + E) | 0.7123 ± 0.0004 (−1.21%) | 0.3187 ± 0.0001 (−0.28%) | 15.14 ± 0.03 (−2.29%) | 58.67 ± 0.09 (+13.72%) | 1.0139 (+8.68%) | 1.3092 (+4.07%) | 171,063 | - |
| OURS_tiny (축소 + S + E) | 0.7119 ± 0.0004 (−1.26%) | 0.3193 ± 0.0002 (−0.11%) | 15.20 ± 0.06 (−1.86%) | 58.56 ± 0.10 (+13.51%) | 1.0129 (+8.57%) | 1.3089 (+4.04%) | 27,159 | - |
| STID | 0.7252 ± 0.0013 | 0.3220 ± 0.0012 | 15.69 ± 0.19 | 58.07 ± 0.99 | 1.0012 | 1.3129 | 106,369 | 103 |
| AGCRN | 0.7298 ± 0.0015 | 0.3275 ± 0.0005 | 15.94 ± 0.08 | 58.67 ± 0.11 | 1.0172 | 1.3268 | 746,705 | 840 |
| MTGNN ‡ | 0.7328 ± 0.0020 | 0.3227 ± 0.0007 | 15.65 ± 0.11 | 56.83 ± 0.71 | 1.0007 | 1.3300 | 348,497 | 1,034 |
| BigST ‡ | 0.7210 ± 0.0019 | 0.3217 ± 0.0003 | 15.49 ± 0.18 | 58.35 ± 0.60 | 1.0097 | 1.3142 | 113,473 | 3,633 |
| BiST ‡ | 0.7259 ± 0.0033 | 0.3196 ± 0.0019 | 15.60 ± 0.13 | 56.08 ± 0.69 | 0.9857 | 1.3145 | 1,274,723 | 1,086 |
| PDFormer | 0.7240 ± 0.0019 | 0.3808 ± 0.0041 | 22.42 ± 0.43 | **51.59 ± 0.50** | **0.9329** | **1.2580** | 531,018 | 1,760 |

### Porto

| 모델 | RMSE | MAE | MAPE+1 | MAPE(y>0) | mMAE | mRMSE | params | 학습 s |
|---|---|---|---|---|---|---|---:|---:|
| **OURS** | **1.6115 ± 0.0315** (−1.79%) | 0.4866 ± 0.0004 (+0.03%) | **14.35 ± 0.16** (−5.44%) | 64.79 ± 0.60 (+22.00%) | 1.6726 (+8.12%) | **3.0672** (−1.24%) | 250,583 | - |
| OURS_dW | 1.5846 ± 0.0359 | 0.4818 ± 0.0007 | 14.36 ± 0.14 | 63.99 ± 0.29 | 1.6487 | 3.0122 | 3,728,473 | - |
| OURS_SE (S + E) | 1.5895 ± 0.0157 (−3.13%) | 0.4805 ± 0.0015 (−1.22%) | 14.10 ± 0.03 (−7.13%) | 64.02 ± 0.27 (+20.56%) | 1.6546 (+6.96%) | 3.0265 (−2.55%) | 250,583 | - |
| OURS_avgSE (평균 주기 + S + E) | 1.5573 ± 0.0212 (−5.09%) | 0.4797 ± 0.0008 (−1.39%) | 14.26 ± 0.04 (−6.05%) | 64.10 ± 0.20 (+20.69%) | 1.6456 (+6.38%) | 2.9605 (−4.67%) | 217,687 | - |
| OURS_cavgSE (축소 + 평균 주기 + S + E) | 1.5682 ± 0.0348 (−4.42%) | 0.4801 ± 0.0007 (−1.31%) | 14.28 ± 0.05 (−5.90%) | 64.06 ± 0.15 (+20.63%) | 1.6461 (+6.41%) | 2.9835 (−3.93%) | 78,551 | - |
| STID | 1.6682 ± 0.0082 | 0.4976 ± 0.0004 | 15.32 ± 0.08 | 59.76 ± 0.16 | 1.6498 | 3.1543 | 107,393 | 268 |
| AGCRN | 1.9332 ± 0.0136 | 0.5067 ± 0.0012 | 15.66 ± 0.09 | 60.31 ± 0.48 | 1.6732 | 3.6857 | 747,025 | 10,435 |
| MTGNN ‡ | 1.7507 ± 0.0483 | 0.4876 ± 0.0017 | 15.27 ± 0.21 | 57.35 ± 0.24 | 1.6005 | 3.3272 | 394,065 | 1,723 |
| BigST ‡ | 1.6545 ± 0.0333 | 0.4925 ± 0.0039 | 15.45 ± 0.13 | 59.98 ± 0.25 | 1.6259 | 3.1355 | 114,497 | 7,737 |
| BiST ‡ | 1.6408 ± 0.0341 | **0.4865 ± 0.0065** | 15.18 ± 0.16 | 58.40 ± 1.26 | 1.6057 | 3.1056 | 1,281,507 | 1,165 |
| PDFormer | 1.6561 ± 0.0334 | 0.5408 ± 0.0052 | 21.42 ± 0.59 | **53.11 ± 0.83** | **1.5469** | 3.1130 | 531,018 | 4,638 |

- OURS: 현재 채택 모델(`tmp` b7f2b44, ΔW 없음). OURS_dW: ΔW 시절 채택 모델(eb92db5). 두 모델 모두 같은 `evaluate_test`로 다시 평가했다.
- `OURS_*`: 채택 전 후보(§7, 브랜치 `night`). 같은 `evaluate_test`로 평가했다. 울산은 `OURS_SE`·`OURS_tiny`, Porto는 `OURS_SE`·`OURS_avgSE`·`OURS_cavgSE`만 돌렸다.
- ‡ 표시는 upstream 학습 코드의 0 mask만 제거한 변형이다(§4). mask를 둔 원본 결과는 §4에 보조로 둔다.
- OURS 괄호 %의 비교 기준(가장 좋은 비교 모델)은 다음과 같다.
  - 울산: RMSE BigST, MAE BiST, MAPE+1 BigST, MAPE(y>0)·mMAE·mRMSE PDFormer.
  - Porto: RMSE BiST, MAE BiST, MAPE+1 BiST, MAPE(y>0)·mMAE PDFormer, mRMSE BiST.
- Porto MAE는 BiST ‡ 0.4865가 OURS 0.4866보다 0.03% 낮다. 차이는 BiST의 시드 표준편차(0.0065)보다 훨씬 작다. 후보 `OURS_*`는 모두 BiST보다 1.2~1.4% 낮다.
- 0이 아닌 셀만 보는 지표(MAPE(y>0), mMAE, mRMSE)는 대부분 PDFormer가 가장 낮다. 대신 PDFormer는 전체 셀 MAE·MAPE+1이 가장 나쁘다.

## 3. ADFormer 비교 — ADFormer test와 같은 시각 (울산 654, Porto 1313)

ADFormer 기록은 test가 울산 654개, Porto 1313개다. 우리 test의 마지막 654/1313 시각과 같다: 저장된 ADFormer truth가 우리 데이터의 해당 시각과 일치함을 확인했다. 즉 ADFormer test에는 우리 test의 첫 2(울산)/1(Porto) 시각이 없다.
그래서 이 표의 다른 모델 값은 저장된 test 예측 중 같은 시각만 골라 다시 계산했다(`_common/summarize.py --adformer-subset`). 학습은 새로 하지 않았다.
굵은 값과 OURS 괄호 %의 규칙은 §2와 같고, 비교 모델에 ADFormer를 포함한다. 비교 기준은 울산 BigST·BiST·BigST, Porto BiST·BiST·BiST다.

| 모델 | 울산 RMSE | 울산 MAE | 울산 MAPE+1 | Porto RMSE | Porto MAE | Porto MAPE+1 |
|---|---|---|---|---|---|---|
| ADFormer (기록) | 0.7290 ± 0.0037 | 0.3251 ± 0.0011 | 15.5920 ± 0.2415 | 1.7542 ± 0.0419 | 0.4977 ± 0.0030 | 15.7666 ± 0.3293 |
| **OURS** | **0.7134 ± 0.0018** (−0.98%) | **0.3188 ± 0.0005** (−0.12%) | **15.1728 ± 0.1703** (−1.92%) | **1.6069 ± 0.0334** (−1.90%) | 0.4862 ± 0.0004 (+0.00%) | **14.3365 ± 0.1605** (−5.43%) |
| OURS_dW | 0.7118 ± 0.0017 | 0.3182 ± 0.0004 | 15.1448 ± 0.0667 | 1.5795 ± 0.0366 | 0.4814 ± 0.0007 | 14.3465 ± 0.1392 |
| OURS_SE | 0.7117 ± 0.0004 (−1.23%) | 0.3182 ± 0.0001 (−0.31%) | 15.1052 ± 0.0318 (−2.36%) | 1.5857 ± 0.0146 (−3.20%) | 0.4802 ± 0.0015 (−1.25%) | 14.0800 ± 0.0255 (−7.12%) |
| OURS_tiny | 0.7113 ± 0.0004 (−1.28%) | 0.3188 ± 0.0002 (−0.14%) | 15.1708 ± 0.0612 (−1.93%) | | | |
| OURS_avgSE | | | | 1.5551 ± 0.0210 (−5.06%) | 0.4795 ± 0.0008 (−1.39%) | 14.2463 ± 0.0438 (−6.03%) |
| OURS_cavgSE | | | | 1.5664 ± 0.0355 (−4.38%) | 0.4799 ± 0.0007 (−1.31%) | 14.2718 ± 0.0455 (−5.86%) |
| STID | 0.7249 ± 0.0013 | 0.3216 ± 0.0012 | 15.6704 ± 0.1889 | 1.6629 ± 0.0073 | 0.4972 ± 0.0005 | 15.2958 ± 0.0782 |
| AGCRN | 0.7292 ± 0.0014 | 0.3271 ± 0.0005 | 15.9145 ± 0.0833 | 1.9333 ± 0.0137 | 0.5066 ± 0.0012 | 15.6546 ± 0.0891 |
| MTGNN ‡ | 0.7321 ± 0.0021 | 0.3223 ± 0.0007 | 15.6213 ± 0.1086 | 1.7508 ± 0.0485 | 0.4875 ± 0.0017 | 15.2678 ± 0.2076 |
| BigST ‡ | 0.7205 ± 0.0020 | 0.3213 ± 0.0003 | 15.4698 ± 0.1815 | 1.6518 ± 0.0334 | 0.4922 ± 0.0039 | 15.4298 ± 0.1333 |
| BiST ‡ | 0.7254 ± 0.0032 | 0.3192 ± 0.0020 | 15.5730 ± 0.1307 | 1.6381 ± 0.0353 | **0.4862 ± 0.0064** | 15.1598 ± 0.1528 |
| PDFormer | 0.7234 ± 0.0018 | 0.3803 ± 0.0041 | 22.3916 ± 0.4297 | 1.6505 ± 0.0350 | 0.5403 ± 0.0052 | 21.3886 ± 0.5924 |

Porto MAE는 BiST ‡ 0.486230, OURS 0.486241로 소수 넷째 자리까지 같다(+0.002%).

OURS ÷ ADFormer − 1:

- 울산: RMSE −2.14%, MAE −1.94%, MAPE+1 −2.69%.
- Porto: RMSE −8.40%, MAE −2.31%, MAPE+1 −9.07%.

후보 ÷ ADFormer − 1:

- 울산 OURS_SE: RMSE −2.37%, MAE −2.11%, MAPE+1 −3.12%. OURS_tiny: −2.43%, −1.94%, −2.70%.
- Porto OURS_SE: RMSE −9.61%, MAE −3.52%, MAPE+1 −10.70%. OURS_avgSE: −11.35%, −3.66%, −9.64%. OURS_cavgSE: −10.71%, −3.58%, −9.48%.

ADFormer 행은 기록의 8시간 입력 값이다. ±는 기록 괄호 안의 표본 표준편차다.
같은 저장 예측(`../ADFormer/log/ADFormer/{ULSAN,PORTO}-Taxi/window8_seed*/predictions.npz`)의 origin 채널을 `gir_common.metrics`로 다시 계산하면 위 세 값이 소수 넷째 자리까지 같다.
같은 계산의 나머지 지표는 다음과 같다.

| 도시 | MAPE(y>0) | mMAE | mRMSE |
|---|---|---|---|
| 울산 | 59.51 ± 0.77 | 1.0265 ± 0.0134 | 1.3307 ± 0.0122 |
| Porto | 59.41 ± 0.80 | 1.6307 ± 0.0103 | 3.3340 ± 0.0813 |

### ADFormer 평가 조건 확인

upstream ADFormer(`decisionintelligence/ADFormer`, 커밋 `c8323fa`)의 기본 평가는 우리와 다르다.

- 지표 mask:
  - `mae`/`rmse`는 기본값 `mask=0.5`로 y≤0.5 셀을 뺀다.
  - `mape`는 기본값 `mask=5.0`으로 y≤5 셀을 빼고, 분모는 y다.
- 평가 대상: trainer가 출발(origin)과 도착(destination) 두 채널을 함께 평가한다.

기록 수치는 이 기본값으로 낸 것이 아니다. 로컬 `../ADFormer` 작업 트리는 지표·평가 부분만 다음과 같이 바꿨다(미커밋 diff).

- mask 기본값을 −1(전체 셀)로 바꿨다.
- MAPE 분모를 y+1로, 대상을 전체 셀로 바꿨다.
- trainer와 `scripts/run_seed_experiments.py`가 origin 채널만 평가한다.

따라서 기록 수치는 우리 지표와 정의가 같고, 위 재계산으로도 확인했다.
그 밖의 로컬 변경은 다음과 같다.

- 데이터셋 등록과 기간 설정.
- `val_rate` 0.15.
- 설정 기본 `batch_size` 24. 실험 스크립트는 `--batch_size 64`를 넘긴다.

남은 차이는 다음과 같다.

- 학습 손실이 origin과 destination 두 채널의 L1이다. 입력에도 도착 수요가 들어간다. 우리 모델과 다른 비교 모델은 승차(origin)만 쓴다.
- test가 우리보다 2/1 시각 적다. 이 차이는 위 subset 표로 맞췄다.
- 학습 시간(8h, 시드당)은 울산 348 s, Porto 1,090 s다.

## 4. 0 값 처리 (‡ 변형의 이유)

0은 결측이 아니라 실제 "수요 없음"이다.

| 모델 | 원본 학습 손실의 0 처리 | 조치 |
|---|---|---|
| STID | BasicTS masked MAE, `NULL_VAL` 설정 가능 | 데이터 설정 `NULL_VAL=NaN` → 0 포함 학습 |
| AGCRN | 일반 L1 | 그대로(0 포함) |
| PDFormer | Huber(δ=2), mask 없음 | 그대로(0 포함) |
| MTGNN | `trainer.py`: `masked_mae(..., 0.0)` 코드 고정 | 코드 수정 → ‡ |
| BigST | `trainer.py`: `masked_mae(..., 0.0)` 코드 고정 | 코드 수정 → ‡ |
| BiST | `engine.py`: 기본 `mask_value=0`, 배치 정답 최솟값이 1 미만이면 그 값을 mask | 코드 수정 → ‡ |

세 모델의 mask는 학습 loss와 검증 loss(best checkpoint 선택·early stop) 양쪽에 걸린다. 입력·정규화·샘플 구성에는 0을 빼는 코드가 없다.

### ‡ 변경 내용 (사용자 지시로 모델 코드 수정)

- MTGNN, BigST: 마스크 없는 `mae(preds, labels, null_val=None) = mean(|preds − labels|)`를 각 `util.py`/`metrics.py`에 추가하고, `trainer.py`의 `self.loss = …masked_mae`를 `…mae`로 바꿨다. 기존 호출의 세 번째 인자 `0.0`은 무시된다.
- BiST: `engine.py`의 학습·평가 두 곳에서 `mask_value = torch.tensor(0)` → `torch.tensor(float('nan'))`, `if label(s).min() < 1` → `< -1`.
  - `< -1`만 바꾸면 기본값 0이 남아 `labels != 0`으로 계속 0이 빠진다. 울산은 역변환한 0이 정확히 0.0이다(545,420개). 그래서 기본값도 NaN으로 바꿨고, upstream `masked_mae`는 NaN이면 `~isnan(labels)` 경로로 모든 셀을 쓴다.
- 학습 전 점검: 실제 train·val loader 배치에서 loss가 전체 셀 MAE와 같고 제외되는 정답이 0개임을 확인해 각 결과 JSON에 기록했다.
- 나머지 설정·epoch·시드는 원본 결과와 같다.
- BigST Porto는 다시 학습하지 않았다. 원본에서도 0이 5.96e−8로 역변환되어 mask가 아무 정답도 빼지 못했다(train·val 모두 0개 확인). 따라서 원본 loss가 mask 없는 MAE와 수치상 같다. 원본 Porto 결과 5개를 `BigST_nomask/`로 복사하고 JSON에 `reused_from`을 남겼다.

### mask를 둔 원본 결과 (보조)

| 모델 | 울산 RMSE / MAE / MAPE+1 | Porto RMSE / MAE / MAPE+1 |
|---|---|---|
| MTGNN | 1.0688 / 0.9518 / 83.25 | 1.9305 / 1.1077 / 81.73 |
| BigST | 1.0755 / 0.9611 / 84.16 | (‡와 같음) |
| BiST | 1.1083 / 0.9801 / 86.19 | 1.9048 / 1.1732 / 88.57 |
| STID `NULL_VAL=0` | 1.0878 / 0.9705 / 85.15 | 1.8439 / 1.1414 / 84.74 |

0을 빼고 학습하면 0 셀에도 양수를 예측해 전체 셀 지표가 크게 나빠진다. mask 제거 후 MAE는 울산 MTGNN 0.9518→0.3227, BigST 0.9611→0.3217, BiST 0.9801→0.3196이다.

## 5. 모델별 브랜치·변경 내역

각 브랜치는 `skeleton`에서 만들었다. ‡ 변형은 원본 브랜치 HEAD에서 `<M>-nomask` 브랜치를 새로 만들었다(worktree `/home/jinsu/PycharmProjects/gir-baselines/<M>-nomask`).

- 첫 커밋: upstream 파일을 `models/<M>/`에 그대로 복사했다. 아래 표의 upstream 커밋을 새로 clone한 것과 `diff -r`로 비교해 동일함을 확인했다.
- 이후 커밋: 데이터 변환(`dataset_frame/convert_gir.py`), 실행 runner, 도시별 설정, `docs/RESULTS.md`·`docs/UPSTREAM.md`를 추가했다.
- `dataset_frame/gir_common.py`는 모든 브랜치에서 공용 원본과 바이트 단위로 같다.
- worktree는 `/home/jinsu/PycharmProjects/gir-baselines/<M>`이다.
- 결과는 `output/baselines/<M>/{city}_seed{s}.json` + `_test_pred.npz`에 있다. ‡ 변형은 `output/baselines/<M>_nomask/`.
- 10개 run 모두 대상 시각을 확인했고, npz에서 지표를 다시 계산해 JSON과 일치함을 확인했다(`_common/verify.py`).

| 모델 | upstream (커밋) | 브랜치 첫 커밋 → HEAD | `models/<M>` 변경 | 기준 설정 |
|---|---|---|---|---|
| STID | zezhishao/STID `e8b313b` | `7c91c0d` → `82b17f2` | 추가만: `stid/ULSAN.py`, `stid/PORTO.py` | PEMS04 |
| AGCRN | LeiBAI/AGCRN `7fbbf2a` | `fcae678` → `39cfe6f` | 추가: 도시별 `.conf`. 수정: `lib/load_dataset.py`(데이터셋 등록), `lib/dataloader.py`(GIR 데이터일 때만: 분할 후 train 구간으로 scaler fit, val loader `drop_last=False`) | PEMSD4 |
| MTGNN | nnzhan/MTGNN `f811746` | `164d23f` → `8d2f709` | 없음 | METR-LA multi-step |
| MTGNN ‡ | 〃 | `MTGNN-nomask` → `4cb2744` | `util.py`(+`mae`), `trainer.py`(loss) | 〃 |
| BigST | usail-hkust/BigST `830cb52` | `655b640` → `d1f1a33` | 없음 | README short-term |
| BigST ‡ | 〃 | `BigST-nomask` → `d249c2c` | `metrics.py`(+`mae`), `trainer.py`(loss) | 〃 |
| BiST | PoorOtterBob/BiST `dd94adf` | `c4b6d08` → `4cf4c67` | 없음 | PeMS08 |
| BiST ‡ | 〃 | `BiST-nomask` → `136aa3c` | `src/base/engine.py`(mask 2곳) | 〃 |
| PDFormer | BUAABIGSCity/PDFormer `f8c8f6a` | `193982b` → `ad23245` | 추가만: `gir_ulsan.json`, `gir_porto.json` | PeMS04 |

AGCRN은 데이터 로딩 코드 두 파일을 바꿨다. 원본 loader는 정규화를 전체 데이터에 먼저 적용해 val/test 통계가 새고, val의 마지막 불완전 batch를 버린다. 그래서 GIR 데이터셋일 때만 분기를 추가했다. 모델·손실·optimizer·학습 루프는 그대로다. ‡ 세 모델은 loss mask만 바꿨다(§4).

### 설정에서 바꾼 점·주의

- STID: PEMS04 기본값 100 epoch를 유지했다. 하루 step 수만 288→24로 바꿨다. 학습 시간은 학습 루프만 잰 값이다.
- AGCRN:
  - 울산은 기본 최대 100 epoch이고 patience는 15다.
  - Porto는 최대 epoch를 60으로 제한했다. 100 epoch 시도에서 60분에 17 epoch만 진행했기 때문이다.
  - Porto 5시드 모두 epoch 44~56에서 early stop했다. best epoch는 29~41이었고 상한에 닿지 않았다.
  - train loader의 `drop_last=True`는 원본 그대로다.
- MTGNN:
  - upstream은 early stop 없이 100 epoch를 모두 돌린 뒤 best-val checkpoint를 쓴다.
  - adapter는 val/test 마지막 batch가 반복 샘플로 채워지지 않게 batch iterator만 교체했다.
  - 8-이웃 격자 adjacency는 필수 입력이다. 기본값 `buildA_true=True`로 실제 학습에는 adaptive graph를 쓴다.
  - ‡ best epoch: 울산 68~87, Porto 75~94.
  - ‡ 4개 run(울산 851, Porto 245·851·5123)은 upstream이 소수 넷째 자리로 출력한 test 지표와 재계산 값 차이가 1e-4를 약간 넘었다(최대 2.6e-4). 출력 반올림 범위이고 최종 지표는 `evaluate_test` 값이다.
- BigST:
  - 기본값은 최대 100 epoch, patience 30이다.
  - 과거 8시간을 시간 축이 아니라 한 시점의 채널 24개(8개 수요 + TOD/DOW + 0 패딩)로 묶어 넣는다. 원본 모델이 출력 길이로 입력 폭을 정하기 때문이다.
  - ‡ best epoch: 울산 90~100, Porto 85~100. 울산 5123·Porto 5123은 상한(100)에서 best였다.
  - 처음 15 epoch로 줄여 돌린 결과는 과소학습이었다(best epoch 대부분 14~15). `output/baselines/BigST_ep15/`에 보조로 남겼다.
- BiST:
  - upstream `main.py`의 `get_config` 재귀 문제 때문에, runner가 원본 클래스(`BiST`, `BaseEngine`, loss, optimizer)를 직접 조립했다.
  - diffusion 반복 수는 parser 기본 2 대신 논문 short-term 값 J=4를 썼다.
  - 모든 train 표본을 쓰도록 train loader 마지막 batch를 마지막 표본 반복으로 채웠다(울산 23개, Porto 20개).
  - ‡ best epoch: 울산 104~298, Porto 37~218 (최대 500, patience 20).
- PDFormer:
  - 울산은 기본 최대 200 epoch다.
  - Porto는 최대 120 epoch로 제한했다(200 epoch 시도가 약 90분에 근접). Porto 5시드 모두 93~108 epoch에서 early stop했고, best epoch는 42~57이었다.
  - upstream PyTorch 1.10.1이 RTX 4090에서 동작하지 않아 PyTorch 2.0.1 환경을 썼다.

## 6. 재현

```bash
# 평균표 (§2)
python /home/jinsu/PycharmProjects/gir-baselines/_common/summarize.py OURS OURS_dW OURS_SE OURS_tiny OURS_avgSE OURS_cavgSE STID AGCRN MTGNN_nomask BigST_nomask BiST_nomask PDFormer
# ADFormer test 시각 subset (§3)
conda run -n DA python /home/jinsu/PycharmProjects/gir-baselines/_common/summarize.py --adformer-subset OURS OURS_dW OURS_SE OURS_tiny OURS_avgSE OURS_cavgSE STID AGCRN MTGNN_nomask BigST_nomask BiST_nomask PDFormer
# 대상 시각·지표 재계산·upstream 변경 확인
conda run -n DA python /home/jinsu/PycharmProjects/gir-baselines/_common/verify.py STID AGCRN MTGNN BigST BiST PDFormer MTGNN_nomask BigST_nomask BiST_nomask OURS OURS_dW OURS_SE OURS_tiny OURS_avgSE OURS_cavgSE
# 후보 평가 (§7, cwd = night 브랜치 worktree gir-cap, 예: OURS_SE 울산)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 conda run -n DA python /home/jinsu/PycharmProjects/gir-baselines/_common/eval_ours.py \
  OURS_SE ulsan "output/merged/night/checkpoints/ulsan_sinkc_ema1_s{s}"
```

모델별 상세(시드별 표, repo 지표 교차 확인, 실행 환경)는 각 브랜치의 `docs/RESULTS.md`에 있다.

## 7. 우리 모델 개선·축소 후보 (10/11)

ΔW 없는 현재 모델에서 ① 파라미터를 줄이면서 성능 유지 ② 성능 향상을 5시드로 탐색했다(46개 설정, 234회 학습).
코드는 브랜치 `night`(worktree `gir-cap`, 옵션 커밋 `96d8fec`, SPEC `90c2c80`·`a1c2f93`)에 있고, 기본값은 바꾸지 않았다.
기록은 `output/merged/night/`, 전체 표·그래프는 Notion 기록 페이지 "파라미터 축소·성능 개선 야간 실험" 절과 `night` 브랜치 `docs/SPEC.md` §4.1에 있다.

새 옵션(둘 다 파라미터 0):

- S `model.sink_carry=true`: 이전 층 sink 출력을 다음 층 sink 입력에 더한다. sink는 모든 노드를 보므로 2층 노드가 격자 전체 요약을 받는다.
- E `ema_epochs=1`: 파라미터 EMA(시간 상수 1 epoch). 평가·best 선택·저장·test에 EMA 가중치를 쓴다.
- 평균 주기: `model.periodic_encoder=average`. 일·주 LSTM 대신 유효 lag 평균 → MLP.

| 후보 | 도시 | 설정 | 파라미터 |
|---|---|---|---:|
| OURS_SE | 울산·Porto | 채택 크기 + S + E | 171,063 / 250,583 |
| OURS_tiny | 울산 | history·주기 hidden 16, fusion 32 + S + E | 27,159 (−84%) |
| OURS_avgSE | Porto | 평균 주기 + S + E | 217,687 (−13%) |
| OURS_cavgSE | Porto | d_model 32, history 32, fusion 64 + 평균 주기 + S + E | 78,551 (−69%) |

같은 코드로 다시 학습한 채택 크기 기준과 시드별로 짝지어 비교한 변화율(RMSE / MAE / MAPE+1, 괄호는 개선 시드 수):

- 울산 OURS_SE −0.25%(5/5) / −0.19% / −0.44%, OURS_tiny −0.30% / −0.02% / +0.00%.
- Porto OURS_SE −0.32% / −1.25%(5/5) / −1.30%, OURS_avgSE −2.35%(5/5) / −1.42%(5/5) / −0.15%, OURS_cavgSE −1.67% / −1.34%(5/5) / +0.01%.
- Porto 재학습 기준은 RMSE 1.5949로 채택 OURS(1.6115, ΔW 시절 stage 1 체크포인트 변환)보다 낮다. 그래서 §2 표의 OURS 대비 차이는 위 짝 비교보다 크게 보인다.
- 울산은 λ(MAPE 가중치)로 RMSE↔MAPE가 교환된다(λ 1.5: −0.91% / +3.46%, λ 3: +1.26% / −3.65%). 같은 MAPE로 환산한 RMSE 이득은 0.2~0.4%로 작다.
- 효과 없음: Porto 크기 키우기(d_model 96, history 128은 3시드), dropout, 도시 전체 수요 context, Porto global node 114.
