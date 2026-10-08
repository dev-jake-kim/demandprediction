# PROJECT SUMMARY — 비교 모델 실험

`docs/crn.pdf`의 비교 모델(CRN 제외) STID, AGCRN, MTGNN, BigST, BiST, PDFormer, ADFormer를 우리 데이터·분할·지표로 다시 잰 결과다.
ADFormer는 기존 실험 기록(`git show tmp:docs/ADFORMER_REFERENCE.md`, 8시간 입력)을 그대로 옮겼고, 나머지 6개는 upstream 코드를 모델별 브랜치에 복사해 새로 학습했다.
MTGNN·BigST·BiST는 upstream 학습 코드가 수요 0을 결측으로 보고 loss에서 빼므로, 그 mask만 제거한 변형(‡, `<M>-nomask` 브랜치)을 주 결과로 쓴다(§4).

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

- **굵은 값**: 열마다 OURS와 비교 모델 중 가장 좋은(낮은) 값이다. OURS_dW는 우리 모델의 이전 버전이라 순위에서 뺐다.
- OURS 옆 괄호: 비교 모델 중 가장 좋은 값 대비 % = (OURS/최고 비교 모델−1)×100. 음수면 OURS가 그만큼 낮고(좋고), 양수면 그만큼 높다(나쁘다).

### 울산

| 모델 | RMSE | MAE | MAPE+1 | MAPE(y>0) | mMAE | mRMSE | params | 학습 s |
|---|---|---|---|---|---|---|---:|---:|
| **OURS** | **0.7141 ± 0.0018** (−0.96%) | **0.3193 ± 0.0005** (−0.09%) | **15.21 ± 0.17** (−1.85%) | 58.96 ± 0.74 (+14.28%) | 1.0149 (+8.78%) | 1.3099 (+4.12%) | 171,063 | - |
| OURS_dW | 0.7124 ± 0.0017 | 0.3186 ± 0.0004 | 15.18 ± 0.07 | 58.34 ± 0.19 | 1.0103 | 1.3081 | 6,111,673 | - |
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
| STID | 1.6682 ± 0.0082 | 0.4976 ± 0.0004 | 15.32 ± 0.08 | 59.76 ± 0.16 | 1.6498 | 3.1543 | 107,393 | 268 |
| AGCRN | 1.9332 ± 0.0136 | 0.5067 ± 0.0012 | 15.66 ± 0.09 | 60.31 ± 0.48 | 1.6732 | 3.6857 | 747,025 | 10,435 |
| MTGNN ‡ | 1.7507 ± 0.0483 | 0.4876 ± 0.0017 | 15.27 ± 0.21 | 57.35 ± 0.24 | 1.6005 | 3.3272 | 394,065 | 1,723 |
| BigST ‡ | 1.6545 ± 0.0333 | 0.4925 ± 0.0039 | 15.45 ± 0.13 | 59.98 ± 0.25 | 1.6259 | 3.1355 | 114,497 | 7,737 |
| BiST ‡ | 1.6408 ± 0.0341 | **0.4865 ± 0.0065** | 15.18 ± 0.16 | 58.40 ± 1.26 | 1.6057 | 3.1056 | 1,281,507 | 1,165 |
| PDFormer | 1.6561 ± 0.0334 | 0.5408 ± 0.0052 | 21.42 ± 0.59 | **53.11 ± 0.83** | **1.5469** | 3.1130 | 531,018 | 4,638 |

- OURS: 현재 채택 모델(`tmp` b7f2b44, ΔW 없음). OURS_dW: ΔW 시절 채택 모델(eb92db5). 두 모델 모두 같은 `evaluate_test`로 다시 평가했다.
- ‡ 표시는 upstream 학습 코드의 0 mask만 제거한 변형이다(§4). mask를 둔 원본 결과는 §4에 보조로 둔다.
- OURS 괄호 %의 비교 기준(가장 좋은 비교 모델)은 다음과 같다.
  - 울산: RMSE BigST, MAE BiST, MAPE+1 BigST, MAPE(y>0)·mMAE·mRMSE PDFormer.
  - Porto: RMSE BiST, MAE BiST, MAPE+1 BiST, MAPE(y>0)·mMAE PDFormer, mRMSE BiST.
- Porto MAE는 BiST ‡ 0.4865가 OURS 0.4866보다 0.03% 낮다. 차이는 BiST의 시드 표준편차(0.0065)보다 훨씬 작다.
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
python /home/jinsu/PycharmProjects/gir-baselines/_common/summarize.py OURS OURS_dW STID AGCRN MTGNN_nomask BigST_nomask BiST_nomask PDFormer
# ADFormer test 시각 subset (§3)
conda run -n DA python /home/jinsu/PycharmProjects/gir-baselines/_common/summarize.py --adformer-subset OURS OURS_dW STID AGCRN MTGNN_nomask BigST_nomask BiST_nomask PDFormer
# 대상 시각·지표 재계산·upstream 변경 확인
conda run -n DA python /home/jinsu/PycharmProjects/gir-baselines/_common/verify.py STID AGCRN MTGNN BigST BiST PDFormer MTGNN_nomask BigST_nomask BiST_nomask OURS OURS_dW
```

모델별 상세(시드별 표, repo 지표 교차 확인, 실행 환경)는 각 브랜치의 `docs/RESULTS.md`에 있다.
