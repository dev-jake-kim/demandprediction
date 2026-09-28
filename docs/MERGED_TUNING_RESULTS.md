# merged 모델 하이퍼파라미터 튜닝 스윕

> **⚠️ porto 결과는 무효다 (2026-09-15).**
>
> porto 런이 잘못된 격자(19×20 = 380노드)로 돌았다. porto의 정본 격자는 **10×20 = 200노드**다
> (`preprocessing/porto/create_graph.py`가 생성하는 `porto_temporal_grid.npy`, `landuse_grid.npy`와
> 같은 shape). 2026-09-10에 `data/raw/porto_temporal_grid.npy`를 복구할 때 전처리의 **중간
> 산출물**(`temporal_grid.npy`, 19×20, 8/19 14:52)을 최종 출력물(`porto_temporal_grid.npy`,
> 10×20, 8/19 15:57) 대신 가져다 쓴 것이 원인이다.
>
> 그 결과 아래 **porto 절의 모든 수치와 판정(`history128` 채택)은 근거를 잃었다.**
> 해당 run JSON과 체크포인트는 삭제했고, `configs/model/merged_porto.yaml`의
> `history_hidden`은 128에서 64(포팅 전 merged_model이 200노드 격자에서 쓰던 값)로 되돌렸다.
> **porto 튜닝 스윕은 재실행이 필요하다.**
>
> **ulsan 결과는 유효하다** — ulsan 격자(14×12 = 168노드)는 바뀐 적이 없다.

`configs/model/merged_ulsan.yaml` / `configs/model/merged_porto.yaml`가 왜 그 값인지의 근거.
baseline(`d_model=64, transformer_ffn=128, num_fourier_bands=8, transformer_layers=2,
transformer_heads=4, history_hidden=64, periodic_hidden=64`)에서 6개 파라미터를 각각 하나씩
키운 변종(one-factor-at-a-time) + baseline, 도시(ulsan/porto) × 시드 245 1회씩, 총 14회.
`transformer_ffn`은 항상 `d_model`의 2배로 유지. loss는 config 기본값(`combined`).

원자료: `output/merged/runs/tune_<city>_<variant>_seed245.json`.

## 변종 정의

| 변종 | 오버라이드 |
|---|---|
| d_model128 | `model.d_model=128 model.transformer_ffn=256` |
| fourier16 | `model.num_fourier_bands=16` |
| layers3 | `model.transformer_layers=3` |
| heads8 | `model.transformer_heads=8` |
| history128 | `model.history_hidden=128` |
| periodic128 | `model.periodic_hidden=128` |

## 결과 — RMSE, MAPE(+1) 기준 (선택 기준으로 이 두 지표만 사용)

### ulsan

| variant | RMSE | Δ | MAPE(+1) | Δ |
|---|---|---|---|---|
| **baseline** | **0.7536** | — | **15.401** | — |
| d_model128 | 0.7527 | -0.1% | 15.618 | +1.4% |
| fourier16 | 0.7550 | +0.2% | 15.622 | +1.4% |
| layers3 | 0.7600 | +0.8% | 15.216 | -1.2% |
| heads8 | 0.7679 | +1.9% | 15.371 | -0.2% |
| history128 | 0.7531 | -0.1% | 15.555 | +1.0% |
| periodic128 | 0.7591 | +0.7% | 15.275 | -0.8% |

### porto — ⚠️ 무효 (위 경고 참고. 잘못된 380노드 격자로 측정됨)

| variant | RMSE | Δ | MAPE(+1) | Δ |
|---|---|---|---|---|
| **baseline** | **1.1227** | — | **7.997** | — |
| d_model128 | 1.2157 | +8.3% | 8.357 | +4.5% |
| fourier16 | 1.1510 | +2.5% | 8.040 | +0.5% |
| layers3 | 1.1242 | +0.1% | 8.450 | +5.7% |
| heads8 | 1.1134 | -0.8% | 8.129 | +1.6% |
| history128 | 1.1045 | -1.6% | 7.912 | -1.1% |
| periodic128 | 1.0774 | -4.0% | 8.114 | +1.5% |

## 판정

RMSE와 MAPE(+1)가 baseline 대비 **둘 다** 개선된 변종만 채택 대상으로 본다.

- **ulsan**: 12개 셀(6변종) 중 둘 다 개선인 경우가 하나도 없다 — 매 변종에서 RMSE와
  MAPE(+1)가 반대 방향으로 움직인다(예: d_model128은 RMSE만, layers3는 MAPE(+1)만 개선).
  하나를 골라 트레이드오프를 감수할 근거가 약해 **baseline을 그대로 채택**한다
  (`configs/model/merged_ulsan.yaml`).
- **porto**: ⚠️ **아래 판정은 무효다.** `history128`(`history_hidden=128`)이 유일하게 RMSE(1.1227→1.1045)와
  MAPE(+1)(7.997→7.912) 둘 다 개선했다. `periodic128`이 RMSE 개선폭은 더 크지만(-4.0%)
  MAPE(+1)은 baseline보다 나빠져(+1.5%) 채택 기준을 통과하지 못한다. **`history128`을
  채택**한다(`configs/model/merged_porto.yaml`).

## 제약사항 (읽기 전 확인)

- **시드 1개**. 대부분의 차이는 1~2%p대라 이 저장소에서 관찰된 시드 재현성 노이즈 범위 안에
  들어올 수 있다. `d_model128`의 porto RMSE 악화(+8.3%)만 노이즈치고 커서 상대적으로 신뢰도가
  높고, 나머지(특히 porto에서 채택한 `history128`의 -1.6%/-1.1%)는 방향성 참고 수준이다.
  여러 시드로 재확인 전까지는 확정적인 결론이 아니다.
- **one-factor-at-a-time**만 시험했다 — 여러 파라미터를 동시에 바꾼 조합(예: history128 +
  periodic128)은 시도하지 않았다.
- **porto는 batch_size=4로 고정**해서 돌렸다(원본 8이 아님) — `configs/config_porto.yaml`의
  `train.per_device_train_batch_size`/`per_device_eval_batch_size` 주석 참고. porto
  (19×20=380노드)는 `batch_size * time_step * nodes`가 커서 PyTorch의 memory-efficient
  SDPA 백엔드가 dropout seed/offset을 못 만드는 한계(65535)를 넘는다(`train.py`가 자동으로
  7까지는 낮추지만, `d_model128`처럼 activation이 큰 변종은 그래도 OOM이 나서 4로 낮췄다).
  이 배치 크기 차이 자체가 ulsan과 porto의 비교를 일부 교란할 수 있다.

## 이전 `tmp` 설정 이식 검증 — 단일-stage, 80/10/10 (기록 보존)

위 단일 시드 스윕과 **별개의 실험**이다. 위의 porto 380노드 결과는 계속 무효지만, 이 절의
porto 실험은 정상 10×20 = 200노드 격자를 사용한다. `tmp`의 retrieval, daily/weekly
LSTM, branch attention, 마스크와 입력 경로를 유지한 채, `tmp-extracted`에서 확인한
학습 설정만 적용했다. 물리 GPU 1만 사용했다 (`CUDA_VISIBLE_DEVICES=1`이므로 결과 JSON의
`cuda:0`은 마스킹된 프로세스 안의 인덱스다).

- 시드: 245, 6835, 851. 도시별 시간 순서 80% 학습, 10% 검증, 10% 테스트;
  retrieval 후보는 학습 시점 이전으로 제한. 단일 단계 학습.
- 공통: AdamW, 최대 120 epoch, lr=0.001, weight_decay=0.05,
  epoch 기준 warmup 5회 (1e-6 → 0.001) + cosine 60회 (0.001 → 0.0001),
  이후 0.0001 고정; 검증 loss 최적 checkpoint, patience=20, min_epochs=0.
  FFN/출력 dropout=0.1, attention-weight dropout=0으로 분리.
- Ulsan: `d_model=16`, batch=24, MAE 목적함수. 이전 `tmp` MAE 기록은
  `output/merged_model/runs/ulsan_mae_seed{seed}.json` (이식 전 구현·설정).
- Porto: `d_model=64`, batch=8, combined 목적함수 유지. 이전 `tmp` combined 기록은
  `output/merged_model/runs/porto_combined_seed{seed}.json` (이식 전 구현·설정).
  이전 기록과의 비교는 설정 묶음 전체에 대한 관찰이며, 개별 설정의 인과효과를 분리하지 않는다.

### Ulsan — 테스트 지표

| seed | 이전 RMSE | 이식 RMSE | 이전 MAE | 이식 MAE | 이전 MAPE(+1) % | 이식 MAPE(+1) % | 이전 MAPE(0 제외) % | 이식 MAPE(0 제외) % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 245 | 0.727543 | 0.726250 | 0.319084 | 0.315965 | 15.088405 | 15.068445 | 57.739176 | 56.033170 |
| 6835 | 0.723758 | 0.726076 | 0.314786 | 0.317404 | 15.072553 | 15.387902 | 56.716873 | 55.356035 |
| 851 | 0.728355 | 0.727865 | 0.318192 | 0.316605 | 14.981227 | 15.212362 | 57.004033 | 55.756410 |
| 평균 ± 표본 SD | 0.726552 ± 0.002453 | 0.726730 ± 0.000986 | 0.317354 ± 0.002268 | 0.316658 ± 0.000721 | 15.047395 ± 0.057849 | 15.222903 ± 0.159989 | 57.153361 ± 0.527257 | 55.715205 ± 0.340443 |

이식 후 RMSE 평균은 +0.025%, MAE는 -0.215%, MAPE(+1)는 +1.168%,
MAPE(0 제외)는 -2.514% (각 시드별 상대 변화율의 평균; 낮을수록 좋음).
RMSE·MAE가 함께 개선된 시드는 245와 851, 6835는 둘 다 악화했다.
선정 epoch는 각각 40, 56, 41. 새 결과 JSON은
`output/experiments/tmp_settings_3seed_ulsan_seed{seed}.json`,
체크포인트는 `output/experiments/checkpoints/tmp_settings_3seed_ulsan_seed{seed}/`.

### Porto — 테스트 지표 (정상 200노드)

| seed | 이전 RMSE | 이식 RMSE | 이전 MAE | 이식 MAE | 이전 MAPE(+1) % | 이식 MAPE(+1) % | 이전 MAPE(0 제외) % | 이식 MAPE(0 제외) % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 245 | 1.492524 | 1.641669 | 0.497310 | 0.501371 | 15.399100 | 14.858440 | 60.184226 | 61.623574 |
| 6835 | 1.537473 | 1.662438 | 0.496737 | 0.502717 | 15.159460 | 14.701127 | 61.446440 | 61.260892 |
| 851 | 1.480463 | 1.682719 | 0.494190 | 0.497362 | 14.897717 | 14.925937 | 61.061819 | 61.055477 |
| 평균 ± 표본 SD | 1.503487 ± 0.030044 | 1.662275 ± 0.020525 | 0.496079 ± 0.001661 | 0.500484 ± 0.002786 | 15.152092 ± 0.250773 | 14.828501 ± 0.115357 | 60.897495 ± 0.646953 | 61.313314 ± 0.287653 |

이식 후 시드별 상대 변화율 평균: RMSE +10.594%, MAE +0.888%,
MAPE(+1) -2.115%, MAPE(0 제외) +0.693%. RMSE·MAE는 **세 시드 모두 악화**했다.
MAPE(+1) 개선만으로 이 설정 묶음을 Porto의 성능 개선으로 판정할 수 없다.
선정 epoch는 각각 33, 21, 28. 새 결과 JSON은
`output/experiments/tmp_settings_3seed_porto_seed{seed}.json`,
체크포인트는 `output/experiments/checkpoints/tmp_settings_3seed_porto_seed{seed}/`.

이전 실험 재현: 현재 기본값은 2-stage·70/15/15로 변경됐으므로, 당시 조건을
**명시적으로** 덮어쓴다. 두 도시·시드를 순차 실행한다.

```bash
for city in ulsan porto; do
  for seed in 245 6835 851; do
    CUDA_VISIBLE_DEVICES=1 conda run -n DA python train.py \
      --config-name "config_${city}" "seed=${seed}" \
      dataset.train_ratio=0.80 dataset.val_ratio=0.10 model.node_adaptive=false \
      "hydra.run.dir=output/experiments/checkpoints/tmp_settings_3seed_${city}_seed${seed}" \
      "run_json=output/experiments/tmp_settings_3seed_${city}_seed${seed}.json"
  done
done
```

기존 기록은 포팅 전 `merged_model`, 신규 기록은 `merged` 구현에서 산출되었고
두 도시 모두 구조는 `tmp` 계열이지만 구현·학습 설정이 동일한 완전 통제 비교는 아니다.
검증 loss로 선택한 checkpoint의 테스트 RMSE를 보고한 것이며, RMSE를
선정 지표로 재선택하지 않았다. 세 시드의 관찰을 다른 데이터/시드로 일반화할 수 없다.

## 요청 실험 — `node_adaptive=true`, 시간순 70/15/15, 2-stage

앞 절의 **단일-stage 80/10/10과 다른 테스트 기간**이므로 그 수치를 이 절의
개선율 기준선으로 쓰지 않는다. 동일 시드의 stage 1과 stage 2만 같은 테스트
기간에서 짝지어 비교한다. 시드는 245, 6835, 851; `stage2.init_from=null`로
각 시드의 stage 1부터 새로 학습했다. 물리 GPU 1만 사용했다
(`CUDA_VISIBLE_DEVICES=1` 이후 결과의 `cuda:0`은 프로세스 안의 인덱스).

- 두 도시 모두 train/validation/test는 각각 70%/15%/15%. 학습 구간에서만 날씨
  정규화 통계와 평균 수요 > 0.8인 ΔW 대상 노드를 계산한다. 검색은
  `observed_past` (`tau < target_time`).
- ΔW는 stage 1에서 0으로 고정하고, 검증 손실이 가장 낮은 공유 가중치에서
  stage 2의 ΔW를 해제한다. 두 stage에서 optimizer와 스케줄러를 새로 만든다.
  stage 1은 Ulsan MAE / Porto combined, stage 2는 공통
  `10 * RMSE + MAPE(+1)` 학습 손실 및 전체 검증 예측의 동일 목적함수로
  best checkpoint를 고른다.
- 두 stage 모두 AdamW·weight_decay=0.05, 최대 120 epoch, patience=20,
  warmup 5 epoch (1e-6 → stage LR) + cosine 60 epoch (stage LR → 1e-4).
  stage 1 lr=1e-3, stage 2 lr=1e-4 (warmup 후 1e-4 유지).
  Ulsan `d_model=16`/batch=24, Porto `d_model=64`/batch=8;
  출력/FFN dropout=0.1, attention-weight dropout=0.

### Ulsan — 정상 168노드 (train_end=3057, val_end=3712)

학습/검증/테스트 샘플 3033/655/656개. 70% 학습 구간에서 45개 노드의
ΔW(1,105,920개 파라미터)를 선택했다. 아래는 각 stage가 검증 기준으로
고른 checkpoint의 **동일 테스트 기간** 지표다.

| seed | stage | RMSE | MAE | MAPE(+1) % | MAPE(0 제외) % | 선정 epoch |
|---:|---|---:|---:|---:|---:|---:|
| 245 | 1 | 0.724387 | 0.316491 | 15.153774 | 54.012018 | 39 |
| 245 | 2 | 0.781254 | 0.330363 | 13.352481 | 64.909479 | 9 |
| 6835 | 1 | 0.725285 | 0.316977 | 15.240681 | 55.366618 | 40 |
| 6835 | 2 | 0.769523 | 0.327366 | 13.578789 | 63.479043 | 8 |
| 851 | 1 | 0.722826 | 0.316650 | 15.288062 | 55.240893 | 37 |
| 851 | 2 | 0.780994 | 0.330757 | 13.433522 | 65.622414 | 10 |
| 평균 ± 표본 SD | 1 | 0.724166 ± 0.001244 | 0.316706 ± 0.000247 | 15.227505 ± 0.068107 | 54.873176 ± 0.748430 | — |
| 평균 ± 표본 SD | 2 | 0.777257 ± 0.006699 | 0.329496 ± 0.001855 | 13.454931 ± 0.114663 | 64.670312 ± 1.091517 | — |

동일 시드 stage 1→2 상대 변화율 평균: RMSE **+7.332%**, MAE **+4.039%**,
MAPE(+1) **-11.641%**, MAPE(0 제외) **+17.874%**.
stage 2가 MAPE(+1)를 세 시드 모두 낮췄지만 다른 세 지표는 모두 악화했다.
원자료: `output/experiments/tmp_two_stage_701515_ulsan_seed{seed}.json`,
최종 체크포인트: `output/experiments/checkpoints/tmp_two_stage_701515_ulsan_seed{seed}/`.

### Porto — 정상 200노드 (train_end=6132, val_end=7446)

학습/검증/테스트 샘플 6108/1314/1314개. 70% 학습 구간에서 52개 노드의
ΔW(1,916,928개 파라미터)를 선택했다. stage 1/2는 같은 테스트 기간이다.

| seed | stage | RMSE | MAE | MAPE(+1) % | MAPE(0 제외) % | 선정 epoch |
|---:|---|---:|---:|---:|---:|---:|
| 245 | 1 | 1.557553 | 0.491282 | 15.192422 | 61.905736 | 18 |
| 245 | 2 | 1.607080 | 0.494100 | 13.470550 | 67.118059 | 4 |
| 6835 | 1 | 1.649147 | 0.503378 | 14.846693 | 61.512803 | 23 |
| 6835 | 2 | 1.687677 | 0.496696 | 13.404102 | 66.793798 | 5 |
| 851 | 1 | 1.678340 | 0.500006 | 15.183083 | 61.523412 | 33 |
| 851 | 2 | 1.715425 | 0.503168 | 13.482880 | 67.539565 | 3 |
| 평균 ± 표본 SD | 1 | 1.628347 ± 0.063023 | 0.498222 ± 0.006242 | 15.074066 ± 0.196966 | 61.647317 ± 0.223860 | — |
| 평균 ± 표본 SD | 2 | 1.670061 ± 0.056280 | 0.497988 ± 0.004670 | 13.452510 ± 0.042374 | 67.150474 ± 0.373939 | — |

동일 시드 stage 1→2 상대 변화율 평균: RMSE **+2.575%**,
MAE **-0.041%** (3개 중 1개 시드만 개선), MAPE(+1) **-10.749%**,
MAPE(0 제외) **+8.928%**. RMSE와 MAPE(0 제외)는 세 시드 모두 악화했고
MAPE(+1)는 세 시드 모두 개선됐다. 원자료:
`output/experiments/tmp_two_stage_701515_porto_seed{seed}.json`,
최종 체크포인트:
`output/experiments/checkpoints/tmp_two_stage_701515_porto_seed{seed}/`.

두 도시에서 stage 2의 ΔW는 0이 아닌 값으로 학습됐지만, 현재
`10 * RMSE + MAPE(+1)` 목적함수의 MAPE(+1) 개선을 위해 RMSE 및
MAPE(0 제외)를 희생했다. 이번 3시드에서 stage 2가 RMSE를 개선했다는
주장은 성립하지 않는다. 이 결과는 **2-stage의 성능 확인**이지,
이전 80/10/10 단일-stage 대비 개선 실험이 아니다.

재현 명령 (각 도시/시드 순차 실행, 물리 GPU 1):

```bash
for city in ulsan porto; do
  for seed in 245 6835 851; do
    CUDA_VISIBLE_DEVICES=1 conda run -n DA python train.py \
      --config-name "config_${city}" "seed=${seed}" \
      "hydra.run.dir=output/experiments/checkpoints/tmp_two_stage_701515_${city}_seed${seed}" \
      "run_json=output/experiments/tmp_two_stage_701515_${city}_seed${seed}.json"
  done
done
```

결과 JSON의 `stage1_test`는 stage 1 체크포인트, `test`는 stage 2
최종 체크포인트의 지표다. stage 1 체크포인트는 각 실행 디렉터리의
`stage1/checkpoint-*`, 최종 저장 모델은 위 루트 체크포인트 경로에 있다.
