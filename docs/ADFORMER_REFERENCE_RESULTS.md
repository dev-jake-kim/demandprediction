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

Ulsan 0.70/0.15 분할, d_model=16 Transformer local encoder, MAE 손실, 동일 학습 설정.
기준 `none`은 앞서 완료한 `output/experiments/var_d16_transformer_seed245.json`이며
주기 입력을 쓰지 않는다. 아래 세 변종은 `model.periodic_mode=lstm|ma|ema`만 바꿔
각각 처음부터 학습했다. 물리 GPU 1(`CUDA_VISIBLE_DEVICES=1`)에서 실행했고
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
Ulsan 기본 설정은 `periodic_mode=ma`로 채택했다. 단일 seed에서 RMSE와
MAPE(+1)는 개선됐지만 MAE(+0.487%)와 MAPE(0 제외)(+3.552%)는 악화했다.
EMA 실행 여부를 test 지표로 결정한 탐색 실험이므로, 이 비교를 독립 test 검증이나
여러 seed에서의 우위 증거로 해석해서는 안 된다.

원자료/체크포인트:
`output/experiments/periodic_{lstm,ma,ema}_d16_ulsan_seed245.json`,
`output/experiments/checkpoints/periodic_{lstm,ma,ema}_d16_ulsan_seed245/`.
세 실행 모두 `seed=245 model.periodic_mode=<mode>`로 재현한다.

### MA의 노드별 게이트 시각화

예측식에 실제 곱해지는 값은 체크포인트의 `fusion.daily_gate` /
`fusion.weekly_gate` 원 파라미터에 sigmoid를 씌운 값이다. 두 게이트는 독립적인
계수이며 합이 1인 attention 비율이 아니다. 행·열은 Ulsan 14×12 격자의
row-major 노드 순서다. 히트맵과 분포를 다시 그리려면:

```bash
conda run -n DA python plot_periodic_gates.py \
  output/experiments/checkpoints/periodic_ma_d16_ulsan_seed245 \
  output/experiments/periodic_ma_gate_ulsan_seed245.png
```
