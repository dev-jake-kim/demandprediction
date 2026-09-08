# 노드별 LSTM weight offset(2-stage) + MoE 실험 결과

## 1. 노드별 LSTM Δw 주입 (`lora` 브랜치, `node_adaptive`)

`self.lstm`과 `self.output_proj`의 weight/bias에 `공유 W + 노드별 ΔW`를 더하는 실험.
저랭크·양자화 없이 노드마다 전체 텐서를 하나씩 갖는다(4종 delta, ulsan 5,558,784 /
porto 6,617,600개 — 기존 모델의 27~32배).

- **1-stage**: 공유 가중치와 offset을 처음부터 함께 학습(loss=`CombinedLoss`).
- **2-stage stage1**: offset을 0으로 고정 — `baseline-weather`와 동일한 학습.
- **2-stage combined**: stage1 결과에서 offset을 풀고 `CombinedLoss`로 finetuning(lr 1e-4).
- **2-stage rmse_mape**: 위와 동일하되 loss만 `10*RMSE + MAPE(+1)`.

모두 **seed 245 단일**. `Δ`는 `baseline-weather` 행 대비 `(row − base) / base × 100`(%p),
양수면 그 지표가 나빠졌다는 뜻이다.

> ⚠ `baseline-weather` 행은 이 실험 중 발견된 저장소 전체의 재현성 결함(`Trainer`가 seed를
> 설정하기 전에 모델이 생성돼 초기 가중치가 프로세스마다 달랐음)이 고쳐지기 **이전**에
> 학습된 값이다. `2-stage stage1`이 offset=0으로 baseline-weather와 수학적으로 동일한
> 학습이므로, 초기화 차이를 배제하고 싶으면 `stage1` 행을 기준으로 보는 것이 더 정확하다.

### ulsan

| 모델 | RMSE | Δ RMSE (%p) | MAE | Δ MAE (%p) | MAPE(+1) | Δ MAPE(+1) (%p) |
|---|---|---|---|---|---|---|
| **baseline-weather (기준)** | **0.76661** | — | **0.35018** | — | **15.523** | — |
| 1-stage (처음부터 함께 학습) | 0.76053 | −0.79%p | 0.34943 | −0.21%p | 15.718 | +1.26%p |
| 2-stage stage1 (offset 고정) | 0.76113 | −0.72%p | 0.34957 | −0.17%p | 15.696 | +1.11%p |
| 2-stage combined | 0.74945 | **−2.24%p** | 0.34607 | −1.17%p | 15.846 | +2.08%p |
| 2-stage rmse_mape | 0.77508 | +1.10%p | 0.33638 | **−3.94%p** | 13.867 | **−10.67%p** |

### porto

| 모델 | RMSE | Δ RMSE (%p) | MAE | Δ MAE (%p) | MAPE(+1) | Δ MAPE(+1) (%p) |
|---|---|---|---|---|---|---|
| **baseline-weather (기준)** | **1.86223** | — | **0.52320** | — | **14.834** | — |
| 1-stage (처음부터 함께 학습) | 1.90753 | +2.43%p | 0.53021 | +1.34%p | 16.502 | +11.25%p |
| 2-stage stage1 (offset 고정) | 1.79264 | −3.74%p | 0.52342 | +0.04%p | 14.970 | +0.92%p |
| 2-stage combined | 1.75492 | **−5.76%p** | 0.51158 | −2.22%p | 15.604 | +5.19%p |
| 2-stage rmse_mape | 1.85730 | −0.26%p | 0.51263 | −2.02%p | 13.924 | **−6.13%p** |

### 요약

- **1-stage(공유 가중치와 offset을 처음부터 함께 학습)는 porto에서 전 지표가 baseline보다
  나쁘다.** 셀 단위로 보면 고수요 셀은 개선되지만 저수요 셀에서 과적합해 손실이 이득을
  덮는다(자세한 근거는 `docs/node_metric_comparison/`).
- **2-stage(공유 가중치를 먼저 수렴시킨 뒤 offset을 푸는 방식)가 이 문제를 크게 줄인다.**
  `combined` loss는 RMSE를 두 도시 모두 개선(ulsan −2.24%p, porto −5.76%p)하지만 MAPE(+1)는
  악화시킨다. `rmse_mape` loss(`10*RMSE+MAPE(+1)`로 최적화)는 반대로 MAE·MAPE(+1)를 크게
  개선(ulsan MAPE −10.67%p, porto −6.13%p)하지만 RMSE 이득은 거의 반납한다.
- 즉 **하나의 설정으로 세 지표를 동시에 이기는 조합은 아직 없다** — loss가 최적화하는
  지표만 확실히 가져가고 나머지는 baseline과 비슷하거나 소폭 밀린다.
- 전부 1시드라 porto RMSE(시드 표준편차 0.0593 ≈ 3.2%)처럼 노이즈가 큰 지표는 해석에 주의.

원자료: `logs/node_adaptive_seed245.log`(1-stage), `logs/two_stage_seed245.log`(2-stage
combined), `logs/stage2_rmsemape_seed245.log`(2-stage rmse_mape),
`docs/node_metric_comparison/`(노드별 분해 그림·CSV).

## 2. MoE 실험 (`MOE` 브랜치 결과)

> ⚠ **이 코드는 현재 어떤 브랜치에도 존재하지 않는다.** 저장소 전체 git 히스토리에서
> `models/config.py`에 `moe_sparse`/`n_experts` 문자열이 들어간 커밋이 하나도 없다 —
> `assemble` 브랜치 체크포인트가 겪었던 것과 같은 종류의 코드 유실로 보인다. 아래 값은
> `output/moe_*/**/train.log`에 학습 직후 자체 기록된 것을 그대로 옮긴 것이고, 코드가 없어
> **재현·재검증이 불가능하다**. 체크포인트의 `config.json`에서 `n_experts`/`moe_sparse` 값만
> 확인 가능하다(모델 클래스명은 `GridDemandModel`로 현재와 동일하게 저장돼 있으나, 실제
> 모듈 구성은 알 수 없다).

전부 **porto, seed 42(저장소 공용 5시드와 다름), 단일 시드**, 2026-08-27 학습.

| 실행 | n_experts | moe_sparse | RMSE | MAE | MAPE(+1) | epoch |
|---|---|---|---|---|---|---|
| moe_node_gate | — (필드 없음) | — (필드 없음) | 1.84446 | 0.51482 | 15.750 | 27 |
| moe_sweep_e2 | 2 | false | 1.81808 | 0.51384 | 15.745 | 25 |
| moe_3expert / moe_sweep_e3 | 3 | false | 1.79252 / 1.80553 | 0.51477 / 0.51533 | 15.390 / 16.081 | 31 / 26 |
| moe_sweep_e4 | 4 | false | 1.79957 | 0.51889 | 15.359 | 24 |
| moe_sweep_e5 | 5 | false | 1.78972 | 0.51771 | 16.169 | 26 |
| moe_sparse_qkv_half | 3 | true | 1.89766 | 0.52049 | 15.400 | 31 |

- `moe_3expert`와 `moe_sweep_e3`는 설정이 동일(`n_experts=3, moe_sparse=false`)한데 값이
  다르다 — 별도 실행이라 시드 42라도 초기화·학습 경로가 갈렸을 가능성이 있다(둘 다 이
  결함이 고쳐지기 전 실행이라 실제로 초기값이 달랐을 것이다, §1 참고).
- `moe_node_gate`는 `n_experts`/`moe_sparse` 필드 자체가 config에 없어 다른 실행들과
  같은 계열의 변형인지 불확실하다. 이름상 "노드별 게이트"를 시사하지만 코드가 없어
  확인할 수 없다.
- 참고용 baseline-weather(porto, seed 42는 아니지만) RMSE 1.86223과 대비하면 `moe_3expert`
  (1.79252)와 `moe_sweep_e5`(1.78972)가 더 낮지만, **seed가 다르고 코드도 재현이 안 돼
  이 비교 자체가 근거로 쓰기엔 약하다.**

원자료: `output/moe_{3expert,node_gate,sparse_qkv_half,sweep_e2,sweep_e3,sweep_e4,sweep_e5}/`
