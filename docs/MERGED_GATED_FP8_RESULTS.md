# merged 모델 게이티드 ΔW + shared-weight FP8 채택 결과

Ulsan의 현재 채택 설정은 `node_adaptive=true`와 `shared_weight_fp8=true`를 함께 켜고,
`combined` loss로 단일 stage 학습하는 구성이다. 이 문서는 그 설정을 baseline 및
게이티드 ΔW만 켠 구성과 3개 seed로 비교한 기록이다. 설계와 실행 경로는
[`MERGED_ARCHITECTURE.md`](MERGED_ARCHITECTURE.md), 삭제된 workflow의 단일 seed 기록은
[`MERGED_NODE_ADAPTIVE_RESULTS.md`](MERGED_NODE_ADAPTIVE_RESULTS.md)에 있다.

## 실험 조건과 채택 설정

- 도시: Ulsan, batch size 24, seeds `2026`, `245`, `6835`
- loss: `combined`, 단일 stage
- adaptive node 선택: train-window mean demand가 `0.8` 초과인 168개 중 44개
- **baseline**: `node_adaptive=false`, `shared_weight_fp8=false`
- **gated ΔW only**: `node_adaptive=true`, `shared_weight_fp8=false`
- **gated ΔW + FP8 (채택)**: `node_adaptive=true`, `shared_weight_fp8=true`

세 구성의 3-seed 수치는 test split에서 계산했다. 원자료는
`output/experiments/exp1_baseline_ulsan_seed{2026,245,6835}.json`,
`output/experiments/exp2_gated_delta_ulsan_seed{2026,245,6835}.json`,
`output/experiments/exp3_fp8_shared_ulsan_seed{2026,245,6835}.json`이며,
체크포인트는 `output/experiments/checkpoints/` 아래에 있다.

## 채택 구성의 seed별 결과

아래는 채택 구성의 seed별 test RMSE, MAE, MAPE(+1), MAPE(0 제외)다.

| seed | RMSE | MAE | MAPE(+1) | MAPE(0 제외) |
|---|---|---|---|---|
| 2026 | 0.759366 | 0.348737 | 15.753004 | 62.567865 |
| 245 | 0.750596 | 0.350102 | 16.296428 | 61.498722 |
| 6835 | 0.762460 | 0.347196 | 15.443913 | 63.599847 |

## 3-seed 평균 ± sample sd

| 설정 | RMSE | MAE | MAPE(+1) | MAPE(0 제외) | test loss |
|---|---|---|---|---|---|
| baseline | 0.769891 ± 0.002274 | 0.351688 ± 0.001376 | 15.604400 ± 0.079968 | 63.652091 ± 0.169864 | 0.820294 ± 0.002858 |
| gated ΔW only | 0.758377 ± 0.005824 | 0.349212 ± 0.001712 | 15.831971 ± 0.425763 | 62.976671 ± 1.053469 | 0.820651 ± 0.000796 |
| **gated ΔW + FP8 (채택)** | **0.757474 ± 0.006154** | **0.348678 ± 0.001454** | 15.831115 ± 0.431592 | **62.555478 ± 1.050617** | **0.819669 ± 0.001140** |

## paired 비교와 seed별 개선 횟수

표의 변화율은 앞의 구성 대비 채택 구성의 paired 변화다. RMSE, MAE,
MAPE(0 제외)는 음수가 개선이며, MAPE(+1)는 양수가 악화다. 괄호는 3개 seed 중
실제로 개선된 seed 수다.

### 채택 구성 vs baseline

| 지표 | 평균 변화율 | 개선 seed 수 |
|---|---:|---:|
| RMSE | -1.613% | 3/3 |
| MAE | -0.855% | 3/3 |
| MAPE(0 제외) | -1.722% | 3/3 |
| MAPE(+1) | +1.456% | 1/3 |

### 채택 구성 vs gated ΔW only

| 지표 | 평균 변화율 | 개선 seed 수 |
|---|---:|---:|
| RMSE | -0.119% | 2/3 |
| MAE | -0.152% | 2/3 |
| MAPE(0 제외) | -0.666% | 2/3 |
| MAPE(+1) | +0.004% | 1/3 |

## 학습된 gate `s`

`s`는 shared weight 쪽의 비율을 나타내는 fp32 scalar다. 채택 구성에서 학습된 값은
다음과 같다.

| seed | learned `s` |
|---|---:|
| 2026 | 0.8324190 |
| 245 | 0.8462719 |
| 6835 | 0.8440180 |
| **평균 ± sample sd** | **0.84090 ± 0.00743** |

즉 평균적으로 약 84%가 shared weight, 약 16%가 ΔW 쪽이다.

## 비용과 학습 길이

| 항목 | baseline | 채택 구성 |
|---|---:|---:|
| parameters | 121,907 | 1,732,660 |
| parameter 배율 | — | 14.21x (기준 대비 +1,610,753) |
| best epoch | 22–27 | 8–9 |
| total epochs | 30–35 | 20 |
| run wall time | 약 29–37분 | 약 19–20분 |

측정 환경은 RTX 4090 한 장, batch size 24이며 채택 구성의 peak memory는 24 GiB 중
약 23.5 GiB였다. 따라서 parameter count는 14배로 늘어나지만 평균 RMSE 개선은 약
1.6%다. 이 비용-효과를 채택 판단에서 분리해 읽어야 한다.

## FP8의 범위와 주의점

`shared_weight_fp8=true`는 네 shared LSTM tensor
(`weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`)를 absmax per-tensor로
`torch.float8_e4m3fn` fake quantize하고 straight-through gradient를 적용한다. ΔW와
`s`는 fp32이며 master weight와 모든 수학 연산도 fp32다. 따라서 이 FP8은 **정확도 QAT
동작만 제공하고 FP8 메모리 절약이나 속도 향상을 제공하지 않는다.** shared weight를
외부 weight로 전달하는 경로 때문에 cuDNN의 non-contiguous RNN weight warning도
발생할 수 있다.

seed 2026에서 최종 shared-weight fake-quantization의 최대 오차는 다음과 같다.

| tensor | max quantization error |
|---|---:|
| `weight_ih` | 0.018561 |
| `weight_hh` | 0.015429 |
| `bias_ih` | 0.006814 |
| `bias_hh` | 0.005964 |

MAPE(+1)는 baseline 대비 평균 +1.456%로 회귀했고 3개 seed 중 1개에서만 개선됐다.
주요 지표가 MAPE(+1)인 run은 `model.node_adaptive=false`와
`model.shared_weight_fp8=false`를 사용해야 한다.

## 재현

현재 Ulsan config 기본값이 두 switch를 모두 켜므로 다음 명령으로 채택 구성을 재현한다.

```bash
python train.py --config-name config_ulsan train.seed=<seed>
```

baseline은 두 switch를 명시적으로 끈다.

```bash
python train.py --config-name config_ulsan train.seed=<seed> \
  model.node_adaptive=false model.shared_weight_fp8=false
```

`<seed>`에는 `2026`, `245`, `6835`를 각각 넣는다.
