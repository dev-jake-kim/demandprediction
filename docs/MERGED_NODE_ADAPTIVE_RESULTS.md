# merged 모델 노드별 ΔW(node_adaptive) + stage2 loss 실험 결과 (역사 기록)
> **역사 기록 — 현재 코드와 분리됨.** 이 문서는 삭제된 2-stage ΔW 학습 workflow의
> seed 245 결과를 보존한다. 이 workflow는 현재 코드에 더 이상 존재하지 않으며,
> `node_adaptive`는 단일 stage의 `combined` 학습으로 동작한다. 따라서 아래의 stage 1/
> stage 2, `stage2.loss`, `stage2.init_from` 표기와 설정은 당시 실행을 설명하는
> 역사적 기록일 뿐 현재 사용할 수 있는 설정이 아니다. 채택한 Ulsan gated ΔW + FP8
> 3-seed 결과와 현재 설정의 재현 방법은 [`MERGED_GATED_FP8_RESULTS.md`](MERGED_GATED_FP8_RESULTS.md)를
> 참고한다.

`model.node_adaptive=true`로 history LSTM에 노드별 weight offset(ΔW)을 붙이고, stage 2의
목적함수를 바꿔가며 측정한 결과다. 설계와 제약은 `docs/MERGED_ARCHITECTURE.md`의
"노드별 LSTM weight offset" 절 참고.

- 전부 **seed 245 단일 시드**, ulsan/porto 각각의 도시별 config 기본값
- stage 1은 `node_adaptive=false` + `combined`으로 학습하고, stage 2는 그 체크포인트를
  `stage2.init_from`으로 이어받아 ΔW만 해제한다. **stage 2에서 바뀌는 것은 ΔW 해제와
  `stage2.loss`뿐**이다.
- 원자료: `output/merged/runs/na_*_seed245.json` (ulsan 4건 + porto 4건)
- 노드별 시각화: `docs/merged_node_metrics/ulsan_loss*.png`

## stage 1 재현 검증

stage 1은 기존 튜닝 기록과 **소수점까지 동일**하다. ΔW 기능 추가가 기본 경로를 건드리지
않았다는 회귀 검증이 된다.

| 도시 | stage1 | 기존 기록 | best ep | RMSE | MAPE(+1) |
|---|---|---|---|---|---|
| ulsan | `na_ulsan_stage1` | `tune_ulsan_baseline` | 8 / 8 | 0.75364 / 0.75364 | 15.401 / 15.401 |
| porto | `na_porto_stage1` | — (200노드 격자의 merged 튜닝 기록이 없어 대조 불가) | 17 | 1.50465 | 14.877 |

## 결과

### ulsan — 44/168 노드, ΔW 1,622,016개

| stage2 loss | best ep | RMSE | Δ | MAPE(+1) | Δ | MAE | Δ | MAPE(0제외) | Δ |
|---|---|---|---|---|---|---|---|---|---|
| **— (stage1, ΔW 없음)** | 8 | **0.75364** | — | **15.401** | — | **0.34308** | — | **62.663** | — |
| `combined` | 1 | 0.76986 | +2.15% | 15.179 | -1.44% | 0.34812 | +1.47% | 63.965 | +2.08% |
| `rmse_mape` | 5 | 0.78237 | +3.81% | 13.459 | -12.61% | 0.33567 | -2.16% | 69.489 | +10.89% |
| `demand_split` | 4 | 0.75010 | -0.47% | 19.969 | +29.66% | 0.35304 | +2.90% | 66.051 | +5.41% |

### porto — 52/200 노드, ΔW 1,916,928개

| stage2 loss | best ep | RMSE | Δ | MAPE(+1) | Δ | MAE | Δ | MAPE(0제외) | Δ |
|---|---|---|---|---|---|---|---|---|---|
| **— (stage1, ΔW 없음)** | 17 | **1.50465** | — | **14.877** | — | **0.49851** | — | **60.888** | — |
| `combined` | 1 | 1.48207 | -1.50% | 15.021 | +0.96% | 0.49411 | -0.88% | 61.761 | +1.43% |
| **`rmse_mape`** | 1 | 1.49030 | -0.95% | 13.456 | -9.55% | 0.49237 | -1.23% | 67.072 | +10.16% |
| `demand_split` | 1 | 1.46431 | -2.68% | 20.135 | +35.34% | 0.51072 | +2.45% | 62.175 | +2.11% |

굵게 표시된 것이 **RMSE와 MAPE(+1)가 둘 다 개선**된 변종이다.

## 판정 — RMSE와 MAPE(+1) 둘 다 개선된 변종

**porto의 `rmse_mape` 하나뿐이다** (RMSE -0.95%, MAPE(+1) -9.55%). ulsan은 3개 loss 중
두 지표가 함께 개선된 경우가 없다.

다만 이 결과는 아래 "유보" 절의 조건을 달고 읽어야 한다 — 1시드이고, porto의 best_epoch가
세 변종 모두 1이다.

## 관찰

### ΔW 단독 효과는 도시마다 반대로 나온다

`stage2.loss=combined`는 stage 1과 목적함수가 같아 **바뀌는 변수가 ΔW 하나뿐**인 유일한
조건이다. 두 도시가 반대 방향으로 나온다:

| 도시 | ΔRMSE | ΔMAPE(+1) | best ep | ΔW norm (weight_ih) |
|---|---|---|---|---|
| ulsan | +2.15% | -1.44% | 1 | 8.07 |
| porto | **-1.50%** | +0.96% | 1 | 3.18 |

**porto에서는 ΔW가 RMSE를 개선하고 ulsan에서는 악화시킨다.** 세 loss 전체로 봐도 porto는
RMSE가 모두 개선(-0.95 ~ -2.68%)인 반면 ulsan은 `demand_split` 하나만 개선(-0.47%)이다.

두 도시가 갈리는 이유는 아직 모른다. 아래 "브랜치별 영향력"에서 보듯 neural 브랜치의
영향력은 ulsan 21.7% / porto 28.2%로 큰 차이가 없어, 구조적 차이로는 설명되지 않는다.

lora 브랜치의 `docs/NODE_ADAPTIVE_AND_MOE_RESULTS.md`는 `combined`에서 RMSE가 두 도시 모두
개선된다고(ulsan -2.24%p, porto -5.76%p) 기록한다. 다만 아키텍처와 데이터가 모두 달라
직접 비교가 성립하지 않는다(아래 제약 참고). lora 기록도 seed 42 단일 시드다.

### loss가 지표를 가른다

ΔW보다 `stage2.loss` 선택이 결과를 훨씬 크게 움직인다. 세 loss가 서로 다른 방향으로 간다:

- `rmse_mape` — MAPE(+1)를 크게 개선(ulsan -12.61%, porto -9.55%)한다. RMSE는 ulsan에서
  악화(+3.81%), porto에서 개선(-0.95%)으로 갈린다. MAPE(0제외)는 두 도시 모두 크게
  악화(+10.89%, +10.16%)한다.
- `demand_split` — **RMSE를 가장 크게 개선**(ulsan -0.47%, porto -2.68%)하고 MAPE(+1)를
  크게 악화(+29.66%, +35.34%)한다. 두 도시에서 방향이 일관된다.
- `combined` — ulsan에서는 거의 안 움직이고, porto에서는 RMSE를 -1.50% 개선한다.

두 MAPE 지표가 반대로 움직이는 것이 이 데이터의 성질을 드러낸다. MAPE(+1)는 분모가
`|y|+1`이라 실제 수요가 0인 셀(ulsan test의 75.1%)에서 `|오차|`가 되어 그 셀들이 지표를
좌우하고, MAPE(0제외)는 그 셀들을 아예 제외한다. `rmse_mape`는 예측을 전반적으로 낮춰
0셀에서 이득을 보고(MAPE(+1) 개선) 수요가 있는 셀에서 손해를 본다(MAPE(0제외) 악화).
`demand_split`은 정확히 그 반대다.

ulsan 노드별 분해(`docs/merged_node_metrics/`)에서 이 구조가 그대로 보인다 —
평균 예측값이 stage1 0.2838, `combined` 0.2607, `rmse_mape` 0.2325, `demand_split` 0.5148
이고 실제 평균은 0.4671이다. 앞의 셋은 과소예측을 더 키웠고 `demand_split`만 실제 수준으로
올렸다(약간 과대예측).

## 브랜치별 영향력

ΔW는 history LSTM(=h_neural)에만 붙는다. 그 신호가 최종 예측에 얼마나 영향을 주는지 재려면
`BranchAttention`의 구조를 정확히 봐야 한다:

```python
# models/merged/modules/attention.py
return self.norm(z_neural + self.output_projection(fused)), weights
```

**residual 연결이 있다.** `z_neural`이 어텐션 출력에 직접 더해지므로, 어텐션 가중치
`weights[..., 2]`는 `fused` 경로만 좌우하고 residual 경로는 가중치와 무관하게 흐른다.
게다가 `query = query_projection(z_neural)`이라 h_neural은 어텐션 가중치를 만드는 쿼리이기도
하다 — `weights[..., 2]`가 작다는 것은 "neural이 무시된다"가 아니라 "neural이 자기 자신을
value로 되가져오지 않고 daily/weekly를 조회한다"는 뜻에 가깝다.

### residual 항과 어텐션 항의 크기

| 도시 | residual `‖z_neural‖` | 어텐션 `‖out_proj(fused)‖` | residual 비중 |
|---|---|---|---|
| ulsan | 6.331 | 5.500 | 53.5% |
| porto | 21.263 | 13.738 | 60.7% |

residual 쪽이 오히려 크다.

### 최종 예측에 대한 브랜치별 민감도

`d(prediction)/d(h_branch)`의 gradient norm. 브랜치 출력을 leaf로 두고 downstream
(브랜치 어텐션 + 검색 게이트)만 미분해 구했다.

| 도시 | neural (ΔW 경로) | daily | weekly |
|---|---|---|---|
| ulsan | 4.704 (**21.7%**) | 8.128 (37.6%) | 8.796 (40.7%) |
| porto | 41.132 (**28.2%**) | 53.289 (36.6%) | 51.284 (35.2%) |

세 브랜치의 영향력이 대체로 비슷하다. **ΔW가 최종 예측에 닿는 경로는 닫혀 있지 않다** —
따라서 ulsan에서 ΔW 효과가 관측되지 않은 것을 "경로가 막혀서"로 설명할 수 없다. 원인은
아직 규명되지 않았다.

> 정정 이력: 이 절은 처음에 어텐션 가중치만으로 "ΔW 도달률 = neural 가중치(0.082) x
> lambda(0.708) = 0.064, 고수요 노드에서는 0.0093"이라고 적었으나 **틀렸다.** residual
> 경로를 빼먹은 계산이었다. 실제 민감도는 ulsan 21.7%로 3배 이상 크다.

## 유보

위 결과를 확정으로 읽으면 안 되는 이유가 둘 있다.

**best_epoch가 porto 세 변종 모두 1이다.** stage 1 체크포인트에서 시작해 첫 epoch가 최적이었다는
뜻이라, ΔW가 학습된 결과라기보다 **초기 미세조정 효과**일 수 있다. porto의 ΔW 이동량도
`weight_ih` norm 3.18로 ulsan(8.07)의 절반 이하다. ulsan도 `combined`은 best_epoch가 1이다.

**전부 1시드(245)다.** lora 브랜치 문서는 porto RMSE의 시드 표준편차를 약 3.2%로 기록하고
있다. porto에서 관찰된 -0.95 ~ -2.68%는 그 범위 안이다. 채택 후보로 나온 `rmse_mape`의
-0.95%도 마찬가지다. **다중 시드 없이는 노이즈와 구분되지 않는다.**

## 제약

- **전부 1시드(245)다.** 위 변화의 상당수가 1~3% 범위이고, 이 저장소에서 관찰된 시드 노이즈와
  구분되지 않는다. 다중 시드 전까지 확정적 결론이 아니다.
- **ΔW 대상 노드는 train 구간 평균 수요 > 0.8인 노드만**이다(ulsan 44/168, porto 49/380).
  전 노드에 적용한 조건은 시험하지 않았다.
- **ulsan의 early stopping 설정이 이 실험 이후 바뀌었다.** 위 ulsan 런들은
  `min_epochs=0 / patience=20`으로 돌았고, 현재 config는 `min_epochs=13 / patience=8`이다.
  기록을 시뮬레이션한 결과 best_epoch가 동일한 것으로 확인됐으나(커밋 f520278), 재현 시
  이 차이를 염두에 둘 것. porto는 `min_epochs=0 / patience=20`으로 그대로다.
- **`stage2.loss`를 바꾼 런은 stage 1과 목적함수가 달라** 두 stage의 val loss를 직접 비교할 수
  없다. best checkpoint 선택 기준도 stage마다 다르다(`loss` vs `rmse_mape_objective`).
- **lora 브랜치의 node_adaptive 결과와 직접 비교할 수 없다.** 아키텍처(ΔW 도달률 위 참고),
  porto 격자(lora/baseline 200노드 vs 당시 merged 380노드), split 비율(lora 0.7/0.15 vs
  merged 0.80/0.10), ΔW 적용 범위(lora 전 노드 vs merged 평균수요 0.8 초과 노드만),
  출력층 ΔW 유무(lora 있음, merged 없음)가 모두 다르다.

## 참고 — dropout=0 / batch=24 (ulsan)

SDPA 한계 우회 목적으로 시험했던 조건이다. 이 실험 계열과는 설정이 달라 위 표와 직접
비교할 수 없어 따로 둔다.

| 런 | dropout | batch | best ep | RMSE | MAPE(+1) | MAE |
|---|---|---|---|---|---|---|
| stage1 | 0.0 | 24 | 12 | 0.76164 | 15.519 | 0.34770 |
| stage2 (`rmse_mape`) | 0.0 | 24 | 6 | 0.79316 | 13.424 | 0.33946 |

기본 설정(dropout 0.1 / batch 8)의 stage1 대비 RMSE +1.06%, MAPE(+1) +0.77%로 세 지표가
모두 악화했다. batch를 3배로 키웠으나 best_epoch가 8에서 12로 늦어져 총 학습 시간은 거의
같았다(29분 vs 30분).
