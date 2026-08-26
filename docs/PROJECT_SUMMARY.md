# 프로젝트 요약

grid 단위 택시 수요 예측 모델 7개(baseline, ADFormer, STResnet, DMVST, aggregator, ir, ir-commute)를 **같은 데이터·같은
loss(`CombinedLoss`)·같은 평가지표(`compute_regression_metrics`)·같은 학습 하네스**
(`train.py`/`test.py`, HuggingFace `Trainer`, Hydra 설정)로 비교하는 프로젝트. 각 모델은 별도 git
브랜치에 구현돼 있고, `master`의 baseline 모델 코드를 지우고 논문을 이식하는 방식(또는 baseline 자체를
개조하는 방식)으로 만들었다.

## 브랜치별 모델

| 브랜치 | 모델 | 핵심 아이디어 | 상세 문서 |
|---|---|---|---|
| `master` | baseline | 노드별 (2a+1)² 로컬 윈도우 + Transformer 인코더 + LSTM, 전체 H×W 노드를 한 스텝에 배치로 동시 예측 | `docs/MODEL_PLAN.md` |
| `ADFormer` | ADFormer (arXiv:2506.02576) | Differential Attention 기반 전역 dense attention(SDA) + DTW 클러스터 attention(SCA) + 시간축 attention(TSA/TAA) | `docs/ADFORMER_PLAN.md` |
| `STResnet` | ST-ResNet (Zhang et al., AAAI 2017) | closeness/period/trend 3-branch CNN(순수 conv, attention 없음) + parametric-matrix fusion | `docs/STRESNET_PLAN.md` |
| `DMVST` | DMVST-Net (Yao et al., AAAI 2018) | Spatial(Local CNN) + Temporal(LSTM) + Semantic(DTW+LINE 그래프 임베딩) 3-view 구조 | `docs/DMVST_PLAN.md` |
| `aggregator` | baseline 강화 | baseline과 동일한 공간(Transformer) 인코더, 시간축 취합만 LSTM → Transformer(learnable positional embedding, causal mask 없음)로 교체 | `docs/MODEL_PLAN.md` |
| `ir` | 검색기 앙상블 baseline | baseline(공간 Transformer+LSTM)은 그대로, 노드별 과거 로컬 윈도우 중 코사인 유사도 top-k를 인과적으로 검색해 다음 시점 수요를 가중평균한 예측을 sigmoid 게이트로 뉴럴 브랜치와 앙상블 | `docs/MODEL_PLAN.md` §1-5 |
| `ir-commute` | `ir` + commute-attention | `ir`에 더해, 노드마다 (2a+1)² 로컬 윈도우 밖에서 DTW 기준 가장 비슷한 n개(기본 3) 노드를 골라 LSTM 출력에 학습 가능한 cross-attention으로 주입(DTW는 attention logit의 learnable bias로만 참여) | `docs/MODEL_PLAN.md` §1-6 |

`ADFormer`/`STResnet`/`DMVST`는 `master`에서 baseline 모델 코드(`models/config.py`,`modeling.py`,
`embeddings.py`,`configs/model/baseline.yaml`,`docs/MODEL_PLAN.md`)를 지우고 새 모델로 교체했다.
`aggregator`/`ir`/`ir-commute`는 baseline 코드를 지우지 않고 부분 수정했다(각각 시간축 취합,
최종 예측 단계, LSTM 출력 이후 commute-attention). 공통으로
`models/losses.py`/`models/metrics.py`, `train.py`/`test.py`의 Trainer 구조는 그대로 재사용한다.
모델 구현 전 항상 논문 PDF를 먼저 읽고 적용 가능성을 판단한 뒤 진행했고, 모든 구현은 `codex exec`을
이용한 외부 리뷰(계획 대비 diff 검토, 최소 1~3라운드)를 통과한 뒤 커밋했다.

## 실험 결과 (5시드 평균: seed=245,6835,851,5123,535 — `baseline-tmp`만 3시드: seed=245,6835,851)

| 데이터 | 모델 | RMSE | MAE | MAPE(+1) |
|---|---|---|---|---|
| ulsan | ir-commute | **0.7629** | 0.3506 | 15.738 |
| ulsan | ADFormer | 0.7638 | 0.3537 | 15.950 |
| ulsan | ir | 0.7672 | 0.3520 | 15.715 |
| ulsan | aggregator | 0.7677 | 0.3526 | 15.915 |
| ulsan | baseline | 0.7752 | 0.3547 | 15.771 |
| ulsan | DMVST | 0.7760 | 0.3575 | 16.107 |
| ulsan | STResnet | 0.8054 | 0.3685 | 16.281 |
| ulsan | baseline-tmp (날씨+캘린더, 3시드) | 0.7636 | **0.3501** | **15.64** |
| porto | ADFormer | **1.6114** | 0.5153 | 16.554 |
| porto | ir | 1.6118 | **0.5049** | 15.709 |
| porto | ir-commute | 1.6355 | 0.5054 | 15.779 |
| porto | baseline | 1.7838 | 0.5136 | 15.663 |
| porto | aggregator | 1.8157 | 0.5179 | 15.599 |
| porto | DMVST | 1.8357 | 0.5375 | 16.922 |
| porto | STResnet | 1.9385 | 0.5368 | 16.578 |
| porto | baseline-tmp (날씨+캘린더, 3시드) | 1.8029 | 0.5158 | **15.57** |

각 지표는 도시(ulsan/porto)별로 가장 좋은 값(낮을수록 좋음)만 볼드 처리했다.

원본 CSV(브랜치/도시별 5개 seed 개별 기록): `docs/{BASELINE,ADFORMER,STRESNET,DMVST,AGGREGATOR,IR,IR_COMMUTE}_RESULTS.csv`
(ulsan), `docs/{BASELINE,ADFORMER,STRESNET,DMVST,AGGREGATOR,IR,IR_COMMUTE}_RESULTS_PORTO.csv`(porto).

### 관찰

- **`ir-commute`(`ir`에 DTW 기반 commute-attention 추가)는 ulsan에서 7개 모델 중 RMSE·MAE 모두
  1위**다(ir 대비 RMSE 0.7672→0.7629, MAE 0.3520→0.3506 개선 — ADFormer보다도 RMSE가 낮음).
  **반면 porto에서는 오히려 소폭 악화**됐다(RMSE 1.6118→1.6355, 약 +1.5%) — 로컬 윈도우 밖 DTW
  유사 노드를 참조하는 게 격자가 작은 ulsan(168노드)에서는 도움이 됐지만, 노드 수가 더 많은
  porto(200노드)에서는 오히려 약간의 노이즈로 작용한 것으로 보인다(`aggregator`가 porto에서
  baseline보다 살짝 나빠졌던 것과 같은 패턴 — 구조 변경이 porto에서 항상 유리하지는 않음).
- **RMSE 순위는 ulsan·porto가 다르다**: ulsan은 ir-commute > ADFormer > ir > aggregator > baseline
  > DMVST > STResnet, porto는 ADFormer > ir > ir-commute > baseline > aggregator > DMVST >
  STResnet. `ir` 계열(ir, ir-commute)이 두 도시 모두 ADFormer와 근접한 RMSE를 내는 것은 공통.
- **`ir`은 MAE 기준으로 porto 1위**(0.5049)를 유지하고, **`ir-commute`는 ulsan MAE 1위**(0.3506)다 —
  검색 브랜치(실제 과거 유사 패턴의 정답값을 직접 가중평균)와 commute-attention(학습된 가중치로
  먼 노드 정보 주입) 둘 다 평균 절대오차를 줄이는 데 특히 효과적인 것으로 보인다.
- **aggregator(baseline의 시간축 취합을 LSTM→Transformer로만 바꾼 버전)는 baseline 대비 RMSE가
  두 도시 모두 개선**됐다(ulsan 0.7752→0.7677, porto 1.7838→1.8157은 예외적으로 소폭 악화 — porto는
  격자 수/샘플 수가 더 커서 causal mask 없는 시간축 self-attention이 LSTM보다 항상 유리하진 않음을
  시사). porto MAPE(+1)는 7개 모델 중 aggregator가 1위.
- ADFormer의 우위 폭은 porto(격자 200개, 8760시간)에서 ulsan(격자 168개, 4368시간)보다 훨씬 크다
  (RMSE 기준 ulsan +0.1% vs porto +1.5% 우위, `ir-commute` 기준) — 전역 dense attention이 로컬
  윈도우 기반 모델보다 큰 데이터에서 더 유리해지는 것으로 보인다.
- **MAPE(+1) 기준으로는 또 다른 순위**: ulsan은 `ir`이 1위(15.72), porto는 aggregator가 1위(15.60).
  RMSE/MAE(절대 오차)와 MAPE(상대 오차)가 다른 모델을 최고로 뽑는다는 뜻은 계속 유효하다.

## 데이터 / 환경

- `ulsan`, `porto` 두 도시의 grid 수요 데이터(`data/raw/{city}_temporal_grid.npy`), 전처리는
  `preprocessing/{ulsan,porto}/create_graph.py`.
- 학습은 `conda activate DA` 환경 사용. DMVST 브랜치만 예외적으로 Semantic View의 LINE 그래프
  임베딩 계산에 `cogdl`이 필요한데, `DA`에는 설치하지 않고 별도 `Torch` conda env로 1회성 전처리만
  분리했다(`preprocessing/build_dmvst_graph.py` → `preprocessing/build_dmvst_line_embeddings.py`).
- 5시드 멀티런 학습은 `python train.py -m dataset={ulsan,porto} seed=245,6835,851,5123,535`
  (`aggregator`/`ir`/`ir-commute`는 GPU 1번만 사용하도록 `CUDA_VISIBLE_DEVICES=1`을 붙여 실행함).
- `ir-commute`는 학습 전 `preprocessing/build_commute_map.py --city {ulsan,porto} --a 2` 실행이
  필요하다(DTW 기반 노드별 참조 목록 사전 계산, `--a`는 모델 config의 `a`와 반드시 같아야 함).

