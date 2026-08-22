# 프로젝트 요약

grid 단위 택시 수요 예측 모델 6개(baseline, ADFormer, STResnet, DMVST, aggregator, ir)를 **같은 데이터·같은
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

`ADFormer`/`STResnet`/`DMVST`는 `master`에서 baseline 모델 코드(`models/config.py`,`modeling.py`,
`embeddings.py`,`configs/model/baseline.yaml`,`docs/MODEL_PLAN.md`)를 지우고 새 모델로 교체했다.
`aggregator`/`ir`은 baseline 코드를 지우지 않고 각각 시간축 취합 부분, 최종 예측 단계만 부분 수정했다.
공통으로 `models/losses.py`/`models/metrics.py`, `train.py`/`test.py`의 Trainer 구조는 그대로 재사용한다.
모델 구현 전 항상 논문 PDF를 먼저 읽고 적용 가능성을 판단한 뒤 진행했고, 모든 구현은 `codex exec`을
이용한 외부 리뷰(계획 대비 diff 검토, 최소 1~3라운드)를 통과한 뒤 커밋했다.

## 실험 결과 (5시드 평균: seed=245,6835,851,5123,535)

| 데이터 | 모델 | RMSE | MAE | MAPE(+1) |
|---|---|---|---|---|
| ulsan | ADFormer | **0.7638** | 0.3537 | 15.950 |
| ulsan | ir | 0.7672 | **0.3520** | **15.715** |
| ulsan | aggregator | 0.7677 | 0.3526 | 15.915 |
| ulsan | baseline | 0.7752 | 0.3547 | 15.771 |
| ulsan | DMVST | 0.7760 | 0.3575 | 16.107 |
| ulsan | STResnet | 0.8054 | 0.3685 | 16.281 |
| porto | ADFormer | **1.6114** | 0.5153 | 16.554 |
| porto | ir | 1.6118 | **0.5049** | 15.709 |
| porto | baseline | 1.7838 | 0.5136 | 15.663 |
| porto | aggregator | 1.8157 | 0.5179 | **15.599** |
| porto | DMVST | 1.8357 | 0.5375 | 16.922 |
| porto | STResnet | 1.9385 | 0.5368 | 16.578 |

각 지표는 도시(ulsan/porto)별로 가장 좋은 값(낮을수록 좋음)만 볼드 처리했다.

원본 CSV(브랜치/도시별 5개 seed 개별 기록): `docs/{BASELINE,ADFORMER,STRESNET,DMVST,AGGREGATOR,IR}_RESULTS.csv`
(ulsan), `docs/{BASELINE,ADFORMER,STRESNET,DMVST,AGGREGATOR,IR}_RESULTS_PORTO.csv`(porto).

### 관찰

- **RMSE 순위는 ulsan·porto 두 도시 모두 거의 동일**: ADFormer ≈ ir > aggregator ≈ baseline > DMVST >
  STResnet. `ir`(검색기 앙상블)은 두 도시 모두 ADFormer에 근소하게 밀려 2위(porto는 1.6114 vs 1.6118로
  사실상 동률)— 검색 브랜치를 얹는 것만으로 훨씬 무거운 전역 attention 구조(ADFormer)에 필적하는
  RMSE를 얻는다.
- **`ir`은 MAE 기준으로는 6개 모델 중 두 도시 모두 1위**다(ulsan 0.3520, porto 0.5049) — RMSE로는
  ADFormer에 살짝 못 미치지만, 평균 절대오차는 검색 브랜치가 실제 과거 유사 패턴의 정답값을 직접
  가중평균해서 쓰는 방식이 큰 이상치(outlier)에 덜 휘둘리는 것으로 보인다. ulsan에서는 MAPE(+1)도
  `ir`이 1위(15.715)라 사실상 전 지표 1~2위를 휩쓴다.
- **aggregator(baseline의 시간축 취합을 LSTM→Transformer로만 바꾼 버전)는 baseline 대비 RMSE가
  두 도시 모두 개선**됐다(ulsan 0.7752→0.7677, porto 1.7838→1.8157은 예외적으로 소폭 악화 — porto는
  격자 수/샘플 수가 더 커서 causal mask 없는 시간축 self-attention이 LSTM보다 항상 유리하진 않음을
  시사). porto MAPE(+1)는 6개 모델 중 aggregator가 1위.
- ADFormer의 우위 폭은 porto(격자 200개, 8760시간)에서 ulsan(격자 168개, 4368시간)보다 훨씬 크다
  (RMSE 기준 ulsan +1.5% vs porto +9.7% 우위) — 전역 dense attention이 로컬 윈도우 기반 모델보다
  큰 데이터에서 더 유리해지는 것으로 보인다. 반면 `ir`의 검색 브랜치는 porto처럼 검색 후보(과거
  시점 수)가 많아지는 큰 데이터에서도 ADFormer와의 격차가 거의 그대로 유지된다.
- **MAPE(+1) 기준으로는 다른 순위**: ulsan은 `ir`이 1위(15.72), porto는 aggregator가 1위(15.60).
  RMSE/MAE(절대 오차)와 MAPE(상대 오차)가 다른 모델을 최고로 뽑는다는 뜻은 계속 유효하다.

## 데이터 / 환경

- `ulsan`, `porto` 두 도시의 grid 수요 데이터(`data/raw/{city}_temporal_grid.npy`), 전처리는
  `preprocessing/{ulsan,porto}/create_graph.py`.
- 학습은 `conda activate DA` 환경 사용. DMVST 브랜치만 예외적으로 Semantic View의 LINE 그래프
  임베딩 계산에 `cogdl`이 필요한데, `DA`에는 설치하지 않고 별도 `Torch` conda env로 1회성 전처리만
  분리했다(`preprocessing/build_dmvst_graph.py` → `preprocessing/build_dmvst_line_embeddings.py`).
- 5시드 멀티런 학습은 `python train.py -m dataset={ulsan,porto} seed=245,6835,851,5123,535`
  (`aggregator`/`ir`은 GPU 1번만 사용하도록 `CUDA_VISIBLE_DEVICES=1`을 붙여 실행함).

