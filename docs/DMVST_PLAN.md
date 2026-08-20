# DMVST-Net 포팅

논문 "Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction"(Yao, Wu, Ke, Tang, Jia,
Lu, Gong, Ye, Li, AAAI 2018, `docs/DMVST.pdf`)을 `master`의 baseline, `ADFormer`, `STResnet` 브랜치에
이어 **같은 데이터·같은 지표·같은 학습 하네스**(`train.py`/`test.py`, HF `Trainer`)로 비교하기 위해
이식했다.

사용자가 예전에 이 논문을 직접 구현해둔 프로젝트가 로컬에 남아있어(`/home/jinsu/PycharmProjects/DMVST`,
GitHub `dev-jake-kim/Lambda-F` main 브랜치와 동일) 그 모델 코드(`models/DMVSTModel.py`의 `LocalCNN`,
`DMVST`)를 가져와 gir의 하네스에 연결했다 — 자체 `main.py`/`loss_fn`/데이터셋은 가져오지 않고,
`CombinedLoss`/`compute_regression_metrics`/`train.py`/`test.py` 구조는 다른 세 브랜치와 동일하게
재사용한다.

## 표기

- `B`: batch, `k`(=`time_step`): lookback 길이(기본 8), `H,W`: 격자 크기(ulsan 14×12, porto 10×20)
- `N=H*W`: 전체 지역(격자셀) 수, `S`(=`patch_size`): 로컬 이웃 crop 크기(기본 9)

## 1. 모델 아키텍처 (`models/modeling.py`)

```
demands (B,k,H,W)
  -> _crop_all_nodes: idx_table 기반으로 N개 노드 전부의 (S,S) 로컬 crop을 한 번에 추출
     (경계는 zero-padding, master의 GridDemandModel과 동일한 이웃 테이블 메커니즘 재사용)
  -> Min-Max [0,1] 정규화(demand_min/max, train split 기준)
  -> LocalCNN: (Conv+ReLU)x num_cnn_layers(층마다 채널 num_filters배로 증가, 기본 1->4->16->64)
              -> Flatten -> Linear -> ReLU                                        # 식(1)-(2)
  -> demand_features (B,k,N,demand_embedding_dim)

hour_of_day,day_of_week (B,k) -> (time_in_day, day_of_week one-hot) -> Linear -> temporal_emb
  -> N 차원으로 broadcast(모든 노드가 같은 시간대 컨텍스트 공유)                        # 식(4)

concat(demand_features, temporal_emb) -> (B*N,k,·) -> LSTM -> 마지막 hidden h_last(B,N,·) # 식(3)-(4)

line_embeddings(N,line_dim, 전처리로 미리 계산) -> Linear -> ReLU -> context_emb(N,·)     # 식(6)

concat(h_last, context_emb) -> Linear -> Sigmoid -> [0,1] 정규화 공간                     # 식(7)-(8)
  -> Min-Max 역정규화 -> logits(B,H,W)
loss = CombinedLoss(logits, labels)                       # baseline/ADFormer/STResnet과 동일 loss
```

### 1-1. Spatial View: Local CNN

`master`의 `GridDemandModel`이 쓰던 `idx_table`(H,W,a로만 정해지는 순수 함수, `a=(patch_size-1)//2`)
+ `_crop_all_nodes()`("N개 노드 전체의 로컬 윈도우를 한 번의 배치 연산으로 추출") 메커니즘을 그대로
재사용한다. `mask_table`은 재사용하지 않음 — 논문 자체가 "we use zero padding for location at
boundaries of the city"라고 명시해서, master의 Transformer처럼 경계를 별도 EDGE 토큰으로 치환할
필요가 없다(zero-padding이 곧 이미지 픽셀 값).

`LocalCNN`(`Conv2d`+`ReLU` 스택 -> `Flatten` -> `Linear`)은 사용자의 기존 구현
(`/home/jinsu/PycharmProjects/DMVST/models/DMVSTModel.py`)을 기반으로 포팅했다. 두 가지가 원본과
다르다: (1) 입력 leading 차원(`B,k,N`)을 모두 배치로 접어 넣도록 확장(원본은 `B,T` 2차원만 지원),
(2) `Flatten`->`Linear` 뒤에 원본엔 없던 `ReLU`를 추가(식(2)/(6) 관련 결정 — 아래 2절 참고). 원본
자체가 논문 대비 BatchNorm이 빠져있는데(논문 6쪽 "Batch normalization is used in the local CNN
component"), 이 누락은 그대로 물려받았다(아래 2절 참고).

### 1-2. Temporal View: LSTM

`hour_of_day`/`day_of_week`(ADFormer와 동일한 산술 계산: `t%24`,`(t//24)%7`, 실제 달력 불필요)로
외부 요인을 구성해 local CNN 임베딩과 concat 후 노드별 독립 시계열로 LSTM에 넣는다(`(B*N,k,·)`로
reshape — `master`가 이미 "노드별 계산은 그대로 유지하되 배치 차원만 늘린다"는 원칙을 확립).

### 1-3. Semantic View: DTW + LINE 그래프 임베딩

- `preprocessing/build_dmvst_graph.py`(**`DA` env**): train split을 Min-Max `[0,1]` 정규화한 뒤
  "average weekly demand time series"(168시간, 논문 Semantic View 절 — 요일별 차이를 보존해야
  하므로 ADFormer의 24시간 daily profile을 그대로 못 씀)를 만들고, `fastdtw`(ADFormer의
  `compute_dtw_matrix`와 동일 로직)로 지역 쌍 DTW 거리를 계산, 식(5) `ω_ij = exp(-DTW(i,j))`로
  유사도 변환해 `{city}_dmvst_graph_edges.csv`(완전연결, 양방향)로 저장한다.
- `preprocessing/build_dmvst_line_embeddings.py`(**반드시 `Torch` conda env로 실행**, `cogdl`
  의존): 위 CSV를 읽어 `cogdl.models.emb.line.LINE`으로 임베딩을 계산, `{city}_dmvst_line_embeddings.npy`
  (`(N, line_dim)`)로 저장한다. `DA` env에는 `cogdl`이 없고 새로 설치하지도 않는다 — 이 1회성
  전처리만 별도 env로 분리하고, 실제 학습(`train.py`, `DA` env)은 결과 `.npy`만 읽는다.
- `DMVSTModel.__init__`은 이 `.npy`를 `np.load` 후 **연산 없이** `register_buffer(persistent=True)`로
  등록하고, `context_embedding_layer`(Linear+ReLU)를 forward마다 적용한다.

## 2. 논문/기존 구현 대비 우리가 내린 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 노드별 샘플링 | per-node-per-timestep(논문/기존 구현 원안) → all-node 배치(한 스텝=한 시간대, N개 노드 동시 예측) | `668aae8`에서 이미 확립한 프로젝트 전체 원칙("노드별 backprop 불필요, 노드별 연산은 유지하되 배치 차원만 늘림") — `master`의 `idx_table`/`_crop_all_nodes` 재사용으로 자연스럽게 해결 |
| Semantic view 그래프 엣지 가중치 | 기존 구현(`Lambda-F`)의 raw DTW distance 그대로 사용 → 식(5) `exp(-DTW)`로 수정 | 원본이 유사도 대신 거리를 그대로 넣어 논문과 반대 의미가 됨(멀수록 강하게 연결) |
| DTW 입력 프로파일 길이 | ADFormer의 24시간(daily) → 168시간(weekly) | 논문이 명시적으로 "average weekly demand time series"를 요구 — daily로 하면 요일별 차이(semantic view의 존재 이유)가 사라짐 |
| DTW 입력 정규화 | raw demand → train split Min-Max `[0,1]` 정규화 후 계산 | 168시간 raw demand로 DTW를 계산하면 거리 스케일이 커져 `exp(-DTW)`가 float32에서 대량 언더플로(0)돼 일부 노드가 고립됨(Codex 리뷰에서 Porto 11개 노드 확인) — 정규화로 논문과 동일한 스케일 유지 |
| 식(2)/(6)의 활성화 함수 | 기존 구현(`Lambda-F`)엔 없음 → ReLU 추가 | 논문이 두 FC 모두 `f(Wx+b)`(f=ReLU)로 명시(Codex 리뷰에서 지적) |
| `line_embeddings_path` | `configs/dataset/*.yaml`의 상대경로 그대로 → `train.py`에서 절대경로로 변환해 config에 저장 | 상대경로면 체크포인트를 다른 작업 디렉터리에서 `from_pretrained()`할 때 파일을 못 찾음(Codex 리뷰에서 `/dev/shm` 이동 재현) |
| cogdl 의존성 | `DA` env에 설치 → **`Torch` env로 분리**한 1회성 전처리 스크립트 | `cogdl`이 무거운 GNN 라이브러리라 `DA` 환경을 깨뜨릴 위험 — 예전에 이 논문을 구현했던 `Torch` env에 이미 설치돼 있어 재사용 |
| loss / 하이퍼파라미터 기본값 | `Lambda-F`의 `criterion.gamma=0.1,eps=1.0` → gir 전체 관례인 `gamma=1.0,eps=0.5` 따름 | baseline/ADFormer/STResnet과 동일 loss로 비교의 유일한 변수를 아키텍처로 한정 |
| 외부 요인 | `Lambda-F`(4개 날씨 피처) → day_of_week/hour_of_day 산술 계산 | 날씨 데이터 없음, ADFormer/STResnet과 동일 정책(공휴일 미사용, 요일 주기성만 학습) |
| Local CNN의 BatchNorm | 논문 본문(6쪽, "Batch normalization is used in the local CNN component") → 미구현 | `Lambda-F` 원본 구현 자체에 BatchNorm이 없어 그대로 물려받음(의도적 결정이 아니라 원본 코드의 논문 대비 누락) — Codex 문서 리뷰에서 지적, 향후 재검토 대상 |

## 3. HuggingFace 통합

- `models/config.py`: `DMVSTConfig(PretrainedConfig)` — `H,W,time_step,patch_size,num_filters,
  num_cnn_layers,kernel_size,demand_embedding_dim,temporal_embedding_dim,context_embedding_dim,
  lstm_hidden_size,lstm_num_layers,lstm_dropout,line_dim,line_embeddings_path,demand_min,
  demand_max,loss_gamma,loss_eps` 등.
- `models/modeling.py`: `DMVSTModel(PreTrainedModel)` — `forward(demands, hour_of_day, day_of_week,
  labels=None, sample_idx=None)` → `{'loss','logits'}`(`logits.shape==(B,H,W)`).
- `idx_table`, `line_embeddings` 모두 `register_buffer(persistent=True)`로 등록해 `save_pretrained`/
  `from_pretrained`로 정확히 복원됨(검증 완료). `__init__`에서 numpy 실데이터와 새로 만든 torch
  텐서를 직접 연산하지 않고(연산 없이 그대로 wrap) buffer로만 등록 — transformers 5.0의
  `from_pretrained` meta-device 버그(STResnet/ADFormer에서 확인)를 처음부터 회피.
- `_init_weights`도 오버라이드하지 않음(같은 이유의 다른 meta-device 버그 회피).

## 4. 엔트리 포인트

**학습 전 1회 필수 전처리** (다른 브랜치엔 없던 단계):
```bash
conda run -n DA python preprocessing/build_dmvst_graph.py --city ulsan   # 또는 porto
conda run -n Torch python preprocessing/build_dmvst_line_embeddings.py --city ulsan
```

`train.py`는 baseline·ADFormer·STResnet과 동일한 구조(Trainer/TrainingArguments/compute_metrics/
LoggingCallback/save/evaluate) — 모델 생성부와 dataset이 반환하는 배치 키만 다르다.
`test.py`는 애초에 Trainer를 안 쓰고 별도 `DataLoader`+평가 루프로 동작하는 독립 스크립트라(다른
세 브랜치도 마찬가지), 여기서도 배치 키(`demands`,`hour_of_day`,`day_of_week`)와 `time_step`을
체크포인트 config에서 읽는 부분만 바뀌었다(STResnet의 `l_c/l_p/l_q` 처리와 동일 원칙).
