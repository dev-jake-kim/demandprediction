# ADFormer 포팅

논문 "ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting"
(arXiv:2506.02576, 공식 구현 https://github.com/decisionintelligence/ADFormer, PDF는
`docs/ADFormer.pdf`)을 `master`의 baseline(로컬 이웃 crop + Transformer + LSTM, `docs`에서 삭제됨)과
**같은 데이터·같은 지표·같은 학습 하네스**(`train.py`/`test.py`, HF `Trainer`)로 비교하기 위해
이식했다. 공식 구현(`model/module.py`, `model/ADFormer.py`, `utils/ADFormer_dataset.py`,
`utils/ADFormer_config.py`)을 직접 읽고 그대로 옮겼다 — 논문 본문 수식만으로는 SCA의 `M_sep^S`
라우팅, TAA의 `M_sep^Tmp` 복원 matmul이 불명확해서 공식 코드로 확인했다.

## 표기

- `B`: batch, `T`(=`time_step`): 입력 시간 길이, `N=H*W`: 전체 지역(격자셀) 수
- `H,W`: 도시별 격자 크기 (ulsan 14×12=168, porto 10×20=200)
- `d`(=`embed_dim`): 임베딩 차원, `M_i`: `i`번째 클러스터 레벨의 클러스터 개수

## 1. 모델 아키텍처 (`models/modeling.py`)

```
demands (B,T,H,W) -> x_raw (B,T,N,1) -> 표준화 (demand_mean/std)
DataEmbedding: value(Linear) + sinusoidal PE + daytime_embedding(Embedding(1440,d)) +
               weekday_embedding(Embedding(7,d)) + spatial_embedding(Linear(SE_dim,d)(학습 파라미터))
             -> x (B,T,N,d)

각 클러스터 레벨 i: agg_raw = einsum('mn,btnd->btmd', cluster_map_i, x_raw_norm)  # (B,T,M_i,1)
                  dtw_agg_x[i] = DataEmbedding_i(agg_raw, ..., spa_cls_emb[i])   # (B,T,M_i,d)

L개 STEncoder 층, 각 층에서:
  SDA  = SpatialDiffAttn(x)                     # N개 지역 전체 dense differential attention
  SCA  = Σ_i SpatialAttn(dtw_agg_x[i], cls_sep[i])   # 클러스터 self-attn -> 지역으로 라우팅
  TSA  = TemporalAttn(x, agg=False)              # 지역별 T 구간 표준 self-attn
  TAA  = TemporalAttn(x, agg=True, tmp_gate)      # 학습 쿼리로 T를 P로 집약 -> day/hour 게이트로 T 복원
  out  = out_proj(concat([SDA,SCA,TSA,TAA]))
  x    = LayerNorm(DropPath(out) + x) -> MLP -> LayerNorm(...)  # post-norm, residual
  skip += skip_conv_l(x)   # 레이어별 1x1 conv, 누적

end_conv1(window T -> horizon 1) -> end_conv2(skip_dim -> 1) -> 역표준화 -> logits (B,H,W)
```

### 1-1. Spatial Differential Attention (SDA)

공식 구현의 `flash_attn_func` 4-way 조합을 `F.scaled_dot_product_attention`으로 재현(수학적으로 동일):
`Q,K,V`를 2등분(`q1,q2,k1,k2,v1,v2`)해서 `attn1=[Attn(q1,k1,v1),Attn(q1,k1,v2)]`,
`attn2=[Attn(q2,k2,v1),Attn(q2,k2,v2)]`를 구하고,
`λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init(depth)`로 `attn1 - λ·attn2` → RMSNorm → `*(1-λ_init)`.
매 timestep마다 N개 지역 전체에 대한 dense attention이라 baseline(로컬 `(2a+1)^2` 윈도우)과
근본적으로 다른 receptive field를 가진다.

### 1-2. Spatial Cluster Attention (SCA)

- `cluster_map_i` (`(M_i,N)`, 이진, `persistent=True` 버퍼): DTW 기반 클러스터링 결과 — 원본 데이터를
  클러스터 단위로 **집계**(`einsum`)할 때만 씀.
- `cls_sep[i]` (`(M_i,N)`, 학습 가능 `nn.Parameter`, 초기값 `randn*cluster_map_i`): SpatialAttn 안에서
  클러스터-클러스터 self-attention 점수를 지역(N) 단위로 **라우팅**할 때 씀 (`cls_map.T @ attn`).
  둘을 분리한 이유: 데이터 집계는 고정된 하드 클러스터 배정을 따라야 하지만, "이 지역이 그 클러스터의
  패턴을 얼마나 참고할지"는 학습되는 게 자연스럽기 때문(공식 구현과 동일한 설계).

### 1-3. Systemic Temporal Attention (TSA/TAA)

`TemporalAttn`(`agg` 플래그로 TSA/TAA 겸용): TSA는 표준 self-attn(쿼리=T), TAA는 지역별
`nn.Parameter(N,P,d)` 학습 쿼리로 T를 P개 세그먼트로 압축한 뒤, `tmp_gate`(day/hour를
`Linear(8,P)`로 투영, `STEncoder`에서 계산)와의 matmul로 **다시 T 길이로 복원**한다 — 그래야 TSA
출력과 길이가 맞아 `STAttention`에서 concat 가능.

## 2. 논문/공식 구현 대비 우리가 내린 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| `flash_attn_func` | `F.scaled_dot_product_attention`으로 대체 | flash-attn은 CUDA 전용 별도 패키지 — 표준 PyTorch만으로 수학적으로 동일하게 재현 가능 |
| `cluster_reg_nums` | ulsan `[40,10]`, porto `[48,12]` (공식 기본 `[64,16]`, N=263 기준) | 우리 N=168/200에 맞게 축소 |
| loss / 평가지표 | `CombinedLoss`(`models/losses.py`), `compute_regression_metrics`(`models/metrics.py`) — baseline과 완전히 동일 재사용 | "동일 지표로 비교"가 목적이라 비교의 유일한 변수를 아키텍처로 한정 |
| 정규화 | z-score(`demand_mean/std`, train split에서만 계산) 후 모델 출력에서 역정규화, loss는 실제 수요 단위로 계산 | 공식 구현의 `StandardScaler`와 동일한 아이디어, baseline과 같은 스케일로 지표 비교 가능하게 함 |
| 출력 활성함수 | 없음 (공식 구현 그대로, baseline의 Softplus와 다름) | 논문 원안 유지 — 음수 예측이 나올 수 있음(비교 시 유의) |
| day/hour 라벨 | 절대 npy 인덱스 `t`로부터 `t%24`, `(t//24)%7` 산술 계산 (실제 달력 날짜 미사용) | ADFormer는 공휴일을 안 쓰고 요일의 "주기성"만 학습 — t=0을 무슨 요일로 보든 실제 요일↔label 매핑이 전역적으로 일관되기만 하면 학습에 무해함 |
| `dtw_map`/`bal_cls` 클러스터링 | `preprocessing/build_cluster_map.py`로 전처리 단계에서 미리 계산(공식 `get_dtw`/`get_cluster`/`hierarchical_clustering` 이식), train split만 사용 | 시간 리크 방지, 공식 구현과 동일 |

## 3. HuggingFace 통합

- `models/config.py`: `ADFormerConfig(PretrainedConfig)` — `H,W,time_step,cluster_map_path,cluster_reg_nums,demand_mean,demand_std` 등.
- `models/modeling.py`: `ADFormerModel(PreTrainedModel)` — `forward(demands, hour_of_day, day_of_week, labels=None, sample_idx=None)` → `{'loss','logits'}` (`logits.shape==(B,H,W)`), baseline과 동일한 HF 반환 규약.
- `cluster_map_i` 버퍼, `cls_sep` 파라미터 모두 `save_pretrained`/`from_pretrained`로 정확히 복원됨 (검증 완료). `__init__`에서 numpy 실데이터와 새로 만든 torch 텐서를 직접 연산(`torch.randn(...)*torch.from_numpy(...)`)하면 transformers 5.0의 meta-device fast-init 경로에서 device mismatch 에러가 나서, numpy로만 계산 후 마지막에 한 번만 `torch.from_numpy`로 감쌌다 (`master`의 baseline 개발 때 겪은 것과 같은 종류의 버그).

## 4. 엔트리 포인트

- `preprocessing/build_cluster_map.py --city {ulsan,porto}`: `data/raw/{city}_cluster_maps.npz` 생성 (학습 전 1회 실행).
- `train.py`: `dataset.cluster_map_path`에서 클러스터 레벨 수를 자동 추론하고, train split grid로 `demand_mean/std`를 계산해 `ADFormerConfig`에 주입. 나머지(Trainer/TrainingArguments/compute_metrics/LoggingCallback/save/evaluate)는 baseline 때와 동일.
- `test.py`: `node_id` 없이 `hour_of_day`/`day_of_week`를 넘기는 것만 baseline과 다름, 나머지 동일.

## 4-1. PDF만 보고 리뷰 시 오탐 나는 두 지점 (공식 코드로 재확인 완료)

논문 PDF 원문(식 12, 16, 17)만 근거로 리뷰하면 아래 두 지점이 "논문과 다르다"고 잘못 지적될 수 있다
(실제로 Codex에 PDF만 첨부해 리뷰시켰을 때 둘 다 CHANGES_NEEDED로 지적됨). 공식 구현
(`github.com/decisionintelligence/ADFormer`, `model/ADFormer.py`)의 실제 코드로 재확인한 결과 둘 다
의도된 설계이며 수정 불필요:

- **SCA `cls_sep`가 매 forward마다 `cluster_map`으로 재마스킹되지 않는 것**: 공식 코드의
  `get_map_param()`도 `forward_map = torch.randn(map.size()) * map`를 `STEncoder.__init__`에서
  **한 번만** 호출해 `nn.Parameter`로 저장하고, `forward()`에서는 그 파라미터를 그대로 재사용한다
  (재마스킹 없음). 식(12) `M_sep^S = M_cls ⊙ M_sep`는 **초기화 공식**이지 forward마다 재적용하는
  제약이 아니다.
- **TAA `tmp_gate`가 순수 캘린더 피처(day/hour)에서만 계산되는 것**: 공식 코드도
  `STEncoder.forward`에서 `tmp_map = self.tmp_map_linear(add_inf.transpose(1, 2))`로 동일하게
  구현 — `add_inf`(외부/캘린더 피처)만 입력으로 쓰고 다른 hidden state는 안 씀. 식(17)의
  `X_full[:,:,D:D']`는 정확히 이 캘린더 피처 슬라이스를 가리키는 것으로 확인.

향후 PDF 기반 재리뷰를 돌릴 때는 이 섹션을 프롬프트에 같이 참고시켜서 같은 오탐이 반복되지 않게
할 것.

## 5. 미확정 / 향후 조정

- `per_device_train/eval_batch_size` 기본값(8/16)은 보수적 시작점 — N×N dense attention이라 baseline의 all-node(4/8→512/2048로 키웠던 것)보다 메모리 프로파일이 다르므로 실측 후 조정 필요.
- `torch_compile`은 우선 꺼둠(`false`) — SDPA + 커스텀 einsum/matmul이 많아 컴파일 그래프가 baseline보다 복잡할 수 있어, eager 모드로 먼저 정확성을 확인한 뒤 켜볼 것.
