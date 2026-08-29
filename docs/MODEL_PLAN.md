# 수요 예측 모델 구현 계획

`(t-k, t-1)` 시간의 격자 수요로 `t` 시점의 **전체 격자(H*W개 노드)** 수요를 한 forward에서 함께
예측하는 baseline 모델. 원래는 노드마다 독립적으로 forward/backward를 도는 구조였지만, 노드별로
따로 backprop할 이유가 없어 샘플을 시간(t) 단위로 바꾸고 모델이 내부에서 N=H*W개 노드에 대해
벡터화 연산을 하도록 변경했다 (개별 노드가 보는 연산 자체는 이전과 동일 — 배치 차원만 늘어남, §2 참고).
`dataset_frame.GridDemandDataset`이 `(demands, labels, sample_idx)` 형태로 샘플을 공급하고
있고(→ `docs/STRUCTURE.md`), 이 문서는 그 위에 올라갈 모델/학습/평가를 설계한다.

## 표기

- `B`: batch, `k`: 입력 시간 길이(`time_step`), `t`: 예측 대상 시각
- `H, W`: 도시별 격자 크기 (ulsan 14×12, porto 19×20 — `grid_size=700m` 기준)
- `a`: 이웃 반경 (한 변 `(2a+1)`인 정사각 윈도우)
- `d_model`: transformer/embedding 차원

## 1. 모델 아키텍처

N=H*W(전체 노드 수). 아래 연산은 노드 0..N-1 전부에 대해 **벡터화**로 한 번에 계산되지만,
각 노드가 보는 입력(자기 자신의 (2a+1)^2 이웃 + 자기 CLS)은 서로 완전히 독립적이라
"노드마다 따로 도는 것"과 수치적으로 동일하다 (§2 참고).

```
노드 0..N-1 각각에 대해 (h,w) = divmod(node, W):
  각 (b, t)에 대해 (2a+1)^2개 이웃 위치 (h+di, w+dj), di,dj ∈ [-a, a]:
      격자 안  → emb = FourierEmbed(log1p(demand[b, t, h+di, w+dj]))   # (d_model,)
      격자 밖  → emb = special_emb[EDGE]                                # (d_model,)

  CLS = special_emb[CLS] + node_emb[node]  # (d_model,) — 노드 고유 임베딩을 CLS에 더함
  seq = concat([CLS, emb_1, ..., emb_(2a+1)^2])           # ((2a+1)^2+1, d_model)
  seq = seq + positional_emb                               # learnable, 길이 (2a+1)^2+1

x: (B, k, N, (2a+1)^2+1, d_model) → reshape (B*k*N, (2a+1)^2+1, d_model)
  → TransformerEncoder → CLS 출력만 추출                  # (B*k*N, d_model)
  → reshape (B, k, N, d_model) → permute → (B*N, k, d_model)
  → LSTM(batch_first=True) → 마지막 timestep 출력          # (B*N, d_model) → reshape (B,N,d_model)
  → Linear(d_model, 1) → Softplus                          # neural_pred (B, N)

병렬로 검색(retrieval) 브랜치가 같은 입력(로컬 윈도우)으로 ir_out (B,N)을 만들고(§1-5),
neural_pred와 ir_out을 sigmoid 게이트로 섞은 뒤 reshape:
  lambda = sigmoid(Linear([last_hidden, ir_out]))          # (B, N), 최종 레이어 직전 concat
  pred = lambda * neural_pred + (1 - lambda) * ir_out
  → reshape                                                 # (B, H, W)  (양쪽 다 비음수라 pred도 비음수)
```

**`assemble-no-ir` 브랜치는 검색(retrieval) 서브시스템 자체가 없다** — 위 검색 브랜치 결합(§1-5
포함)은 이 브랜치에 해당 안 되며, `pred = neural_pred`(daily/weekly 주기 브랜치까지 반영한 `fused`를
`output_proj`+`Softplus`에 통과시킨 값)를 그대로 최종 출력으로 쓴다(§8, `models/modeling.py`).

### 1-1. 스칼라 임베딩 `FourierEmbed` (교체 가능하게 설계)

```
γ(x) = [cos(2π b_1 x), sin(2π b_1 x), ..., cos(2π b_m x), sin(2π b_m x)]   # 2m = d_model
```

- 입력은 `log1p(demand)`로 정규화 후 통과 (raw demand는 skew가 커서 그대로 넣으면 aliasing 위험).
- `freqs (b_1..b_m)`는 `nn.Parameter`로 학습.
- 다른 임베딩 방식으로 바꿀 수 있게 `ScalarEmbedding` 인터페이스(예: `nn.Module` + `forward(x) -> (..., d_model)`)로 감싸서, `FourierScalarEmbedding`을 기본 구현체로 둔다.

### 1-2. 특수 토큰

- `nn.Embedding(2, d_model)`: index 0 = CLS, index 1 = EDGE(격자 밖 이웃 대체).
- EDGE 토큰은 `demand=0`과 의미가 다름(실제 데이터가 없는 위치) → Fourier로 계산하지 않고 이 토큰으로 대체.

### 1-2-1. 노드 고유 임베딩

- `nn.Embedding(H*W, d_model)`. 모든 노드를 한 번에 예측하므로 `node_embed.weight`(N,d_model) 전체를
  그대로 `special_emb[CLS]`에 더해 N개 노드 각각의 CLS를 구성한다 (`CLS = special_emb[CLS] + node_embed.weight`).
- 이웃 수요 패턴만으로는 "이 노드가 원래 어떤 위치인지"(상습 hotspot vs 조용한 곳 등)를 구분 못 하는 문제를 보완 — CLS 자체가 "이 노드 전용 CLS"가 되어 스스로 attention부터 노드 정체성을 반영.
- 노드 정체성은 각 timestep(`k`)마다 동일하므로, CLS에 한 번 실어서 encoder 전체 시퀀스에 전파되게 함.

### 1-3. Positional embedding

- `nn.Parameter(shape=((2a+1)^2 + 1, d_model))`, learnable, 시퀀스에 elementwise 덧셈.
- CLS 포함 전체 시퀀스 길이만큼 슬롯을 두고, 인덱스 0을 CLS 자리로 고정.

### 1-4. 하이퍼파라미터 (기본값, `configs/model/*.yaml`에서 조정)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `a` | 2 | 이웃 반경 → 5×5=25 이웃 + CLS = 26 토큰 |
| `d_model` | 64 | 임베딩/transformer 차원 |
| `n_freqs` (m) | 32 | `2m = d_model` |
| `n_layers` | 2 | TransformerEncoder layer 수 |
| `n_heads` | 4 | attention head 수 |
| `dim_feedforward` | 128 | transformer FFN 차원 |
| `dropout` | 0.1 | transformer/lstm dropout |
| `lstm_hidden` | 64 (=`d_model`) | LSTM hidden size |
| `lstm_layers` | 1 | LSTM layer 수 |
| `retrieval_k` | 20 | 검색(retrieval) top-k — **`assemble-no-ir`에는 없는 필드**(§1-5 참고) |
| `time_step` | 24 | 검색 DB 슬라이딩 윈도우 길이(`train.py`가 `dataset.time_step`으로 주입, yaml에 직접 안 둠) — **`assemble-no-ir`에는 없는 필드** |
| `npy_path` | (없음, 필수) | 검색 DB를 만들 원본 grid npy **절대경로**(`train.py`가 `dataset.npy_path`를 resolve해서 주입) — **`assemble-no-ir`에는 없는 필드** |

## 1-5. 검색(retrieval) 앙상블 브랜치

**`assemble-no-ir` 브랜치에는 이 절 전체(검색 DB, `_retrieve`, `lambda` 게이트)가 존재하지 않는다**
— `GridDemandModel`에 `build_retrieval_db`/`_retrieve`/`lambda_layer`가 전부 제거됐고,
`GridDemandConfig`/`configs/model/baseline.yaml`에도 `time_step`/`npy_path`/`retrieval_k` 필드가
없다. 아래 설명은 이 서브시스템이 있는 다른 브랜치(`ir`, `ir-weather`, `assemble`,
`assemble-random-init` 등) 기준이다.

`/home/jinsu/PycharmProjects/DMVST` 저장소 `ir` 브랜치(`IRModule`)에서 아이디어를 가져온
브랜치(이 저장소도 브랜치명이 `ir`). 뉴럴 브랜치(위 1절)와 별도로, 같은 위치(node)의 **과거** 로컬
윈도우 중 지금 입력과 코사인 유사도가 가장 높은 top-k를 찾아 그 시점 실제 수요값을 softmax
가중평균한 값(`ir_out`)을 만들고, `sigmoid` 게이트로 뉴럴 브랜치 예측과 섞는다
(`GridDemandModel.build_retrieval_db`/`_retrieve`, `models/modeling.py`).

**검색 DB(`build_retrieval_db`)**: `config.npy_path`의 원본 grid 전체 시계열(T,H,W)을
`_crop_all_nodes`로 한 번에 크롭(B=1,k=T로 호출)한 뒤, `time_step` 길이 슬라이딩 윈도우로 잘라
절대 시간 인덱스 `t`(0..T-1)로 인덱싱되는 3개 텐서를 만든다:
- `retrieval_keys (N,T,flat_dim)`, `flat_dim = n_neighbors * time_step`
- `retrieval_values (N,T)` — 그 시점 실제 수요
- `retrieval_norms (N,T)` — key 벡터 L2 norm(코사인 유사도 분모 캐시)

`t < time_step` 구간은 유효한 윈도우가 없어 0으로 남겨두고, 후보 슬라이스가 항상
`[time_step, t)`(자기 자신 미만)로 제한되므로 미래 시점을 참조하지 않는다. 검색(`_retrieve`)은
gir가 이미 한 샘플 = 한 시각(t)에 N개 노드 전부를 담고 있다는 점을 이용해, 원본처럼 배치 안
`node_id`별로 루프 도는 대신 **배치 원소(B)별로만** 루프를 돈다(후보 구간이 노드에 무관하게
b 하나당 하나로 통일되기 때문 — B~4-8회로 원본보다 훨씬 적은 반복).

**중요 — `persistent=False` + 수동 재로드**: 위 3개 버퍼는 `config.npy_path`의 순수 함수지만
크기가 커서(ulsan ~1.7GB, porto ~4GB, fp32) 다른 buffer(`idx_table`/`mask_table`)처럼
`persistent=True`로 체크포인트에 통째로 저장하지 않는다. 대신:
- `GridDemandModel(config)` fresh construction(`train.py`) 시엔 `__init__`이 자동으로
  `build_retrieval_db()`를 호출해 정상적으로 채워짐.
- **`GridDemandModel.from_pretrained(...)`로 로드한 경우, 반드시 그 직후 `model.build_retrieval_db()`
  를 다시 호출해야 함** — transformers 5.0의 meta-device fast-init이 `persistent=False` buffer를
  체크포인트에서 복원하지 않아 깨진 채로 남기 때문(이 프로젝트가 4개 브랜치 내내 확인해온 것과 동일
  원인). `test.py`가 이미 이렇게 구현돼 있음: `from_pretrained(...)` → (config.npy_path/time_step과
  `--npy_path`/`--time_step` 일치 검증) → `build_retrieval_db()`(CPU 상태에서) → `.to(device)` 순서.
  이 순서(재구성을 `.to(device)` **전에**)가 중요함 — 반대로 하면 깨진 buffer가 GPU로 옮겨진 뒤
  재구성돼서 old+intermediate+new 버퍼가 동시에 존재하는 메모리 스파이크가 생김.
- 새 checkpoint를 다른 방식으로 로드하는 코드를 추가할 때도 반드시 `build_retrieval_db()`를 그
  직후에 호출해야 한다 — 빠뜨리면 검색 브랜치가 조용히 0만 반환하는 위험한 실패 모드가 됨.

## 2. 배치 크롭 구현 방식 (`GridDemandModel._crop_all_nodes`)

N=H*W개 노드 전부를, 그것도 배치 전체에 대해 파이썬 for-loop 없이 한 번에 잘라낸다:

1. `demands: (B, k, H, W)`를 `F.pad`로 사방 `a`만큼 0-padding → `(B, k, H+2a, W+2a)` → `(B,k,-1)`로 펼침.
2. `__init__`에서 노드 0..N-1 전부에 대해 미리 계산해둔 `idx_table`/`mask_table`(N, n_neighbors) —
   각 노드의 `(2a+1)×(2a+1)` 이웃이 padded grid의 어느 flat index에 해당하는지, 격자 밖인지 여부.
3. `padded_flat[:, :, idx_table.reshape(-1)].reshape(B,k,N,n_neighbors)` — advanced indexing 한 번으로
   B,k 전체 × N개 노드의 이웃 값을 동시에 뽑아낸다 (`torch.gather` 없이 기본 인덱싱만으로 충분).
4. `mask_table`(N,n_neighbors, 배치/시간 축 없음)을 그대로 브로드캐스트해서 EDGE 토큰 대체 여부를 결정한다
   (0-padding 값과 EDGE 토큰 의미를 분리하기 위해 마스크는 padding 여부로 직접 계산하지, 값이 0인지로 유추하지 않는다).

`idx_table`/`mask_table`은 H,W,a로만 정해지는 순수 함수라 `node_id`를 인자로 받지 않고도(=모든 노드에 대해)
그대로 재사용 가능 — 예전에 특정 `node_id`의 행만 골라 쓰던 것을 전체 사용으로만 바꾼 것.

## 3. Loss

```python
class CombinedLoss(nn.Module):
    def __init__(self, gamma=1.0, eps=0.5, reduction='mean'):
        ...
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        term1 = diff ** 2                        # 절대 오차 제곱
        term2 = (diff / (y_true + self.eps)) ** 2  # 상대 오차 제곱
        loss = term1 + self.gamma * term2
        return loss.mean()  # or .sum()
```

- `y_true=0`(전체 샘플의 큰 비중)일 때 `term2 = y_pred² / eps²`로 0이 아닌 예측에 강하게 벌점 → `gamma`는 학습 초반 손실 스케일을 보고 튜닝 필요.
- 출력에 `Softplus`를 씌워 음수 예측 자체를 구조적으로 막는다.

## 4. 평가 지표

- **RMSE**: `sqrt(mean((y_true - y_pred)^2))`
- **MAE**: `mean(|y_true - y_pred|)`
- **MAPE(+1)**: `mean(|y_true - y_pred| / (|y_true| + 1)) * 100` — 0-수요가 많아 분모에 +1 스무딩. 표준 MAPE가 아니므로 로그/리포트에 `MAPE(+1)`로 명시.

## 5. HuggingFace 통합

- `models/config.py`: `GridDemandConfig(PretrainedConfig)` — 위 1-4 하이퍼파라미터 + `H, W`(도시별 격자 크기)를 필드로.
- `models/modeling.py`: `GridDemandModel(PreTrainedModel)` — `forward(demands, labels=None, sample_idx=None, ...)` → `logits`는 `(B,H,W)`(전체 노드), `labels`가 있으면 `{'loss', 'logits'}`, 없으면 `{'logits'}` 반환 (HF `Trainer` 호환). 검색 브랜치(§1-5)가 있는 브랜치에서는 `sample_idx`가 사실상 필수(`None`이면 `ValueError`)지만, **`assemble-no-ir`은 검색 브랜치가 없어 `sample_idx`를 시그니처로 받기만 하고 forward 내부에서 쓰지 않는다**(`remove_unused_columns: false`로 `Trainer`가 넘기는 배치 키를 그냥 다 받아주는 것 — `None`이어도 에러 없음). `GridDemandDataset`이 반환하는 `sample_idx`는 어느 브랜치든 split-local idx가 아니라 **절대 시간 인덱스** `t`임에 유의.
- 노드별 backprop을 없애고 한 forward에서 N=H*W개 노드를 다 예측하도록 바꾸면서 step당 연산량이 N배로
  늘어남 (ulsan N=168, porto N=200) → `configs/config.yaml`의 `per_device_train/eval_batch_size`를
  그만큼 낮춰야 함 (기본값 4/8, 실제 GPU 메모리에 맞춰 조정).
- `trainer.save_model(output_dir)` / `model.save_pretrained(output_dir)` → `config.json` + 가중치가 한 번에 저장되어, `test.py`에서 `GridDemandModel.from_pretrained(checkpoint_path)`만으로 아키텍처+가중치 복원 가능 (hydra model config 재참조 불필요).

## 6. Hydra 설정 구조

```
configs/
├── config.yaml          # defaults: [dataset: ulsan, model: baseline]
├── dataset/
│   └── *.yaml           # city(→ data/raw/{city}_temporal_grid.npy), time_step, train/val/test 시간 분할
└── model/
    └── baseline.yaml    # 위 1-4 표의 하이퍼파라미터 + loss(gamma, eps)
```

## 7. 엔트리 포인트

- `train.py`: `@hydra.main(config_path="configs", config_name="config")` → dataset/model 생성 → `TrainingArguments`(`cfg.train`) → HF `Trainer` → `trainer.train()` → `trainer.save_model()` (hydra run dir = `output/${project_name}/...`, `configs/config.yaml`의 `hydra.run.dir`과 일치).
- `test.py`: `checkpoint_path` 인자만 받아 `GridDemandModel.from_pretrained(checkpoint_path)`로 복원 후, 별도 test 시간 구간 `GridDemandDataset`으로 RMSE/MAE/MAPE(+1) 계산. 학습 프로세스와 완전히 분리된 독립 실행.

## 8. daily/weekly 주기 브랜치 + null-option attention 융합 (`assemble`/`assemble-random-init` 브랜치)

`ir-weather`(검색기 앙상블 + 날씨/캘린더) 위에, 사용자가 별도 제공한 참고 구현("main model",
N2MSDWGateTarget)에서 두 아이디어만 가져와 얹은 브랜치다.

- **daily/weekly 주기 브랜치**: "정확히 같은 시각"의 과거 6일치(daily)/4주치(weekly) 수요를
  `dataset_frame/grid_demand_dataset.py`가 `daily_demands`/`weekly_demands`(오래된 것→최근 것 순,
  `[t-144,...,t-24]`/`[t-672,...,t-168]`)로 반환한다. 가장 먼 daily/weekly lag 둘 다 항상 유효해야
  하므로 `t_start` 하한이 `max(time_step, daily_lag_count*24, weekly_lag_count*168)`로 올라간다
  (기본값 6/4 기준 672시간) — 부족분을 0-패딩하지 않고 무효 구간 자체를 샘플에서 제외하는 이
  클래스의 기존 관례를 그대로 따름. 이 때문에 train 샘플 수가 ulsan -21%, porto -11% 감소한다.
- **공유 vs 독립 가중치**: 로컬 `(2a+1)²` 공간 인코딩(`scalar_embed`/CLS 구성/`pos_embed`/
  `encoder`)은 recent/daily/weekly 세 브랜치가 `GridDemandModel._spatial_encode()`라는 단일 공유
  메서드를 통해 **동일 가중치**로 처리한다. 반면 그 뒤 시간축 취합은 `self.lstm`(recent)/
  `self.daily_lstm`/`self.weekly_lstm`로 **브랜치마다 독립된 가중치**를 쓴다. 날씨/캘린더는
  recent에만 주입되고(`_spatial_encode`의 `weather` 인자가 `None`이면 CLS에 안 더함), daily/weekly는
  순수 수요값만 본다 — main model 자체도 주기 브랜치엔 날씨/캘린더를 안 넣는 것과 동일한 설계.
- **null-option attention 융합**: recent LSTM 출력(`last`, vec_1)을 Query, daily/weekly LSTM
  출력(각각 `daily_projection`/`weekly_projection`으로 투영한 것)을 Key/Value 후보로 하고,
  학습 가능한 "null" key를 하나 더 둬서 attention이 "daily도 weekly도 안 쓰겠다"를 선택할 자유를
  준다. `periodic_correction`(vec_2)은 이 softmax 가중치로 만든 daily/weekly 가중합이고, 이후
  `neural_pred`/검색기 게이트 입력 전부 `last` 대신 `fused`를 쓴다(검색기 앙상블이 있는 브랜치에서는
  그 자체, 즉 `retrieval_query`/`_retrieve`는 recent 윈도우만 그대로 사용 — daily/weekly와 무관,
  스코프 밖. **`assemble-no-ir`은 검색기 앙상블이 아예 없으므로 `fused`가 `output_proj`+`Softplus`를
  거쳐 바로 최종 `pred`가 된다** — §1 참고).
- **`assemble`(0-init)**: `daily_projection`/`weekly_projection`을 0-init해서, 학습 시작 시점엔
  `daily_last`/`weekly_last`가 정확히 0 → `periodic_correction=0` → `fused==last`, 즉 순수
  `ir-weather`와 동일하게 시작하고 daily/weekly의 기여는 학습되며 서서히 커진다("중립 시작").
  **주의**: `PreTrainedModel.post_init()`이 내부적으로 호출하는 `init_weights()`가 (이
  코드베이스가 `_init_weights`를 오버라이드하지 않으므로) 기본 구현을 통해 모든
  `nn.Linear.weight`를 `normal_(0,0.02)`로 재초기화한다(bias는 0으로) — `__init__` 중간에
  0-init을 하면 뒤이은 `self.post_init()`이 지워버리므로, 0-init은 반드시
  **`self.post_init()` 호출 이후**에 해야 실제로 유지된다(`models/modeling.py` 개발 중
  확인/수정함). 이 버전의 융합식은 `fused = last + residual_scale * periodic_correction`.
- **`assemble-random-init`(랜덤 초기화 + 스케일 보정, ablation)**: `daily_projection`/
  `weekly_projection`/`periodic_null_key`를 0-init하지 않고 `post_init()`의 기본 랜덤 초기화
  그대로 둔다 — "중립 시작" 없이 학습 시작부터 daily/weekly 신호가 섞인 채로 출발하면 결과가
  어떻게 달라지는지 보는 ablation. 여기에 추가로, `vec_1=last`는 항상 풀스케일로 더해지고
  `vec_2=periodic_correction`만 null-option 가중치에 따라 크기가 흔들리다 보니 `fused`의 노름이
  샘플마다 들쭉날쭉해지는 문제가 있어 스케일 보정을 넣었다:
  ```python
  scale_correction = (1 - residual_scale * ||periodic_correction|| / (||last|| + eps)).clamp(min=0)
  fused = scale_correction * last + residual_scale * periodic_correction
  ```
  `periodic_correction`이 0에 가까울수록 `scale_correction→1`(`fused≈last`)이고, daily/weekly가
  강하게 관여할수록 `last`의 기여가 줄어 `fused`의 스케일이 안정된다(두 벡터가 평행하지 않아
  완벽한 노름 보존은 아님). 단, 이 브랜치는 랜덤 초기화라 학습 시작 시점에 `periodic_correction`이
  정확히 0이 되는 건 아니므로 `assemble`의 "중립 시작" 보장은 없음 — 오직
  `periodic_correction=0`이 되는 매 순간(예: null-option이 완전히 이길 때)에 `fused=last`가
  된다는 구조적 성질만 보장된다.

## 9. 검색(retrieval) 서브시스템 제거 ablation (`assemble-no-ir` 브랜치)

`assemble-random-init`(§8)은 검색기 앙상블(`ir`)과 daily/weekly 주기 브랜치가 동시에 들어가 있어서
"주기 브랜치 자체가 도움이 되는지"를 검색기 효과와 분리해서 볼 수 없다. `assemble-no-ir`은
`assemble-random-init`에서 분기해 **검색(retrieval) 서브시스템만 제거**한 것 — 순수하게
"baseline-weather(주기 브랜치도 검색기도 없는 원본) + daily/weekly 주기 브랜치"가 baseline-weather
대비 나은지 확인하기 위한 ablation이다.

- `GridDemandModel`에서 `build_retrieval_db`/`_retrieve`/`lambda_layer` 제거 → `pred = neural_pred`
  (daily/weekly 융합 이후의 `fused`를 `output_proj`+`Softplus`에 통과시킨 값)를 그대로 최종 출력으로
  사용(§1, §8 참고).
- `GridDemandConfig`/`configs/model/baseline.yaml`에서 `time_step`/`npy_path`/`retrieval_k` 제거.
- `sample_idx`는 시그니처에 남아있지만(`Trainer`의 `remove_unused_columns: false` 때문에 받아야 함)
  forward 내부에서 쓰이지 않음.
- `test.py`도 검색 서브시스템의 소비자였음 — 체크포인트 `config.npy_path`/`config.time_step`과
  `--npy_path`/`--time_step` 일치 검증, `build_retrieval_db()` 호출을 제거하고
  `from_pretrained(...).to(device)`로 바로 로드하도록 단순화(`baseline-weather`의 `test.py`와
  동일한 형태). `--npy_path`/`--time_step` 자체는 평가 데이터셋 생성에 여전히 필요하지만, 값이
  학습 때와 어긋나도 더 이상 명확한 에러로 막아주지 않는다(검색 DB 보호용이었을 뿐이라 이 ablation
  범위 밖 — `baseline-weather`도 원래 이 검증이 없었다).
- 기존 `assemble-random-init` 체크포인트와는 `lambda_layer` 제거로 파라미터 구조가 달라져 완전
  비호환 — 새로 학습해야 한다.

## 10. 아직 정해지지 않은 것 / 기본값으로 진행할 것

- `a`, `d_model`, `n_freqs` 등 정확한 하이퍼파라미터 값 — 위 표를 기본값으로 두고 `configs/model/baseline.yaml`에서 조정.
- train/val/test 시간 분할 비율 — 별도 지시 없으면 시간순 70/15/15로 가정.
- `CombinedLoss`의 `gamma` 초기값 — 1.0으로 시작, 학습 로그 보고 조정.
- 출력 비음수 제약 방식 — `Softplus` 기본 채택.
