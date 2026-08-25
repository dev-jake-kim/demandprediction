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
  → Linear(d_model, 1) → Softplus → reshape                # (B, H, W)  (수요는 음수 불가)
```

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
- `models/modeling.py`: `GridDemandModel(PreTrainedModel)` — `forward(demands, labels=None, sample_idx=None)` → `logits`는 `(B,H,W)`(전체 노드), `labels`가 있으면 `{'loss', 'logits'}`, 없으면 `{'logits'}` 반환 (HF `Trainer` 호환).
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

## 8. 날씨 + 캘린더 피처 (`baseline-weather` 브랜치)

`master`의 baseline은 순수 과거 수요값만 입력으로 쓴다(캘린더/날씨 전혀 없음). `baseline-weather`
브랜치는 여기에 날씨(기온/강수/적설)와 캘린더(시간대/요일) 정보를 추가한다.

- **날씨 임베딩**: Lambda-F 스타일 — `nn.Linear(3, d_model)` 한 번으로 투영(Fourier 미사용).
  입력 전 train split 통계로 표준화(`(weather - weather_mean) / weather_std`).
- **캘린더 임베딩**: ADFormer의 `DataEmbedding`과 동일 — `daytime_embedding: nn.Embedding(1440, d_model)`
  (`round(hour_of_day/24*1440).clamp(0,1439)`로 인덱싱) + `weekday_embedding: nn.Embedding(7, d_model)`.
- **주입 위치**: 기존 `CLS = special_emb[CLS] + node_embed[node]`에 날씨/캘린더 임베딩 합을 그대로
  더한다(`cls_emb += weather_emb + daytime_emb + weekday_emb`, 노드 축엔 무관하게 브로드캐스트) —
  이 프로젝트가 이미 쓰고 있는 "서로 다른 의미의 벡터를 CLS에 더해 합성"하는 패턴의 연장.
- **날씨 윈도우는 의도적으로 한 칸 밀려 있다**: 수요 입력 윈도우가 `[t-k, t-1]`이면 날씨는
  `[t-k+1, t]`를 쓴다(예측 대상 시점 `t` 자체의 날씨 포함). "그 시점 예보는 이미 안다"는 가정 —
  수요 자체를 미리 아는 것과는 다르며, 단기 기상 예보 정확도가 높다는 근거로 채택했다.
  **이건 leakage 버그가 아니라 의도된 설계다.** 캘린더 정보는 예보가 필요 없는 결정론적 정보라
  수요와 동일한 비이동 윈도우 `[t-k, t-1]`을 그대로 쓴다(ADFormer와 동일 계산).
- **날씨 CSV 로딩 시 주의(`dataset_frame/grid_demand_dataset.py`)**: 기상청 관측 데이터 관례상
  강수량/적설 빈 셀은 "관측값 없음"이 아니라 "0"을 의미한다(비/눈이 없으면 값을 아예 안 채움).
  `강수량(mm)`/`적설(cm)` 두 컬럼만 `fillna(0.0)`을 적용하고, 기온을 포함한 3개 컬럼 전체에 대해
  `np.isfinite` 검증을 둬서 진짜 결측/무한값은 zero-fill로 숨기지 않고 명시적 `ValueError`로 드러낸다.
- **모델 필수 인자**: `forward(demands, labels=None, sample_idx=None, weather=None, hour_of_day=None,
  day_of_week=None)` — `weather`/`hour_of_day`/`day_of_week` 중 하나라도 `None`이면 `ValueError`
  (ADFormer의 관례와 동일하게 이 브랜치에서는 세 값 다 필수 취급).
- `weather_mean`/`weather_std`는 `GridDemandConfig`에 저장되고(`train.py`가 train split에서만 계산해
  주입, leakage 방지), `GridDemandModel`에는 `persistent=True` buffer로 등록되어 `from_pretrained`
  이후 자동 복원된다(`idx_table`/`mask_table`과 같은 범주 — transformers 5.0 meta-device fast-init
  이슈 회피, §1-2-1 참고 패턴과 동일 이유).

## 9. 아직 정해지지 않은 것 / 기본값으로 진행할 것

- `a`, `d_model`, `n_freqs` 등 정확한 하이퍼파라미터 값 — 위 표를 기본값으로 두고 `configs/model/baseline.yaml`에서 조정.
- train/val/test 시간 분할 비율 — 별도 지시 없으면 시간순 70/15/15로 가정.
- `CombinedLoss`의 `gamma` 초기값 — 1.0으로 시작, 학습 로그 보고 조정.
- 출력 비음수 제약 방식 — `Softplus` 기본 채택.
