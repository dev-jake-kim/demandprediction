# 수요 예측 모델 구현 계획

`(t-k, t-1)` 시간의 격자 수요로 `t` 시점의 특정 노드(격자셀) 수요를 예측하는 baseline 모델.
`dataset_frame.GridDemandDataset`이 이미 `(demands, labels, node_id, sample_idx)` 형태로 샘플을
공급하고 있고(→ `docs/STRUCTURE.md`), 이 문서는 그 위에 올라갈 모델/학습/평가를 설계한다.

## 표기

- `B`: batch, `k`: 입력 시간 길이(`time_step`), `t`: 예측 대상 시각
- `H, W`: 도시별 격자 크기 (ulsan 14×12, porto 19×20 — `grid_size=700m` 기준)
- `a`: 이웃 반경 (한 변 `(2a+1)`인 정사각 윈도우)
- `d_model`: transformer/embedding 차원

## 1. 모델 아키텍처

```
node_id → (h, w) = divmod(node_id, W)

각 (b, t)에 대해 (2a+1)^2개 이웃 위치 (h+di, w+dj), di,dj ∈ [-a, a]:
    격자 안  → emb = FourierEmbed(log1p(demand[b, t, h+di, w+dj]))   # (d_model,)
    격자 밖  → emb = special_emb[EDGE]                                # (d_model,)

CLS = special_emb[CLS] + node_emb[node_id]  # (d_model,) — 노드 고유 임베딩을 CLS에 더함
seq = concat([CLS, emb_1, ..., emb_(2a+1)^2])           # ((2a+1)^2+1, d_model)
seq = seq + positional_emb                               # learnable, 길이 (2a+1)^2+1

x: (B, k, (2a+1)^2+1, d_model) → reshape (B*k, (2a+1)^2+1, d_model)
  → TransformerEncoder → CLS 출력만 추출                  # (B*k, d_model)
  → reshape (B, k, d_model)
  → LSTM(batch_first=True) → 마지막 timestep 출력          # (B, d_model)
  → Linear(d_model, 1) → Softplus                         # (B,)  (수요는 음수 불가)
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

- `nn.Embedding(H*W, d_model)`을 `node_id`로 조회해서 `special_emb[CLS]`에 더함 (`CLS = special_emb[CLS] + node_emb[node_id]`).
- 이웃 수요 패턴만으로는 "이 노드가 원래 어떤 위치인지"(상습 hotspot vs 조용한 곳 등)를 구분 못 하는 문제를 보완 — CLS 자체가 "이 노드 전용 CLS"가 되어 스스로 attention부터 노드 정체성을 반영.
- `node_id`는 각 timestep(`k`)마다 동일하므로, CLS에 한 번 실어서 encoder 전체 시퀀스에 전파되게 함.

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

## 2. 배치 크롭 구현 방식

`node_id`마다 크롭 중심이 다르므로 배치 전체를 파이썬 for-loop로 자르면 느림. 대신:

1. `demands: (B, k, H, W)`를 `F.pad`로 사방 `a`만큼 0-padding → `(B, k, H+2a, W+2a)`.
2. `h, w = divmod(node_id, W)`로 배치별 중심 좌표 계산 (padding 후 좌표는 `h+a, w+a`).
3. `unfold` 또는 advanced indexing(`torch.gather`/좌표 기반 슬라이싱을 벡터화)으로 배치별 `(2a+1)×(2a+1)` 윈도우를 한 번에 추출.
4. padding으로 채워진 0 값 자체는 버리고, **padding 여부(격자 밖인지)**를 별도 boolean 마스크로 계산해 EDGE 토큰 대체 여부를 결정한다 (0-padding 값과 EDGE 토큰 의미를 분리하기 위해 마스크는 padding 여부로 직접 계산하지, 값이 0인지로 유추하지 않는다).

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
- `models/modeling.py`: `GridDemandModel(PreTrainedModel)` — `forward(demands, node_id, labels=None, sample_idx=None)` → `labels`가 있으면 `{'loss', 'logits'}`, 없으면 `{'logits'}` 반환 (HF `Trainer` 호환).
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

## 8. 아직 정해지지 않은 것 / 기본값으로 진행할 것

- `a`, `d_model`, `n_freqs` 등 정확한 하이퍼파라미터 값 — 위 표를 기본값으로 두고 `configs/model/baseline.yaml`에서 조정.
- train/val/test 시간 분할 비율 — 별도 지시 없으면 시간순 70/15/15로 가정.
- `CombinedLoss`의 `gamma` 초기값 — 1.0으로 시작, 학습 로그 보고 조정.
- 출력 비음수 제약 방식 — `Softplus` 기본 채택.
