# ST-ResNet 포팅

논문 "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction"(Zhang, Zheng, Qi,
AAAI 2017, `docs/STResnet.pdf`)을 `master`의 baseline, `ADFormer` 브랜치에 이어 **같은 데이터·같은
지표·같은 학습 하네스**(`train.py`/`test.py`, HF `Trainer`)로 비교하기 위해 이식했다.

## 표기

- `B`: batch, `H,W`: 격자 크기(ulsan 14×12, porto 10×20), `D=1`: 수요 단일 채널
- `l_c,l_p,l_q`: closeness/period/trend 시퀀스 길이 (기본 3/1/1), `period=24h`, `trend_span=168h`(고정)

## 1. 모델 아키텍처 (`models/modeling.py`)

```
demands_closeness (B,l_c,H,W) ─┐
demands_period    (B,l_p,H,W) ─┼─ 각각 Min-Max [-1,1] 정규화(demand_min/max, train split 기준)
demands_trend     (B,l_q,H,W) ─┘

각 시퀀스 -> STResNetBranch(Conv1(l→num_filters) -> ResUnit x L -> ReLU -> Conv2(num_filters→1))
           -> (B,1,H,W)

X_Res = w_c∘branch_c + w_p∘branch_p + w_q∘branch_q     # 식(4), w_*는 (1,H,W) 학습 파라미터
day_of_week(B,) -> one-hot(7) -> Linear(7,10)+ReLU -> Linear(10,H*W) -> (B,1,H,W) = X_Ext

X_hat = tanh(X_Res + X_Ext)                             # 식(5), [-1,1]
logits = (X_hat+1)/2 * (demand_max-demand_min) + demand_min   # 실제 수요 단위로 역정규화
loss = CombinedLoss(logits, labels)                      # baseline/ADFormer와 동일 loss 재사용
```

### 1-1. ResUnit (Figure 4b, pre-activation residual)

`x -> BN?+ReLU+Conv(3x3) -> BN?+ReLU+Conv(3x3) -> +x` (identity residual). `use_bn=true`(기본값)면
각 ReLU 앞에 BatchNorm2d — 논문 최고 성능 변형 L12-E-BN 기준. 모든 conv는 `padding=1`(zero-padding,
논문 footnote 1)이라 depth를 아무리 쌓아도 spatial 크기(H,W)가 유지된다.

### 1-2. closeness/period/trend 데이터 구성 (`dataset_frame/grid_demand_dataset.py`)

- closeness: `t` 직전 `l_c`개 연속 시간
- period: 하루(24h) 간격으로 `l_p`개 (`t-24, t-48, ...`)
- trend: 일주일(168h) 간격으로 `l_q`개 (`t-168, t-336, ...`)
- `l_c/l_p/l_q=0`이면 해당 브랜치 자체를 생성/실행하지 않음(`models/modeling.py`의 `branch_c/p/q`가
  `None`) — dataset도 그 키를 아예 안 돌려줘서 `forward()`의 Optional 인자가 `None`으로 채워짐.
  트렌드처럼 특정 시간 특성을 끄고 실험하고 싶을 때 쓰는 옵션(Codex 리뷰에서 이 경로가 실제로
  0-채널 Conv로 죽던 버그를 잡아서 고쳤음).

## 2. 논문 대비 우리가 내린 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 채널 수 | inflow/outflow 2채널 → 수요 1채널(`D=1`) | 우리 데이터가 단일 수요값만 있음 |
| 외부 요인 | day-of-week만 사용, 날씨·공휴일·hour-of-day 제외 | porto엔 날씨 데이터 없음, ADFormer와 범위를 맞춰 비교 단순화(단 ADFormer와 달리 hour-of-day는 애초에 안 씀 — 논문 자체가 day-of-week/weekend 메타데이터만 외부 요인으로 씀). day-of-week는 `(t//24)%7` 산술 계산(실제 달력 불필요 — 공휴일 미사용이라 요일 주기성만 일관되면 무해함, ADFormer와 동일 논리) |
| loss | 논문의 단순 MSE 대신 `CombinedLoss`(baseline/ADFormer와 동일) | 비교의 유일한 변수를 아키텍처로 고정 |
| `l_c/l_p/l_q` 기본값 | 논문 예시(3~5/1~4/1~4)보다 작게(3/1/1) | 우리 데이터가 논문 실험(1년+)보다 짧아(ulsan 182일) trend lookback으로 인한 학습 샘플 손실을 줄임 |
| 정규화 | Min-Max `[-1,1]`(demand_min/max, train 구간 `grid[:t_end]` 전체 — target뿐 아니라 closeness/period/trend가 참조하는 `t_start` 이전 구간도 포함해야 함, 처음엔 이 부분을 빠뜨렸다가 Codex 리뷰에서 잡음) | 논문의 tanh 출력 범위와 직접 대응 |
| ResUnit 마지막 Conv2 앞 ReLU | 표준 pre-activation ResNet 관례를 따라 추가 | 논문 본문/그림에 이 세부사항이 명시돼 있지 않음 |

## 3. HuggingFace 통합

- `models/config.py`: `STResNetConfig(PretrainedConfig)` — `H,W,l_c,l_p,l_q,num_filters,num_res_units,use_bn,demand_min,demand_max` 등.
- `models/modeling.py`: `STResNetModel(PreTrainedModel)` — `forward(demands_closeness=None, demands_period=None, demands_trend=None, day_of_week=None, labels=None, sample_idx=None)` → `{'loss','logits'}`(`logits.shape==(B,H,W)`).
- 이 모델은 ADFormer와 달리 `__init__`에서 외부 numpy 아티팩트(클러스터 맵 등)를 안 씀 — 모든 파라미터가
  순수 `torch.zeros/randn`/기본 `nn.Module` 초기화라서, ADFormer 개발 때 반복적으로 겪었던
  `from_pretrained`의 meta-device 버그(새 torch 텐서 x 외부 실데이터 직접 연산 시 발생) 자체가
  발생할 여지가 없음 — 실제로 저장/재로드 라운드트립이 별문제 없이 바로 통과함.

## 4. 엔트리 포인트

`train.py`는 baseline·ADFormer와 동일한 구조(Trainer/TrainingArguments/compute_metrics/
LoggingCallback/save/evaluate) — 모델 생성부와 dataset이 반환하는 배치 키만 다르다.
`test.py`는 애초에 Trainer를 안 쓰고 별도 `DataLoader`+평가 루프로 동작하는 독립 스크립트라(baseline·
ADFormer 때도 마찬가지), 여기서도 배치 키(`demands_closeness/period/trend`, `day_of_week`)만 바뀌었다.
