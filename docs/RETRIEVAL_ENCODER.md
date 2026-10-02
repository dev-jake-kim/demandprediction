# 검색 encoder 단독 학습 SPEC

검색에 쓸 질의·후보 embedding을 본 모델과 따로 학습한다. VAE처럼 각 입력을 표준 정규분포 안의
Gaussian `q(z|x) = N(μ, diag σ²)`로 encode하고, 같은 정답 라벨 집단은 latent에서 가깝게, 다른 집단은
멀게 두되 라벨 차이가 클수록 더 강하게 떨어뜨린다. 이 문서는 encoder 단독 학습과 평가까지를 다루며,
학습한 encoder를 본 모델의 검색기에 붙이는 일은 평가 결과를 본 뒤 정한다.

---

## 1. 학습 샘플

샘플 하나 = (예측 시점 t, 노드 n).

| 항목 | 정의 |
|---|---|
| 입력 x | 노드 n 주변 3×3 창의 raw 수요 `[t−8, t)` (격자 밖 0), `(8, 3, 3)`을 행 우선으로 편 `[8, 9]` |
| 정답 y | 노드 n 자신의 t 시점 수요 (스칼라) |
| 노드 id | n (frozen node embedding 조회용) |

- 분할은 본 모델과 같다: `configs/dataset/<city>.yaml`(시간순 70/15/15, `time_step=8`),
  경계는 `UnifiedDemandDataset`의 `train_end`, `val_end`를 그대로 쓴다.
- 학습은 train 구간 `t ∈ [8, train_end)`의 모든 (t, n) 쌍, 검증은 val 구간, 평가는 test 구간이다.
- **입력이 전부 0인 쌍은 학습·평가 질의에서 제외한다** (울산 약 15%). 입력이 같아 embedding이
  같을 수밖에 없는데 라벨은 제각각이라 분리할 수 없다.

## 2. 라벨 집단 (bucket)

- cap = train 구간 셀·시간 수요에서 `value ≥ cap`의 비율이 0.5% 이하가 되는 최소 정수
  (울산 7, Porto 21).
- bucket 하한: `0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 25, …`(5 이후 간격 2, 3, 4, …)를 cap 미만까지 쓰고,
  cap 이상은 마지막 bucket 하나. 라벨은 하한 기준(내림)으로 배정한다.
  - 울산: `{0},{1},{2},{3},{4},{5,6},{≥7}` (7개)
  - Porto: `{0},{1},{2},{3},{4},{5,6},{7–9},{10–13},{14–18},{19,20},{≥21}` (11개)
- 같은 bucket = 같은 집단(positive), 다른 bucket = negative.

## 3. Encoder (Gaussian posterior)

```text
x [B, 8, 9] → log1p → Linear(9 → D) → + node_embedding[n] (frozen, 8 시점에 같은 값) → [B, 8, D]
            → LSTM(D → 16, batch_first) → 마지막 hidden h [B, 16]
            → μ = Linear(16 → 16)(h),  log σ² = Linear(16 → 16)(h)       (latent 차원 L = 16)
학습:  z = μ + σ ⊙ ε,  ε ~ N(0, I)   (reparameterization)
검색:  z = μ                          (결정적)
```

- `node_embedding`은 채택 모델 체크포인트의 `local_history.node_embedding`을 그대로 가져온다
  (`output/adopted/checkpoints/timestep8_<city>_seed245_39185a1`). `D`는 그 차원이다
  (울산 16, Porto 64). 본 모델과 같이 입력 projection에 **더하며**, 학습 파라미터가 아닌
  buffer로 두어 gradient를 받지 않고 optimizer에도 들어가지 않는다.
- 학습 파라미터: 입력 Linear, LSTM, μ·log σ² head. decoder(입력 복원)는 두지 않는다.
- `log σ²`는 `[−6, 2]`로 clamp해 수치 안정성을 지킨다.

## 4. Loss: Gaussian 거리 기반 contrastive + KL

$$
L = L_{\text{con}} + \beta\, L_{\text{KL}}
$$

### 4.1 `L_con`: 라벨 거리 가중 contrastive (latent 거리 기반)

latent가 단위 구가 아니라 Gaussian 공간이므로 유사도를 cosine 대신 **유클리드 거리**로 둔다.
batch 안에서 샘플 z로

$$
s_{ij} = -\frac{\lVert z_i - z_j \rVert^2}{L \cdot T}\quad(T = 0.1),
$$

`c_i` = i의 bucket, `P(i) = {j ≠ i : c_j = c_i}`, `N(i) = {j : c_j ≠ c_i}`.

$$
L_i = -\frac{1}{|P(i)|}\sum_{p\in P(i)}\left[s_{ip} - \log\Big(\sum_{p'\in P(i)} e^{s_{ip'}} + \sum_{n\in N(i)} w_{in}\, e^{s_{in}}\Big)\right],
\qquad L_{\text{con}} = \text{mean}_{i:\,|P(i)|>0}\; L_i
$$

- `e^{s_ij}`는 두 점 사이의 Gaussian kernel이다. 같은 집단은 kernel 값이 커지도록(가깝게),
  다른 집단은 작아지도록(멀게) 학습된다. `L·T`로 나누는 것은 차원 수와 무관하게 거리 척도를 맞추기
  위해서다(prior N(0, I)에서 두 점의 평균 제곱거리는 `2L`이라 `d²/L ≈ 2`).
- `w_in = |log1p(y_i) − log1p(y_n)|` (raw 라벨 기준). 예: 0–1 0.69, 0–5 1.79, 0–7 2.08, 5–7 0.29.
  negative를 밀어내는 gradient가 `w_in·e^{s_in}`에 비례하므로 0–5 쌍이 0–1 쌍보다 약 2.6배 강하게
  떨어진다.
- `|P(i)| = 0`인 샘플은 anchor에서 빠지고 negative로만 쓰인다.
- 샘플 z(μ가 아닌)로 계산하므로, 집단이 σ 크기의 잡음에도 분리되도록(여유를 두고 뭉치도록) 학습된다.

### 4.2 `L_KL`: 표준 정규 prior

$$
L_{\text{KL}} = \text{mean}_i\; \tfrac{1}{2}\sum_{d=1}^{L}\big(\mu_{id}^2 + \sigma_{id}^2 - \log\sigma_{id}^2 - 1\big)
= \text{mean}_i\, \mathrm{KL}\big(N(\mu_i, \sigma_i^2)\,\Vert\, N(0, I)\big)
$$

- 모든 embedding이 원점 주변의 표준 정규분포 안에 머물게 한다. contrastive만 쓰면 집단을 떼어 놓으려고
  거리가 끝없이 커질 수 있는데, KL이 latent의 크기를 묶는다.
- **β = 0.1, 처음 5 epoch 동안 0에서 선형으로 올린다(warm-up).** 처음부터 KL을 크게 주면 μ가 원점으로
  몰리고 σ가 1로 가는 posterior collapse가 생겨 contrastive가 분리할 거리가 사라진다.
- collapse 감시: epoch마다 active unit 수(데이터에 걸친 `Var(μ_d) > 0.01`인 차원 수), 평균 KL,
  평균 σ를 기록한다.

## 5. batch 구성

- batch 128, **라벨 균형 샘플링**: bucket g의 뽑힐 확률 ∝ `count_g^0.5`(복원 추출).
  샘플 가중치 = `count_g^{-0.5}`.
- epoch = `학습 쌍 수 ÷ 128` 스텝.

## 6. 학습 설정

AdamW lr 1e-3, weight decay 0.05, 시드 245, 울산부터. early stopping은 val 검색 MAE(7절) 기준.

## 7. 평가 (encoder 단독)

val·test의 각 (t, n) 질의(입력 0 제외)에 대해:

- 후보: 같은 노드 n의 `τ ∈ [8, t)` (본 모델 eval 모드 검색과 같은 규칙, 후보 입력 0 포함).
- 유사도: μ 사이의 `s = −‖μ_q − μ_τ‖² / (L·T)` (학습과 같은 척도). top-20을 골라 `softmax(s)`
  가중평균 라벨로 예측.
- 지표: 위 예측의 MAE·RMSE, top-20 중 질의와 같은 bucket 비율(bucket별로도).
- 비교 기준: 같은 질의·후보·top-20에서 raw 입력 cosine(기존 검색기 유사도)으로 한 결과.

## 8. 코드와 출력

| 경로 | 역할 |
|---|---|
| `models/retrieval_encoder/` | encoder, loss, (t, n) 데이터셋, 균형 샘플러 |
| `train_retrieval_encoder.py` | Hydra 학습 스크립트 (`configs/retrieval_encoder_<city>.yaml`, `configs/dataset/<city>.yaml` 공유) |
| `output/retrieval_encoder/<city>_seed<seed>/` | encoder 가중치, 학습 로그, 평가 JSON |
