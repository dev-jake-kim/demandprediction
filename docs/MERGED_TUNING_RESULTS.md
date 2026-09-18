# merged 모델 하이퍼파라미터 튜닝 스윕

> **⚠️ porto 결과는 무효다 (2026-09-15).**
>
> porto 런이 잘못된 격자(19×20 = 380노드)로 돌았다. porto의 정본 격자는 **10×20 = 200노드**다
> (`preprocessing/porto/create_graph.py`가 생성하는 `porto_temporal_grid.npy`, `landuse_grid.npy`와
> 같은 shape). 2026-09-10에 `data/raw/porto_temporal_grid.npy`를 복구할 때 전처리의 **중간
> 산출물**(`temporal_grid.npy`, 19×20, 8/19 14:52)을 최종 출력물(`porto_temporal_grid.npy`,
> 10×20, 8/19 15:57) 대신 가져다 쓴 것이 원인이다.
>
> 그 결과 아래 **porto 절의 모든 수치와 판정(`history128` 채택)은 근거를 잃었다.**
> 해당 run JSON과 체크포인트는 삭제했고, `configs/model/merged_porto.yaml`의
> `history_hidden`은 128에서 64(포팅 전 merged_model이 200노드 격자에서 쓰던 값)로 되돌렸다.
> **porto 튜닝 스윕은 재실행이 필요하다.**
>
> **ulsan 결과는 유효하다** — ulsan 격자(14×12 = 168노드)는 바뀐 적이 없다.

`configs/model/merged_ulsan.yaml` / `configs/model/merged_porto.yaml`가 왜 그 값인지의 근거.
baseline(`d_model=64, transformer_ffn=128, num_fourier_bands=8, transformer_layers=2,
transformer_heads=4, history_hidden=64, periodic_hidden=64`)에서 6개 파라미터를 각각 하나씩
키운 변종(one-factor-at-a-time) + baseline, 도시(ulsan/porto) × 시드 245 1회씩, 총 14회.
`transformer_ffn`은 항상 `d_model`의 2배로 유지. loss는 config 기본값(`combined`).

원자료: `output/merged/runs/tune_<city>_<variant>_seed245.json`.

## 변종 정의

| 변종 | 오버라이드 |
|---|---|
| d_model128 | `model.d_model=128 model.transformer_ffn=256` |
| fourier16 | `model.num_fourier_bands=16` |
| layers3 | `model.transformer_layers=3` |
| heads8 | `model.transformer_heads=8` |
| history128 | `model.history_hidden=128` |
| periodic128 | `model.periodic_hidden=128` |

## 결과 — RMSE, MAPE(+1) 기준 (선택 기준으로 이 두 지표만 사용)

### ulsan

| variant | RMSE | Δ | MAPE(+1) | Δ |
|---|---|---|---|---|
| **baseline** | **0.7536** | — | **15.401** | — |
| d_model128 | 0.7527 | -0.1% | 15.618 | +1.4% |
| fourier16 | 0.7550 | +0.2% | 15.622 | +1.4% |
| layers3 | 0.7600 | +0.8% | 15.216 | -1.2% |
| heads8 | 0.7679 | +1.9% | 15.371 | -0.2% |
| history128 | 0.7531 | -0.1% | 15.555 | +1.0% |
| periodic128 | 0.7591 | +0.7% | 15.275 | -0.8% |

### porto — ⚠️ 무효 (위 경고 참고. 잘못된 380노드 격자로 측정됨)

| variant | RMSE | Δ | MAPE(+1) | Δ |
|---|---|---|---|---|
| **baseline** | **1.1227** | — | **7.997** | — |
| d_model128 | 1.2157 | +8.3% | 8.357 | +4.5% |
| fourier16 | 1.1510 | +2.5% | 8.040 | +0.5% |
| layers3 | 1.1242 | +0.1% | 8.450 | +5.7% |
| heads8 | 1.1134 | -0.8% | 8.129 | +1.6% |
| history128 | 1.1045 | -1.6% | 7.912 | -1.1% |
| periodic128 | 1.0774 | -4.0% | 8.114 | +1.5% |

## 판정

RMSE와 MAPE(+1)가 baseline 대비 **둘 다** 개선된 변종만 채택 대상으로 본다.

- **ulsan**: 12개 셀(6변종) 중 둘 다 개선인 경우가 하나도 없다 — 매 변종에서 RMSE와
  MAPE(+1)가 반대 방향으로 움직인다(예: d_model128은 RMSE만, layers3는 MAPE(+1)만 개선).
  하나를 골라 트레이드오프를 감수할 근거가 약해 **baseline을 그대로 채택**한다
  (`configs/model/merged_ulsan.yaml`).
- **porto**: ⚠️ **아래 판정은 무효다.** `history128`(`history_hidden=128`)이 유일하게 RMSE(1.1227→1.1045)와
  MAPE(+1)(7.997→7.912) 둘 다 개선했다. `periodic128`이 RMSE 개선폭은 더 크지만(-4.0%)
  MAPE(+1)은 baseline보다 나빠져(+1.5%) 채택 기준을 통과하지 못한다. **`history128`을
  채택**한다(`configs/model/merged_porto.yaml`).

## 제약사항 (읽기 전 확인)

- **시드 1개**. 대부분의 차이는 1~2%p대라 이 저장소에서 관찰된 시드 재현성 노이즈 범위 안에
  들어올 수 있다. `d_model128`의 porto RMSE 악화(+8.3%)만 노이즈치고 커서 상대적으로 신뢰도가
  높고, 나머지(특히 porto에서 채택한 `history128`의 -1.6%/-1.1%)는 방향성 참고 수준이다.
  여러 시드로 재확인 전까지는 확정적인 결론이 아니다.
- **one-factor-at-a-time**만 시험했다 — 여러 파라미터를 동시에 바꾼 조합(예: history128 +
  periodic128)은 시도하지 않았다.
- **porto는 batch_size=4로 고정**해서 돌렸다(원본 8이 아님) — `configs/config_porto.yaml`의
  `train.per_device_train_batch_size`/`per_device_eval_batch_size` 주석 참고. porto
  (19×20=380노드)는 `batch_size * time_step * nodes`가 커서 PyTorch의 memory-efficient
  SDPA 백엔드가 dropout seed/offset을 못 만드는 한계(65535)를 넘는다(`train.py`가 자동으로
  7까지는 낮추지만, `d_model128`처럼 activation이 큰 변종은 그래도 OOM이 나서 4로 낮췄다).
  이 배치 크기 차이 자체가 ulsan과 porto의 비교를 일부 교란할 수 있다.
