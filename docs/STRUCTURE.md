# 폴더 구조 (branch: STResnet)

`master`의 baseline 모델은 이 브랜치에서 삭제하고 ST-ResNet(`docs/STRESNET_PLAN.md`)으로 대체했다.
데이터 전처리(`preprocessing/{ulsan,porto}/`)와 학습 하네스(`train.py`/`test.py`)는 공유한다.

```
gir/
├── configs/
│   ├── config.yaml            # hydra 메인 설정 (defaults: dataset=ulsan, model=stresnet)
│   ├── dataset/
│   │   └── *.yaml             # city, npy_path, l_c/l_p/l_q, train/val/test 분할 비율
│   └── model/
│       └── stresnet.yaml      # STResNetConfig 하이퍼파라미터
├── data/
│   └── raw/
│       └── {city}_temporal_grid.npy
├── dataset_frame/
│   ├── __init__.py
│   └── grid_demand_dataset.py # GridDemandDataset (npy -> demands_closeness/period/trend, day_of_week, labels, sample_idx)
├── docs/
│   └── STRESNET_PLAN.md       # 이 브랜치의 모델 설계 문서
├── models/
│   ├── config.py              # STResNetConfig (PretrainedConfig)
│   ├── modeling.py            # STResNetModel (PreTrainedModel)
│   ├── losses.py              # CombinedLoss (baseline과 공유)
│   └── metrics.py             # compute_regression_metrics (baseline과 공유)
├── output/                    # hydra.run.dir 대상 (git 추적 안 함)
├── preprocessing/
│   ├── ulsan/                 # create_graph.py + modules/ (원본 데이터 -> temporal_grid.npy 등)
│   └── porto/
├── train.py                   # @hydra.main, HF Trainer로 학습, output/에 체크포인트 저장
└── test.py                    # checkpoint_path + --npy_path로 from_pretrained 복원 후 평가
                                # (학습 프로세스와 분리, 학습 종료 후 아무때나 단독 실행)
```
