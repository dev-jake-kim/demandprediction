# 폴더 구조

## 이미 있음

```
gir/
├── configs/
│   └── config.yaml            # hydra 메인 설정 (dataset/model/train/callbacks)
├── data/
│   └── raw/
│       ├── ulsan_temporal_grid.npy
│       └── porto_temporal_grid.npy
├── dataset_frame/
│   ├── __init__.py
│   └── grid_demand_dataset.py # GridDemandDataset (npy -> demands/labels/node_id/sample_idx)
├── models/                    # 비어있음
├── output/                    # 비어있음 (hydra.run.dir 대상)
├── preprocessing/
│   ├── ulsan/                 # create_graph.py + modules/ (원본 데이터 -> temporal_grid.npy 등)
│   └── porto/
└── temp/
```

## 추가 예정 (모델 아키텍처 미정이라 보류 중)

```
gir/
├── configs/
│   ├── config.yaml            # defaults: [dataset, model] 추가
│   ├── dataset/
│   │   └── *.yaml             # city, time_step, train/val/test 시간 분할 비율
│   └── model/
│       └── *.yaml             # 모델 하이퍼파라미터
├── models/
│   └── *.py                   # PretrainedConfig + PreTrainedModel 서브클래스
│                               # (save_pretrained/from_pretrained로 config+가중치 같이 저장)
├── train.py                   # @hydra.main, HF Trainer로 학습, output/에 체크포인트 저장
└── test.py                    # checkpoint_path만 받아 from_pretrained로 복원 후 평가
                                # (학습 프로세스와 분리, 학습 종료 후 아무때나 단독 실행)
```
