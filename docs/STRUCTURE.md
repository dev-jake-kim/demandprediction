# 폴더 구조 (branch: DMVST)

`master`의 baseline 모델은 이 브랜치에서 삭제하고 DMVST-Net(`docs/DMVST_PLAN.md`)으로 대체했다.
데이터 전처리(`preprocessing/{ulsan,porto}/`)와 학습 하네스(`train.py`/`test.py`)는 공유한다.

아래 트리는 이 브랜치를 이해하는 데 필요한 주요 파일만 표시한 것으로, 전체 파일 목록이 아니다
(예: `__init__.py`, 실험 결과 CSV, `docs/*.pdf`, `preprocessing/`의 보조 스크립트/README 등은 생략).

```
gir/
├── configs/
│   ├── config.yaml            # hydra 메인 설정 (defaults: dataset=ulsan, model=dmvst)
│   ├── dataset/
│   │   └── *.yaml             # city, npy_path, time_step, patch_size, line_embeddings_path,
│   │                           # train/val/test 분할 비율
│   └── model/
│       └── dmvst.yaml         # DMVSTConfig 하이퍼파라미터
├── data/
│   └── raw/
│       ├── {city}_temporal_grid.npy
│       ├── {city}_dmvst_graph_edges.csv       # build_dmvst_graph.py 출력
│       └── {city}_dmvst_line_embeddings.npy   # build_dmvst_line_embeddings.py 출력(Torch env)
├── dataset_frame/
│   ├── __init__.py
│   └── grid_demand_dataset.py # GridDemandDataset (npy -> demands/hour_of_day/day_of_week/labels/sample_idx)
├── docs/
│   └── DMVST_PLAN.md          # 이 브랜치의 모델 설계 문서
├── models/
│   ├── config.py               # DMVSTConfig (PretrainedConfig)
│   ├── modeling.py             # DMVSTModel (PreTrainedModel) — LocalCNN + LSTM + LINE 임베딩
│   ├── losses.py                # CombinedLoss (baseline과 공유)
│   └── metrics.py               # compute_regression_metrics (baseline과 공유)
├── output/                    # hydra.run.dir 대상 (git 추적 안 함)
├── preprocessing/
│   ├── ulsan/                 # create_graph.py + modules/ (원본 데이터 -> temporal_grid.npy 등)
│   ├── porto/
│   ├── build_dmvst_graph.py             # DTW 기반 지역 유사도 그래프(DA env 실행)
│   └── build_dmvst_line_embeddings.py   # LINE 그래프 임베딩(**Torch env 실행**, cogdl 의존)
├── train.py                   # @hydra.main, HF Trainer로 학습, output/에 체크포인트 저장
└── test.py                    # checkpoint_path + --npy_path로 from_pretrained 복원 후 평가
                                # (학습 프로세스와 분리, 학습 종료 후 아무때나 단독 실행)
```
