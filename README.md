# gir (branch: DMVST)

grid 단위 택시 수요 예측. `preprocessing/{ulsan,porto}`가 원본 데이터를 전처리해 `temporal_grid.npy`를
만들고(`data/raw/{city}_temporal_grid.npy`로 복사해서 씀 — 아래 "전처리" 절 참고), `dataset_frame`/
`models`가 그 npy로 학습/평가한다. 이 브랜치는 `master`의 baseline 모델을
지우고 논문 DMVST-Net(Yao et al., AAAI 2018)을 구현했다. 구조는 `docs/STRUCTURE.md`, 모델 설계는
`docs/DMVST_PLAN.md` 참고.

## 환경

```bash
conda activate DA
```

DMVST-Net은 Semantic View(LINE 그래프 임베딩)에 `cogdl`이 필요한데, `DA` env에는 설치돼 있지 않다.
학습 전 1회, **`Torch` conda env**로 아래 전처리를 실행해야 한다(자세한 내용은 다음 절).

## 사전 전처리 (학습 전 1회 필수)

```bash
conda run -n DA python preprocessing/build_dmvst_graph.py --city ulsan             # DTW 기반 그래프 CSV 생성
conda run -n Torch python preprocessing/build_dmvst_line_embeddings.py --city ulsan  # LINE 임베딩 계산(cogdl)
```

porto도 `--city porto`로 동일하게 실행한다. 결과는 `data/raw/{city}_dmvst_graph_edges.csv`,
`data/raw/{city}_dmvst_line_embeddings.npy`로 저장되고, `configs/dataset/*.yaml`의
`line_embeddings_path`가 이 파일을 가리킨다.

## 학습 (train.py)

Hydra로 설정을 관리한다 (`configs/config.yaml` + `configs/dataset/*.yaml` + `configs/model/*.yaml`).

```bash
python train.py                      # 기본값: dataset=ulsan, model=dmvst
python train.py dataset=porto         # porto로 학습
python train.py train.num_train_epochs=50 model.lstm_hidden_size=64   # 하이퍼파라미터 오버라이드
```

- 결과(로그, 체크포인트)는 `output/${project_name}/{날짜}/{시간}/`에 저장된다 (`configs/config.yaml`의 `hydra.run.dir`).
- train/val만 여기서 나누고 학습한다 (시간순 `train_ratio`/`val_ratio`, 기본 70%/15%, 나머지 15%는 test).
- 학습이 끝나면 `trainer.save_model(output_dir)`로 `config.json`+가중치가 저장되어, test.py가 이 폴더 경로만으로 모델을 복원할 수 있다.

## 평가 (test.py)

학습 프로세스와 무관하게, 체크포인트 경로 + 평가할 npy 데이터만 있으면 언제든 독립 실행된다.

```bash
python test.py <checkpoint_path> --npy_path data/raw/ulsan_temporal_grid.npy --t_start <test 구간 시작 timestep>
```

- `<checkpoint_path>`: train.py가 저장한 `output/.../` 디렉토리 (`config.json` + `model.safetensors`가 있는 곳)
- `time_step`: CLI로 받지 않고 체크포인트의 `config.json`(=학습 때 쓴 값)을 그대로 사용함
- `--t_start`: train.py 로그에 찍힌 val 구간 끝 지점(=test 구간 시작)을 그대로 넣으면 됨. 생략하면 전체 구간을 평가.
- 출력: RMSE, MAE, MAPE(+1, 0-수요 스무딩)

## 전처리 (원본 데이터 -> npy)

```bash
cd preprocessing/ulsan && python create_graph.py   # 또는 preprocessing/porto
```

`temporal_grid.npy`는 `preprocessing/{city}/output/`에 저장된다 — `train.py`/전처리 스크립트들이
읽는 `data/raw/{city}_temporal_grid.npy`로는 별도로 복사(또는 심볼릭 링크)해야 한다.

porto는 `preprocessing/porto/README.md`에 세부 옵션(다운로드, crop 설정 등) 설명이 있다.
