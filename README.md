# gir

grid 단위 택시 수요 예측. `preprocessing/{ulsan,porto}`가 원본 데이터를 `data/raw/{city}_temporal_grid.npy`로
전처리하고, `dataset_frame`/`models`가 그 npy로 학습/평가한다. 구조는 `docs/STRUCTURE.md`, 모델 설계는
`docs/MODEL_PLAN.md` 참고.

## 환경

```bash
conda activate DA
```

## 모델

로컬 히스토리 인코더 + daily/weekly 주기 브랜치 + 인과적 검색 + 브랜치 어텐션 + 신경망/검색
게이트를 하나로 합친 모델(`MergedDemandModel`) 하나뿐이다 — 이 브랜치(tmp)에는 다른 모델
구현이 없다(`models/config.py`, `models/modeling.py` 부재, `models/__init__.py` 참고).
원래 자기 완결형 `merged_model/` 패키지였던 것을 저장소 공용 관례(HuggingFace
`PreTrainedModel` + Hydra + `Trainer`)로 포팅한 것이며, 계산 그래프는 그대로다. 설계와 코드
지도는 `docs/MERGED_ARCHITECTURE.md`, ablation 결과는 `docs/MERGED_ABLATION_RESULTS.md`.

## 학습 (train.py)

Hydra로 설정을 관리한다. **도시마다 최적 하이퍼파라미터가 달라 루트 config가 도시별로 분리돼
있다** (`configs/config_ulsan.yaml` + `configs/model/merged_ulsan.yaml`, `configs/config_porto.yaml`
+ `configs/model/merged_porto.yaml`, 둘 다 `configs/dataset/*.yaml` 공유). 근거는
`docs/MERGED_TUNING_RESULTS.md`.

```bash
python train.py                                            # ulsan (기본값, full)
python train.py --config-name config_porto model.loss_type=mae   # porto
python train.py model.use_retrieval=false ablation=no-ir   # ablation (ulsan)
python validate_merged.py --device cpu                     # 학습 전 빠른 구조/인과 검사
```

- 결과(로그, 체크포인트)는 `output/${project_name}/{날짜}/{시간}/`에 저장된다 (`hydra.run.dir`).
- 결과 요약 JSON은 `output/merged/runs/<city>_<loss_type>_<ablation>_seed<seed>.json`에도 남는다
  (`run_ablation.sh`/`run_seeds.sh` 등 기존 도구가 존재 여부로 진행 상황을 판단하는 파일).
- 학습이 끝나면 `trainer.save_model(output_dir)`로 `config.json`+가중치가 저장되어, test.py가
  이 폴더 경로만으로 모델을 복원할 수 있다.
- 9개 ablation 스위치(`use_daily`/`use_weekly`/`use_retrieval`/`use_weather`/`weather_injection`/
  `use_calendar`/`use_branch_attention`/`use_neighbors`/`use_softplus`)는 전부 Hydra 오버라이드로
  켜고 끈다. `run_ablation.sh`(큐 러너) / `run_seeds.sh`(다중 시드)가 그 매핑을 들고 있다.
- 포팅 충실도 검증: `python tests/test_merged_parity.py --device cuda`
  (구현 교체 전후 forward/loss 일치, save/load 왕복, 짧은 학습 loss 궤적 비교).

## 평가 (test.py)

학습 프로세스와 무관하게, 체크포인트 경로만 있으면 언제든 독립 실행된다.

```bash
python test.py <checkpoint_path> --city ulsan --weather_csv_path data/raw/ulsan_meteorological_data.csv
```

- `<checkpoint_path>`: train.py가 저장한 `output/.../` 디렉토리 (`config.json` + `model.safetensors`가 있는 곳)
- 출력: RMSE, MAE, MAPE(+1, 0-수요 스무딩), MAPE(0제외)

## 날씨 데이터

날씨 CSV(기온/강수량/적설 3개 컬럼 필수, `cp949` 인코딩, row i = grid의 시간 인덱스 i와 정확히
일치)는 `configs/dataset/{ulsan,porto}.yaml`의 `weather_csv_path`가 가리킨다. `.gitignore`
(`*.csv`) 대상이라 저장소엔 없다 — ulsan은 `/home/jinsu/PycharmProjects/DMVST/data/raw/
meteorological_data.csv`를 복사, porto는 `python preprocessing/porto/fetch_weather.py`로 생성.

## 전처리 (원본 데이터 -> npy)

```bash
cd preprocessing/ulsan && python create_graph.py   # 또는 preprocessing/porto
```

porto는 `preprocessing/porto/README.md`에 세부 옵션(다운로드, crop 설정 등) 설명이 있다.
