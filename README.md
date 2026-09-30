# gir

grid 단위 택시 수요 예측. `preprocessing/{ulsan,porto}`가 원본 데이터를 `data/raw/{city}_temporal_grid.npy`로
전처리하고, `dataset_frame`/`models`가 그 npy로 학습/평가한다. 모델·데이터·학습 사양은
[`docs/SPEC.md`](docs/SPEC.md)가 기준이다.

## 환경

```bash
conda activate DA
```

## 모델

로컬 히스토리 인코더 + daily/weekly 주기 브랜치 + 브랜치 어텐션을
`MergedDemandModel`로 합쳤다. 기본값은 검색 pass(neural 예측만 사용)이고,
`model.use_retrieval=true`를 명시하면 인과적 검색과 출력 게이트를 켠다.

## 학습 (train.py)

Hydra로 설정을 관리한다. 루트 config가 도시별로 분리돼 있다
(`configs/config_ulsan.yaml` + `configs/model/merged_ulsan.yaml`, `configs/config_porto.yaml`
+ `configs/model/merged_porto.yaml`, 둘 다 `configs/dataset/*.yaml` 공유).

```bash
python train.py                                            # ulsan, 검색 pass (기본값)
python train.py --config-name config_porto                  # porto, 검색 pass
python train.py model.use_retrieval=true ablation=full     # 검색 켬 (ulsan)
python validate_merged.py --device cpu                     # 학습 전 빠른 구조/인과 검사
```

- 결과(로그, 체크포인트)는 `output/${project_name}/{날짜}/{시간}/`에 저장된다 (`hydra.run.dir`).
- 결과 요약 JSON 경로는 `run_json`, 없으면 `output/merged/runs/`. 이전 실험 기록은 `output/past/`.
- 기능 스위치와 2-stage 학습은 `docs/SPEC.md` 4·8절 참고.
- 포팅 충실도 검증: `python tests/test_merged_parity.py --device cuda`

## 평가 (test.py)

학습 프로세스와 무관하게, 체크포인트 경로만 있으면 언제든 독립 실행된다.

```bash
python test.py <checkpoint_path> --city ulsan --weather_csv_path data/raw/ulsan_meteorological_data.csv \
  --train_ratio 0.70 --val_ratio 0.15
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
