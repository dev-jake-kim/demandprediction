# Porto Taxi Preprocessing

[UCI Taxi Service Trajectory Prediction Challenge (ECML/PKDD 2015)](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)
데이터를 `preprocessing/ulsan`과 동일한 시공간 그래프(graph_data.json) 포맷으로 전처리한다.

## 준비

conda 환경 `DA`에서 실행한다.

```bash
conda activate DA
```

`raw/train.csv`가 이미 없다면 원본 zip을 받아 풀어야 한다 (`raw/`에 `train.csv`가 있으면 생략):

```bash
cd preprocessing/porto/raw
curl -L -o taxi.zip "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip"
unzip taxi.zip
unzip train.csv.zip
```

## 실행

```bash
cd preprocessing/porto
python create_graph.py
```

`raw/train.csv`(약 170만 행)를 읽어 `output/`에 결과를 생성한다. 크롭 필터(이상치 제거)와 MILP 기반 패치 선택 때문에 수 분~수십 분 걸릴 수 있다.

승차지점만 빠르게 지도에 뿌려보고 싶다면:

```bash
python plot_pickups.py   # output/pickup_locations.png 생성
```

## 출력물 (`output/`)

- `graph_data.json` — 최종 시공간 그래프 (nodes + 시간대별 demand/OD)
- `step1_cropped_pickup_points.png` — 전처리 후 승차지점 분포
- `patch_near_demands.png` — 선택된 패치/인접 셀 시각화
- `node_demand_density_curves.png` — 노드별 수요 평균/표준편차 밀도
- `landuse_grid.npy`, `temporal_grid.npy` — 중간 산출 격자 배열

## ulsan과 다른 점

`preprocessing/ulsan/create_graph.py`와 동일한 파이프라인 구조(전처리 → 시공간 그리드 → MILP 패치 선택 → 노드/OD/시간특성 → JSON 저장)를 따르지만, Porto 원본 데이터에 없는 부분은 아래처럼 처리했다.

| 항목 | ulsan | porto |
|---|---|---|
| 승차 위치/시각 | `origin_data.csv`의 `xpos/ypos`, `call_date` | `POLYLINE`의 첫 좌표(승차지점) + `TIMESTAMP` |
| 중복 호출 제거 (`eleminate_duplicates`) | 고객 `clientid` 기준 10분 내 중복 제거 | **미적용** — Porto엔 대응하는 승객 식별자가 없음(`ORIGIN_CALL`은 콜센터 호출 건에만 존재) |
| 용도지역(land use) | `UPIS_C_UQ111.shp` 기반 `landuse_grid` | 대응 shapefile이 없어 **동일 shape의 빈 배열(전부 0)** 로 유지 |
| POI | `poi_data.csv` 기반 노드별 POI 구성 | 대응 데이터가 없어 `poi_gdf=None` → 각 노드 `composition.poi`는 항상 `{}` |
| OD flow | `dest_xpos/dest_ypos` 컬럼으로 목적지 추출 | 대응 컬럼이 없어 `extract_od_flows`가 기존 폴백 경로를 타서 **타임스텝 수만큼의 빈 리스트**(`[[], [], ...]`) 구조만 유지 |
| 공휴일 리스트 | 한국 공휴일(`korean_holidays`) | 포르투갈 공휴일(`portuguese_holidays`, 데이터 기간 2013-07~2014-06 + 포르투 지역 공휴일 상 주앙 06-24 포함) |
| 좌표계 | `EPSG:5174` | `EPSG:3763` (ETRS89 / Portugal TM06) |

즉 `nodes[].composition.land_use/poi`와 `x[].OD`는 스키마는 ulsan과 동일하지만 실제 값은 비어 있거나(`{}`, `[]`) `land_use`는 전부 `Unclassified`로 채워진다.
