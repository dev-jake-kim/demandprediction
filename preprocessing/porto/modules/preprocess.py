from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import GraphBuildConfig
from .filters import crop_filter, crop_to_projected_bounds, remove_other_region

REQUIRED_COLUMNS = ('TIMESTAMP', 'MISSING_DATA', 'POLYLINE')
FIRST_POINT_PATTERN = re.compile(r'\[\[(-?\d+\.?\d*),(-?\d+\.?\d*)\]')
CHUNK_SIZE = 200_000


def validate_required_columns(df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {list(required_columns)}")


def extract_pickup_records(csv_path: Path) -> pd.DataFrame:
    """POLYLINE의 첫 좌표(승차지점)와 TIMESTAMP(호출/승차 시각)만 추출.

    ulsan의 xpos/ypos와 동일한 스케일(도(degree) * 1e6, 정수)로 맞춰서
    filters.py의 remove_other_region/crop_filter를 그대로 재사용할 수 있게 한다.
    """
    frames: list[pd.DataFrame] = []
    total_rows = 0
    collected = 0

    reader = pd.read_csv(
        csv_path,
        usecols=['TIMESTAMP', 'MISSING_DATA', 'POLYLINE'],
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        total_rows += len(chunk)
        chunk = chunk[chunk['MISSING_DATA'] == False]  # noqa: E712
        coords = chunk['POLYLINE'].str.extract(FIRST_POINT_PATTERN)
        valid = coords.dropna()

        sub = chunk.loc[valid.index, ['TIMESTAMP']].copy()
        sub['xpos'] = (valid[0].astype(float) * 1_000_000).round().astype('int64')
        sub['ypos'] = (valid[1].astype(float) * 1_000_000).round().astype('int64')
        frames.append(sub)
        collected += len(sub)
        print(f"  Processed {total_rows:,} rows, collected {collected:,} pickup points so far")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total pickup points extracted: {len(df):,}")
    return df


def parse_call_dates(df: pd.DataFrame, time_col: str = 'TIMESTAMP') -> pd.DataFrame:
    parsed = df.copy()
    parsed['call_date'] = pd.to_datetime(parsed[time_col], unit='s', errors='coerce')
    return parsed


def build_projected_geodataframe(df: pd.DataFrame, config: GraphBuildConfig) -> gpd.GeoDataFrame:
    df_geo = df[['xpos', 'ypos']].copy()
    df_geo['lon'] = df_geo['xpos'] / 1_000_000
    df_geo['lat'] = df_geo['ypos'] / 1_000_000

    gdf = gpd.GeoDataFrame(
        df_geo,
        geometry=gpd.points_from_xy(df_geo['lon'], df_geo['lat']),
        crs='EPSG:4326'
    )
    return gdf.to_crs(config.target_crs)


def load_and_preprocess_data(csv_path: Path, config: GraphBuildConfig) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    print("=" * 60)
    print("Loading and preprocessing Porto pickup data")
    print("=" * 60)

    validate_required_columns(pd.read_csv(csv_path, nrows=0))

    print("\n[1/4] Extracting pickup points from POLYLINE...")
    df = extract_pickup_records(csv_path)
    df = parse_call_dates(df)

    print("\n[2/4] Applying remove_other_region...")
    df = remove_other_region(
        df,
        x_col='xpos',
        y_col='ypos',
        x_range=config.porto_range_x,
        y_range=config.porto_range_y,
    )

    # NOTE: ulsan 파이프라인의 eleminate_duplicates(승객 clientid 기준 중복 호출 제거)는
    # Porto 데이터에 대응하는 승객 식별자가 없어 적용하지 않음.

    print("\n[3/4] Projecting to target CRS...")
    gdf = build_projected_geodataframe(df, config)

    # NOTE: crop_filter(반복적 이상치 제거)는 Porto가 해안을 따라 동서로 길게 뻗어있어
    # 정사각형으로 수렴시키는 crop_filter 대신 투영 좌표 기준 하드캡으로 대체.
    # crop_filter 자체는 filters.py에 그대로 남겨둠.
    print("\n[4/4] Applying hard cap on projected bounds...")
    gdf = crop_to_projected_bounds(gdf, config.porto_projected_x_range, config.porto_projected_y_range)
    df = df.loc[gdf.index]

    print(f"\nFinal preprocessed data: {len(df):,} rows")

    return df, gdf
