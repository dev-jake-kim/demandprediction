from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import GraphBuildConfig
from .filters import crop_filter, eleminate_duplicates, remove_other_region


REQUIRED_COLUMNS = ('xpos', 'ypos', 'clientid', 'call_date')


def validate_required_columns(df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {list(required_columns)}")


def report_missing_values(df: pd.DataFrame, focus_columns: tuple[str, ...] | None = None) -> dict[str, int]:
    columns = focus_columns if focus_columns is not None else tuple(df.columns)
    return {column: int(df[column].isna().sum()) for column in columns if column in df.columns}


def parse_call_dates(
    df: pd.DataFrame,
    time_col: str = 'call_date',
    fmt: str = '%Y-%m-%d %H:%M:%S.%f',
) -> pd.DataFrame:
    parsed = df.copy()
    parsed[time_col] = pd.to_datetime(parsed[time_col], format=fmt, errors='coerce')
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
    print("Loading and preprocessing data with timestamps")
    print("=" * 60)

    print("\n[1/4] Loading CSV...")
    df = pd.read_csv(csv_path, encoding='cp949')
    print(f"  Initial rows: {len(df):,}")

    validate_required_columns(df)

    required_missing = report_missing_values(df, REQUIRED_COLUMNS)
    print(f"  Required-column missing counts: {required_missing}")

    df = parse_call_dates(df)

    print("\n[2/4] Applying remove_other_region...")
    df = remove_other_region(
        df,
        x_col='xpos',
        y_col='ypos',
        x_range=config.us_range_x,
        y_range=config.us_range_y,
    )

    print("\n[3/4] Applying eleminate_duplicates...")
    df = eleminate_duplicates(df, client_col='clientid', time_col='call_date', window_seconds=600)

    print("\n[4/4] Applying crop_filter...")
    df = crop_filter(df, threshold_rate=0.85, step_rate=0.001)

    print(f"\nFinal preprocessed data: {len(df):,} rows")

    gdf = build_projected_geodataframe(df, config)

    if 'dest_xpos' in df.columns and 'dest_ypos' in df.columns:
        print("  Destination columns preserved")

    return df, gdf
