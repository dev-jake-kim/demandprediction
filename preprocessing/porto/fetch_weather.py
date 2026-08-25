"""porto 격자 데이터와 동일한 기간(2013-07-01 ~ 2014-06-30, UTC, 8760시간)의 시간당 날씨를
Open-Meteo Historical Weather API(무료, API 키 불필요)에서 받아 data/raw/porto_meteorological_data.csv
로 저장한다. row i가 data/raw/porto_temporal_grid.npy의 시간 인덱스 i와 1:1로 대응한다.

울산 쪽(`/home/jinsu/PycharmProjects/DMVST/data/raw/meteorological_data.csv`)과 같은 컬럼 이름
(기온(°C), 강수량(mm), 적설(cm))을 써서, 이후 두 도시를 같은 파서로 읽을 수 있게 맞췄다.

Usage:
    python preprocessing/porto/fetch_weather.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).parent.parent.parent
API_URL = 'https://archive-api.open-meteo.com/v1/archive'
HOURLY_VARS = ['temperature_2m', 'precipitation', 'snowfall']

# 포르투 시내 중심 좌표. porto_temporal_grid.npy의 실제 기간(preprocessing/porto/README.md:
# "데이터 기간 2013-07~2014-06")과 일치.
DEFAULT_LAT = 41.1579
DEFAULT_LON = -8.6291
START_DATE = '2013-07-01'
END_DATE = '2014-06-30'
EXPECTED_HOURS = 8760


def fetch(lat: float, lon: float) -> pd.DataFrame:
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'hourly': ','.join(HOURLY_VARS),
        'timezone': 'UTC',
    }
    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()['hourly'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lat', type=float, default=DEFAULT_LAT)
    parser.add_argument('--lon', type=float, default=DEFAULT_LON)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output_path) if args.output_path else REPO_ROOT / 'data' / 'raw' / 'porto_meteorological_data.csv'

    print(f'Fetching Porto weather: lat={args.lat}, lon={args.lon}, {START_DATE}..{END_DATE} (UTC)')
    df = fetch(args.lat, args.lon)
    print(f'Got {len(df)} hourly rows (expected {EXPECTED_HOURS})')
    if len(df) != EXPECTED_HOURS:
        raise ValueError(
            f'받아온 행 수({len(df)})가 porto grid 시간 수({EXPECTED_HOURS})와 다름 — '
            f'날짜 범위/DST 처리를 확인해야 함'
        )

    out_df = pd.DataFrame({
        '지점': 'PORTO',
        '지점명': 'Porto',
        '일시': df['time'],
        '기온(°C)': df['temperature_2m'],
        '강수량(mm)': df['precipitation'],
        '적설(cm)': df['snowfall'],
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding='cp949')
    print(f'Saved {output_path} ({len(out_df)} rows)')


if __name__ == '__main__':
    main()
