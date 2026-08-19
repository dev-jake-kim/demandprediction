from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

try:
    import contextily as ctx
except ImportError:
    ctx = None

BASE_DIR = Path(__file__).parent
RAW_CSV = BASE_DIR / 'raw' / 'train.csv'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_PNG = OUTPUT_DIR / 'pickup_locations.png'

FIRST_POINT_PATTERN = re.compile(r'\[\[(-?\d+\.?\d*),(-?\d+\.?\d*)\]')

# Porto 도심 인근으로 좌표를 한정 (GPS 오류로 인한 원거리 이상치 제거용)
LON_RANGE = (-8.75, -8.50)
LAT_RANGE = (41.05, 41.25)

CHUNK_SIZE = 200_000


def extract_pickup_points(csv_path: Path) -> pd.DataFrame:
    lons: list[float] = []
    lats: list[float] = []

    reader = pd.read_csv(
        csv_path,
        usecols=['MISSING_DATA', 'POLYLINE'],
        chunksize=CHUNK_SIZE,
    )

    total_rows = 0
    for chunk in reader:
        total_rows += len(chunk)
        chunk = chunk[chunk['MISSING_DATA'] == False]  # noqa: E712
        coords = chunk['POLYLINE'].str.extract(FIRST_POINT_PATTERN)
        coords = coords.dropna()
        lons.extend(coords[0].astype(float).tolist())
        lats.extend(coords[1].astype(float).tolist())
        print(f"  Processed {total_rows:,} rows, collected {len(lons):,} pickup points so far")

    df = pd.DataFrame({'lon': lons, 'lat': lats})
    print(f"  Total pickup points extracted: {len(df):,}")
    return df


def main() -> None:
    print("\n" + "=" * 60)
    print("Loading Porto taxi pickup points from raw train.csv")
    print("=" * 60)

    df = extract_pickup_points(RAW_CSV)

    before = len(df)
    df = df[
        df['lon'].between(*LON_RANGE) & df['lat'].between(*LAT_RANGE)
    ]
    print(f"  Filtered to Porto bounding box: {len(df):,} / {before:,} points kept")

    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    print("\n" + "=" * 60)
    print("Plotting pickup locations")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_web.plot(ax=ax, markersize=0.3, color='#d7301f', alpha=0.03)

    if ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution=False)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            print(f"  Failed to load basemap: {exc}")
    else:
        print("  contextily is not installed, basemap was skipped")

    ax.set_title(f'Porto Taxi Pickup Locations (n={len(gdf):,})')
    ax.set_axis_off()
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved pickup location map to: {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
