"""울산 원본 호출 기록 → 노드·시각별 배차 상태 격자 ``data/raw/ulsan_dispatch_grid.npy`` ``[T, H, W, 4]``.

채널: raw_calls(지역 필터만 적용한 전체 호출 수, 중복 제거 전), failed(call_state=3), wait_min(성공 호출의
평균 board−call 대기 분, 0~120분만), high_send(send_level≥3). 좌표·시각은 수요 격자(``ulsan_temporal_grid.npy``)와 같다:
EPSG:5174, 700m, 격자 원점 min_x=407034.756, max_y=234826.179(전처리 산출물 셀 중심에서 역산), 호출 시각 floor('h').
docs/SPEC.md §4 ``data.aux_grid``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from modules.config import GraphBuildConfig

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MIN_X, MAX_Y = 407034.756, 234826.179
START = pd.Timestamp('2024-10-01 00:00')
CHANNELS = ('raw_calls', 'failed', 'wait_min', 'high_send')


def main() -> None:
    config = GraphBuildConfig()
    demand = np.load(REPO / 'data/raw/ulsan_temporal_grid.npy')
    steps, rows, cols = demand.shape
    df = pd.read_csv(ROOT / 'raw' / config.origin_csv_name, encoding='cp949',
                     usecols=['xpos', 'ypos', 'call_date', 'call_state', 'board_time', 'send_level'])
    df['call_date'] = pd.to_datetime(df['call_date'], errors='coerce')
    df['board_time'] = pd.to_datetime(df['board_time'], errors='coerce')
    (x_lo, x_hi), (y_hi, y_lo) = config.us_range_x, config.us_range_y
    df = df[df.xpos.between(x_lo, x_hi) & df.ypos.between(y_lo, y_hi) & df.call_date.notna()].copy()

    to_xy = Transformer.from_crs('EPSG:4326', config.target_crs, always_xy=True)
    x, y = to_xy.transform(df.xpos.values / 1e6, df.ypos.values / 1e6)
    col = np.floor((x - MIN_X) / config.grid_size).astype(int)
    row = np.floor((MAX_Y - y) / config.grid_size).astype(int)
    t = ((df.call_date.dt.floor('h') - START) / pd.Timedelta(hours=1)).astype(int).values
    keep = (row >= 0) & (row < rows) & (col >= 0) & (col < cols) & (t >= 0) & (t < steps)
    df, row, col, t = df[keep], row[keep], col[keep], t[keep]

    wait = (df.board_time - df.call_date).dt.total_seconds().values / 60
    served = df.call_state.values == 2
    valid_wait = served & (wait > 0) & (wait < 120)
    out = np.zeros((steps, rows, cols, len(CHANNELS)), dtype=np.float32)
    np.add.at(out[..., 0], (t, row, col), 1.0)
    np.add.at(out[..., 1], (t[df.call_state.values == 3], row[df.call_state.values == 3], col[df.call_state.values == 3]), 1.0)
    wait_sum = np.zeros((steps, rows, cols), dtype=np.float32)
    wait_n = np.zeros((steps, rows, cols), dtype=np.float32)
    np.add.at(wait_sum, (t[valid_wait], row[valid_wait], col[valid_wait]), wait[valid_wait])
    np.add.at(wait_n, (t[valid_wait], row[valid_wait], col[valid_wait]), 1.0)
    out[..., 2] = np.where(wait_n > 0, wait_sum / np.maximum(wait_n, 1), 0.0)
    high = df.send_level.values >= 3
    np.add.at(out[..., 3], (t[high], row[high], col[high]), 1.0)

    corr = np.corrcoef(out[..., 0].ravel(), demand.ravel())[0, 1]
    print(f'호출 {int(out[..., 0].sum()):,} (수요 격자 {int(demand.sum()):,}), 셀·시각 상관 {corr:.4f}, 실패 비율 {out[..., 1].sum() / out[..., 0].sum():.3f}')
    np.save(REPO / 'data/raw/ulsan_dispatch_grid.npy', out)


if __name__ == '__main__':
    main()
