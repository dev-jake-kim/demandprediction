from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def save_graph_json(
    output_path: Path,
    nodes: list[dict],
    demands: np.ndarray,
    near_demands: list[np.ndarray],
    temporal_features: list[dict],
    od_flows: list[list[dict]],
) -> None:
    print("\n" + "=" * 60)
    print("Saving graph to JSON")
    print("=" * 60)

    x = []
    for time_idx in range(len(demands)):
        x_t = {
            'demand': demands[time_idx].tolist(),
            # 노드별 near cell을 합산하지 않고, 셀 단위 해상도 그대로 유지
            # (near_patches[node] 순서와 동일: 위치1의 수요, 위치2의 수요, ...)
            'near_demands': [node_near[time_idx].tolist() for node_near in near_demands],
            'day': temporal_features[time_idx]['day'],
            'time': temporal_features[time_idx]['time'],
            'holiday': temporal_features[time_idx]['holiday'],
            'OD': od_flows[time_idx],
        }
        x.append(x_t)

    data = {
        'nodes': nodes,
        'x': x,
    }

    json_str = json.dumps(data, indent=2, ensure_ascii=False, separators=(',', ': '))

    def compact_list(match: re.Match[str]) -> str:
        return match.group(0).replace('\n', '').replace(' ', '')

    compact_json = re.sub(r'\[[\s\d,]+\]', compact_list, json_str)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(compact_json)

    print(f"\nJSON saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nData structure:")
    print(f"  nodes: {len(nodes)} nodes")
    print(f"  x: {len(x)} timesteps")
    print(f"    - Each timestep has: demand (list of {len(demands[0])}), near_demands (per-node list of per-cell demand), day, time, holiday, OD (list)")
