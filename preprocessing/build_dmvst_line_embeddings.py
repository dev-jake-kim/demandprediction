"""DMVST-Net Semantic View용 LINE 그래프 임베딩을 미리 계산한다.

**주의: 이 스크립트는 `DA` conda env가 아니라 `Torch` conda env로 실행해야 한다**
(`conda run -n Torch python preprocessing/build_dmvst_line_embeddings.py --city ulsan`).
LINE 구현(`cogdl`)이 `DA` env에는 설치돼 있지 않고, gir의 실제 학습 파이프라인(`train.py`, `DA` env)이
cogdl에 의존하지 않도록 이 전처리만 별도 env로 분리했다 — 결과는 `.npy`로만 저장하고,
`models/modeling.py`(DMVSTModel)는 산술 없이 그 배열을 `np.load`해서 buffer로 등록할 뿐이다.

`/home/jinsu/PycharmProjects/DMVST/models/DMVSTModel.py`의 `line()` 함수 로직을 그대로 가져왔다.
입력은 `preprocessing/build_dmvst_graph.py`가 만든 `{u,v,w}` 엣지 CSV(w=논문 식(5)의 유사도,
exp(-DTW)).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cogdl.data import Graph
from cogdl.models.emb.line import LINE

REPO_ROOT = Path(__file__).parent.parent

if not hasattr(np, 'int'):
    np.int = int


def compute_line_embeddings(
    graph_path: Path,
    dimension: int = 64,
    walk_length: int = 40,
    walk_num: int = 10,
    negative: int = 5,
    batch_size: int = 100,
    alpha: float = 0.025,
    order: int = 2,
) -> np.ndarray:
    df = pd.read_csv(graph_path)
    edges = df[['u', 'v', 'w']].values
    nodes = set()
    for u, v, w in edges:
        nodes.add(u)
        nodes.add(v)

    node_to_id = {node: i for i, node in enumerate(sorted(nodes))}
    num_nodes = len(nodes)

    src_list, dst_list, edge_weights = [], [], []
    for src, dst, w in edges:
        src_list.append(node_to_id[src])
        dst_list.append(node_to_id[dst])
        edge_weights.append(w)

    edge_index = torch.LongTensor([src_list, dst_list])
    edge_weight = torch.FloatTensor(edge_weights)
    data = Graph(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)

    model = LINE(
        dimension=dimension,
        walk_length=walk_length,
        walk_num=walk_num,
        negative=negative,
        batch_size=batch_size,
        alpha=alpha,
        order=order,
    )
    embeddings = model(data)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True, choices=['ulsan', 'porto'])
    parser.add_argument('--graph_path', type=str, default=None, help='기본값: data/raw/{city}_dmvst_graph_edges.csv')
    parser.add_argument('--output_path', type=str, default=None, help='기본값: data/raw/{city}_dmvst_line_embeddings.npy')
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--walk_num', type=int, default=10)
    parser.add_argument('--negative', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.025, help='LINE 학습률(그래프 엣지 가중치 exp(-DTW)와는 별개)')
    parser.add_argument('--order', type=int, default=2)
    args = parser.parse_args()

    graph_path = Path(args.graph_path) if args.graph_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_dmvst_graph_edges.csv'
    output_path = Path(args.output_path) if args.output_path else REPO_ROOT / 'data' / 'raw' / f'{args.city}_dmvst_line_embeddings.npy'

    print(f"[{args.city}] loading graph from {graph_path}")
    embeddings = compute_line_embeddings(
        graph_path,
        dimension=args.dimension,
        walk_length=args.walk_length,
        walk_num=args.walk_num,
        negative=args.negative,
        batch_size=args.batch_size,
        alpha=args.alpha,
        order=args.order,
    )
    print(f"  embeddings shape: {embeddings.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"  saved to: {output_path}")


if __name__ == '__main__':
    main()
