"""Reusable neural blocks for :mod:`models.merged`.

Each file owns one architectural responsibility. 원본 ``merged_model/modules``를 그대로
옮긴 것이며, 계산은 바뀌지 않았다(import 경로와 meta-device 안전성만 조정).
"""

from .attention import BranchAttention
from .embeddings import FourierScalarEmbedding
from .fusion import NeuralRetrievalGate
from .history import LocalHistoryEncoder
from .periodic import PeriodicLSTMEncoder
from .retrieval import CausalRetrieval

__all__ = [
    'BranchAttention',
    'CausalRetrieval',
    'FourierScalarEmbedding',
    'LocalHistoryEncoder',
    'NeuralRetrievalGate',
    'PeriodicLSTMEncoder',
]
