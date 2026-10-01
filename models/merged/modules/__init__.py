"""Reusable neural blocks for :mod:`models.merged`."""

from .embeddings import FourierScalarEmbedding
from .fusion import GATE_INIT, NodeHourGate
from .history import LocalHistoryEncoder
from .periodic import LinearTrendForecaster

__all__ = [
    'FourierScalarEmbedding',
    'GATE_INIT',
    'LinearTrendForecaster',
    'LocalHistoryEncoder',
    'NodeHourGate',
]
