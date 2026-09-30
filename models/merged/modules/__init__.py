"""Reusable neural blocks for :mod:`models.merged`."""

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
