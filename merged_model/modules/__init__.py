"""Reusable neural blocks for :mod:`comparison_models.merged_model`.

Each file owns one architectural responsibility.  The parent
``components.py`` module re-exports these names for backwards compatibility
with existing imports.
"""

from .attention import BranchAttention
from .embeddings import FourierScalarEmbedding
from .fusion import NeuralRetrievalGate
from .history import LocalHistoryEncoder
from .periodic import PeriodicLSTMEncoder
from .retrieval import CausalRetrieval

__all__ = [
    "BranchAttention",
    "CausalRetrieval",
    "FourierScalarEmbedding",
    "LocalHistoryEncoder",
    "NeuralRetrievalGate",
    "PeriodicLSTMEncoder",
]
