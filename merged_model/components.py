"""Compatibility exports for the merged model's reusable components.

New code should import from ``merged_model.modules`` when it needs a specific
block.  Keeping this facade preserves the stable ``.components`` import used
by ``model.py`` and ``validate.py``.
"""

from .modules import (
    BranchAttention,
    CausalRetrieval,
    FourierScalarEmbedding,
    LocalHistoryEncoder,
    NeuralRetrievalGate,
    PeriodicLSTMEncoder,
)

__all__ = [
    "BranchAttention",
    "CausalRetrieval",
    "FourierScalarEmbedding",
    "LocalHistoryEncoder",
    "NeuralRetrievalGate",
    "PeriodicLSTMEncoder",
]
