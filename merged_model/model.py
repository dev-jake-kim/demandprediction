"""High-level orchestration for the unified demand model.

Branch-specific code lives in :mod:`components`; this file intentionally
contains only model construction, data flow, and the final MAE calculation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modules import (
    BranchAttention,
    CausalRetrieval,
    LocalHistoryEncoder,
    NeuralRetrievalGate,
    PeriodicLSTMEncoder,
)


class UnifiedDemandModel(nn.Module):
    """One end-to-end model for neural, periodic, and retrieval information."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        time_step: int = 24,
        local_radius: int = 2,
        d_model: int = 64,
        num_fourier_bands: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
        history_hidden: int = 64,
        periodic_hidden: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        retrieval_grid_path: str | Path | None = None,
        retrieval_k: int = 20,
        retrieval_chunk_size: int = 256,
        retrieval_scope: Literal["observed_past", "train_prefix"] = "observed_past",
        retrieval_train_end: int | None = None,
    ) -> None:
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        if local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if fusion_dim <= 0 or retrieval_k <= 0 or retrieval_chunk_size <= 0:
            raise ValueError("fusion_dim, retrieval_k, and retrieval_chunk_size must be positive")
        if transformer_heads <= 0 or d_model % transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads")

        self.height = height
        self.width = width
        self.num_nodes = height * width
        self.time_step = time_step

        self.local_history = LocalHistoryEncoder(
            height=height,
            width=width,
            time_step=time_step,
            local_radius=local_radius,
            d_model=d_model,
            num_fourier_bands=num_fourier_bands,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn=transformer_ffn,
            history_hidden=history_hidden,
            dropout=dropout,
        )
        self.daily_branch = PeriodicLSTMEncoder(periodic_hidden)
        self.weekly_branch = PeriodicLSTMEncoder(periodic_hidden)
        self.branch_attention = BranchAttention(history_hidden, periodic_hidden, fusion_dim)
        self.retrieval = CausalRetrieval(
            height=height,
            width=width,
            time_step=time_step,
            local_radius=local_radius,
            retrieval_grid_path=retrieval_grid_path,
            retrieval_k=retrieval_k,
            retrieval_chunk_size=retrieval_chunk_size,
            retrieval_scope=retrieval_scope,
            retrieval_train_end=retrieval_train_end,
        )
        self.output_gate = NeuralRetrievalGate(fusion_dim)

    # Keep the old debugging entry points available while the implementation
    # is organized under named components.
    def _crop_all_nodes(self, demands: Tensor) -> Tensor:
        return self.local_history.crop(demands)

    def _retrieve(self, local_crop: Tensor, sample_idx: Tensor) -> Tensor:
        return self.retrieval(local_crop, sample_idx)

    def forward(
        self,
        *,
        demand_history: Tensor,
        daily_demand: Tensor,
        daily_mask: Tensor,
        weekly_demand: Tensor,
        weekly_mask: Tensor,
        sample_idx: Tensor,
        target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        local_crop, h_neural = self.local_history(demand_history)
        h_daily, daily_valid = self.daily_branch(daily_demand, daily_mask)
        h_weekly, weekly_valid = self.weekly_branch(weekly_demand, weekly_mask)
        h_attn, attention_weights = self.branch_attention(
            h_neural, h_daily, h_weekly, daily_valid, weekly_valid
        )

        ir_out = self.retrieval(local_crop, sample_idx)
        neural_pred, lambda_weight, prediction = self.output_gate(h_attn, ir_out)
        prediction_grid = prediction.reshape(-1, self.height, self.width)

        output = {
            "prediction": prediction_grid,
            "prediction_flat": prediction,
            "neural_pred": neural_pred,
            "ir_out": ir_out,
            "lambda_weight": lambda_weight,
            "attention_weights": attention_weights,
            "h_neural": h_neural,
            "h_attn": h_attn,
            "daily_valid": daily_valid,
            "weekly_valid": weekly_valid,
        }
        if target is not None:
            output["loss"] = F.l1_loss(prediction_grid, target)
        return output


__all__ = ["UnifiedDemandModel"]
