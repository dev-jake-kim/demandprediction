"""Daily and weekly lag encoders with explicit invalid-lag handling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .node_delta import NodeLSTMDelta, run_node_lstm


class PeriodicLSTMEncoder(nn.Module):
    """Encode valid daily/weekly lag values in chronological order."""

    def __init__(
        self,
        hidden_size: int,
        extra_dim: int = 0,
        node_adaptive_indices: Sequence[int] | None = None,
        node_adaptive_mode: str = "additive",
        node_delta_rank: int = 8,
        num_nodes: int = 0,
        node_embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        if extra_dim < 0:
            raise ValueError("extra_dim must be non-negative")
        self.extra_dim = extra_dim
        # 노드 임베딩 입력(선택): 공용 주기 LSTM에 노드 정체성을 준다(stage 1부터 학습, 노드별 게이트 bias와 같은 효과).
        self.node_embedding = (
            nn.Parameter(torch.randn(num_nodes, node_embedding_dim) * 0.1) if node_embedding_dim > 0 else None
        )
        input_size = 1 + extra_dim + node_embedding_dim
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 선택 노드만 LSTM ΔW(docs/SPEC.md §6.7 node_adaptive_scope=periodic). 노드 id는 리스트로 보관.
        self._node_index_list = None if node_adaptive_indices is None else [int(v) for v in node_adaptive_indices]
        self._cached_node_index: Tensor | None = None
        self.node_delta = (
            None if self._node_index_list is None
            else NodeLSTMDelta(len(self._node_index_list), input_size, hidden_size, node_adaptive_mode, node_delta_rank)
        )

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        return () if self.node_delta is None else self.node_delta.node_delta_parameters()

    def forward(
        self, values: Tensor, invalid_mask: Tensor, extra: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if values.ndim != 4 or values.shape[-1] != 1:
            raise ValueError(f"Periodic values must be [B,L,N,1], got {tuple(values.shape)}")
        if invalid_mask.ndim != 2 or invalid_mask.shape[:2] != values.shape[:2]:
            raise ValueError("Periodic mask must be [B,L] matching the first two value dimensions")

        batch, length, nodes, _ = values.shape
        valid_mask = ~invalid_mask.bool()
        lengths = valid_mask.sum(dim=1)
        sequence = torch.log1p(torch.clamp(values, min=0.0))

        if self.extra_dim > 0:
            if extra is None:
                raise ValueError(f"extra_dim={self.extra_dim}인데 extra가 None임")
            if extra.shape != (batch, length, self.extra_dim):
                raise ValueError(
                    f"extra must be [B,L,{self.extra_dim}], got {tuple(extra.shape)}"
                )
            sequence = torch.cat(
                [sequence, extra[:, :, None, :].expand(batch, length, nodes, self.extra_dim)],
                dim=-1,
            )
        elif extra is not None:
            raise ValueError("extra_dim=0인데 extra가 주어짐")
        if self.node_embedding is not None:
            if self.node_embedding.shape[0] != nodes:
                raise ValueError(f"node_embedding 노드 수 {self.node_embedding.shape[0]} != 입력 {nodes}")
            sequence = torch.cat(
                [sequence, self.node_embedding[None, None].expand(batch, length, -1, -1).to(sequence.dtype)], dim=-1
            )

        # 유효 lag를 원래 순서대로 앞에 모은다. lag 유효성은 샘플 단위라 노드 축에 공유된다.
        order = valid_mask.to(torch.long).argsort(dim=1, descending=True, stable=True)
        feature_dim = sequence.shape[-1]
        compact = sequence.gather(1, order[:, :, None, None].expand(-1, -1, nodes, feature_dim))
        compact = compact.permute(0, 2, 1, 3).reshape(batch * nodes, length, feature_dim)
        # 전체 길이로 돌리고 각 행의 마지막 유효 시점 출력을 읽는다(packed LSTM의 마지막 hidden과
        # 같다). host 동기화가 필요한 bool index·packing을 쓰지 않는다.
        steps, _ = self.lstm(compact)
        if self.node_delta is not None:
            index = self._cached_node_index
            if index is None or index.device != compact.device:
                index = torch.as_tensor(self._node_index_list, dtype=torch.long, device=compact.device)
                self._cached_node_index = index
            node_sequence = compact.reshape(batch, nodes, length, feature_dim)[:, index]
            node_steps = run_node_lstm(node_sequence, *self.node_delta.weights(self.lstm))
            steps = steps.reshape(batch, nodes, length, -1).index_copy(1, index, node_steps.to(steps.dtype))
            steps = steps.reshape(batch * nodes, length, -1)
        last = (lengths.clamp_min(1) - 1).repeat_interleave(nodes)
        index = last[:, None, None].expand(-1, 1, steps.shape[-1])
        hidden = steps.gather(1, index).squeeze(1)
        row_valid = (lengths > 0).repeat_interleave(nodes)
        output = hidden * row_valid[:, None].to(hidden.dtype)
        return output.reshape(batch, nodes, -1), lengths > 0


__all__ = ["PeriodicLSTMEncoder"]
