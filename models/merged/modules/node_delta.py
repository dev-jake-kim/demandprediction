"""노드별 LSTM 가중치 적응(ΔW) 공용 부품. docs/SPEC.md §6.7.

``NodeLSTMDelta``는 선택 노드 A개에 대해 공유 ``nn.LSTM``(1층) 가중치에 더할 ΔW를 가진다.
``additive``: 노드마다 독립 full-rank ΔW. ``basis``: ΔW[n] = Σ_m c[n,m]·B[m] (공유 기저 M개).
모든 파라미터 이름은 ``node_delta_``로 시작한다(train.py의 stage 1 고정·검사·init_from이 이름으로 찾는다).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def run_node_lstm(sequence: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias: Tensor) -> Tensor:
    """노드별 가중치 LSTM. ``sequence [B,A,L,F]``, ``weight_ih [A,4h,F]``, ``weight_hh [A,4h,h]``,
    ``bias [A,4h]`` -> 시점별 hidden ``[B,A,L,h]`` (PyTorch 게이트 순서 i, f, g, o)."""

    batch, num_nodes, steps, _ = sequence.shape
    hidden_size = weight_hh.shape[-1]
    gates_x = torch.einsum("bakf,agf->bakg", sequence, weight_ih) + bias.unsqueeze(1)
    hidden = sequence.new_zeros(batch, num_nodes, hidden_size)
    cell = sequence.new_zeros(batch, num_nodes, hidden_size)
    outputs = []
    for step in range(steps):
        gates = gates_x[:, :, step] + torch.einsum("bah,agh->bag", hidden, weight_hh)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
        cell = forget_gate.sigmoid() * cell + in_gate.sigmoid() * cell_gate.tanh()
        hidden = out_gate.sigmoid() * cell.tanh()
        outputs.append(hidden)
    return torch.stack(outputs, dim=2)


class NodeLSTMDelta(nn.Module):
    """선택 노드의 LSTM ΔW (``additive`` | ``basis``). 생성 시 전부 0이라 공유 LSTM과 같은 함수다."""

    def __init__(self, num_nodes: int, input_size: int, hidden_size: int, mode: str, rank: int = 8) -> None:
        super().__init__()
        if mode not in ("additive", "basis"):
            raise ValueError(f"NodeLSTMDelta mode는 'additive'|'basis' (받음: {mode!r})")
        self.mode = mode
        gates = 4 * hidden_size
        if mode == "basis":
            if rank < 1:
                raise ValueError("node_delta_rank는 1 이상이어야 함")
            self.rank = int(rank)
            self.node_delta_basis_ih = nn.Parameter(torch.zeros(rank, gates, input_size))
            self.node_delta_basis_hh = nn.Parameter(torch.zeros(rank, gates, hidden_size))
            self.node_delta_basis_bias = nn.Parameter(torch.zeros(rank, gates))
            self.node_delta_coef = nn.Parameter(torch.zeros(num_nodes, rank))
        else:
            self.node_delta_weight_ih = nn.Parameter(torch.zeros(num_nodes, gates, input_size))
            self.node_delta_weight_hh = nn.Parameter(torch.zeros(num_nodes, gates, hidden_size))
            self.node_delta_bias = nn.Parameter(torch.zeros(num_nodes, gates))

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.mode == "basis":
            return (self.node_delta_basis_ih, self.node_delta_basis_hh, self.node_delta_basis_bias, self.node_delta_coef)
        return (self.node_delta_weight_ih, self.node_delta_weight_hh, self.node_delta_bias)

    @torch.no_grad()
    def begin(self) -> None:
        """basis: 기저 B=0 유지, 계수 C ~ N(0, 1/M) (출력 불변). additive: 아무것도 안 함."""

        if self.mode == "basis":
            self.node_delta_coef.normal_(0.0, (1.0 / self.rank) ** 0.5)

    def weights(self, lstm: nn.LSTM) -> tuple[Tensor, Tensor, Tensor]:
        """노드별 ``(W_ih [A,4h,F], W_hh [A,4h,h], b [A,4h])`` = 공유 + ΔW."""

        shared_bias = lstm.bias_ih_l0 + lstm.bias_hh_l0
        if self.mode == "basis":
            coef = self.node_delta_coef
            return (
                lstm.weight_ih_l0 + torch.einsum("am,mgf->agf", coef, self.node_delta_basis_ih),
                lstm.weight_hh_l0 + torch.einsum("am,mgh->agh", coef, self.node_delta_basis_hh),
                shared_bias + coef @ self.node_delta_basis_bias,
            )
        return (
            lstm.weight_ih_l0 + self.node_delta_weight_ih,
            lstm.weight_hh_l0 + self.node_delta_weight_hh,
            shared_bias + self.node_delta_bias,
        )


__all__ = ["NodeLSTMDelta", "run_node_lstm"]
