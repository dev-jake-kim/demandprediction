"""노드별 LSTM 가중치 적응(ΔW) 공용 부품. docs/SPEC.md §6.7.

``NodeLSTMDelta``는 선택 노드 A개에 대해 공유 ``nn.LSTM``(1층) 가중치에 더할 ΔW를 가진다.
``additive``: 노드마다 독립 full-rank ΔW. ``basis``: ΔW[n] = Σ_m c[n,m]·B[m] (공유 기저 M개).
모든 파라미터 이름은 ``node_delta_``로 시작한다(train.py의 stage 1 고정·검사·init_from이 이름으로 찾는다).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


def run_dynamic_basis_lstm(
    sequence: Tensor,
    lstm: nn.LSTM,
    coef: Tensor,
    basis_ih: Tensor,
    basis_hh: Tensor,
    basis_bias: Tensor,
) -> Tensor:
    """샘플·노드별 계수 ``coef [B,A,M]``의 basis ΔW LSTM 마지막 hidden ``[B,A,h]``.

    ``W(b,a) = W + Σ_m coef[b,a,m]·B[m]``를 가중치로 만들지 않고 활성값 공간에서 계산한다:
    ``x·W(b,a)ᵀ = x·Wᵀ + Σ_m coef[b,a,m]·(x·B[m]ᵀ)``.
    """

    batch, num_nodes, steps, _ = sequence.shape
    hidden_size = lstm.hidden_size
    shared_x = torch.einsum("bakf,gf->bakg", sequence, lstm.weight_ih_l0) + (lstm.bias_ih_l0 + lstm.bias_hh_l0)
    basis_x = torch.einsum("bakf,mgf->bakmg", sequence, basis_ih) + basis_bias[None, None, None]
    gates_x = shared_x + torch.einsum("bam,bakmg->bakg", coef, basis_x)
    hidden = sequence.new_zeros(batch, num_nodes, hidden_size)
    cell = sequence.new_zeros(batch, num_nodes, hidden_size)
    for step in range(steps):
        recurrent = hidden @ lstm.weight_hh_l0.T + torch.einsum(
            "bam,bamg->bag", coef, torch.einsum("bah,mgh->bamg", hidden, basis_hh)
        )
        gates = gates_x[:, :, step] + recurrent
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=-1)
        cell = forget_gate.sigmoid() * cell + in_gate.sigmoid() * cell_gate.tanh()
        hidden = out_gate.sigmoid() * cell.tanh()
    return hidden


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


class NodeLSTMMoE(nn.Module):
    """선택 노드의 LSTM을 공유 전문가 K개(W + ΔW_k)의 router 혼합으로 바꾼다. docs/SPEC.md §6.7 ``moe``.

    파라미터는 전부 0으로 생성(stage 1에서 공유 LSTM과 같음). ``begin``이 router 표 R을 N(0,1)로 채운다.
    """

    def __init__(
        self,
        num_nodes: int,
        input_size: int,
        hidden_size: int,
        num_experts: int = 4,
        router: str = "static",
        top_k: int = 0,
        assign: Sequence[int] | None = None,
        residual: bool = False,
        routing: str = "soft",
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("node_moe_experts는 2 이상이어야 함")
        if router not in ("static", "dynamic"):
            raise ValueError(f"node_moe_router는 'static'|'dynamic' (받음: {router!r})")
        if not 0 <= top_k <= num_experts:
            raise ValueError("node_moe_top_k는 0(전체)~전문가 수")
        if routing not in ("soft", "gumbel"):
            raise ValueError(f"node_moe_routing은 'soft'|'gumbel' (받음: {routing!r})")
        if routing == "gumbel" and (router != "static" or top_k != 0):
            raise ValueError("node_moe_routing=gumbel은 node_moe_router=static, node_moe_top_k=0에서만 쓴다")
        if tau <= 0:
            raise ValueError("node_moe_tau는 양수여야 함")
        # gumbel: 노드별 범주형 분포 softmax(R[n])에서 전문가 하나를 고른다(학습: straight-through Gumbel 표본, 추론: argmax).
        self.routing = routing
        self.tau = float(tau)
        # 노드 → 전문가 초기 배정(train 구간 노드 기술자 k-means, 선택). 없으면 router 표를 난수로 시작.
        self._assign_list = None if assign is None else [int(v) for v in assign]
        if self._assign_list is not None and (
            len(self._assign_list) != num_nodes or min(self._assign_list) < 0 or max(self._assign_list) >= num_experts
        ):
            raise ValueError(f"node_moe_assign은 노드 {num_nodes}개의 0..{num_experts - 1} 전문가 id여야 함")
        gates = 4 * hidden_size
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.node_delta_expert_ih = nn.Parameter(torch.zeros(num_experts, gates, input_size))
        self.node_delta_expert_hh = nn.Parameter(torch.zeros(num_experts, gates, hidden_size))
        self.node_delta_expert_bias = nn.Parameter(torch.zeros(num_experts, gates))
        self.node_delta_router = nn.Parameter(torch.zeros(num_nodes, num_experts))
        self.node_delta_router_input = (
            nn.Parameter(torch.zeros(input_size, num_experts)) if router == "dynamic" else None
        )
        # residual: 공유 전문가 위에 노드 고유 ΔW_n을 더한다(전문가 k의 노드 n 가중치 = W + ΔW_k + ΔW_n).
        self.residual = bool(residual)
        if self.residual:
            self.node_delta_node_ih = nn.Parameter(torch.zeros(num_nodes, gates, input_size))
            self.node_delta_node_hh = nn.Parameter(torch.zeros(num_nodes, gates, hidden_size))
            self.node_delta_node_bias = nn.Parameter(torch.zeros(num_nodes, gates))

    def node_delta_parameters(self) -> tuple[nn.Parameter, ...]:
        params = (self.node_delta_expert_ih, self.node_delta_expert_hh, self.node_delta_expert_bias, self.node_delta_router)
        if self.node_delta_router_input is not None:
            params += (self.node_delta_router_input,)
        if self.residual:
            params += (self.node_delta_node_ih, self.node_delta_node_hh, self.node_delta_node_bias)
        return params

    @torch.no_grad()
    def begin(self) -> None:
        """router 표 R: 배정이 있으면 R[n, assign[n]] = 2(나머지 0), 없으면 N(0,1). dynamic이면 W_r ~ N(0, 1/F).
        전문가 ΔW=0이라 출력은 불변이고, 노드별 게이트 차이로 전문가가 갈라진다."""

        if self._assign_list is None:
            self.node_delta_router.normal_(0.0, 1.0)
        else:
            self.node_delta_router.zero_()
            index = torch.as_tensor(self._assign_list, dtype=torch.long, device=self.node_delta_router.device)
            self.node_delta_router.scatter_(1, index[:, None], 2.0)
        if self.node_delta_router_input is not None:
            self.node_delta_router_input.normal_(0.0, self.node_delta_router_input.shape[0] ** -0.5)

    def gates(self, sequence: Tensor) -> Tensor:
        """``sequence [B,A,L,F]`` -> 게이트 ``[B,A,K]``.

        soft: softmax(top_k>0이면 상위 k개만). gumbel: 순전파 값은 one-hot(학습은 Gumbel 표본, 추론은 argmax R),
        학습 중 역전파는 softmax((R + ε)/τ)를 거친다(straight-through).
        """

        logits = self.node_delta_router.unsqueeze(0).expand(sequence.shape[0], -1, -1)
        if self.routing == "gumbel":
            if not self.training:
                return F.one_hot(logits.argmax(dim=-1), self.num_experts).to(logits.dtype)
            uniform = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-7)
            noisy = (logits - torch.log(-torch.log(uniform))) / self.tau
            soft = torch.softmax(noisy, dim=-1)
            hard = F.one_hot(soft.argmax(dim=-1), self.num_experts).to(soft.dtype)
            return hard - soft.detach() + soft
        if self.node_delta_router_input is not None:
            logits = logits + sequence.mean(dim=2) @ self.node_delta_router_input
        if 0 < self.top_k < self.num_experts:
            threshold = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def forward(self, lstm: nn.LSTM, sequence: Tensor) -> Tensor:
        """``sequence [B,A,L,F]`` -> 시점별 혼합 hidden ``[B,A,L,h]``."""

        batch, num_nodes, steps, features = sequence.shape
        flat = sequence.reshape(batch * num_nodes, steps, features)
        zeros = flat.new_zeros(1, flat.shape[0], lstm.hidden_size)
        gate = self.gates(sequence)
        mixed = None
        for expert in range(self.num_experts):
            if self.residual:
                output = run_node_lstm(
                    sequence,
                    lstm.weight_ih_l0 + self.node_delta_expert_ih[expert] + self.node_delta_node_ih,
                    lstm.weight_hh_l0 + self.node_delta_expert_hh[expert] + self.node_delta_node_hh,
                    lstm.bias_ih_l0 + lstm.bias_hh_l0 + self.node_delta_expert_bias[expert] + self.node_delta_node_bias,
                )
            else:
                weights = [
                    lstm.weight_ih_l0 + self.node_delta_expert_ih[expert],
                    lstm.weight_hh_l0 + self.node_delta_expert_hh[expert],
                    lstm.bias_ih_l0 + self.node_delta_expert_bias[expert],
                    lstm.bias_hh_l0,
                ]
                output = torch._VF.lstm(flat, (zeros, zeros), weights, True, 1, 0.0, self.training, False, True)[0]
            term = gate[..., expert, None, None] * output.reshape(batch, num_nodes, steps, -1)
            mixed = term if mixed is None else mixed + term
        return mixed


__all__ = ["NodeLSTMDelta", "NodeLSTMMoE", "run_dynamic_basis_lstm", "run_node_lstm"]
