"""검색 encoder: 3×3 창 수요 이력 → Gaussian posterior ``N(μ, σ²)`` (docs/RETRIEVAL_ENCODER.md)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class RetrievalEncoder(nn.Module):
    """``[B, k, P]`` raw 창 + 노드 id → ``(μ [B, L], log σ² [B, L])``.

    ``log1p`` 입력에 학습 때만 Gaussian 잡음을 더하고, Linear(P → D) 뒤에 frozen node embedding을
    더해 LSTM에 넣는다. node embedding은 buffer라 gradient·optimizer 대상이 아니다.
    """

    def __init__(
        self,
        node_embedding: Tensor,
        *,
        time_step: int = 8,
        num_neighbors: int = 9,
        hidden_dim: int = 16,
        latent_dim: int = 16,
        input_noise_std: float = 0.1,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if node_embedding.ndim != 2:
            raise ValueError(f'node_embedding must be [N, D], got {tuple(node_embedding.shape)}')
        self.time_step = time_step
        self.num_neighbors = num_neighbors
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_noise_std = float(input_noise_std)
        self.temperature = float(temperature)
        self.register_buffer('node_embedding', node_embedding.detach().clone().float())
        self.input_projection = nn.Linear(num_neighbors, node_embedding.shape[1])
        self.lstm = nn.LSTM(node_embedding.shape[1], hidden_dim, batch_first=True)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, windows: Tensor, nodes: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.log1p(windows.clamp_min(0.0))
        if self.training and self.input_noise_std > 0:
            # eval(검색·평가)에서는 잡음을 끈다.
            x = x + torch.randn_like(x) * self.input_noise_std
        x = self.input_projection(x) + self.node_embedding[nodes].unsqueeze(1)
        _, (hidden, _) = self.lstm(x)
        h = hidden[-1]
        return self.mu_head(h), self.logvar_head(h).clamp(-6.0, 2.0)

    def similarity(self, query: Tensor, keys: Tensor) -> Tensor:
        """``s = −‖q − k‖² / (L·T)``. ``query [..., L]``와 ``keys [..., L]``는 broadcast된다."""

        return -((query - keys) ** 2).sum(dim=-1) / (self.latent_dim * self.temperature)

    def config_dict(self) -> dict:
        return {
            'num_nodes': int(self.node_embedding.shape[0]),
            'node_dim': int(self.node_embedding.shape[1]),
            'time_step': self.time_step,
            'num_neighbors': self.num_neighbors,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'input_noise_std': self.input_noise_std,
            'temperature': self.temperature,
        }


def save_retrieval_encoder(encoder: RetrievalEncoder, path: str | Path, **extra) -> None:
    torch.save({'config': encoder.config_dict(), 'state_dict': encoder.state_dict(), **extra}, path)


def load_retrieval_encoder(path: str | Path, device: torch.device | str = 'cpu') -> RetrievalEncoder:
    """``save_retrieval_encoder``가 쓴 파일을 eval 모드 encoder로 읽는다."""

    payload = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=False)
    config = payload['config']
    # from_pretrained의 meta-device 기본값과 무관하게 실제 장치에서 만든다.
    with torch.device(device):
        encoder = RetrievalEncoder(
            torch.zeros(config['num_nodes'], config['node_dim']),
            time_step=config['time_step'],
            num_neighbors=config['num_neighbors'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            input_noise_std=config['input_noise_std'],
            temperature=config['temperature'],
        )
    encoder.load_state_dict(payload['state_dict'])
    return encoder.to(device).eval()


__all__ = ['RetrievalEncoder', 'load_retrieval_encoder', 'save_retrieval_encoder']
