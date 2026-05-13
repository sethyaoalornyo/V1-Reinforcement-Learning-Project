from __future__ import annotations
import json
import os
from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple multi-layer perceptron Q-network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingMLP(nn.Module):
    """
    Dueling network architecture.
    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        V = self.value_stream(shared)
        A = self.advantage_stream(shared)
        return V + (A - A.mean(dim=-1, keepdim=True))


def save_checkpoint(
    online: nn.Module,
    target: nn.Module,
    meta: dict,
    directory: str,
) -> None:
    os.makedirs(directory, exist_ok=True)
    torch.save(online.state_dict(), os.path.join(directory, "online.pt"))
    torch.save(target.state_dict(), os.path.join(directory, "target.pt"))
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    online: nn.Module,
    target: nn.Module,
    directory: str,
) -> dict:
    online.load_state_dict(torch.load(os.path.join(directory, "online.pt")))
    target.load_state_dict(torch.load(os.path.join(directory, "target.pt")))
    with open(os.path.join(directory, "meta.json")) as f:
        return json.load(f)
