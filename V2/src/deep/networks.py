"""
Neural Network Architectures
==============================
Architectures suitable for the DDoS mitigation environment.

The DDoS environment has a SMALL discrete state space (nS=5, nA=3), so
we use compact MLPs.  The one-hot encoding turns state integer s into
a vector of length nS, which feeds into the network.

Architectures
-------------
1. MLP        : Simple multi-layer perceptron → used for standard DQN.
2. DuelingMLP : Dueling network (Wang et al., 2016) — splits the Q-head
                into a Value stream V(s) and an Advantage stream A(s,a),
                then combines: Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))
                This stabilises training by allowing V to be learned
                without needing samples for every (s,a) pair.

Usage
-----
    net = MLP(input_dim=5, hidden_dim=64, output_dim=3)
    q_values = net(state_tensor)   # shape (batch, 3)

Checkpoint convention
---------------------
Checkpoints are saved to:
    checkpoints/<algorithm>/<task>/<run_tag>/
        online.pt   ← weights of the online network
        target.pt   ← weights of the target network (DQN)
        meta.json   ← hyperparameters and training step
"""

from __future__ import annotations
import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Helper — one-hot state encoding                                      #
# ------------------------------------------------------------------ #

def state_to_tensor(state: int, nS: int, device: str = "cpu") -> torch.Tensor:
    """Convert integer state to one-hot float tensor of shape (1, nS)."""
    one_hot = torch.zeros(1, nS, device=device)
    one_hot[0, state] = 1.0
    return one_hot


def batch_states_to_tensor(states: List[int], nS: int, device: str = "cpu") -> torch.Tensor:
    """Convert a list of integer states to a batched one-hot tensor (B, nS)."""
    batch = torch.zeros(len(states), nS, device=device)
    for i, s in enumerate(states):
        batch[i, s] = 1.0
    return batch


# ------------------------------------------------------------------ #
#  1. Simple MLP                                                        #
# ------------------------------------------------------------------ #

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for DQN.

    Architecture
    ------------
    Input (nS)  →  Linear + ReLU  →  Linear + ReLU  →  Output (nA)

    Parameters
    ----------
    input_dim  : size of input (= nS for one-hot encoding)
    hidden_dim : number of units in each hidden layer
    output_dim : number of outputs (= nA, one Q-value per action)
    n_layers   : total number of Linear layers (including output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → Q-values: (batch, output_dim)"""
        return self.net(x)


# ------------------------------------------------------------------ #
#  2. Dueling MLP                                                       #
# ------------------------------------------------------------------ #

class DuelingMLP(nn.Module):
    """
    Dueling Network for DQN (Wang et al., 2016).

    Splits the hidden representation into:
      - Value stream  : V(s)       — scalar
      - Advantage stream: A(s, a) — vector of length nA

    Combined: Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

    The mean subtraction ensures identifiability (otherwise V and A are
    redundant and can cancel in arbitrary ways).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream: → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream: → nA values
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        V    = self.value_stream(feat)               # (B, 1)
        A    = self.advantage_stream(feat)           # (B, nA)
        # Identifiable dueling combination
        Q    = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# ------------------------------------------------------------------ #
#  Checkpoint utilities                                                  #
# ------------------------------------------------------------------ #

def save_checkpoint(
    model: nn.Module,
    path: str,
    meta: Optional[dict] = None,
) -> None:
    """
    Save model weights (and optional metadata) to disk.

    Convention: path ends in .pt, metadata saved to same dir as meta.json.

    Example
    -------
    save_checkpoint(online_net, "checkpoints/dqn/ddos/run1/online.pt",
                    meta={"step": 5000, "epsilon": 0.1})
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    if meta is not None:
        meta_path = os.path.join(os.path.dirname(path), "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load weights from a .pt checkpoint into `model`."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Checkpoint] Loaded ← {path}")
    return model
