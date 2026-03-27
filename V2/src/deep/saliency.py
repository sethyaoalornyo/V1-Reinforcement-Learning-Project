"""
Saliency Analysis
==================
Saliency methods explain WHICH parts of the input the neural network
is "attending to" when making a decision.  For our DDoS environment the
input is a one-hot state vector, so saliency tells us which state
dimensions drive the Q-value predictions most.

Methods implemented
-------------------
1. gradient_saliency    : vanilla gradient saliency (∂Q/∂input)
2. integrated_gradients : path-integral attribution (Sundararajan 2017)
3. plot_saliency_heatmap: matplotlib visualisation of saliency maps

For higher-dimensional inputs (e.g. raw packet feature vectors)
gradient_saliency and integrated_gradients generalise directly.
"""

from __future__ import annotations
import os
from typing import List, Optional

import torch
import torch.nn as nn

from src.deep.networks import state_to_tensor


# ------------------------------------------------------------------ #
#  1. Vanilla Gradient Saliency                                         #
# ------------------------------------------------------------------ #

def gradient_saliency(
    model: nn.Module,
    state: int,
    nS: int,
    action: Optional[int] = None,
    device: str = "cpu",
) -> List[float]:
    """
    Compute the gradient of Q(s, action) w.r.t. the INPUT state vector.

    A large |∂Q/∂x_i| means input dimension i strongly influences the
    predicted Q-value for the chosen action.

    Parameters
    ----------
    model  : trained DQN / DuelingMLP
    state  : integer state index
    nS     : total number of states (input dimension)
    action : which action's Q-value to differentiate.
             If None, uses the greedy (argmax) action.

    Returns
    -------
    saliency : list of length nS — absolute gradient values
    """
    model.eval()
    x = state_to_tensor(state, nS, device).requires_grad_(True)

    q_values = model(x)                                # (1, nA)

    if action is None:
        action = int(q_values.argmax(dim=1).item())

    # Backprop through the chosen action's Q-value
    q_a = q_values[0, action]
    model.zero_grad()
    q_a.backward()

    saliency = x.grad.squeeze(0).abs().tolist()
    return saliency


# ------------------------------------------------------------------ #
#  2. Integrated Gradients                                              #
# ------------------------------------------------------------------ #

def integrated_gradients(
    model: nn.Module,
    state: int,
    nS: int,
    action: Optional[int] = None,
    n_steps: int = 50,
    device: str = "cpu",
) -> List[float]:
    """
    Integrated Gradients attribution (Sundararajan et al., 2017).

    Instead of a single gradient at the input, IG integrates the
    gradient along the straight-line path from a BASELINE (all-zeros)
    to the actual input x:

        IG_i(x) = (x_i − x̄_i) × ∫₀¹ ∂Q/∂x_i (x̄ + α(x−x̄)) dα

    This satisfies the *completeness* axiom: ΣᵢIG_i = Q(x) − Q(x̄),
    meaning attributions sum to the actual output difference — making
    them more trustworthy than vanilla gradients.

    Parameters
    ----------
    n_steps : number of interpolation steps (higher = more accurate)

    Returns
    -------
    ig_values : list of length nS — integrated gradient per input dim
    """
    model.eval()
    x        = state_to_tensor(state, nS, device)
    baseline = torch.zeros_like(x)

    if action is None:
        with torch.no_grad():
            action = int(model(x).argmax(dim=1).item())

    # Accumulate gradients along interpolation path
    integrated_grads = torch.zeros(nS, device=device)

    for step in range(1, n_steps + 1):
        alpha     = step / n_steps
        interp    = (baseline + alpha * (x - baseline)).requires_grad_(True)
        q_values  = model(interp)
        q_a       = q_values[0, action]
        model.zero_grad()
        q_a.backward()
        integrated_grads += interp.grad.squeeze(0)

    # Scale by (x - baseline) and divide by n_steps
    ig_values = ((x - baseline).squeeze(0) * integrated_grads / n_steps).tolist()
    return ig_values


# ------------------------------------------------------------------ #
#  3. Visualisation                                                      #
# ------------------------------------------------------------------ #

def plot_saliency_heatmap(
    saliency_by_state: List[List[float]],
    state_names: List[str],
    action_names: Optional[List[str]] = None,
    title: str = "Saliency Map",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a heatmap of saliency values for each (state, input_dim) pair.

    Parameters
    ----------
    saliency_by_state : list of saliency vectors, one per state
                        shape: (nS, input_dim)
    state_names       : label for each row (query state)
    action_names      : if provided, label columns as actions
    save_path         : if given, save PNG to this path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[Saliency] matplotlib not installed — skipping plot.")
        return

    n_rows = len(saliency_by_state)
    n_cols = len(saliency_by_state[0])

    # Build 2D array
    data = [saliency_by_state[i] for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(4, n_rows * 0.8)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="|Gradient|")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"Query: {s}" for s in state_names])

    if action_names and n_cols == len(action_names):
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(action_names, rotation=30, ha="right")
    else:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"dim {i}" for i in range(n_cols)], rotation=30)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Saliency] Plot saved → {save_path}")

    plt.show()
    plt.close()


# ------------------------------------------------------------------ #
#  Convenience: compute saliency for all states                         #
# ------------------------------------------------------------------ #

def full_saliency_report(
    model: nn.Module,
    nS: int,
    state_names: List[str],
    method: str = "gradient",
    device: str = "cpu",
    save_dir: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute saliency for every state and optionally save a heatmap.

    Parameters
    ----------
    method : "gradient" or "integrated_gradients"

    Returns
    -------
    saliency_matrix : shape (nS, nS) — saliency of each input dim
                      for each query state
    """
    saliency_matrix = []
    for s in range(nS):
        if method == "integrated_gradients":
            sal = integrated_gradients(model, s, nS, device=device)
        else:
            sal = gradient_saliency(model, s, nS, device=device)
        saliency_matrix.append(sal)

    if save_dir:
        save_path = os.path.join(save_dir, f"saliency_{method}.png")
        plot_saliency_heatmap(
            saliency_matrix,
            state_names=state_names,
            title=f"DQN Saliency ({method.replace('_', ' ').title()})",
            save_path=save_path,
        )

    return saliency_matrix
