"""
Temporal Difference Methods
============================
TD learning bootstraps from intermediate estimates — unlike MC it does
NOT need complete episodes and can run in continuing tasks.

Algorithms
----------
1.  td0_prediction          : TD(0) — simplest 1-step bootstrapping
2.  td_n_prediction         : n-step TD (FORWARD VIEW)
                              G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k+1} + γ^n V(s_{t+n})
3.  td_lambda_forward       : TD(λ) FORWARD VIEW (λ-return)
                              G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^(n)
                              Uses episodes and computes the full λ-return offline.
4.  td_lambda_backward      : TD(λ) BACKWARD VIEW (eligibility traces, online)
                              δ_t = r + γV(s') - V(s)
                              e(s) ← γλ e(s) + 1   (accumulating traces)
                              V(s) ← V(s) + α δ_t e(s) for all s

Why both views?
---------------
* Forward view is conceptually clearer — it defines what TD(λ) computes.
* Backward view is computationally efficient — one pass per step,
  updating ALL states whose trace is non-zero.  The two views are
  equivalent in the linear case (Sutton & Barto, Ch. 12).

N-cutoff
--------
The n-step methods naturally implement n-cutoff by only looking n steps
ahead.  Setting n=1 recovers TD(0); n=∞ recovers MC.
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple

from src.utils.env_wrapper import TabularEnv


# ------------------------------------------------------------------ #
#  1. TD(0) Prediction                                                  #
# ------------------------------------------------------------------ #

def td0_prediction(
    env: TabularEnv,
    policy,                    # list[int] or callable(s) → a
    num_episodes: int = 2000,
    gamma: float = 0.95,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    TD(0) policy evaluation.  For each step:
        δ = r + γ V(s') − V(s)
        V(s) ← V(s) + α δ

    Returns
    -------
    V        : state-value estimates, shape (nS,)
    td_errors: list of |δ| per step (for diagnostics)
    """
    env.seed(seed)
    V  = [0.0] * env.nS
    td_errors = []

    for _ in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a           = policy(s) if callable(policy) else policy[s]
            s2, r, done, _ = env.step(a)
            v_next      = 0.0 if done else V[s2]
            delta       = r + gamma * v_next - V[s]
            V[s]       += alpha * delta
            td_errors.append(abs(delta))
            s = s2

    return V, td_errors


# ------------------------------------------------------------------ #
#  2. n-Step TD Prediction (Forward View)                               #
# ------------------------------------------------------------------ #

def td_n_prediction(
    env: TabularEnv,
    policy,
    num_episodes: int = 2000,
    gamma: float = 0.95,
    alpha: float = 0.05,
    n: int = 4,
    seed: int = 0,
) -> List[float]:
    """
    n-Step TD prediction (forward view).

    At each step t, we buffer n rewards and then compute:
        G = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V(s_{t+n})
    and update V(s_t) with this n-step return.

    Parameters
    ----------
    n : look-ahead steps.  n=1 ≡ TD(0), n=∞ ≡ MC.

    Returns
    -------
    V : state-value estimates
    """
    env.seed(seed)
    V = [0.0] * env.nS

    for _ in range(num_episodes):
        s      = env.reset()
        states  = [s]
        rewards = []
        dones   = []
        done    = False

        # Roll out the full episode first (forward view needs the future)
        while not done:
            a = policy(s) if callable(policy) else policy[s]
            s2, r, done, _ = env.step(a)
            states.append(s2)
            rewards.append(r)
            dones.append(done)
            s = s2

        T = len(rewards)

        # Update V(s_t) using n-step return for each t
        for t in range(T):
            G = 0.0
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                G += (gamma ** k) * rewards[idx]
                if dones[idx]:
                    break
            else:
                # Bootstrap from V(s_{t+n}) if episode not yet done
                idx_next = t + n
                if idx_next < len(states) and not dones[min(t + n - 1, T - 1)]:
                    G += (gamma ** n) * V[states[idx_next]]

            V[states[t]] += alpha * (G - V[states[t]])

    return V


# ------------------------------------------------------------------ #
#  3. TD(λ) Forward View                                                #
# ------------------------------------------------------------------ #

def td_lambda_forward(
    env: TabularEnv,
    policy,
    num_episodes: int = 2000,
    gamma: float = 0.95,
    alpha: float = 0.05,
    lam: float = 0.8,
    seed: int = 0,
) -> List[float]:
    """
    TD(λ) prediction using the FORWARD VIEW (λ-return).

    For each episode, compute all n-step returns G_t^(n) and combine:
        G_t^λ = (1 - λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t^(n) + λ^{T-t-1} G_t^{MC}

    This is offline (needs complete episodes) but exactly realises
    the λ-weighted average of n-step returns.

    Returns
    -------
    V : state-value estimates
    """
    env.seed(seed)
    V = [0.0] * env.nS

    for _ in range(num_episodes):
        s      = env.reset()
        states  = [s]
        rewards = []
        done    = False

        while not done:
            a = policy(s) if callable(policy) else policy[s]
            s2, r, done, _ = env.step(a)
            states.append(s2)
            rewards.append(r)
            s = s2

        T = len(rewards)

        for t in range(T):
            # Compute λ-return G_t^λ
            G_lambda   = 0.0
            lambda_pow = 1.0

            for n in range(1, T - t + 1):
                # n-step return from t
                G_n = sum((gamma ** k) * rewards[t + k] for k in range(n))
                if t + n < len(states):
                    G_n += (gamma ** n) * V[states[t + n]]

                if n < T - t:
                    # intermediate term: (1-λ) λ^{n-1} G_t^(n)
                    G_lambda   += (1.0 - lam) * lambda_pow * G_n
                    lambda_pow *= lam
                else:
                    # terminal term: λ^{T-t-1} G_t^{MC}
                    G_lambda += lambda_pow * G_n

            V[states[t]] += alpha * (G_lambda - V[states[t]])

    return V


# ------------------------------------------------------------------ #
#  4. TD(λ) Backward View (Eligibility Traces, online)                  #
# ------------------------------------------------------------------ #

def td_lambda_backward(
    env: TabularEnv,
    policy,
    num_episodes: int = 2000,
    gamma: float = 0.95,
    alpha: float = 0.05,
    lam: float = 0.8,
    trace_type: str = "accumulating",   # "accumulating" or "replacing"
    seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    TD(λ) prediction using the BACKWARD VIEW (eligibility traces).

    At every step:
        δ   = r + γ V(s') − V(s)           # TD error
        e(s) ← γλ e(s) + 1                # accumulating traces
             OR  e(s) ← 1  for s == s_t   # replacing traces
        V(s) ← V(s) + α δ e(s)  ∀ s      # credit assignment

    Parameters
    ----------
    trace_type : "accumulating" (standard) or "replacing" (clip at 1)

    Returns
    -------
    V  : state-value estimates
    td_errors: list of |δ| per step
    """
    env.seed(seed)
    V  = [0.0] * env.nS
    td_errors = []

    for _ in range(num_episodes):
        s    = env.reset()
        e    = [0.0] * env.nS     # eligibility traces, reset each episode
        done = False

        while not done:
            a               = policy(s) if callable(policy) else policy[s]
            s2, r, done, _  = env.step(a)
            v_next          = 0.0 if done else V[s2]
            delta           = r + gamma * v_next - V[s]
            td_errors.append(abs(delta))

            # Update eligibility trace for visited state
            for ss in range(env.nS):
                e[ss] *= gamma * lam

            if trace_type == "replacing":
                e[s] = 1.0
            else:                             # accumulating
                e[s] += 1.0

            # Update ALL states weighted by their trace
            for ss in range(env.nS):
                V[ss] += alpha * delta * e[ss]

            s = s2

    return V, td_errors
