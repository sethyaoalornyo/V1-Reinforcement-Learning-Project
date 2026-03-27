"""
SARSA — On-Policy TD Control
==============================
SARSA extends TD prediction to control by learning Q(s, a) instead of
V(s), using the on-policy tuple (S, A, R, S', A') — hence the name.

Algorithms
----------
1. sarsa             : SARSA(0) — standard on-policy TD control
2. sarsa_n           : n-Step SARSA (FORWARD VIEW, n-cutoff variant)
3. sarsa_lambda_fwd  : SARSA(λ) FORWARD VIEW (offline, per-episode)
4. sarsa_lambda_bwd  : SARSA(λ) BACKWARD VIEW (online eligibility traces)

All algorithms use ε-greedy exploration and return (Q, policy, returns).

Key differences from Q-learning
---------------------------------
* SARSA is ON-POLICY: the next action A' is chosen by the SAME ε-greedy
  policy used during training, so Q converges to the ε-greedy policy.
* Q-learning is OFF-POLICY: the next action is the greedy (max) action,
  so Q converges to the optimal policy even under an exploratory policy.
"""

from __future__ import annotations
import random
from typing import List, Tuple

from src.utils.env_wrapper import TabularEnv


# ------------------------------------------------------------------ #
#  Helper                                                               #
# ------------------------------------------------------------------ #

def _eps_greedy(Q: List[List[float]], s: int, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randint(0, len(Q[s]) - 1)
    return max(range(len(Q[s])), key=lambda a: Q[s][a])


# ------------------------------------------------------------------ #
#  1. SARSA(0)                                                          #
# ------------------------------------------------------------------ #

def sarsa(
    env: TabularEnv,
    num_episodes: int = 3000,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    epsilon_decay: float = 1.0,
    epsilon_min: float = 0.01,
    seed: int = 0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    Standard SARSA(0): On-policy one-step TD control.

    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]
    where a' is chosen by the SAME ε-greedy policy.

    Returns
    -------
    Q        : action-value table
    policy   : greedy policy derived from Q
    ep_returns: episode returns
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q   = [[0.0] * env.nA for _ in range(env.nS)]
    eps = epsilon
    ep_returns = []

    for _ in range(num_episodes):
        s    = env.reset()
        a    = _eps_greedy(Q, s, eps, rng)
        done = False
        G    = 0.0
        step = 0

        while not done:
            s2, r, done, _ = env.step(a)
            G += (gamma ** step) * r
            step += 1
            a2 = _eps_greedy(Q, s2, eps, rng) if not done else 0
            # SARSA update: bootstrap from Q(s', a')
            td_target = r + (gamma * Q[s2][a2] if not done else 0.0)
            Q[s][a]  += alpha * (td_target - Q[s][a])
            s, a = s2, a2

        ep_returns.append(G)
        eps = max(epsilon_min, eps * epsilon_decay)

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, ep_returns


# ------------------------------------------------------------------ #
#  2. n-Step SARSA (Forward View, n-cutoff)                             #
# ------------------------------------------------------------------ #

def sarsa_n(
    env: TabularEnv,
    num_episodes: int = 3000,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    n: int = 4,
    seed: int = 0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    n-Step SARSA (forward view).

    Buffer n (s, a, r) transitions, then update Q(s_t, a_t) using:
        G = r_{t+1} + … + γ^{n-1} r_{t+n} + γ^n Q(s_{t+n}, a_{t+n})

    n=1 recovers SARSA(0); n→∞ recovers MC control.
    The n-CUTOFF is enforced naturally — we look exactly n steps ahead.

    Returns
    -------
    Q, policy, ep_returns
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q = [[0.0] * env.nA for _ in range(env.nS)]
    ep_returns = []

    for _ in range(num_episodes):
        s = env.reset()
        # Pre-roll the episode
        buf_s, buf_a, buf_r = [s], [], []
        done = False
        while not done:
            a = _eps_greedy(Q, s, epsilon, rng)
            s2, r, done, _ = env.step(a)
            buf_s.append(s2)
            buf_a.append(a)
            buf_r.append(r)
            s = s2

        T = len(buf_r)
        G_ep = sum((gamma ** t) * buf_r[t] for t in range(T))
        ep_returns.append(G_ep)

        for t in range(T):
            # n-step return from t
            G = 0.0
            for k in range(n):
                if t + k >= T:
                    break
                G += (gamma ** k) * buf_r[t + k]
            else:
                s_next = t + n
                if s_next < len(buf_s) and not (t + n - 1 >= T):
                    a_next = _eps_greedy(Q, buf_s[s_next], epsilon, rng)
                    G     += (gamma ** n) * Q[buf_s[s_next]][a_next]

            Q[buf_s[t]][buf_a[t]] += alpha * (G - Q[buf_s[t]][buf_a[t]])

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, ep_returns


# ------------------------------------------------------------------ #
#  3. SARSA(λ) Forward View                                             #
# ------------------------------------------------------------------ #

def sarsa_lambda_fwd(
    env: TabularEnv,
    num_episodes: int = 3000,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lam: float = 0.8,
    seed: int = 0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    SARSA(λ) FORWARD VIEW — offline λ-return version.

    For each (s_t, a_t) pair, compute the λ-weighted average of all
    n-step SARSA returns and use that as the TD target.

    This is episodic (needs complete trajectories) and is the
    conceptually clean definition of SARSA(λ).

    Returns
    -------
    Q, policy, ep_returns
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q = [[0.0] * env.nA for _ in range(env.nS)]
    ep_returns = []

    for _ in range(num_episodes):
        s = env.reset()
        buf_s, buf_a, buf_r = [s], [], []
        done = False
        while not done:
            a = _eps_greedy(Q, s, epsilon, rng)
            s2, r, done, _ = env.step(a)
            buf_s.append(s2)
            buf_a.append(a)
            buf_r.append(r)
            s = s2

        T = len(buf_r)
        ep_returns.append(sum((gamma ** t) * buf_r[t] for t in range(T)))

        for t in range(T):
            G_lambda   = 0.0
            lambda_pow = 1.0

            for step_n in range(1, T - t + 1):
                # n-step SARSA return from t
                G_n = sum((gamma ** k) * buf_r[t + k] for k in range(step_n))
                s_n = t + step_n
                if s_n < len(buf_s):
                    a_n  = _eps_greedy(Q, buf_s[s_n], epsilon, rng)
                    G_n += (gamma ** step_n) * Q[buf_s[s_n]][a_n]

                if step_n < T - t:
                    G_lambda   += (1.0 - lam) * lambda_pow * G_n
                    lambda_pow *= lam
                else:
                    G_lambda += lambda_pow * G_n

            Q[buf_s[t]][buf_a[t]] += alpha * (G_lambda - Q[buf_s[t]][buf_a[t]])

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, ep_returns


# ------------------------------------------------------------------ #
#  4. SARSA(λ) Backward View (Eligibility Traces, online)               #
# ------------------------------------------------------------------ #

def sarsa_lambda_bwd(
    env: TabularEnv,
    num_episodes: int = 3000,
    gamma: float = 0.95,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    lam: float = 0.8,
    trace_type: str = "accumulating",   # "accumulating" or "replacing"
    epsilon_decay: float = 1.0,
    epsilon_min: float = 0.01,
    seed: int = 0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    SARSA(λ) BACKWARD VIEW — online eligibility trace version.

    At every step (s, a, r, s', a'):
        δ       = r + γ Q(s', a') − Q(s, a)
        e(s, a) ← γλ e(s, a) + 1              # accumulating
        Q(s, a) ← Q(s, a) + α δ e(s, a)  ∀(s,a)

    This is the ONLINE equivalent of the forward view and is much more
    efficient computationally — it runs in O(nS × nA) per step.

    Parameters
    ----------
    trace_type : "accumulating" or "replacing" (e(s,a) ← 1 if visited)

    Returns
    -------
    Q, policy, ep_returns
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q   = [[0.0] * env.nA for _ in range(env.nS)]
    eps = epsilon
    ep_returns = []

    for _ in range(num_episodes):
        s    = env.reset()
        a    = _eps_greedy(Q, s, eps, rng)
        e    = [[0.0] * env.nA for _ in range(env.nS)]  # traces per episode
        done = False
        G    = 0.0
        step = 0

        while not done:
            s2, r, done, _ = env.step(a)
            G += (gamma ** step) * r
            step += 1
            a2 = _eps_greedy(Q, s2, eps, rng) if not done else 0

            # TD error
            q_next = Q[s2][a2] if not done else 0.0
            delta  = r + gamma * q_next - Q[s][a]

            # Decay all traces, then update visited (s, a)
            for ss in range(env.nS):
                for aa in range(env.nA):
                    e[ss][aa] *= gamma * lam

            if trace_type == "replacing":
                e[s][a] = 1.0
            else:
                e[s][a] += 1.0

            # Credit assignment to ALL (s, a) via trace
            for ss in range(env.nS):
                for aa in range(env.nA):
                    Q[ss][aa] += alpha * delta * e[ss][aa]

            s, a = s2, a2

        ep_returns.append(G)
        eps = max(epsilon_min, eps * epsilon_decay)

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, ep_returns
