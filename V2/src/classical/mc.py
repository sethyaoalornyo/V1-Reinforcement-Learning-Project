"""
Monte Carlo Methods
===================
Model-free prediction and control using complete episode returns.
MC methods only work with episodic tasks (the DDoS MDP has a Terminal
absorbing state, so episodes naturally end).

Algorithms implemented
----------------------
1. first_visit_mc_prediction  : estimate V(s) under a fixed policy
2. every_visit_mc_prediction  : same but counts every visit to s
3. mc_control_epsilon_greedy  : on-policy first-visit MC control (ε-greedy)
4. mc_control_off_policy_is   : off-policy MC control (importance sampling)

Key ideas
---------
* Return G_t = r_{t+1} + γ r_{t+2} + … + γ^{T-1} r_T
* First-visit: only the FIRST occurrence of s in an episode counts.
* Every-visit: every occurrence counts (more data, slight bias).
* ε-greedy policy: with prob ε choose random action (explore),
  otherwise greedy (exploit).
* Off-policy IS: use behaviour policy b to collect data, then correct
  with importance weights ρ = π(a|s) / b(a|s) to estimate π's value.
"""

from __future__ import annotations
import random
from typing import List, Optional, Tuple

from src.utils.env_wrapper import TabularEnv


# ------------------------------------------------------------------ #
#  Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _generate_episode(
    env: TabularEnv,
    policy,          # callable(state) → action  OR  list[int]
    max_steps: int = 1000,
    start_state: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """
    Roll out one episode using `policy`.

    Returns
    -------
    trajectory: list of (state, action, reward)
    """
    s = env.reset(start_state)
    trajectory = []
    for _ in range(max_steps):
        a = policy(s) if callable(policy) else policy[s]
        s2, r, done, _ = env.step(a)
        trajectory.append((s, a, r))
        s = s2
        if done:
            break
    return trajectory


def _epsilon_greedy(Q: List[List[float]], s: int, epsilon: float, rng: random.Random) -> int:
    """ε-greedy action selection."""
    if rng.random() < epsilon:
        return rng.randint(0, len(Q[s]) - 1)
    return max(range(len(Q[s])), key=lambda a: Q[s][a])


def _compute_returns(
    rewards: List[float], gamma: float, start: int
) -> List[float]:
    """Compute discounted returns G_t for each timestep t >= start."""
    G = [0.0] * len(rewards)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        G[t] = g
    return G


# ------------------------------------------------------------------ #
#  1. First-Visit MC Prediction                                         #
# ------------------------------------------------------------------ #

def first_visit_mc_prediction(
    env: TabularEnv,
    policy,                   # list[int] or callable
    num_episodes: int = 5000,
    gamma: float = 0.95,
    seed: int = 0,
) -> List[float]:
    """
    Estimate V(s) for a fixed policy using first-visit MC.

    For each episode, for each state visited, add G at the FIRST
    visit to a running average.

    Returns
    -------
    V : state-value estimates, shape (nS,)
    """
    rng = random.Random(seed)
    env.seed(seed)

    returns_sum   = [0.0] * env.nS
    returns_count = [0]   * env.nS

    for _ in range(num_episodes):
        traj    = _generate_episode(env, policy)
        states  = [t[0] for t in traj]
        rewards = [t[2] for t in traj]
        G       = _compute_returns(rewards, gamma, 0)

        visited = set()
        for t, s in enumerate(states):
            if s not in visited:
                visited.add(s)
                returns_sum[s]   += G[t]
                returns_count[s] += 1

    V = [
        returns_sum[s] / returns_count[s] if returns_count[s] > 0 else 0.0
        for s in range(env.nS)
    ]
    return V


# ------------------------------------------------------------------ #
#  2. Every-Visit MC Prediction                                         #
# ------------------------------------------------------------------ #

def every_visit_mc_prediction(
    env: TabularEnv,
    policy,
    num_episodes: int = 5000,
    gamma: float = 0.95,
    seed: int = 0,
) -> List[float]:
    """
    Estimate V(s) using every-visit MC.  Every occurrence of s in the
    episode contributes its return to the average (more data than
    first-visit, slight positive bias for continuing tasks).

    Returns
    -------
    V : state-value estimates, shape (nS,)
    """
    env.seed(seed)
    returns_sum   = [0.0] * env.nS
    returns_count = [0]   * env.nS

    for _ in range(num_episodes):
        traj    = _generate_episode(env, policy)
        states  = [t[0] for t in traj]
        rewards = [t[2] for t in traj]
        G       = _compute_returns(rewards, gamma, 0)

        for t, s in enumerate(states):
            returns_sum[s]   += G[t]
            returns_count[s] += 1

    V = [
        returns_sum[s] / returns_count[s] if returns_count[s] > 0 else 0.0
        for s in range(env.nS)
    ]
    return V


# ------------------------------------------------------------------ #
#  3. On-Policy First-Visit MC Control (ε-greedy)                       #
# ------------------------------------------------------------------ #

def mc_control_epsilon_greedy(
    env: TabularEnv,
    num_episodes: int = 5000,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    epsilon_decay: float = 1.0,    # set < 1 to anneal ε over episodes
    epsilon_min: float = 0.01,
    seed: int = 0,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    On-policy first-visit Monte Carlo control.

    Improves an ε-greedy policy by accumulating first-visit returns
    for Q(s, a) and greedy-improving at the end of each episode.

    Parameters
    ----------
    epsilon_decay : multiply ε by this factor after each episode.

    Returns
    -------
    Q        : action-value estimates, shape (nS, nA)
    policy   : greedy policy derived from Q
    returns  : list of episode returns (for plotting)
    """
    rng = random.Random(seed)
    env.seed(seed)

    Q      = [[0.0] * env.nA for _ in range(env.nS)]
    counts = [[0]   * env.nA for _ in range(env.nS)]
    eps    = epsilon
    episode_returns = []

    def behaviour(s):
        return _epsilon_greedy(Q, s, eps, rng)

    for ep in range(num_episodes):
        traj    = _generate_episode(env, behaviour)
        rewards = [t[2] for t in traj]
        G_list  = _compute_returns(rewards, gamma, 0)
        ep_return = G_list[0] if G_list else 0.0
        episode_returns.append(ep_return)

        visited_sa = set()
        for t, (s, a, _) in enumerate(traj):
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                counts[s][a] += 1
                # Incremental mean update
                Q[s][a] += (G_list[t] - Q[s][a]) / counts[s][a]

        # Anneal exploration
        eps = max(epsilon_min, eps * epsilon_decay)

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, episode_returns


# ------------------------------------------------------------------ #
#  4. Off-Policy MC Control (Weighted Importance Sampling)              #
# ------------------------------------------------------------------ #

def mc_control_off_policy_is(
    env: TabularEnv,
    num_episodes: int = 10_000,
    gamma: float = 0.95,
    epsilon_b: float = 0.3,   # behaviour policy ε (must be > 0)
    seed: int = 0,
) -> Tuple[List[List[float]], List[int]]:
    """
    Off-policy MC control using weighted importance sampling.

    The behaviour policy b is soft (ε-greedy), the target policy π
    is deterministic greedy.  Weighted IS gives lower variance than
    ordinary IS but is biased for finite samples.

    The weight for a trajectory (a_t, …, a_T) is:
        ρ = ∏_{t} π(a_t|s_t) / b(a_t|s_t)
    where π(a|s)=1 if a is greedy, else 0.

    Returns
    -------
    Q      : action-value estimates, shape (nS, nA)
    policy : derived greedy policy
    """
    rng = random.Random(seed)
    env.seed(seed)

    Q = [[0.0] * env.nA for _ in range(env.nS)]
    C = [[0.0] * env.nA for _ in range(env.nS)]   # cumulative weight sums

    def behaviour(s):
        return _epsilon_greedy(Q, s, epsilon_b, rng)

    def greedy(s):
        return max(range(env.nA), key=lambda a: Q[s][a])

    for _ in range(num_episodes):
        traj    = _generate_episode(env, behaviour)
        G       = 0.0
        W       = 1.0

        # Backward update
        for t in reversed(range(len(traj))):
            s, a, r = traj[t]
            G = gamma * G + r

            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            # IS weight update: if a ≠ greedy(s), ρ = 0 → stop
            if a != greedy(s):
                break

            # b(a|s) = ε/nA + (1-ε) if greedy, else ε/nA
            b_prob = epsilon_b / env.nA + (1.0 - epsilon_b)
            W /= b_prob

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy
