"""
Q-Learning — Off-Policy TD Control
=====================================
Q-learning is the canonical off-policy TD control algorithm.  Unlike
SARSA it bootstraps from the GREEDY (max) action rather than the action
actually taken, so its Q-table converges to Q* regardless of the
exploration policy used.

Algorithms
----------
1. q_learning        : Standard Q-learning (Watkins, 1989)
2. double_q_learning : Double Q-learning (van Hasselt, 2010)
                       Uses two Q-tables to decouple action selection
                       from action evaluation, reducing maximisation bias.

Update rules
------------
Q-learning:
    Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

Double Q-learning (each step randomly uses Table A or B):
    Q_A(s,a) ← Q_A(s,a) + α [r + γ Q_B(s', argmax_{a'} Q_A(s',a')) − Q_A(s,a)]
    (or swap A and B)
"""

from __future__ import annotations
import random
from typing import List, Tuple

from src.utils.env_wrapper import TabularEnv


# ------------------------------------------------------------------ #
#  Helper                                                               #
# ------------------------------------------------------------------ #

def _eps_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randint(0, len(Q[s]) - 1)
    return max(range(len(Q[s])), key=lambda a: Q[s][a])


# ------------------------------------------------------------------ #
#  1. Q-Learning                                                        #
# ------------------------------------------------------------------ #

def q_learning(
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
    Standard Q-learning.

    The key difference from SARSA: the TD target uses max_{a'} Q(s', a')
    — the best possible next action — NOT the action that was actually
    chosen.  This makes Q-learning off-policy.

    Parameters
    ----------
    epsilon_decay : multiply ε by this after every episode (1.0 = no decay).

    Returns
    -------
    Q        : action-value table, shape (nS, nA)
    policy   : greedy policy derived from Q
    ep_returns: episode returns (for plotting)
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q   = [[0.0] * env.nA for _ in range(env.nS)]
    eps = epsilon
    ep_returns = []

    for _ in range(num_episodes):
        s    = env.reset()
        done = False
        G    = 0.0
        step = 0

        while not done:
            a               = _eps_greedy(Q, s, eps, rng)
            s2, r, done, _  = env.step(a)
            G              += (gamma ** step) * r
            step           += 1

            # OFF-POLICY target: max over next Q values
            q_next = max(Q[s2]) if not done else 0.0
            target = r + gamma * q_next
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2

        ep_returns.append(G)
        eps = max(epsilon_min, eps * epsilon_decay)

    policy = [max(range(env.nA), key=lambda a: Q[s][a]) for s in range(env.nS)]
    return Q, policy, ep_returns


# ------------------------------------------------------------------ #
#  2. Double Q-Learning                                                 #
# ------------------------------------------------------------------ #

def double_q_learning(
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
    Double Q-Learning — reduces maximisation bias.

    Maintains two independent Q-tables (Q_A, Q_B).  At each step,
    one table is chosen at random (50/50) to select the best action,
    and the OTHER table evaluates it.  This prevents the agent from
    being over-optimistic about noisy Q estimates.

    Action selection uses the AVERAGE of Q_A and Q_B (ε-greedy).

    Returns
    -------
    Q_avg    : (Q_A + Q_B) / 2 for each (s, a)
    policy   : greedy policy derived from Q_avg
    ep_returns
    """
    rng = random.Random(seed)
    env.seed(seed)
    Q_A = [[0.0] * env.nA for _ in range(env.nS)]
    Q_B = [[0.0] * env.nA for _ in range(env.nS)]
    eps = epsilon
    ep_returns = []

    for _ in range(num_episodes):
        s    = env.reset()
        done = False
        G    = 0.0
        step = 0

        while not done:
            # Action selection: ε-greedy on Q_A + Q_B
            Q_avg_s = [Q_A[s][a] + Q_B[s][a] for a in range(env.nA)]
            if rng.random() < eps:
                a = rng.randint(0, env.nA - 1)
            else:
                a = max(range(env.nA), key=lambda aa: Q_avg_s[aa])

            s2, r, done, _ = env.step(a)
            G += (gamma ** step) * r
            step += 1

            # Randomly update one of the two tables
            if rng.random() < 0.5:
                # Update Q_A: select action via Q_A, evaluate via Q_B
                a_star = max(range(env.nA), key=lambda aa: Q_A[s2][aa]) if not done else 0
                target = r + (gamma * Q_B[s2][a_star] if not done else 0.0)
                Q_A[s][a] += alpha * (target - Q_A[s][a])
            else:
                # Update Q_B: select action via Q_B, evaluate via Q_A
                a_star = max(range(env.nA), key=lambda aa: Q_B[s2][aa]) if not done else 0
                target = r + (gamma * Q_A[s2][a_star] if not done else 0.0)
                Q_B[s][a] += alpha * (target - Q_B[s][a])

            s = s2

        ep_returns.append(G)
        eps = max(epsilon_min, eps * epsilon_decay)

    # Average both tables for the final Q estimate
    Q_final = [
        [(Q_A[s][a] + Q_B[s][a]) / 2.0 for a in range(env.nA)]
        for s in range(env.nS)
    ]
    policy = [max(range(env.nA), key=lambda a: Q_final[s][a]) for s in range(env.nS)]
    return Q_final, policy, ep_returns
