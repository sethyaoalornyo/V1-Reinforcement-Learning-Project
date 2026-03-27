"""
Dynamic Programming (V1 carry-over)
=====================================
Policy Iteration and Value Iteration for the tabular MDP.
These assume full knowledge of the transition model P.

Functions
---------
policy_evaluation      : evaluate a fixed policy → V
q_from_v               : convert V → Q
greedy_policy_from_q   : argmax over Q → deterministic policy
policy_improvement     : one-step greedy improvement of a policy
policy_iteration       : PI loop until convergence
value_iteration        : VI loop until convergence
"""

from __future__ import annotations
import math
from typing import List, Tuple

from src.mdp.ddos_mdp import TabularMDP, State, Action


# ------------------------------------------------------------------ #
#  Helpers                                                              #
# ------------------------------------------------------------------ #

def q_from_v(mdp: TabularMDP, V: List[float]) -> List[List[float]]:
    """Compute Q(s, a) from V using the Bellman expectation equation."""
    Q = [[0.0] * mdp.nA for _ in range(mdp.nS)]
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            q_sa = 0.0
            for prob, s2, r, done in mdp.transitions(s, a):
                v_next = 0.0 if done else V[s2]
                q_sa += prob * (r + mdp.gamma * v_next)
            Q[s][a] = q_sa
    return Q


def greedy_policy_from_q(Q: List[List[float]]) -> List[Action]:
    """Return the deterministic greedy policy: π(s) = argmax_a Q(s, a)."""
    return [max(range(len(Q[s])), key=lambda a: Q[s][a]) for s in range(len(Q))]


# ------------------------------------------------------------------ #
#  Policy Evaluation                                                    #
# ------------------------------------------------------------------ #

def policy_evaluation(
    mdp: TabularMDP,
    policy: List[Action],
    theta: float = 1e-8,
    max_iters: int = 100_000,
) -> List[float]:
    """
    Iterative policy evaluation (Bellman expectation sweeps).

    Parameters
    ----------
    mdp     : the MDP
    policy  : list mapping state → action
    theta   : convergence tolerance (max |V_new - V_old| < theta)
    max_iters: safety cap on iterations

    Returns
    -------
    V : state-value function under `policy`
    """
    V = [0.0] * mdp.nS
    for _ in range(max_iters):
        delta = 0.0
        for s in range(mdp.nS):
            a = policy[s]
            v_old = V[s]
            v_new = sum(
                prob * (r + mdp.gamma * (0.0 if done else V[s2]))
                for prob, s2, r, done in mdp.transitions(s, a)
            )
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        if delta < theta:
            break
    return V


# ------------------------------------------------------------------ #
#  Policy Iteration                                                     #
# ------------------------------------------------------------------ #

def policy_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_policy_iters: int = 10_000,
) -> Tuple[List[Action], List[float], List[List[float]]]:
    """
    Policy Iteration: evaluate → improve → repeat until stable.

    Returns (policy, V, Q)
    """
    policy = [0] * mdp.nS          # start with all-zero policy
    for _ in range(max_policy_iters):
        V          = policy_evaluation(mdp, policy, theta=theta)
        Q          = q_from_v(mdp, V)
        new_policy = greedy_policy_from_q(Q)
        if new_policy == policy:
            return policy, V, Q
        policy = new_policy
    return policy, V, Q


# ------------------------------------------------------------------ #
#  Value Iteration                                                      #
# ------------------------------------------------------------------ #

def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iters: int = 100_000,
) -> Tuple[List[Action], List[float], List[List[float]]]:
    """
    Value Iteration: apply Bellman optimality operator until convergence.

    Returns (policy, V, Q)
    """
    V = [0.0] * mdp.nS
    for _ in range(max_iters):
        delta = 0.0
        for s in range(mdp.nS):
            v_old = V[s]
            best = max(
                sum(prob * (r + mdp.gamma * (0.0 if done else V[s2]))
                    for prob, s2, r, done in mdp.transitions(s, a))
                for a in range(mdp.nA)
            )
            V[s] = best
            delta = max(delta, abs(v_old - best))
        if delta < theta:
            break
    Q      = q_from_v(mdp, V)
    policy = greedy_policy_from_q(Q)
    return policy, V, Q
