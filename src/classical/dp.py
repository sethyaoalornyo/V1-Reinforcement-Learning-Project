from __future__ import annotations
import math
from typing import List, Tuple

from src.mdp.ddos_mdp import TabularMDP, State, Action


# ── helpers ──────────────────────────────────────────────────────────────────

def q_from_v(mdp: TabularMDP, V: List[float]) -> List[List[float]]:
    Q = [[0.0] * mdp.nA for _ in range(mdp.nS)]
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, s2, r, done in mdp.transitions(s, a):
                Q[s][a] += prob * (r + mdp.gamma * (0.0 if done else V[s2]))
    return Q


def greedy_policy(Q: List[List[float]]) -> List[Action]:
    return [max(range(len(Q[s])), key=lambda a: Q[s][a]) for s in range(len(Q))]


# ── policy evaluation ─────────────────────────────────────────────────────────

def policy_evaluation(
    mdp: TabularMDP,
    policy: List[Action],
    theta: float = 1e-8,
    max_iters: int = 100_000,
) -> List[float]:
    V = [0.0] * mdp.nS
    for _ in range(max_iters):
        delta = 0.0
        for s in range(mdp.nS):
            v_old = V[s]
            v_new = 0.0
            for prob, s2, r, done in mdp.transitions(s, policy[s]):
                v_new += prob * (r + mdp.gamma * (0.0 if done else V[s2]))
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        if delta < theta:
            break
    return V


# ── policy iteration ──────────────────────────────────────────────────────────

def policy_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_policy_iters: int = 10_000,
) -> Tuple[List[Action], List[float], List[List[float]]]:
    policy = [0] * mdp.nS
    for _ in range(max_policy_iters):
        V  = policy_evaluation(mdp, policy, theta=theta)
        Q  = q_from_v(mdp, V)
        new_policy = greedy_policy(Q)
        if new_policy == policy:
            return policy, V, Q
        policy = new_policy
    return policy, V, Q


# ── value iteration ───────────────────────────────────────────────────────────

def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iters: int = 100_000,
) -> Tuple[List[Action], List[float], List[List[float]]]:
    V = [0.0] * mdp.nS
    for _ in range(max_iters):
        delta = 0.0
        for s in range(mdp.nS):
            v_old = V[s]
            best  = max(
                sum(prob * (r + mdp.gamma * (0.0 if done else V[s2]))
                    for prob, s2, r, done in mdp.transitions(s, a))
                for a in range(mdp.nA)
            )
            V[s]  = best
            delta = max(delta, abs(v_old - best))
        if delta < theta:
            break
    Q      = q_from_v(mdp, V)
    policy = greedy_policy(Q)
    return policy, V, Q
