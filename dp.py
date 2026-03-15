from __future__ import annotations
from typing import List, Tuple
import math

from mdp import TabularMDP, State, Action


def greedy_policy_from_q(Q: List[List[float]]) -> List[Action]:
    policy = []
    for s in range(len(Q)):
        best_a = max(range(len(Q[s])), key=lambda a: Q[s][a])
        policy.append(best_a)
    return policy


def q_from_v(mdp: TabularMDP, V: List[float]) -> List[List[float]]:
    Q = [[0.0 for _ in range(mdp.nA)] for _ in range(mdp.nS)]

    for s in range(mdp.nS):
        for a in range(mdp.nA):
            q_sa = 0.0

            for prob, s2, r, done in mdp.transitions(s, a):
                v_next = 0.0 if done else V[s2]
                q_sa += prob * (r + mdp.gamma * v_next)

            Q[s][a] = q_sa

    return Q


def policy_evaluation(
    mdp: TabularMDP,
    policy: List[Action],
    theta: float = 1e-8,
    max_iters: int = 100000
) -> List[float]:

    V = [0.0 for _ in range(mdp.nS)]

    for _ in range(max_iters):

        delta = 0.0

        for s in range(mdp.nS):

            a = policy[s]
            v_old = V[s]
            v_new = 0.0

            for prob, s2, r, done in mdp.transitions(s, a):

                v_next = 0.0 if done else V[s2]
                v_new += prob * (r + mdp.gamma * v_next)

            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))

        if delta < theta:
            break

    return V


def policy_improvement_from_v(mdp: TabularMDP, V: List[float]) -> List[Action]:

    new_policy = []

    for s in range(mdp.nS):

        best_a = 0
        best_val = -math.inf

        for a in range(mdp.nA):

            q_sa = 0.0

            for prob, s2, r, done in mdp.transitions(s, a):

                v_next = 0.0 if done else V[s2]
                q_sa += prob * (r + mdp.gamma * v_next)

            if q_sa > best_val:
                best_val = q_sa
                best_a = a

        new_policy.append(best_a)

    return new_policy


def policy_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_policy_iters: int = 10000
) -> Tuple[List[Action], List[float], List[List[float]]]:

    policy = [0 for _ in range(mdp.nS)]

    for _ in range(max_policy_iters):

        V = policy_evaluation(mdp, policy, theta=theta)

        Q = q_from_v(mdp, V)

        new_policy = greedy_policy_from_q(Q)

        if new_policy == policy:
            return policy, V, Q

        policy = new_policy

    return policy, V, Q


def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iters: int = 100000
) -> Tuple[List[Action], List[float], List[List[float]]]:

    V = [0.0 for _ in range(mdp.nS)]

    for _ in range(max_iters):

        delta = 0.0

        for s in range(mdp.nS):

            v_old = V[s]
            best = -math.inf

            for a in range(mdp.nA):

                q_sa = 0.0

                for prob, s2, r, done in mdp.transitions(s, a):

                    v_next = 0.0 if done else V[s2]
                    q_sa += prob * (r + mdp.gamma * v_next)

                best = max(best, q_sa)

            V[s] = best
            delta = max(delta, abs(v_old - best))

        if delta < theta:
            break

    Q = q_from_v(mdp, V)
    policy = greedy_policy_from_q(Q)

    return policy, V, Q