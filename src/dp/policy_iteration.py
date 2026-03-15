from __future__ import annotations
from typing import Dict, Tuple
from mdp import GridWorldMDP

State = Tuple[int, int]

def policy_evaluation(mdp: GridWorldMDP, policy: Dict[State, str], theta: float = 1e-8):
    V = {s: 0.0 for s in mdp.states}
    while True:
        delta = 0.0
        for s in mdp.states:
            old_v = V[s]
            a = policy[s]
            s2, r = mdp.step(s, a)
            V[s] = r + mdp.gamma * V[s2]
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    return V

def q_from_v(mdp: GridWorldMDP, V: Dict[State, float]):
    Q = {s: {} for s in mdp.states}
    for s in mdp.states:
        for a in mdp.actions:
            s2, r = mdp.step(s, a)
            Q[s][a] = r + mdp.gamma * V[s2]
    return Q

def policy_improvement_from_v(mdp: GridWorldMDP, V: Dict[State, float]):
    # Greedy improvement using V via one-step lookahead
    policy = {}
    for s in mdp.states:
        best_a = None
        best_val = float("-inf")
        for a in mdp.actions:
            s2, r = mdp.step(s, a)
            val = r + mdp.gamma * V[s2]
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a
    return policy

def policy_improvement_from_q(mdp: GridWorldMDP, Q):
    # Greedy improvement directly from Q
    policy = {}
    for s in mdp.states:
        policy[s] = max(Q[s], key=Q[s].get)
    return policy

def policy_iteration(mdp: GridWorldMDP, theta: float = 1e-8, max_iters: int = 10_000):
    # Initialize arbitrary policy
    policy = {s: mdp.actions[0] for s in mdp.states}

    for _ in range(max_iters):
        V = policy_evaluation(mdp, policy, theta=theta)

        # Improvement on values (V)
        new_policy_v = policy_improvement_from_v(mdp, V)

        # Improvement on Q-values (compute Q from V and greedify)
        Q = q_from_v(mdp, V)
        new_policy_q = policy_improvement_from_q(mdp, Q)

        # If both improvements match and policy is stable, stop
        if new_policy_v == policy and new_policy_q == policy:
            return policy, V, Q

        # Update policy (either is fine; keep Q-greedy for explicitness)
        policy = new_policy_q

    return policy, V, Q
