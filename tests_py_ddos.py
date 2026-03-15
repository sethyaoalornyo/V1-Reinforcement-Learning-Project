from mdp import build_ddos_style_mdp
from dp import policy_iteration, value_iteration

def test_policy_iteration_runs():
    mdp = build_ddos_style_mdp()
    policy, V, Q = policy_iteration(mdp)
    assert len(policy) == mdp.nS
    assert len(V) == mdp.nS
    assert len(Q) == mdp.nS

def test_value_iteration_runs():
    mdp = build_ddos_style_mdp()
    policy, V, Q = value_iteration(mdp)
    assert len(policy) == mdp.nS
    assert len(V) == mdp.nS
    assert len(Q) == mdp.nS
