"""
tests/test_v3.py
----------------
Test suite covering MDP, classical DP, DQN, and Bayesian tuning.
Run with: python -m pytest tests/test_v3.py -v
"""
from __future__ import annotations
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdp.ddos_mdp import build_ddos_mdp, save_mdp, load_mdp
from src.classical.dp import policy_iteration, value_iteration, q_from_v
from src.utils.env_wrapper import TabularEnv
from src.utils.replay_buffer import ReplayBuffer


# ── MDP tests ────────────────────────────────────────────────────────────────

class TestMDP:
    def test_build_returns_correct_dimensions(self):
        mdp = build_ddos_mdp()
        assert mdp.nS == 5
        assert mdp.nA == 3

    def test_transition_probs_sum_to_one(self):
        mdp = build_ddos_mdp()
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                total = sum(p for p, *_ in mdp.transitions(s, a))
                assert abs(total - 1.0) < 1e-9, f"Prob sum ≠ 1 for s={s}, a={a}"

    def test_terminal_state_absorbing(self):
        mdp = build_ddos_mdp()
        terminal = mdp.nS - 1
        for a in range(mdp.nA):
            for prob, s2, r, done in mdp.transitions(terminal, a):
                assert done is True
                assert s2 == terminal

    def test_save_and_load_roundtrip(self, tmp_path):
        mdp  = build_ddos_mdp()
        path = str(tmp_path / "mdp.json")
        save_mdp(mdp, path)
        mdp2 = load_mdp(path)
        assert mdp.nS == mdp2.nS
        assert mdp.nA == mdp2.nA
        assert mdp.gamma == mdp2.gamma


# ── DP tests ─────────────────────────────────────────────────────────────────

class TestDP:
    def test_policy_iteration_shape(self):
        mdp = build_ddos_mdp()
        policy, V, Q = policy_iteration(mdp)
        assert len(policy) == mdp.nS
        assert len(V) == mdp.nS
        assert len(Q) == mdp.nS
        assert all(len(q) == mdp.nA for q in Q)

    def test_value_iteration_shape(self):
        mdp = build_ddos_mdp()
        policy, V, Q = value_iteration(mdp)
        assert len(policy) == mdp.nS
        assert len(V) == mdp.nS

    def test_both_methods_agree_on_policy(self):
        mdp = build_ddos_mdp()
        p_pi, V_pi, _ = policy_iteration(mdp)
        p_vi, V_vi, _ = value_iteration(mdp)
        assert p_pi == p_vi, "Policy iteration and value iteration disagree on optimal policy"

    def test_terminal_state_value_zero(self):
        mdp = build_ddos_mdp()
        _, V, _ = value_iteration(mdp)
        assert abs(V[mdp.nS - 1]) < 1e-6, "Terminal state should have V=0"

    def test_q_from_v_consistency(self):
        mdp = build_ddos_mdp()
        _, V, Q_direct = policy_iteration(mdp)
        Q_computed = q_from_v(mdp, V)
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                assert abs(Q_direct[s][a] - Q_computed[s][a]) < 1e-6


# ── environment wrapper tests ─────────────────────────────────────────────────

class TestEnv:
    def test_reset_returns_start_state(self):
        mdp = build_ddos_mdp()
        env = TabularEnv(mdp, seed=0)
        s = env.reset(start_state=2)
        assert s == 2

    def test_step_returns_valid_state(self):
        mdp = build_ddos_mdp()
        env = TabularEnv(mdp, seed=0)
        env.reset()
        s2, r, done = env.step(0)
        assert 0 <= s2 < mdp.nS
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_terminal_action_ends_episode(self):
        mdp = build_ddos_mdp()
        env = TabularEnv(mdp, seed=0)
        terminal = mdp.nS - 1
        env.reset(start_state=terminal)
        _, _, done = env.step(0)
        assert done is True


# ── replay buffer tests ───────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100, seed=0)
        for i in range(10):
            buf.push(i % 5, i % 3, float(i), (i + 1) % 5, False)
        assert len(buf) == 10
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_capacity_enforced(self):
        buf = ReplayBuffer(capacity=5, seed=0)
        for i in range(20):
            buf.push(0, 0, 0.0, 0, False)
        assert len(buf) == 5

    def test_save_and_load(self, tmp_path):
        buf  = ReplayBuffer(capacity=50, seed=0)
        buf.push(1, 2, 3.0, 4, False)
        path = str(tmp_path / "buffer.json")
        buf.save(path)
        buf2 = ReplayBuffer(capacity=50)
        buf2.load(path)
        assert len(buf2) == 1


# ── DQN smoke test ────────────────────────────────────────────────────────────

class TestDQN:
    def test_dqn_trains_without_error(self):
        import torch
        from src.deep.dqn import DQNAgent

        mdp   = build_ddos_mdp()
        env   = TabularEnv(mdp, seed=0)
        agent = DQNAgent(nS=mdp.nS, nA=mdp.nA, hidden_dim=32, seed=0)

        s = env.reset()
        for _ in range(200):
            s_vec = torch.zeros(mdp.nS); s_vec[s] = 1.0
            a     = agent.select_action_onehot(s_vec)
            s2, r, done = env.step(a)
            agent.push(s, a, r, s2, done)
            agent.learn(mdp.nS)
            s = env.reset() if done else s2

    def test_dueling_dqn_trains_without_error(self):
        import torch
        from src.deep.dqn import DQNAgent

        mdp   = build_ddos_mdp()
        env   = TabularEnv(mdp, seed=0)
        agent = DQNAgent(nS=mdp.nS, nA=mdp.nA, hidden_dim=32, dueling=True, seed=0)

        s = env.reset()
        for _ in range(200):
            s_vec = torch.zeros(mdp.nS); s_vec[s] = 1.0
            a     = agent.select_action_onehot(s_vec)
            s2, r, done = env.step(a)
            agent.push(s, a, r, s2, done)
            agent.learn(mdp.nS)
            s = env.reset() if done else s2


# ── Bayesian tuning smoke test ────────────────────────────────────────────────

class TestBayesianTuning:
    def test_study_runs_and_returns_params(self, tmp_path):
        from src.tuning.bayesian_tuning import run_study
        best = run_study(
            n_trials=3,
            n_episodes=20,
            out_dir=str(tmp_path),
            verbose=False,
        )
        assert "lr" in best
        assert "hidden_dim" in best
        assert "best_value" in best
