"""
tests/test_v2.py
=================
Unit and smoke tests for all V2 implementations.
Run with:   python -m pytest tests/test_v2.py -v
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.mdp.ddos_mdp import build_ddos_mdp, TabularMDP
from src.utils.env_wrapper import TabularEnv, make_ddos_env
from src.utils.replay_buffer import ReplayBuffer
from src.classical.dp import policy_iteration, value_iteration
from src.classical.mc import (
    first_visit_mc_prediction, every_visit_mc_prediction,
    mc_control_epsilon_greedy, mc_control_off_policy_is,
)
from src.classical.td import (
    td0_prediction, td_n_prediction, td_lambda_forward, td_lambda_backward,
)
from src.classical.sarsa import sarsa, sarsa_n, sarsa_lambda_fwd, sarsa_lambda_bwd
from src.classical.qlearning import q_learning, double_q_learning


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def mdp():
    return build_ddos_mdp(gamma=0.95)

@pytest.fixture
def env(mdp):
    return TabularEnv(mdp, start_state=0, max_steps=100, seed=0)

@pytest.fixture
def opt_policy(mdp):
    policy, _, _ = policy_iteration(mdp)
    return policy


# ================================================================== #
#  MDP / Environment tests                                             #
# ================================================================== #

class TestMDP:
    def test_build(self, mdp):
        assert mdp.nS == 5
        assert mdp.nA == 3

    def test_transition_probabilities_sum_to_one(self, mdp):
        for s in range(mdp.nS):
            for a in range(mdp.nA):
                total = sum(p for p, _, _, _ in mdp.transitions(s, a))
                assert abs(total - 1.0) < 1e-6, f"P[{s}][{a}] sums to {total}"

    def test_terminal_state(self, mdp):
        TERMINAL = 4
        for a in range(mdp.nA):
            for prob, s2, r, done in mdp.transitions(TERMINAL, a):
                assert done, "Terminal state must always return done=True"
                assert s2 == TERMINAL

    def test_env_reset(self, env):
        s = env.reset()
        assert 0 <= s < env.nS

    def test_env_step(self, env):
        env.reset()
        s2, r, done, info = env.step(0)
        assert 0 <= s2 < env.nS
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_env_max_steps(self, mdp):
        env = TabularEnv(mdp, max_steps=5, seed=1)
        env.reset()
        for _ in range(10):
            _, _, done, _ = env.step(0)
            if done:
                break
        assert done, "Episode should terminate by max_steps"

    def test_save_load_mdp(self, mdp, tmp_path):
        path = str(tmp_path / "mdp.json")
        mdp.save(path)
        loaded = TabularMDP.load(path)
        assert loaded.nS == mdp.nS
        assert loaded.nA == mdp.nA


# ================================================================== #
#  Replay Buffer tests                                                  #
# ================================================================== #

class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(i % 5, 0, float(i), (i + 1) % 5, False)
        assert len(buf) == 10

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(0, 0, 0.0, 0, False)
        assert len(buf) == 5

    def test_sample(self):
        buf = ReplayBuffer(capacity=100, seed=0)
        for i in range(50):
            buf.push(i % 5, i % 3, float(i), (i + 1) % 5, i % 7 == 0)
        batch = buf.sample(16)
        assert len(batch) == 16

    def test_sample_raises_if_too_small(self):
        buf = ReplayBuffer(capacity=100)
        buf.push(0, 0, 0.0, 0, False)
        with pytest.raises(ValueError):
            buf.sample(32)

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(30):
            buf.push(0, 0, 0.0, 0, False)
        assert buf.is_ready(30)
        assert not buf.is_ready(31)

    def test_save_load(self, tmp_path):
        buf = ReplayBuffer(capacity=100, seed=7)
        for i in range(20):
            buf.push(i % 5, i % 3, float(i), (i + 1) % 5, False)
        path = str(tmp_path / "buf.json")
        buf.save(path)
        buf2 = ReplayBuffer(capacity=100)
        buf2.load(path)
        assert len(buf2) == 20

    def test_stats(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(0, 0, float(i), 0, False)
        stats = buf.stats()
        assert "size" in stats and "fill_pct" in stats


# ================================================================== #
#  Dynamic Programming                                                  #
# ================================================================== #

class TestDP:
    def test_policy_iteration_shape(self, mdp):
        policy, V, Q = policy_iteration(mdp)
        assert len(policy) == mdp.nS
        assert len(V) == mdp.nS
        assert len(Q) == mdp.nS

    def test_value_iteration_shape(self, mdp):
        policy, V, Q = value_iteration(mdp)
        assert len(policy) == mdp.nS
        assert len(V) == mdp.nS

    def test_pi_vi_agree(self, mdp):
        pi_pol, pi_V, _ = policy_iteration(mdp)
        vi_pol, vi_V, _ = value_iteration(mdp)
        for s in range(mdp.nS):
            assert abs(pi_V[s] - vi_V[s]) < 1e-4, f"V mismatch at state {s}"

    def test_terminal_state_value_zero(self, mdp):
        _, V, _ = value_iteration(mdp)
        assert abs(V[4]) < 1e-6, "Terminal state must have V=0"

    def test_non_negative_values_for_nonterminal(self, mdp):
        _, V, _ = value_iteration(mdp)
        # Low-threat states should have positive value (agent can avoid attacks)
        assert V[0] > V[2], "Low traffic state should be valued higher than High attack state"


# ================================================================== #
#  Monte Carlo                                                          #
# ================================================================== #

class TestMC:
    EPISODES = 500   # keep fast for CI

    def test_first_visit_pred_shape(self, env, opt_policy):
        V = first_visit_mc_prediction(env, opt_policy, num_episodes=self.EPISODES)
        assert len(V) == env.nS

    def test_every_visit_pred_shape(self, env, opt_policy):
        V = every_visit_mc_prediction(env, opt_policy, num_episodes=self.EPISODES)
        assert len(V) == env.nS

    def test_mc_onpolicy_shape(self, env):
        Q, policy, returns = mc_control_epsilon_greedy(env, num_episodes=self.EPISODES)
        assert len(Q) == env.nS
        assert len(policy) == env.nS
        assert len(returns) == self.EPISODES

    def test_mc_offpolicy_shape(self, env):
        Q, policy = mc_control_off_policy_is(env, num_episodes=self.EPISODES)
        assert len(Q) == env.nS
        assert len(policy) == env.nS

    def test_policy_valid_actions(self, env):
        _, policy, _ = mc_control_epsilon_greedy(env, num_episodes=self.EPISODES)
        for a in policy:
            assert 0 <= a < env.nA


# ================================================================== #
#  TD Methods                                                           #
# ================================================================== #

class TestTD:
    EPISODES = 500

    def test_td0_shape(self, env, opt_policy):
        V, errors = td0_prediction(env, opt_policy, num_episodes=self.EPISODES)
        assert len(V) == env.nS
        assert len(errors) > 0

    def test_td_n_shape(self, env, opt_policy):
        V = td_n_prediction(env, opt_policy, num_episodes=self.EPISODES, n=4)
        assert len(V) == env.nS

    def test_td_lambda_fwd_shape(self, env, opt_policy):
        V = td_lambda_forward(env, opt_policy, num_episodes=self.EPISODES, lam=0.8)
        assert len(V) == env.nS

    def test_td_lambda_bwd_accumulating(self, env, opt_policy):
        V, errors = td_lambda_backward(env, opt_policy, num_episodes=self.EPISODES,
                                        lam=0.8, trace_type="accumulating")
        assert len(V) == env.nS

    def test_td_lambda_bwd_replacing(self, env, opt_policy):
        V, errors = td_lambda_backward(env, opt_policy, num_episodes=self.EPISODES,
                                        lam=0.8, trace_type="replacing")
        assert len(V) == env.nS


# ================================================================== #
#  SARSA                                                                #
# ================================================================== #

class TestSARSA:
    EPISODES = 500

    def test_sarsa_shape(self, env):
        Q, policy, returns = sarsa(env, num_episodes=self.EPISODES)
        assert len(Q) == env.nS and len(policy) == env.nS

    def test_sarsa_n_shape(self, env):
        Q, policy, _ = sarsa_n(env, num_episodes=self.EPISODES, n=3)
        assert len(policy) == env.nS

    def test_sarsa_lambda_fwd_shape(self, env):
        Q, policy, _ = sarsa_lambda_fwd(env, num_episodes=self.EPISODES, lam=0.8)
        assert len(policy) == env.nS

    def test_sarsa_lambda_bwd_accumulating(self, env):
        Q, policy, _ = sarsa_lambda_bwd(env, num_episodes=self.EPISODES,
                                         lam=0.8, trace_type="accumulating")
        assert len(policy) == env.nS

    def test_sarsa_lambda_bwd_replacing(self, env):
        Q, policy, _ = sarsa_lambda_bwd(env, num_episodes=self.EPISODES,
                                         lam=0.8, trace_type="replacing")
        assert len(policy) == env.nS


# ================================================================== #
#  Q-Learning                                                           #
# ================================================================== #

class TestQLearning:
    EPISODES = 500

    def test_qlearning_shape(self, env):
        Q, policy, returns = q_learning(env, num_episodes=self.EPISODES)
        assert len(Q) == env.nS and len(policy) == env.nS

    def test_double_qlearning_shape(self, env):
        Q, policy, _ = double_q_learning(env, num_episodes=self.EPISODES)
        assert len(policy) == env.nS

    def test_qlearning_policy_valid(self, env):
        _, policy, _ = q_learning(env, num_episodes=self.EPISODES)
        for a in policy:
            assert 0 <= a < env.nA


# ================================================================== #
#  DQN (light smoke test — no full training)                           #
# ================================================================== #

class TestDQN:
    def test_dqn_forward_pass(self, mdp):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.deep.networks import MLP, DuelingMLP, state_to_tensor, batch_states_to_tensor

        mlp     = MLP(input_dim=5, hidden_dim=32, output_dim=3)
        dueling = DuelingMLP(input_dim=5, hidden_dim=32, output_dim=3)

        x = state_to_tensor(0, nS=5)
        assert mlp(x).shape == (1, 3)
        assert dueling(x).shape == (1, 3)

        batch = batch_states_to_tensor([0, 1, 2], nS=5)
        assert mlp(batch).shape == (3, 3)

    def test_dqn_agent_smoke(self, env):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.deep.dqn import DQNAgent
        agent = DQNAgent(env, hidden_dim=32, buffer_capacity=200, batch_size=16)
        returns = agent.train(num_episodes=30, log_every=10, checkpoint_dir="/tmp/ckpt_test")
        assert len(returns) == 30
        policy = agent.get_policy()
        assert len(policy) == env.nS

    def test_saliency_gradient(self, env):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.deep.networks import MLP
        from src.deep.saliency import gradient_saliency, integrated_gradients

        net = MLP(input_dim=env.nS, hidden_dim=32, output_dim=env.nA)
        sal = gradient_saliency(net, state=0, nS=env.nS)
        assert len(sal) == env.nS
        ig = integrated_gradients(net, state=0, nS=env.nS, n_steps=10)
        assert len(ig) == env.nS
