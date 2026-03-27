"""
Environment Wrapper
===================
Wraps a TabularMDP into a gym-style interface so all RL algorithms
(classical and deep) can interact with it through the same API.

    env = DDosEnv()
    s   = env.reset()
    s2, r, done, info = env.step(action)
"""

from __future__ import annotations
import random
from typing import Optional, Tuple

from src.mdp.ddos_mdp import TabularMDP, State, build_ddos_mdp


class TabularEnv:
    """
    Gym-style wrapper around any TabularMDP.

    Parameters
    ----------
    mdp        : TabularMDP instance
    start_state: fixed start state (default 0 = Low traffic)
    max_steps  : episode terminates after this many steps even if not done
    seed       : optional random seed for reproducibility
    """

    def __init__(
        self,
        mdp: TabularMDP,
        start_state: int = 0,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ) -> None:
        self.mdp = mdp
        self.start_state = start_state
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        self.current_state: int = start_state
        self._step_count: int = 0

    # ------------------------------------------------------------------ #
    #  Properties matching gym conventions                                  #
    # ------------------------------------------------------------------ #
    @property
    def nS(self) -> int:
        return self.mdp.nS

    @property
    def nA(self) -> int:
        return self.mdp.nA

    @property
    def state_names(self):
        return self.mdp.state_names

    @property
    def action_names(self):
        return self.mdp.action_names

    # ------------------------------------------------------------------ #
    #  Core API                                                             #
    # ------------------------------------------------------------------ #
    def reset(self, start_state: Optional[int] = None) -> State:
        """Reset the environment and return the initial state."""
        self.current_state = start_state if start_state is not None else self.start_state
        self._step_count = 0
        return self.current_state

    def step(self, action: int) -> Tuple[State, float, bool, dict]:
        """
        Apply action and return (next_state, reward, done, info).

        Transitions are sampled according to P[s][a].
        """
        transitions = self.mdp.transitions(self.current_state, action)

        # Sample next state from the transition distribution
        rnum = self._rng.random()
        cumulative = 0.0
        chosen = transitions[-1]           # default: last outcome
        for t in transitions:
            prob, s2, r, done = t
            cumulative += prob
            if rnum <= cumulative:
                chosen = t
                break

        prob, s2, r, done = chosen
        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = True

        self.current_state = s2
        return s2, r, done, {"prob": prob, "step": self._step_count}

    def sample_action(self) -> int:
        """Uniformly random action."""
        return self._rng.randint(0, self.mdp.nA - 1)

    def seed(self, s: int) -> None:
        self._rng = random.Random(s)

    def __repr__(self) -> str:
        return (
            f"TabularEnv(nS={self.nS}, nA={self.nA}, "
            f"state={self.state_names[self.current_state]})"
        )


# ------------------------------------------------------------------ #
#  Convenience factory                                                  #
# ------------------------------------------------------------------ #
def make_ddos_env(gamma: float = 0.95, max_steps: int = 200, seed: int = 42) -> TabularEnv:
    """Return a ready-to-use DDoS environment."""
    mdp = build_ddos_mdp(gamma=gamma)
    return TabularEnv(mdp, start_state=0, max_steps=max_steps, seed=seed)
