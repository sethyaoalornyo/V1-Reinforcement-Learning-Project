from __future__ import annotations
import random
from typing import Optional, Tuple

from src.mdp.ddos_mdp import TabularMDP, State, Action


class TabularEnv:
    """Gym-style wrapper around a TabularMDP."""

    def __init__(self, mdp: TabularMDP, seed: Optional[int] = None) -> None:
        self.mdp = mdp
        self.nS  = mdp.nS
        self.nA  = mdp.nA
        self._rng = random.Random(seed)
        self._state: State = 0

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, start_state: int = 0) -> State:
        self._state = start_state
        return self._state

    def step(self, action: Action) -> Tuple[State, float, bool]:
        transitions = self.mdp.transitions(self._state, action)
        r_val       = self._rng.random()
        cumulative  = 0.0
        chosen      = transitions[-1]
        for t in transitions:
            cumulative += t[0]
            if r_val <= cumulative:
                chosen = t
                break
        prob, s2, reward, done = chosen
        self._state = s2
        return s2, reward, done

    def sample_action(self) -> Action:
        return self._rng.randint(0, self.nA - 1)

    @property
    def state(self) -> State:
        return self._state
