from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

State = int
Action = int
Transition = Tuple[float, State, float, bool]


@dataclass
class TabularMDP:
    """
    Tabular MDP:
      - states : 0 .. nS-1
      - actions: 0 .. nA-1
      - P[s][a] -> list of (prob, s_next, reward, done)
      - gamma in [0, 1)
    """
    nS: int
    nA: int
    P: List[List[List[Transition]]]
    gamma: float
    state_names: List[str]
    action_names: List[str]

    def transitions(self, s: State, a: Action) -> List[Transition]:
        return self.P[s][a]

    def is_terminal(self, s: State) -> bool:
        for a in range(self.nA):
            for prob, s2, r, done in self.P[s][a]:
                if not done or s2 != s:
                    return False
        return True

    def to_json(self) -> Dict:
        return {
            "nS": self.nS,
            "nA": self.nA,
            "gamma": self.gamma,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "P": self.P,
        }

    @staticmethod
    def from_json(obj: Dict) -> "TabularMDP":
        return TabularMDP(
            nS=obj["nS"],
            nA=obj["nA"],
            P=obj["P"],
            gamma=obj["gamma"],
            state_names=obj["state_names"],
            action_names=obj["action_names"],
        )


def build_ddos_mdp(
    gamma: float = 0.95,
    p_attack_by_state: Optional[List[float]] = None,
) -> TabularMDP:
    """
    DDoS mitigation tabular MDP.

    States  : Low(0), Medium(1), High(2), Critical(3), Terminal(4)
    Actions : ALLOW(0), RATE_LIMIT(1), BLOCK(2)
    """
    state_names  = ["Low", "Medium", "High", "Critical", "Terminal"]
    action_names = ["ALLOW", "RATE_LIMIT", "BLOCK"]
    nS, nA = 5, 3
    terminal = 4

    if p_attack_by_state is None:
        p_attack_by_state = [0.01, 0.08, 0.25, 0.55, 0.0]

    # Reward constants
    ALLOW_ATTACK  = -25.0;  ALLOW_LEGIT   = +2.0
    RL_ATTACK     = -6.0;   RL_LEGIT      = +1.0
    BLOCK_ATTACK  = +4.0;   BLOCK_LEGIT   = -6.0

    def expected_reward(s: int, a: int) -> float:
        pa = p_attack_by_state[s]
        pl = 1.0 - pa
        if a == 0: return pa * ALLOW_ATTACK  + pl * ALLOW_LEGIT
        if a == 1: return pa * RL_ATTACK     + pl * RL_LEGIT
        if a == 2: return pa * BLOCK_ATTACK  + pl * BLOCK_LEGIT
        raise ValueError(f"Unknown action {a}")

    P: List[List[List[Transition]]] = [[[] for _ in range(nA)] for _ in range(nS)]

    for s in range(nS):
        for a in range(nA):
            if s == terminal:
                P[s][a] = [(1.0, terminal, 0.0, True)]
                continue
            r = expected_reward(s, a)
            if a == 0:   # ALLOW — slight escalation risk
                P[s][a] = [
                    (0.15, max(s - 1, 0), r, False),
                    (0.65, s,             r, False),
                    (0.20, min(s + 1, 3), r, False),
                ]
            elif a == 1: # RATE_LIMIT — moderate de-escalation
                P[s][a] = [
                    (0.30, max(s - 1, 0), r, False),
                    (0.60, s,             r, False),
                    (0.10, min(s + 1, 3), r, False),
                ]
            else:        # BLOCK — strong de-escalation, may end episode
                P[s][a] = [
                    (0.55, max(s - 2, 0), r, False),
                    (0.25, max(s - 1, 0), r, False),
                    (0.20, terminal,      r, True),
                ]

    return TabularMDP(nS=nS, nA=nA, P=P, gamma=gamma,
                      state_names=state_names, action_names=action_names)


def save_mdp(mdp: TabularMDP, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mdp.to_json(), f, indent=2)


def load_mdp(path: str) -> TabularMDP:
    with open(path, "r", encoding="utf-8") as f:
        return TabularMDP.from_json(json.load(f))
