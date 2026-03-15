from __future__ import annotations
from typing import List, Tuple, Dict, TypeAlias, Optional
import json

State: TypeAlias = int
Action: TypeAlias = int
Transition: TypeAlias = Tuple[float, State, float, bool]


class TabularMDP:
    def __init__(
        self,
        nS: int,
        nA: int,
        gamma: float,
        P: Dict[State, Dict[Action, List[Transition]]],
        state_names: Optional[List[str]] = None,
        action_names: Optional[List[str]] = None,
    ) -> None:
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.P = P
        self.state_names = state_names if state_names is not None else [f"S{s}" for s in range(nS)]
        self.action_names = action_names if action_names is not None else [f"A{a}" for a in range(nA)]

    def transitions(self, s: State, a: Action) -> List[Transition]:
        return self.P[s][a]


def build_ddos_style_mdp(gamma: float = 0.95) -> TabularMDP:
    """
    Small DDoS-style tabular MDP.

    States:
        0 = Normal traffic
        1 = Suspicious traffic
        2 = DDoS attack active
        3 = Mitigated
    Actions:
        0 = Monitor
        1 = Filter
        2 = Block
    """

    nS = 4
    nA = 3

    state_names = [
        "Normal Traffic",
        "Suspicious Traffic",
        "DDoS Attack",
        "Mitigated",
    ]

    action_names = [
        "Monitor",
        "Filter",
        "Block",
    ]

    P: Dict[State, Dict[Action, List[Transition]]] = {
        0: {
            0: [(0.80, 0, 2.0, False), (0.20, 1, 0.0, False)],
            1: [(0.85, 0, 1.0, False), (0.15, 1, -1.0, False)],
            2: [(0.90, 0, -2.0, False), (0.10, 1, -3.0, False)],
        },
        1: {
            0: [(0.20, 0, 1.0, False), (0.50, 1, 0.0, False), (0.30, 2, -4.0, False)],
            1: [(0.50, 0, 2.0, False), (0.40, 1, 0.0, False), (0.10, 2, -2.0, False)],
            2: [(0.70, 0, 1.0, False), (0.20, 3, 3.0, False), (0.10, 2, -1.0, False)],
        },
        2: {
            0: [(0.10, 1, -6.0, False), (0.90, 2, -10.0, False)],
            1: [(0.30, 1, -2.0, False), (0.40, 2, -4.0, False), (0.30, 3, 5.0, False)],
            2: [(0.20, 1, -1.0, False), (0.70, 3, 8.0, False), (0.10, 2, -3.0, False)],
        },
        3: {
            0: [(1.0, 3, 0.0, True)],
            1: [(1.0, 3, 0.0, True)],
            2: [(1.0, 3, 0.0, True)],
        },
    }

    return TabularMDP(
        nS=nS,
        nA=nA,
        gamma=gamma,
        P=P,
        state_names=state_names,
        action_names=action_names,
    )


def save_mdp(mdp: TabularMDP, filename: str = "ddos_mdp.json") -> None:
    data = {
        "nS": mdp.nS,
        "nA": mdp.nA,
        "gamma": mdp.gamma,
        "state_names": mdp.state_names,
        "action_names": mdp.action_names,
        "P": {
            str(s): {
                str(a): [
                    {
                        "prob": prob,
                        "next_state": s2,
                        "reward": r,
                        "done": done,
                    }
                    for (prob, s2, r, done) in mdp.P[s][a]
                ]
                for a in mdp.P[s]
            }
            for s in mdp.P
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)