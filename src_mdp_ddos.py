from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import random
import matplotlib.pyplot as plt

State = int
Action = int


@dataclass
class TabularMDP:
    """
    Tabular MDP:
      - states: 0..(nS-1)
      - actions: 0..(nA-1)
      - P[s][a] -> list of (prob, s_next, reward, done)
      - gamma in [0,1)
    """
    nS: int
    nA: int
    P: List[List[List[Tuple[float, State, float, bool]]]]
    gamma: float
    state_names: List[str]
    action_names: List[str]

    def transitions(self, s: State, a: Action):
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


def build_ddos_style_mdp(
    gamma: float = 0.95,
    p_attack_by_state: List[float] | None = None,
    seed: int = 7,
) -> TabularMDP:
    """
    Small MDP inspired by DDoS mitigation.
    """
    random.seed(seed)

    state_names = ["Low", "Medium", "High", "Critical", "Terminal"]
    action_names = ["ALLOW", "RATE_LIMIT", "BLOCK"]

    nS = 5
    nA = 3

    if p_attack_by_state is None:
        p_attack_by_state = [0.01, 0.08, 0.25, 0.55, 0.0]

    allow_attack_cost = -25.0
    allow_legit_reward = +2.0

    block_legit_cost = -6.0
    block_attack_reward = +4.0

    ratelimit_legit_reward = +1.0
    ratelimit_attack_cost = -6.0

    P: List[List[List[Tuple[float, int, float, bool]]]] = [
        [[] for _ in range(nA)] for _ in range(nS)
    ]

    terminal = 4

    def expected_reward(s: int, a: int) -> float:
        p_attack = p_attack_by_state[s]
        p_legit = 1.0 - p_attack

        if a == 0:
            return p_attack * allow_attack_cost + p_legit * allow_legit_reward
        if a == 1:
            return p_attack * ratelimit_attack_cost + p_legit * ratelimit_legit_reward
        if a == 2:
            return p_attack * block_attack_reward + p_legit * block_legit_cost
        raise ValueError("Invalid action")

    for s in range(nS):
        for a in range(nA):
            if s == terminal:
                P[s][a] = [(1.0, terminal, 0.0, True)]
                continue

            r = expected_reward(s, a)

            if a == 0:
                s_down = max(s - 1, 0)
                s_same = s
                s_up = min(s + 1, 3)
                P[s][a] = [
                    (0.15, s_down, r, False),
                    (0.65, s_same, r, False),
                    (0.20, s_up, r, False),
                ]
            elif a == 1:
                s_down = max(s - 1, 0)
                s_same = s
                s_up = min(s + 1, 3)
                P[s][a] = [
                    (0.30, s_down, r, False),
                    (0.60, s_same, r, False),
                    (0.10, s_up, r, False),
                ]
            else:
                s_down2 = max(s - 2, 0)
                s_down1 = max(s - 1, 0)
                P[s][a] = [
                    (0.55, s_down2, r, False),
                    (0.25, s_down1, r, False),
                    (0.20, terminal, r, True),
                ]

    return TabularMDP(
        nS=nS,
        nA=nA,
        P=P,
        gamma=gamma,
        state_names=state_names,
        action_names=action_names,
    )


def save_mdp(mdp: TabularMDP, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mdp.to_json(), f, indent=2)


def load_mdp(path: str) -> TabularMDP:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return TabularMDP.from_json(obj)

if __name__ == "__main__":
    mdp = build_ddos_style_mdp()
    print("MDP created successfully")
    print("States:", mdp.state_names)
    print("Actions:", mdp.action_names)

    if __name__ == "__main__":
    
     from dp import value_iteration

    mdp = build_ddos_style_mdp()

    print("MDP created successfully")
    print("States:", mdp.state_names)
    print("Actions:", mdp.action_names)

    policy, V, Q = value_iteration(mdp)

    print("\nLearned Policy:")
    for s in range(mdp.nS):
        print(mdp.state_names[s], "->", mdp.action_names[policy[s]])

    # Plot the state values
    plt.bar(mdp.state_names, V)
    plt.title("State Values Learned by the Agent")
    plt.xlabel("States")
    plt.ylabel("Value")
    plt.xticks(rotation=20)
    plt.show()