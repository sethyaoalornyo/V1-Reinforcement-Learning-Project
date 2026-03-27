"""
DDoS Mitigation MDP — Foundation Environment for V2
=====================================================
A 5-state tabular MDP that models a network under varying levels
of DDoS threat. The agent learns when to allow, rate-limit, or
block traffic to maximise cumulative reward.

States  : Low (0), Medium (1), High (2), Critical (3), Terminal (4)
Actions : ALLOW (0), RATE_LIMIT (1), BLOCK (2)
Terminal: State 4 absorbs all transitions (done=True).
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

State  = int
Action = int
Transition = Tuple[float, State, float, bool]   # (prob, s', r, done)


@dataclass
class TabularMDP:
    """
    Generic tabular MDP container.

    Attributes
    ----------
    nS           : number of states
    nA           : number of actions
    P            : transition table P[s][a] → list of (prob, s', r, done)
    gamma        : discount factor in [0, 1)
    state_names  : human-readable state labels
    action_names : human-readable action labels
    """
    nS: int
    nA: int
    P: List[List[List[Transition]]]
    gamma: float
    state_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #
    def transitions(self, s: State, a: Action) -> List[Transition]:
        """Return all (prob, s', r, done) outcomes for (s, a)."""
        return self.P[s][a]

    def is_terminal(self, s: State) -> bool:
        """True when every outgoing transition from s is marked done."""
        return all(done for _, _, _, done in self.P[s][0])

    # ------------------------------------------------------------------ #
    #  Serialisation                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict:
        return {
            "nS": self.nS,
            "nA": self.nA,
            "gamma": self.gamma,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "P": self.P,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "TabularMDP":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return TabularMDP(
            nS=obj["nS"],
            nA=obj["nA"],
            P=obj["P"],
            gamma=obj["gamma"],
            state_names=obj["state_names"],
            action_names=obj["action_names"],
        )


# ------------------------------------------------------------------ #
#  Factory function                                                    #
# ------------------------------------------------------------------ #
def build_ddos_mdp(
    gamma: float = 0.95,
    p_attack_by_state: Optional[List[float]] = None,
    seed: int = 7,
) -> TabularMDP:
    """
    Build the DDoS mitigation MDP.

    Reward structure
    ----------------
    Each expected reward blends attack-cost and legit-benefit based
    on p_attack_by_state, the probability the current traffic is
    genuinely malicious.

    ALLOW    : good for legit (+2), terrible for attacks (-25)
    RATE_LIMIT: modest legit reward (+1), moderate attack penalty (-6)
    BLOCK    : great for attacks (+4), punishes legit (-6)

    Transition dynamics
    -------------------
    ALLOW      : biased toward staying or escalating (0.15/0.65/0.20)
    RATE_LIMIT : moderate de-escalation (0.30/0.60/0.10)
    BLOCK      : strong de-escalation, 20 % chance to reach Terminal
    """
    random.seed(seed)

    state_names  = ["Low", "Medium", "High", "Critical", "Terminal"]
    action_names = ["ALLOW", "RATE_LIMIT", "BLOCK"]
    nS, nA = 5, 3
    TERMINAL = 4

    if p_attack_by_state is None:
        p_attack_by_state = [0.01, 0.08, 0.25, 0.55, 0.0]

    # Per-outcome rewards
    ALLOW_ATTACK   = -25.0
    ALLOW_LEGIT    =  +2.0
    RL_ATTACK      =  -6.0
    RL_LEGIT       =  +1.0
    BLOCK_ATTACK   =  +4.0
    BLOCK_LEGIT    =  -6.0

    def expected_reward(s: int, a: int) -> float:
        p_a = p_attack_by_state[s]
        p_l = 1.0 - p_a
        if a == 0:   return p_a * ALLOW_ATTACK  + p_l * ALLOW_LEGIT
        if a == 1:   return p_a * RL_ATTACK     + p_l * RL_LEGIT
        if a == 2:   return p_a * BLOCK_ATTACK  + p_l * BLOCK_LEGIT
        raise ValueError(f"Unknown action {a}")

    P: List[List[List[Transition]]] = [
        [[] for _ in range(nA)] for _ in range(nS)
    ]

    for s in range(nS):
        for a in range(nA):
            if s == TERMINAL:
                P[s][a] = [(1.0, TERMINAL, 0.0, True)]
                continue
            r = expected_reward(s, a)
            s_dn2 = max(s - 2, 0)
            s_dn1 = max(s - 1, 0)
            s_up1 = min(s + 1, TERMINAL - 1)

            if a == 0:   # ALLOW  – traffic can escalate easily
                P[s][a] = [(0.15, s_dn1, r, False),
                           (0.65, s,     r, False),
                           (0.20, s_up1, r, False)]
            elif a == 1: # RATE_LIMIT – moderate de-escalation
                P[s][a] = [(0.30, s_dn1, r, False),
                           (0.60, s,     r, False),
                           (0.10, s_up1, r, False)]
            else:        # BLOCK – strong de-escalation; 20 % terminal
                P[s][a] = [(0.55, s_dn2, r, False),
                           (0.25, s_dn1, r, False),
                           (0.20, TERMINAL, r, True)]

    return TabularMDP(
        nS=nS, nA=nA, P=P, gamma=gamma,
        state_names=state_names, action_names=action_names,
    )
