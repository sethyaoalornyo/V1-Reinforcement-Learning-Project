"""
scripts/run_classical.py
------------------------
Run Policy Iteration or Value Iteration on the DDoS MDP.

Usage
-----
    python scripts/run_classical.py --method policy_iteration
    python scripts/run_classical.py --method value_iteration --gamma 0.99
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdp.ddos_mdp import build_ddos_mdp, save_mdp
from src.classical.dp import policy_iteration, value_iteration


def save_artifacts(policy, V, Q, mdp, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_mdp(mdp, os.path.join(out_dir, "mdp.json"))

    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump({mdp.state_names[s]: mdp.action_names[policy[s]] for s in range(mdp.nS)}, f, indent=2)
    with open(os.path.join(out_dir, "V.json"), "w") as f:
        json.dump({mdp.state_names[s]: round(V[s], 6) for s in range(mdp.nS)}, f, indent=2)

    # State value bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(mdp.state_names, V)
    plt.title("State Values — Learned by DP"); plt.xlabel("State"); plt.ylabel("V(s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "state_values.png"))
    plt.close()

    # Q-value grouped bar chart
    width = 0.25
    x = list(range(mdp.nS))
    plt.figure(figsize=(10, 5))
    for a in range(mdp.nA):
        plt.bar([xi + (a - 1) * width for xi in x], [Q[s][a] for s in x], width=width, label=mdp.action_names[a])
    plt.xticks(x, mdp.state_names, rotation=15)
    plt.title("Q-Values per State and Action"); plt.xlabel("State"); plt.ylabel("Q(s,a)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "q_values.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",  choices=["policy_iteration", "value_iteration"], default="policy_iteration")
    parser.add_argument("--gamma",   type=float, default=0.95)
    parser.add_argument("--theta",   type=float, default=1e-8)
    parser.add_argument("--out",     type=str,   default="artifacts/classical")
    args = parser.parse_args()

    mdp = build_ddos_mdp(gamma=args.gamma)

    if args.method == "policy_iteration":
        policy, V, Q = policy_iteration(mdp, theta=args.theta)
    else:
        policy, V, Q = value_iteration(mdp, theta=args.theta)

    print(f"\nResults ({args.method}):")
    for s in range(mdp.nS):
        print(f"  {mdp.state_names[s]:12s} -> {mdp.action_names[policy[s]]:10s} | V={V[s]:.4f}")

    save_artifacts(policy, V, Q, mdp, args.out)
    print(f"\nArtifacts saved to {args.out}")


if __name__ == "__main__":
    main()
