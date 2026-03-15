from __future__ import annotations
import argparse
import os
import json

from mdp import GridWorldMDP
from dp import policy_iteration

def main():
    parser = argparse.ArgumentParser(description="V1: GridWorld DP Policy Iteration")
    parser.add_argument("--size", type=int, default=4, help="Grid size (NxN)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--theta", type=float, default=1e-8, help="Evaluation tolerance")
    args = parser.parse_args()

    mdp = GridWorldMDP(size=args.size, gamma=args.gamma)
    policy, V, Q = policy_iteration(mdp, theta=args.theta)

    os.makedirs("artifacts", exist_ok=True)

    # Save as JSON so it's easy to view in GitHub
    with open("artifacts/policy.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in policy.items()}, f, indent=2)

    with open("artifacts/V.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(vv) for k, vv in V.items()}, f, indent=2)

    print("\n=== Optimal Policy (state -> action) ===")
    for s in mdp.states:
        print(f"{s} -> {policy[s]}")

if __name__ == "__main__":
    main()
