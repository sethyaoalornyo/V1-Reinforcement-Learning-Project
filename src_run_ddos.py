from __future__ import annotations
import argparse
import json
import os
from typing import List

from mdp import build_ddos_style_mdp, save_mdp
from dp import policy_iteration, value_iteration


def save_artifacts(policy: List[int], V, Q, mdp, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save MDP definition too (helps grading)
    save_mdp(mdp, os.path.join(out_dir, "mdp.json"))

    # Human-readable policy
    pretty_policy = {
        mdp.state_names[s]: mdp.action_names[a] for s, a in enumerate(policy)
    }

    with open(os.path.join(out_dir, "policy.json"), "w", encoding="utf-8") as f:
        json.dump(pretty_policy, f, indent=2)

    with open(os.path.join(out_dir, "V.json"), "w", encoding="utf-8") as f:
        json.dump({mdp.state_names[s]: float(V[s]) for s in range(mdp.nS)}, f, indent=2)

    with open(os.path.join(out_dir, "Q.json"), "w", encoding="utf-8") as f:
        q_dump = {
            mdp.state_names[s]: {mdp.action_names[a]: float(Q[s][a]) for a in range(mdp.nA)}
            for s in range(mdp.nS)
        }
        json.dump(q_dump, f, indent=2)


def print_results(policy, V, Q, mdp):
    print("\n=== Results ===")
    for s in range(mdp.nS):
        sname = mdp.state_names[s]
        aname = mdp.action_names[policy[s]]
        print(f"{sname:>8} -> {aname:>10} | V={V[s]: .3f} | Q={ {mdp.action_names[a]: round(Q[s][a],3) for a in range(mdp.nA)} }")


def main():
    parser = argparse.ArgumentParser(description="V1 Term Project: Tabular RL via Dynamic Programming")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--method", choices=["policy_iteration", "value_iteration"], default="policy_iteration")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence tolerance")
    parser.add_argument("--out", type=str, default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    mdp = build_ddos_style_mdp(gamma=args.gamma)

    if args.method == "policy_iteration":
        policy, V, Q = policy_iteration(mdp, theta=args.theta)
    else:
        policy, V, Q = value_iteration(mdp, theta=args.theta)

    print_results(policy, V, Q, mdp)

    # Save best policy/value artifacts (required)
    save_artifacts(policy, V, Q, mdp, args.out)
    print(f"\nSaved artifacts to: {args.out}")


if __name__ == "__main__":
    main()
