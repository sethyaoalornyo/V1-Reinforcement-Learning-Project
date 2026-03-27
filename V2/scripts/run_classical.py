"""
scripts/run_classical.py
========================
Train and evaluate ALL classical RL algorithms on the DDoS MDP.

Usage
-----
    python scripts/run_classical.py --algo all --episodes 3000
    python scripts/run_classical.py --algo sarsa_lambda_bwd --episodes 5000
    python scripts/run_classical.py --algo mc_onpolicy --gamma 0.99

Outputs (saved to artifacts/<algo>/)
------
    policy.json      ← best action per state
    V.json           ← state-value estimates
    Q.json           ← action-value estimates (where applicable)
    returns.png      ← learning curve
    q_values.png     ← bar chart of Q(s,a) for all (s,a)
"""

from __future__ import annotations
import argparse
import json
import os
import sys

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.mdp.ddos_mdp import build_ddos_mdp
from src.utils.env_wrapper import TabularEnv
from src.utils.logger import TrainingLogger
from src.classical.dp import policy_iteration, value_iteration
from src.classical.mc import (
    first_visit_mc_prediction,
    every_visit_mc_prediction,
    mc_control_epsilon_greedy,
    mc_control_off_policy_is,
)
from src.classical.td import (
    td0_prediction,
    td_n_prediction,
    td_lambda_forward,
    td_lambda_backward,
)
from src.classical.sarsa import (
    sarsa,
    sarsa_n,
    sarsa_lambda_fwd,
    sarsa_lambda_bwd,
)
from src.classical.qlearning import q_learning, double_q_learning


# ------------------------------------------------------------------ #
#  Save helpers                                                         #
# ------------------------------------------------------------------ #

def save_policy(policy, V, Q, mdp, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump({mdp.state_names[s]: mdp.action_names[a] for s, a in enumerate(policy)}, f, indent=2)
    if V:
        with open(os.path.join(out_dir, "V.json"), "w") as f:
            json.dump({mdp.state_names[s]: round(float(V[s]), 4) for s in range(mdp.nS)}, f, indent=2)
    if Q:
        with open(os.path.join(out_dir, "Q.json"), "w") as f:
            q_dump = {
                mdp.state_names[s]: {mdp.action_names[a]: round(float(Q[s][a]), 4) for a in range(mdp.nA)}
                for s in range(mdp.nS)
            }
            json.dump(q_dump, f, indent=2)


def plot_q_values(Q, mdp, out_dir: str, title: str = "Q-Values") -> None:
    if Q is None:
        return
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for a in range(mdp.nA):
        x = [s + (a - 1) * width for s in range(mdp.nS)]
        y = [Q[s][a] for s in range(mdp.nS)]
        ax.bar(x, y, width=width, label=mdp.action_names[a])
    ax.set_xticks(range(mdp.nS))
    ax.set_xticklabels(mdp.state_names, rotation=20)
    ax.set_title(title)
    ax.set_xlabel("State")
    ax.set_ylabel("Q(s, a)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "q_values.png"))
    plt.close()


def plot_returns(ep_returns, out_dir, title="Episode Returns", window=50) -> None:
    if not ep_returns:
        return
    smoothed = []
    for i in range(len(ep_returns)):
        start = max(0, i - window + 1)
        smoothed.append(sum(ep_returns[start:i+1]) / (i - start + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep_returns, alpha=0.3, label="Raw")
    ax.plot(smoothed, linewidth=2, label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "returns.png"))
    plt.close()


def print_results(policy, V, Q, mdp, algo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Algorithm: {algo}")
    print(f"{'='*60}")
    for s in range(mdp.nS):
        v_str = f"V={V[s]:.3f}" if V else ""
        q_str = ""
        if Q:
            q_str = " | Q=" + str({mdp.action_names[a]: round(Q[s][a], 3) for a in range(mdp.nA)})
        print(f"  {mdp.state_names[s]:>14} → {mdp.action_names[policy[s]]:<12} {v_str} {q_str}")
    print()


# ------------------------------------------------------------------ #
#  Algorithm dispatcher                                                 #
# ------------------------------------------------------------------ #

ALL_ALGOS = [
    "policy_iteration", "value_iteration",
    "mc_firstvisit", "mc_everyvisit", "mc_onpolicy", "mc_offpolicy",
    "td0", "td_n", "td_lambda_fwd", "td_lambda_bwd",
    "sarsa", "sarsa_n", "sarsa_lambda_fwd", "sarsa_lambda_bwd",
    "qlearning", "double_qlearning",
]


def run_algo(algo: str, mdp, env: TabularEnv, args) -> None:
    out_dir = os.path.join(args.out, algo)
    os.makedirs(out_dir, exist_ok=True)
    V, Q, policy, ep_returns = None, None, None, []

    # DP methods (need the MDP model)
    if algo == "policy_iteration":
        policy, V, Q = policy_iteration(mdp, theta=args.theta)

    elif algo == "value_iteration":
        policy, V, Q = value_iteration(mdp, theta=args.theta)

    # MC methods
    elif algo == "mc_firstvisit":
        # Use greedy PI policy for evaluation
        pi_policy, _, _ = policy_iteration(mdp)
        V = first_visit_mc_prediction(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma)
        policy = pi_policy

    elif algo == "mc_everyvisit":
        pi_policy, _, _ = policy_iteration(mdp)
        V = every_visit_mc_prediction(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma)
        policy = pi_policy

    elif algo == "mc_onpolicy":
        Q, policy, ep_returns = mc_control_epsilon_greedy(
            env, num_episodes=args.episodes, gamma=args.gamma,
            epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
        )

    elif algo == "mc_offpolicy":
        Q, policy = mc_control_off_policy_is(
            env, num_episodes=args.episodes, gamma=args.gamma,
        )

    # TD methods
    elif algo == "td0":
        pi_policy, _, _ = policy_iteration(mdp)
        V, _ = td0_prediction(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma, alpha=args.alpha)
        policy = pi_policy

    elif algo == "td_n":
        pi_policy, _, _ = policy_iteration(mdp)
        V = td_n_prediction(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma, alpha=args.alpha, n=args.n)
        policy = pi_policy

    elif algo == "td_lambda_fwd":
        pi_policy, _, _ = policy_iteration(mdp)
        V = td_lambda_forward(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma, alpha=args.alpha, lam=args.lam)
        policy = pi_policy

    elif algo == "td_lambda_bwd":
        pi_policy, _, _ = policy_iteration(mdp)
        V, _ = td_lambda_backward(env, pi_policy, num_episodes=args.episodes, gamma=args.gamma, alpha=args.alpha, lam=args.lam)
        policy = pi_policy

    # SARSA variants
    elif algo == "sarsa":
        Q, policy, ep_returns = sarsa(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
        )

    elif algo == "sarsa_n":
        Q, policy, ep_returns = sarsa_n(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, n=args.n,
        )

    elif algo == "sarsa_lambda_fwd":
        Q, policy, ep_returns = sarsa_lambda_fwd(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, lam=args.lam,
        )

    elif algo == "sarsa_lambda_bwd":
        Q, policy, ep_returns = sarsa_lambda_bwd(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, lam=args.lam,
            epsilon_decay=args.epsilon_decay,
        )

    # Q-learning variants
    elif algo == "qlearning":
        Q, policy, ep_returns = q_learning(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
        )

    elif algo == "double_qlearning":
        Q, policy, ep_returns = double_q_learning(
            env, num_episodes=args.episodes, gamma=args.gamma,
            alpha=args.alpha, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay,
        )

    else:
        print(f"[WARNING] Unknown algorithm: {algo}")
        return

    # Save results
    save_policy(policy, V, Q, mdp, out_dir)
    plot_q_values(Q, mdp, out_dir, title=f"{algo} — Q-Values")
    plot_returns(ep_returns, out_dir, title=f"{algo} — Episode Returns")
    print_results(policy, V, Q, mdp, algo)
    print(f"  Saved artifacts → {out_dir}")


# ------------------------------------------------------------------ #
#  Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="V2 Classical RL — DDoS Environment")
    parser.add_argument("--algo",           default="all",  help=f"One of: {ALL_ALGOS} or 'all'")
    parser.add_argument("--gamma",    type=float, default=0.95)
    parser.add_argument("--alpha",    type=float, default=0.1,   help="Learning rate (TD/SARSA/Q)")
    parser.add_argument("--epsilon",  type=float, default=0.1,   help="Exploration probability")
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--lam",      type=float, default=0.8,   help="λ for TD(λ)/SARSA(λ)")
    parser.add_argument("--n",        type=int,   default=4,     help="n for n-step methods")
    parser.add_argument("--episodes", type=int,   default=3000,  help="Number of training episodes")
    parser.add_argument("--theta",    type=float, default=1e-8,  help="DP convergence threshold")
    parser.add_argument("--out",      default="artifacts",       help="Output directory root")
    args = parser.parse_args()

    mdp = build_ddos_mdp(gamma=args.gamma)
    env = TabularEnv(mdp, start_state=0, max_steps=200, seed=42)

    target_algos = ALL_ALGOS if args.algo == "all" else [args.algo]

    print(f"\n{'='*60}")
    print(f"  V2 Classical RL — DDoS MDP")
    print(f"  Training {len(target_algos)} algorithm(s) × {args.episodes} episodes")
    print(f"{'='*60}\n")

    for algo in target_algos:
        print(f"► Running: {algo} …")
        run_algo(algo, mdp, env, args)

    print(f"\n✓ All results saved to: {args.out}/")


if __name__ == "__main__":
    main()
