"""
scripts/run_dqn.py
==================
Train the DQN agent on the DDoS environment, save checkpoints,
replay buffer, saliency maps, and a comparison with tabular Q-learning.

Usage
-----
    python scripts/run_dqn.py
    python scripts/run_dqn.py --episodes 1000 --dueling
    python scripts/run_dqn.py --episodes 500 --hidden 128 --lr 5e-4

Outputs
-------
    artifacts/dqn/
        returns.png             ← learning curve
        q_values.png            ← DQN Q-values vs tabular comparison
        saliency_gradient.png   ← gradient saliency heatmap
        saliency_ig.png         ← integrated gradients heatmap
        policy.json
        Q.json
    checkpoints/dqn/ddos/run1/
        online.pt               ← final online network weights
        target.pt               ← final target network weights
        meta.json               ← training metadata
    replay_buffer/ddos/dqn/fresh/
        buffer.json             ← final replay buffer snapshot
"""

from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.mdp.ddos_mdp import build_ddos_mdp
from src.utils.env_wrapper import TabularEnv
from src.utils.logger import TrainingLogger
from src.deep.dqn import DQNAgent
from src.deep.saliency import full_saliency_report
from src.classical.qlearning import q_learning


# ------------------------------------------------------------------ #
#  Comparison plot: DQN vs Tabular Q-learning                           #
# ------------------------------------------------------------------ #

def plot_comparison(dqn_Q, tabular_Q, mdp, out_dir: str) -> None:
    """Side-by-side Q-value comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.25

    for idx, (Q, title) in enumerate([(tabular_Q, "Tabular Q-Learning"), (dqn_Q, "DQN")]):
        ax = axes[idx]
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

    plt.suptitle("Q-Value Comparison: Tabular vs DQN")
    plt.tight_layout()
    path = os.path.join(out_dir, "q_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"[DQN] Comparison plot saved → {path}")


def print_policy_table(policy, Q, mdp, label: str) -> None:
    print(f"\n{label}")
    print("-" * 50)
    for s in range(mdp.nS):
        q_str = {mdp.action_names[a]: round(Q[s][a], 3) for a in range(mdp.nA)}
        print(f"  {mdp.state_names[s]:>12} → {mdp.action_names[policy[s]]:<12} Q={q_str}")


# ------------------------------------------------------------------ #
#  Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="V2 DQN — DDoS Environment")
    parser.add_argument("--gamma",          type=float, default=0.95)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--episodes",       type=int,   default=600)
    parser.add_argument("--epsilon_start",  type=float, default=1.0)
    parser.add_argument("--epsilon_end",    type=float, default=0.01)
    parser.add_argument("--epsilon_decay",  type=float, default=0.990)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--buffer_cap",     type=int,   default=10_000)
    parser.add_argument("--target_update",  type=int,   default=20)
    parser.add_argument("--dueling",        action="store_true", help="Use Dueling DQN")
    parser.add_argument("--out",            default="artifacts/dqn")
    parser.add_argument("--ckpt_dir",       default="checkpoints/dqn/ddos/run1")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- Build environment ----
    mdp = build_ddos_mdp(gamma=args.gamma)
    env = TabularEnv(mdp, start_state=0, max_steps=200, seed=42)

    arch = "DuelingDQN" if args.dueling else "DQN"
    print(f"\n{'='*60}")
    print(f"  {arch} — DDoS Environment")
    print(f"  Episodes: {args.episodes} | γ={args.gamma} | lr={args.lr}")
    print(f"{'='*60}\n")

    # ---- Save hyperparameters ----
    hypers = vars(args)
    hypers["architecture"] = arch
    with open(os.path.join(args.out, "hyperparameters.json"), "w") as f:
        json.dump(hypers, f, indent=2)

    # ---- Logger ----
    logger = TrainingLogger(run_dir=args.out)
    logger.log_hypers(hypers)

    # ---- DQN Agent ----
    agent = DQNAgent(
        env=env,
        hidden_dim=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_cap,
        batch_size=args.batch_size,
        target_update=args.target_update,
        dueling=args.dueling,
    )

    # ---- Train ----
    print(f"► Training {arch}…")
    ep_returns = agent.train(
        num_episodes=args.episodes,
        log_every=max(1, args.episodes // 10),
        checkpoint_dir=args.ckpt_dir,
        logger=logger,
    )

    # ---- Evaluate ----
    mean_return = agent.evaluate(num_episodes=100)
    logger.save()
    logger.plot_returns(title=f"{arch} — Episode Returns", filename="returns.png")

    # ---- Extract DQN policy ----
    dqn_Q      = agent.get_q_table()
    dqn_policy = agent.get_policy()
    print_policy_table(dqn_policy, dqn_Q, mdp, f"{arch} Policy")

    # Save
    with open(os.path.join(args.out, "policy.json"), "w") as f:
        json.dump({mdp.state_names[s]: mdp.action_names[dqn_policy[s]] for s in range(mdp.nS)}, f, indent=2)
    with open(os.path.join(args.out, "Q.json"), "w") as f:
        json.dump({
            mdp.state_names[s]: {mdp.action_names[a]: round(dqn_Q[s][a], 4) for a in range(mdp.nA)}
            for s in range(mdp.nS)
        }, f, indent=2)

    # ---- Baseline: Tabular Q-learning ----
    print("\n► Running tabular Q-learning as baseline…")
    tab_env = TabularEnv(mdp, start_state=0, max_steps=200, seed=0)
    tab_Q, tab_policy, _ = q_learning(tab_env, num_episodes=3000, gamma=args.gamma)
    print_policy_table(tab_policy, tab_Q, mdp, "Tabular Q-Learning Policy")

    # ---- Comparison plot ----
    plot_comparison(dqn_Q, tab_Q, mdp, args.out)

    # ---- Saliency ----
    print("\n► Computing saliency maps…")
    full_saliency_report(
        agent.online, nS=mdp.nS, state_names=mdp.state_names,
        method="gradient", device="cpu",
        save_dir=args.out,
    )
    full_saliency_report(
        agent.online, nS=mdp.nS, state_names=mdp.state_names,
        method="integrated_gradients", device="cpu",
        save_dir=args.out,
    )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Mean eval return: {mean_return:.4f}")
    print(f"  Artifacts → {args.out}/")
    print(f"  Checkpoints → {args.ckpt_dir}/")
    print(f"  Replay buffer → replay_buffer/ddos/dqn/fresh/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
