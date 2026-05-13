"""
scripts/run_dqn.py
------------------
Train DQN or Dueling DQN on the DDoS MDP.

Usage
-----
    python scripts/run_dqn.py                          # default params
    python scripts/run_dqn.py --dueling                # Dueling DQN
    python scripts/run_dqn.py --use-tuned-params       # load from Bayesian search
    python scripts/run_dqn.py --episodes 1000 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mdp.ddos_mdp import build_ddos_mdp, save_mdp
from src.utils.env_wrapper import TabularEnv
from src.utils.logger import TrainingLogger
from src.deep.dqn import DQNAgent
from src.deep.networks import save_checkpoint
from src.tuning.bayesian_tuning import load_best_params


def onehot(state: int, nS: int) -> torch.Tensor:
    t = torch.zeros(nS)
    t[state] = 1.0
    return t


def train(args: argparse.Namespace) -> None:
    mdp = build_ddos_mdp(gamma=args.gamma)
    env = TabularEnv(mdp, seed=args.seed)

    # Load tuned hyperparameters if requested
    hp = {}
    if args.use_tuned_params:
        hp = load_best_params(out_dir="artifacts/tuning") or {}
        if hp:
            print("Loaded tuned hyperparameters:")
            for k, v in hp.items():
                if k != "best_value":
                    print(f"  {k}: {v}")
        else:
            print("No tuned params found — using defaults. Run scripts/run_tuning.py first.")

    agent = DQNAgent(
        nS=mdp.nS,
        nA=mdp.nA,
        hidden_dim=hp.get("hidden_dim", args.hidden_dim),
        lr=hp.get("lr", args.lr),
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=hp.get("epsilon_decay", args.epsilon_decay),
        batch_size=hp.get("batch_size", args.batch_size),
        target_update_freq=hp.get("target_update_freq", args.target_update_freq),
        dueling=hp.get("dueling", args.dueling),
        seed=args.seed,
    )

    out_dir   = args.out
    ckpt_dir  = os.path.join("checkpoints", "dqn", "ddos", "run1")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger  = TrainingLogger(os.path.join(out_dir, "dqn_log.json"))
    returns = []

    for ep in range(args.episodes):
        s         = env.reset(start_state=0)
        ep_return = 0.0
        for _ in range(50):
            s_vec = onehot(s, mdp.nS)
            a     = agent.select_action_onehot(s_vec)
            s2, r, done = env.step(a)
            agent.push(s, a, r, s2, done)
            loss = agent.learn(mdp.nS)
            ep_return += r
            s = s2
            if done:
                break

        returns.append(ep_return)
        logger.log(episode=ep, return_=ep_return, epsilon=agent.epsilon)

        if (ep + 1) % 50 == 0:
            mean_r = sum(returns[-50:]) / 50
            print(f"Ep {ep+1:4d}/{args.episodes} | mean_return(50)={mean_r:7.3f} | eps={agent.epsilon:.3f}")

    logger.save()

    # Save checkpoint
    save_checkpoint(
        agent.online, agent.target,
        meta={"episodes": args.episodes, "epsilon": agent.epsilon, "architecture": "dueling" if args.dueling else "mlp"},
        directory=ckpt_dir,
    )

    # Plot learning curve
    window = 20
    smoothed = [sum(returns[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(returns))]
    plt.figure(figsize=(10, 5))
    plt.plot(returns,  alpha=0.3, label="Episode return")
    plt.plot(smoothed, label=f"Smoothed (window={window})")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title(f"{'Dueling ' if args.dueling else ''}DQN Training — DDoS MDP")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dqn_learning_curve.png"))
    plt.close()

    print(f"\nTraining complete. Artifacts saved to {out_dir}")
    print(f"Checkpoint saved to {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",          type=int,   default=600)
    parser.add_argument("--gamma",             type=float, default=0.95)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--hidden-dim",        type=int,   default=128, dest="hidden_dim")
    parser.add_argument("--batch-size",        type=int,   default=64,  dest="batch_size")
    parser.add_argument("--epsilon-decay",     type=int,   default=500, dest="epsilon_decay")
    parser.add_argument("--target-update-freq",type=int,   default=50,  dest="target_update_freq")
    parser.add_argument("--dueling",           action="store_true")
    parser.add_argument("--use-tuned-params",  action="store_true", dest="use_tuned_params")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--out",               type=str,   default="artifacts/dqn")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
