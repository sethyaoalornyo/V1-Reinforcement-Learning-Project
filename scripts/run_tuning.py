"""
scripts/run_tuning.py
---------------------
Run Bayesian hyperparameter tuning for the DQN agent on the DDoS MDP.

Usage
-----
    python scripts/run_tuning.py                        # 50 trials, 300 eps each
    python scripts/run_tuning.py --trials 30 --episodes 200
    python scripts/run_tuning.py --trials 10 --quiet    # fast sanity check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tuning.bayesian_tuning import run_study, load_best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian DQN hyperparameter tuning")
    parser.add_argument("--trials",   type=int,  default=50,              help="Number of Optuna trials")
    parser.add_argument("--episodes", type=int,  default=300,             help="Training episodes per trial")
    parser.add_argument("--out",      type=str,  default="artifacts/tuning", help="Output directory")
    parser.add_argument("--seed",     type=int,  default=42,              help="Random seed")
    parser.add_argument("--quiet",    action="store_true",                help="Suppress progress output")
    args = parser.parse_args()

    print(f"Starting Bayesian hyperparameter search:")
    print(f"  Trials   : {args.trials}")
    print(f"  Episodes : {args.episodes} per trial")
    print(f"  Output   : {args.out}")
    print()

    best = run_study(
        n_trials=args.trials,
        n_episodes=args.episodes,
        out_dir=args.out,
        verbose=not args.quiet,
        seed=args.seed,
    )

    print(f"\nBest params saved to {args.out}/best_params.json")
    print("Re-run DQN training with these params using:")
    print(f"  python scripts/run_dqn.py --use-tuned-params")


if __name__ == "__main__":
    main()
