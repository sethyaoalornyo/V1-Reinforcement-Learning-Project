"""
Bayesian Hyperparameter Tuning for DQN — V3 Addition
=====================================================
Uses Optuna's Tree-structured Parzen Estimator (TPE) to search the
DQN hyperparameter space efficiently.  Replaces manual grid/random
search from V2 and finds better configurations with fewer trials.

Why Bayesian over grid/random search?
- Grid search is exponential in the number of hyperparameters.
- Random search ignores previous trial results entirely.
- Bayesian optimisation (TPE) builds a probabilistic model of
  performance as a function of hyperparameters, then samples from
  regions predicted to be promising.  It typically finds good
  configurations in 30-50 trials where grid search needs hundreds.

Why relevant to this environment?
- DQN has at least 5 interacting hyperparameters (lr, epsilon_decay,
  hidden_dim, batch_size, target_update_freq).
- The DDoS environment is small enough that each trial is fast,
  making an automated sweep practical.
- Bayesian tuning was explicitly discussed in class as the preferred
  method for single-agent environments where you have one learnable
  agent and want to squeeze the most performance from it.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import optuna
import torch

from src.mdp.ddos_mdp import build_ddos_mdp
from src.utils.env_wrapper import TabularEnv


# ── objective ────────────────────────────────────────────────────────────────

def _make_onehot(state: int, nS: int) -> torch.Tensor:
    t = torch.zeros(nS)
    t[state] = 1.0
    return t


def _run_trial(
    env: TabularEnv,
    hp: Dict[str, Any],
    n_episodes: int = 300,
    seed: int = 0,
) -> float:
    """
    Train a DQN with the given hyperparameters and return the mean
    episode return over the final 50 episodes (the objective to maximise).
    Imported inline to avoid circular imports at module load time.
    """
    from src.deep.dqn import DQNAgent

    random.seed(seed)
    torch.manual_seed(seed)

    nS = env.nS
    nA = env.nA

    agent = DQNAgent(
        nS=nS, nA=nA,
        hidden_dim=hp["hidden_dim"],
        lr=hp["lr"],
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=hp["epsilon_decay"],
        batch_size=hp["batch_size"],
        target_update_freq=hp["target_update_freq"],
        dueling=hp.get("dueling", False),
        seed=seed,
    )

    returns = []
    for ep in range(n_episodes):
        s   = env.reset(start_state=0)
        ep_return = 0.0
        for _ in range(50):  # max steps per episode
            s_vec  = _make_onehot(s, nS)
            a      = agent.select_action_onehot(s_vec)
            s2, r, done = env.step(a)
            agent.push(s, a, r, s2, done)
            agent.learn(nS)
            ep_return += r
            s = s2
            if done:
                break
        returns.append(ep_return)

    # Objective: mean return over the final 50 episodes
    return float(sum(returns[-50:]) / 50)


def objective(trial: optuna.Trial, env: TabularEnv, n_episodes: int = 300) -> float:
    hp = {
        "lr":                trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_dim":        trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        "batch_size":        trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epsilon_decay":     trial.suggest_int("epsilon_decay", 100, 1000),
        "target_update_freq":trial.suggest_int("target_update_freq", 10, 200),
        "dueling":           trial.suggest_categorical("dueling", [False, True]),
    }
    return _run_trial(env, hp, n_episodes=n_episodes, seed=42)


# ── public API ────────────────────────────────────────────────────────────────

def run_study(
    n_trials: int       = 50,
    n_episodes: int     = 300,
    out_dir: str        = "artifacts/tuning",
    verbose: bool       = True,
    seed: int           = 0,
) -> Dict[str, Any]:
    """
    Run a Bayesian hyperparameter search and save the best configuration.

    Parameters
    ----------
    n_trials   : Number of Optuna trials (30–50 recommended).
    n_episodes : Training episodes per trial.
    out_dir    : Directory to save best_params.json and study results.
    verbose    : Print progress to stdout.
    seed       : Random seed for reproducibility.

    Returns
    -------
    best_params : dict of best hyperparameter values found.
    """
    os.makedirs(out_dir, exist_ok=True)

    mdp = build_ddos_mdp()
    env = TabularEnv(mdp, seed=seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    direction = "maximize"

    study = optuna.create_study(direction=direction, sampler=sampler)

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        lambda trial: objective(trial, env, n_episodes=n_episodes),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    best = study.best_params
    best["best_value"] = study.best_value

    # Save results
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Save all trial results
    trials_data = [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in study.trials
        if t.value is not None
    ]
    with open(os.path.join(out_dir, "all_trials.json"), "w") as f:
        json.dump(trials_data, f, indent=2)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best mean return: {study.best_value:.3f}")
        print("Best hyperparameters:")
        for k, v in best.items():
            if k != "best_value":
                print(f"  {k:25s}: {v}")
        print(f"Results saved to: {out_dir}")

    return best


def load_best_params(out_dir: str = "artifacts/tuning") -> Optional[Dict[str, Any]]:
    """Load previously saved best hyperparameters."""
    path = os.path.join(out_dir, "best_params.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
