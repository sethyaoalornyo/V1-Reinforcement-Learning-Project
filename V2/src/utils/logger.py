"""
Training Logger
===============
Lightweight logger that records episode returns, policy snapshots and
hyperparameters to JSON files.  Works for both classical and deep RL.

Usage
-----
    log = TrainingLogger(run_dir="artifacts/dqn_run1")
    log.log_hypers({"lr": 1e-3, "gamma": 0.95, "epsilon_start": 1.0})

    for ep in range(num_eps):
        ...
        log.log_episode(ep, total_return, steps)

    log.save()
    log.plot_returns()
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """
    Logs training statistics to disk as JSON + optional matplotlib plots.

    Parameters
    ----------
    run_dir : directory where all log artefacts will be stored.
    """

    def __init__(self, run_dir: str = "artifacts/run") -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self._hypers: Dict[str, Any] = {}
        self._episodes: List[Dict] = []
        self._start_time: float = time.time()

    # ------------------------------------------------------------------ #
    #  Logging API                                                          #
    # ------------------------------------------------------------------ #
    def log_hypers(self, hypers: Dict[str, Any]) -> None:
        """Record hyperparameters (call once before training)."""
        self._hypers = hypers
        path = os.path.join(self.run_dir, "hyperparameters.json")
        with open(path, "w") as f:
            json.dump(hypers, f, indent=2)
        print(f"[Logger] Hyperparameters saved → {path}")

    def log_episode(
        self,
        episode: int,
        total_return: float,
        steps: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one episode's statistics."""
        record = {
            "episode": episode,
            "return": round(total_return, 4),
            "steps": steps,
            "elapsed_s": round(time.time() - self._start_time, 2),
        }
        if extra:
            record.update(extra)
        self._episodes.append(record)

    def log_step(self, step: int, loss: float, epsilon: float = None) -> None:
        """Optionally log per-step info (e.g. DQN loss)."""
        record = {"step": step, "loss": round(loss, 6)}
        if epsilon is not None:
            record["epsilon"] = round(epsilon, 4)
        # Append to a separate step log to avoid memory blowup
        path = os.path.join(self.run_dir, "step_log.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """Write all episode logs to disk."""
        path = os.path.join(self.run_dir, "episodes.json")
        with open(path, "w") as f:
            json.dump(self._episodes, f, indent=2)
        print(f"[Logger] Episode log saved ({len(self._episodes)} eps) → {path}")

    # ------------------------------------------------------------------ #
    #  Plotting                                                             #
    # ------------------------------------------------------------------ #
    def plot_returns(
        self,
        window: int = 20,
        title: str = "Episode Returns",
        filename: str = "returns.png",
    ) -> None:
        """Plot raw returns + smoothed moving average."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Logger] matplotlib not installed — skipping plot.")
            return

        returns = [ep["return"] for ep in self._episodes]
        episodes = [ep["episode"] for ep in self._episodes]

        # Compute moving average
        smoothed = []
        for i in range(len(returns)):
            start = max(0, i - window + 1)
            smoothed.append(sum(returns[start : i + 1]) / (i - start + 1))

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, returns, alpha=0.3, label="Raw return")
        plt.plot(episodes, smoothed, linewidth=2, label=f"{window}-ep avg")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        path = os.path.join(self.run_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"[Logger] Returns plot saved → {path}")

    # ------------------------------------------------------------------ #
    #  Convenience                                                          #
    # ------------------------------------------------------------------ #
    def recent_mean(self, n: int = 10) -> float:
        """Mean return over the last n episodes (useful for early stopping)."""
        if not self._episodes:
            return float("-inf")
        tail = self._episodes[-n:]
        return sum(ep["return"] for ep in tail) / len(tail)

    def summary(self) -> Dict[str, Any]:
        if not self._episodes:
            return {}
        returns = [ep["return"] for ep in self._episodes]
        return {
            "total_episodes": len(self._episodes),
            "mean_return": round(sum(returns) / len(returns), 4),
            "best_return": max(returns),
            "last_return": returns[-1],
            "elapsed_s": round(time.time() - self._start_time, 2),
        }
