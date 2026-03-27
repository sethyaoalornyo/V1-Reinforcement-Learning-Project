"""
Replay Buffer
=============
A circular experience replay buffer used by off-policy deep RL methods
(e.g. DQN).  Experiences are stored as (s, a, r, s', done) tuples and
can be saved/loaded from disk for persistence across runs.

Organisation on disk
--------------------
replay_buffer/
  <task>/               e.g. ddos/
    <algorithm>/        e.g. dqn/
      <tag>/            e.g. fresh/ or stale/
        buffer.json

Usage
-----
    buf = ReplayBuffer(capacity=10_000)
    buf.push(s, a, r, s2, done)
    batch = buf.sample(32)
    buf.save("replay_buffer/ddos/dqn/fresh/buffer.json")
    buf.load("replay_buffer/ddos/dqn/fresh/buffer.json")
"""

from __future__ import annotations
import json
import os
import random
from collections import deque
from typing import List, NamedTuple, Optional, Tuple


class Experience(NamedTuple):
    state:      int
    action:     int
    reward:     float
    next_state: int
    done:       bool


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Parameters
    ----------
    capacity : maximum number of experiences to store.
               When full, the oldest experience is evicted.
    seed     : optional random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None) -> None:
        assert capacity > 0, "Capacity must be positive"
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Core operations                                                      #
    # ------------------------------------------------------------------ #
    def push(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Add a single experience.  Evicts oldest if at capacity."""
        self._buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample `batch_size` experiences uniformly at random.

        Raises
        ------
        ValueError if the buffer has fewer experiences than batch_size.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self)} experiences "
                f"but batch_size={batch_size} was requested."
            )
        return self._rng.sample(list(self._buffer), batch_size)

    def sample_as_tensors(self, batch_size: int):
        """
        Sample a batch and return separate numpy-friendly lists.
        Returns (states, actions, rewards, next_states, dones).
        """
        batch = self.sample(batch_size)
        states      = [e.state      for e in batch]
        actions     = [e.action     for e in batch]
        rewards     = [e.reward     for e in batch]
        next_states = [e.next_state for e in batch]
        dones       = [e.done       for e in batch]
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True when enough experiences are stored to sample a batch."""
        return len(self) >= batch_size

    def clear(self) -> None:
        """Remove all stored experiences."""
        self._buffer.clear()

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """
        Serialise buffer to JSON.

        The directory tree is created automatically, so you can use
        paths like 'replay_buffer/ddos/dqn/fresh/buffer.json'.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "capacity": self.capacity,
            "size": len(self),
            "experiences": [list(e) for e in self._buffer],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[ReplayBuffer] Saved {len(self)} experiences → {path}")

    def load(self, path: str) -> None:
        """
        Load experiences from JSON, respecting the buffer's capacity.
        If the file contains more than `capacity` experiences, only the
        most recent `capacity` are kept.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._buffer.clear()
        for e in data["experiences"]:
            self._buffer.append(Experience(*e))
        print(f"[ReplayBuffer] Loaded {len(self)} experiences ← {path}")

    # ------------------------------------------------------------------ #
    #  Stats                                                                #
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Return summary statistics for logging."""
        if len(self) == 0:
            return {"size": 0, "capacity": self.capacity, "fill_pct": 0.0}
        rewards = [e.reward for e in self._buffer]
        return {
            "size": len(self),
            "capacity": self.capacity,
            "fill_pct": round(100 * len(self) / self.capacity, 1),
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"


# ------------------------------------------------------------------ #
#  Buffer rotation utility                                             #
# ------------------------------------------------------------------ #
def rotate_replay(
    fresh_path: str,
    stale_path: str,
    max_total: int = 50_000,
    seed: int = 0,
) -> None:
    """
    Replace the stale replay buffer with the fresh one, then enforce
    a total-size cap across both combined (keeping the newest entries).

    Parameters
    ----------
    fresh_path : path to the newer buffer JSON
    stale_path : path to the older buffer JSON (overwritten on exit)
    max_total  : maximum combined number of experiences to retain
    seed       : rng seed for the combined buffer

    This is called from scripts/manage_replay.py.
    """
    fresh = ReplayBuffer(capacity=max_total, seed=seed)
    stale = ReplayBuffer(capacity=max_total, seed=seed)

    if os.path.exists(stale_path):
        stale.load(stale_path)
    if os.path.exists(fresh_path):
        fresh.load(fresh_path)

    # Merge: stale first, then fresh (so fresh overwrites when at cap)
    combined = ReplayBuffer(capacity=max_total, seed=seed)
    for e in stale._buffer:
        combined.push(*e)
    for e in fresh._buffer:
        combined.push(*e)

    # Save merged result back to stale_path
    combined.save(stale_path)
    print(f"[rotate_replay] Combined buffer: {len(combined)}/{max_total}")
