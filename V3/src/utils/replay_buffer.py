from __future__ import annotations
import json
import random
from collections import deque
from typing import List, Optional, Tuple

Transition = Tuple[int, int, float, int, bool]


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return self._rng.choices(list(self._buf), k=batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self._buf), f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                self._buf.append(tuple(item))


def rotate_replay(fresh_path: str, stale_path: str, max_total: int = 50_000) -> None:
    """Merge fresh buffer into stale, enforcing a total size cap."""
    fresh, stale = [], []
    try:
        with open(fresh_path) as f:
            fresh = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open(stale_path) as f:
            stale = json.load(f)
    except FileNotFoundError:
        pass

    merged = stale + fresh
    if len(merged) > max_total:
        merged = merged[-max_total:]   # keep most recent

    with open(stale_path, "w") as f:
        json.dump(merged, f)
    with open(fresh_path, "w") as f:
        json.dump([], f)
