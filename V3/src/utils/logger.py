from __future__ import annotations
import json
import os
from typing import Any, Dict, List


class TrainingLogger:
    """Logs episode-level training metrics to JSON."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._records: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    def log(self, **kwargs: Any) -> None:
        self._records.append(kwargs)

    def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)

    def __len__(self) -> int:
        return len(self._records)
