"""Promotion state transition log."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

EVENT_PROMOTE = "PROMOTE"
EVENT_KEEP_LIGHT = "KEEP_LIGHT"
EVENT_DEMOTE = "DEMOTE"
EVENT_DETAIL_USED = "DETAIL_USED"


@dataclass
class StateLogEntry:
    event: str
    node_id: str
    query_idx: int
    query_id: str = ""
    score: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


class StateLog:
    """Append-only JSONL state transition log."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: StateLogEntry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def record(
        self,
        event: str,
        node_id: str,
        query_idx: int,
        query_id: str = "",
        score: float = 0.0,
        reason: str = "",
        **extra: Any,
    ) -> None:
        self.append(
            StateLogEntry(
                event=event,
                node_id=node_id,
                query_idx=query_idx,
                query_id=query_id,
                score=score,
                reason=reason,
                extra=extra,
            )
        )
