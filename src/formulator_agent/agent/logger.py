"""Tiny JSONL observability log."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent / "run_logs" / "logs.jsonl"
# The service runs each ticket on its own thread. One writer at a time keeps
# two events from interleaving inside one line.
_lock = threading.Lock()


class ObsLogger:
    def log_event(self, request_id: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "request_id": request_id,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
