"""Lightweight structured logger. One JSON line per event to stdout + an
append-only jsonl file under `logs/`.

Naming `logger.py` (not `logging.py`) to avoid shadowing the stdlib module."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")


def _ensure_dir() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    return LOG_DIR


def log_event(node: str, session_id: str = "-", **fields: Any) -> None:
    """Emit a structured event. Best-effort; never raises."""
    record = {"node": node, "session_id": session_id, **fields}
    line = json.dumps(record, default=str, ensure_ascii=False)
    print(line, file=sys.stdout, flush=True)
    try:
        path = _ensure_dir() / f"run_{session_id}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass  # logging failures are not user-visible
