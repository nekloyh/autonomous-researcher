"""SQLite-backed checkpointer for resuming graph runs after crash."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB = Path(".checkpoints.db")

_saver: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    """Return a process-wide SqliteSaver bound to .checkpoints.db.

    LangGraph's SqliteSaver normally needs to live inside a `with` block; we
    use a long-lived sqlite3 connection (check_same_thread=False) so the saver
    can be used across the application's lifetime.
    """
    global _saver
    if _saver is not None:
        return _saver
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _saver = SqliteSaver(conn)
    return _saver


def cleanup_old_threads(days: int = 7) -> int:
    """Purge checkpoint rows older than `days`. Returns count removed.

    Best-effort: schema is owned by langgraph-checkpoint-sqlite; we just delete
    rows whose `ts` (Unix seconds) is older than the cutoff if such a column exists.
    """
    if not CHECKPOINT_DB.exists():
        return 0
    cutoff = days * 86400
    with sqlite3.connect(str(CHECKPOINT_DB)) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "DELETE FROM checkpoints WHERE strftime('%s','now') - "
                "strftime('%s', metadata) > ?",
                (cutoff,),
            )
        except sqlite3.OperationalError:
            return 0
        return cur.rowcount or 0
