"""Regression: each named identity-index migration must be reachable
independently, not nested inside the v1 gate.

The v3 backstop (idx_msg_identity_ts_fallback) shipped nested inside
`if not _has_named_migration_step(conn, "messages_identity_index_v1")`.
Any store that had already completed v1 therefore skipped v3 forever, so
the NULL-observed_at backstop never existed on a live install and
timestamp-less duplicate rows were written on every replay.

Each test here builds a store whose migration state looks like a long-lived
install, then asserts the backstop still lands.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_trove import db_bootstrap as db


def _bare_conn() -> sqlite3.Connection:
    """Minimal messages table with the columns the migration reads."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, ingested_at REAL,
            observed_at REAL)"""
    )
    db.ensure_migration_state_table(conn)
    return conn


def _index_names(conn) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_msg_identity%'"
        )
    }


def _steps(conn) -> set:
    return {r[0] for r in conn.execute("SELECT step_name FROM trove_migration_state")}


def _run_migration(conn):
    """Invoke exactly what a store open invokes."""
    db.run_message_identity_migration(conn)
    conn.commit()


def test_v3_runs_on_a_store_that_already_completed_v1():
    """The regression: v1 done + v3 missing must still build the backstop."""
    conn = _bare_conn()
    _run_migration(conn)
    assert "messages_identity_index_v1" in _steps(conn)

    # Roll the store back to the long-lived-install shape: v1 and v2 recorded,
    # the v3 backstop never created.
    conn.execute("DROP INDEX IF EXISTS idx_msg_identity_ts_fallback")
    conn.execute("DELETE FROM trove_migration_state WHERE step_name = "
                 "'messages_identity_index_v3'")
    conn.commit()

    assert "idx_msg_identity_ts_fallback" not in _index_names(conn)

    _run_migration(conn)

    assert "idx_msg_identity_ts_fallback" in _index_names(conn), (
        "v3 stayed nested inside the v1 gate: a store that already completed "
        "v1 never builds the NULL-observed_at backstop"
    )
    assert "messages_identity_index_v3" in _steps(conn)


def test_each_step_is_gated_only_on_its_own_marker():
    """A store with v2 recorded but v1/v3 missing must still reach v1 and v3."""
    conn = _bare_conn()
    _run_migration(conn)
    conn.execute("DELETE FROM trove_migration_state")
    conn.execute(
        "INSERT INTO trove_migration_state (step_name, completed_at) "
        "VALUES ('messages_identity_index_v2', 1.0)"
    )
    conn.commit()

    _run_migration(conn)

    steps = _steps(conn)
    assert "messages_identity_index_v1" in steps
    assert "messages_identity_index_v3" in steps


def test_v3_backstop_rejects_a_null_timestamp_replay():
    """The index that failed to land must do its job once it does."""
    conn = _bare_conn()
    _run_migration(conn)
    conn.execute("DROP INDEX IF EXISTS idx_msg_identity_ts_fallback")
    conn.execute("DELETE FROM trove_migration_state WHERE step_name = "
                 "'messages_identity_index_v3'")
    conn.commit()
    _run_migration(conn)

    row = (
        "INSERT INTO messages (session_id, role, content, tool_call_id, "
        "tool_calls, ingested_at, observed_at) VALUES (?,?,?,?,?,?,?)"
    )
    conn.execute(row, ("s", "user", "hello", None, None, 1000.0, None))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(row, ("s", "user", "hello", None, None, 1000.0, None))

    # A genuine repeat a second later must still survive.
    conn.execute(row, ("s", "user", "hello", None, None, 1001.0, None))
