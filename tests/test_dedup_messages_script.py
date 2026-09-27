"""Tests for scripts/dedup_messages.py — the operator message dedup tool.

Uses a real MessageStore (not a hand-built schema) so the identity
columns, the FTS triggers, and trove_chunk_meta match production.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_trove.store import MessageStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dedup_messages.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("dedup_messages", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dedup_messages"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dedup_module():
    return _load_script()


def _store(tmp_path) -> tuple[MessageStore, Path]:
    db_path = tmp_path / "dedup-test.db"
    store = MessageStore(db_path)
    return store, db_path


def _append(store, session_id, role, content, tool_calls=None, observed=None):
    msg = {"role": role, "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if observed is not None:
        msg["timestamp"] = observed
    return store.append(session_id, msg)


def test_dry_run_removes_nothing(dedup_module, tmp_path):
    """The default must never write — this is a destructive tool."""
    store, db_path = _store(tmp_path)
    try:
        _append(store, "s1", "user", "hello")
        _append(store, "s1", "user", "hello")
        assert dedup_module.main(["--db", str(db_path)]) == 0
    finally:
        store.close()

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    finally:
        conn.close()


def test_collapses_reingested_rows_keeping_earliest(dedup_module, tmp_path):
    """The 8-copies-of-one-message shape: same content, no host timestamp."""
    store, db_path = _store(tmp_path)
    try:
        ids = [_append(store, "s1", "assistant", "**2 cores, 11GB RAM.**") for _ in range(8)]
    finally:
        store.close()

    assert dedup_module.main(["--db", str(db_path), "--apply"]) == 0

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT store_id FROM messages").fetchall()
        assert [r[0] for r in rows] == [ids[0]]
    finally:
        conn.close()


def test_keeps_genuinely_distinct_messages(dedup_module, tmp_path):
    """Different content, role, session, or tool_calls must all survive."""
    store, db_path = _store(tmp_path)
    try:
        _append(store, "s1", "user", "one")
        _append(store, "s1", "user", "two")
        _append(store, "s2", "user", "one")
        _append(store, "s1", "assistant", "one")
        _append(store, "s1", "user", "one", tool_calls=[{"id": "tc1"}])
        # Same content, different observed_at: NOT a duplicate. The identity
        # key includes the timestamp, so both rows must survive.
        _append(store, "s1", "user", "stamped", observed=1_700_000_000.0)
        _append(store, "s1", "user", "stamped", observed=1_700_000_500.0)
    finally:
        store.close()

    assert dedup_module.main(["--db", str(db_path), "--apply"]) == 0

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 7
    finally:
        conn.close()


def test_repoints_chunk_refs_before_delete(dedup_module, tmp_path):
    """A chunk pointing at a victim must follow it to the keeper."""
    store, db_path = _store(tmp_path)
    try:
        keep = _append(store, "s1", "assistant", "referenced body")
        victim = _append(store, "s1", "assistant", "referenced body")
        assert keep != victim
        # trove_chunk_meta is opt-in (vector side-feature); a stock DB has
        # no such table, so the tool must work with and without it.
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS trove_chunk_meta("
            "chunk_id TEXT, identity_hash TEXT, store_id INTEGER, chunk_index INTEGER,"
            " char_start INTEGER, char_end INTEGER, token_estimate INTEGER,"
            " embedded_at TEXT, archived INTEGER)"
        )
        store._conn.execute(
            "INSERT INTO trove_chunk_meta"
            "(chunk_id, identity_hash, store_id, chunk_index, char_start, char_end,"
            " token_estimate, embedded_at, archived) VALUES (?,?,?,?,?,?,?,?,?)",
            ("chunk-1", "hash-1", victim, 0, 0, 10, 5, None, 0),
        )
        store._conn.commit()
    finally:
        store.close()

    assert dedup_module.main(["--db", str(db_path), "--apply"]) == 0

    conn = sqlite3.connect(db_path)
    try:
        survivor = conn.execute("SELECT store_id FROM messages").fetchone()[0]
        assert survivor == keep
        pointed = conn.execute("SELECT store_id FROM trove_chunk_meta").fetchall()
        assert [r[0] for r in pointed] == [keep]
        orphans = conn.execute(
            "SELECT COUNT(*) FROM trove_chunk_meta c LEFT JOIN messages m "
            "ON m.store_id = c.store_id WHERE m.store_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0
    finally:
        conn.close()


def test_is_idempotent(dedup_module, tmp_path):
    """Running twice must be a no-op the second time."""
    store, db_path = _store(tmp_path)
    try:
        _append(store, "s1", "user", "dup")
        _append(store, "s1", "user", "dup")
    finally:
        store.close()

    assert dedup_module.main(["--db", str(db_path), "--apply"]) == 0
    assert dedup_module.main(["--db", str(db_path), "--apply"]) == 0

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_missing_db_is_an_error_not_a_crash(dedup_module, tmp_path):
    assert dedup_module.main(["--db", str(tmp_path / "nope.db")]) == 1
