"""``embed status`` must not count lease rows from an ARCHIVED profile.

Repro (2026-09-27, live store): the chunk profile was switched from bge-base
(768-dim) to bge-small (384-dim), which archived the old profile but left its
64 lease rows in ``trove_embedding_backfill_inflight`` under the OLD
identity_hash. ``embed status`` printed ``in_flight: 64`` while
``/trove embed backfill`` - which does filter by identity - correctly printed
0. Two counters, one table, one of them wrong, and the report said ``status: ok``.

Drives the real ``embed status`` report end to end. A helper-level test cannot
catch this: the fix is a MISSING ARGUMENT at one call site, so the test has to
go through that call site.

Schema follows tests/test_chunk_backfill.py - no production code creates this
table, it is written by the backfill worker at runtime.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from hermes_trove.command import handle_trove_command  # noqa: E402
from hermes_trove.config import TROVEConfig  # noqa: E402
from hermes_trove.engine import TROVEEngine  # noqa: E402
from hermes_trove.vector_store import VectorStore  # noqa: E402


def _engine(tmp_path) -> TROVEEngine:
    return TROVEEngine(
        config=TROVEConfig(
            database_path=str(tmp_path / "status.db"),
            embeddings_enabled=True,
        ),
        hermes_home=str(tmp_path / "home"),
    )


def _seed(engine: TROVEEngine) -> None:
    conn = sqlite3.connect(engine._store.db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
               store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
               source TEXT DEFAULT '', role TEXT NOT NULL, content TEXT,
               timestamp REAL NOT NULL)"""
    )
    conn.executemany(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        [(i, "s", "test", "user", f"message {i} " * 40, 0.0) for i in range(3)],
    )
    conn.commit()
    conn.close()
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("model-a", "ollama", 2, task="chunk")
    finally:
        store.close()


def _leases(engine: TROVEEngine, identity: str, count: int) -> None:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trove_embedding_backfill_inflight (
                   embedded_id TEXT, identity_hash TEXT, state TEXT,
                   updated_at REAL,
                   PRIMARY KEY(embedded_id, identity_hash)
               )"""
        )
        conn.executemany(
            "INSERT OR REPLACE INTO trove_embedding_backfill_inflight"
            "(embedded_id, identity_hash, state, updated_at) VALUES(?, ?, 'uncertain', 1.0)",
            [(f"{i}:0", identity) for i in range(count)],
        )
        conn.commit()
    finally:
        conn.close()


def _identities(engine: TROVEEngine) -> dict[str, str]:
    conn = sqlite3.connect(engine._store.db_path)
    try:
        rows = conn.execute(
            "SELECT active, identity_hash FROM trove_embedding_profile WHERE task='chunk'"
        ).fetchall()
    finally:
        conn.close()
    return {("active" if a else "archived"): h for a, h in rows}


def test_status_hides_archived_identity_leases(tmp_path):
    """64 dead leases from a retired model must not read as in-flight work."""
    engine = _engine(tmp_path)
    _seed(engine)

    # Model switch: register_profile archives the previous chunk profile
    # automatically, so model-a's identity becomes the archived one.
    first = _identities(engine)["active"]
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile("model-old", "ollama", 2, task="chunk")
    finally:
        store.close()

    ids = _identities(engine)
    assert "archived" in ids and "active" in ids, f"expected both, got {list(ids)}"
    assert first == ids["archived"], "the first profile should now be archived"
    # 64 leases stranded under the RETIRED identity.
    _leases(engine, ids["archived"], 64)

    result = handle_trove_command("embed status", engine)

    assert "in_flight: 64" not in result, (
        f"embed status counted lease rows from an ARCHIVED profile:\n{result}"
    )
    assert "in_flight: 0" in result, f"expected a clean count:\n{result}"


def test_status_still_reports_live_leases(tmp_path):
    """The fix must not blind the report: ACTIVE-identity leases still show."""
    engine = _engine(tmp_path)
    _seed(engine)
    ids = _identities(engine)
    _leases(engine, ids["active"], 5)

    result = handle_trove_command("embed status", engine)
    assert "in_flight: 5" in result, f"active leases must still be counted:\n{result}"


if __name__ == "__main__":  # ponytail: runnable, no pytest needed
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        eng = _engine(Path(tmp))
        _seed(eng)
        store = VectorStore(eng._store.db_path, config=eng._config)
        try:
            store.register_profile("model-old", "ollama", 2, task="chunk")
        finally:
            store.close()
        c = sqlite3.connect(eng._store.db_path)
        c.execute(
            "UPDATE trove_embedding_profile SET active=0 WHERE task='chunk' AND active=1"
        )
        c.commit()
        c.close()
        _leases(eng, _identities(eng)["archived"], 64)
        out = handle_trove_command("embed status", eng)
        assert "in_flight: 64" not in out, out
        assert "in_flight: 0" in out, out
        print("64 archived-model leases no longer read as in-flight  OK")
    print("PASS")
