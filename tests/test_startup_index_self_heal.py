"""Tests for the startup index self-heal and transient
ingest retry (multi-process WAL split-brain hardening, Sep 2026).

Production fault being covered: ``sqlite_autoindex_metadata_1`` loses a row
entry while the row survives. ``PRAGMA quick_check`` misses it;
``PRAGMA integrity_check`` reports ``row N missing from index`` /
``wrong # of entries in index``. The store must heal that in place with
REINDEX on open — never rename or move the database file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_trove.sqlite_util import (
    _is_index_corruption_detail,
    startup_index_self_heal,
)
from hermes_trove.store import _prepare_private_sqlite_storage


PROD_DETAILS = [
    "row 5 missing from index sqlite_autoindex_metadata_1",
    "wrong # of entries in index sqlite_autoindex_metadata_1",
]


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _StubConn:
    """Replay canned integrity_check responses; count REINDEX calls."""

    def __init__(self, script):
        self.script = list(script)
        self.reindex_calls = 0

    def execute(self, sql):
        op = sql.strip().upper()
        assert op in ("PRAGMA INTEGRITY_CHECK", "REINDEX"), sql
        if op == "REINDEX":
            self.reindex_calls += 1
            return _Rows([])
        return _Rows([(d,) for d in self.script.pop(0)])


class TestIndexCorruptionDetailMatcher:
    def test_matches_production_strings(self):
        for detail in PROD_DETAILS:
            assert _is_index_corruption_detail(detail), detail

    def test_rejects_non_index_damage(self):
        assert not _is_index_corruption_detail("ok")
        assert not _is_index_corruption_detail("Page 12 is never used")
        assert not _is_index_corruption_detail("*** in database main ***")
        assert not _is_index_corruption_detail("")


class TestStartupIndexSelfHeal:
    def test_healthy_db_is_noop(self, tmp_path: Path):
        db = tmp_path / "healthy.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO metadata VALUES ('a','1'),('b','2')")
            conn.commit()
            assert startup_index_self_heal(conn) == {"healed": False, "details": []}
        finally:
            conn.close()

    def test_production_details_heal_and_reindex_once(self):
        conn = _StubConn([PROD_DETAILS, ["ok"]])
        assert startup_index_self_heal(conn) == {"healed": True, "details": []}
        assert conn.reindex_calls == 1

    def test_persistent_damage_reports_without_heal(self):
        conn = _StubConn([PROD_DETAILS, PROD_DETAILS])
        result = startup_index_self_heal(conn)
        assert result == {"healed": False, "details": PROD_DETAILS}
        assert conn.reindex_calls == 1

    def test_non_index_damage_never_reindexes(self):
        conn = _StubConn([["*** in database main ***", "Page 12 is never used"]])
        result = startup_index_self_heal(conn)
        assert result["healed"] is False
        assert conn.reindex_calls == 0


class TestOpenNeverUnlinksSidecars:
    """Opening a store must not unlink a sibling's ``-shm``.

    A t3 "stale shm cleanup" used to do exactly that before every open, and
    unlinking the shared WAL-index region under a live sibling *is* the
    split-brain it was meant to cure. It was removed: SQLite rebuilds a stale
    ``-shm`` itself, so the delete bought nothing and cost a WAL. These pin
    the removal — re-adding an unlink here fails the first test.
    """

    def _wal_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(20)])
        conn.commit()
        conn.close()
        return db

    def test_prepare_storage_leaves_shm_alone(self, tmp_path: Path):
        db = self._wal_db(tmp_path)
        shm = db.with_name(db.name + "-shm")
        shm.write_bytes(b"\x00" * 32768)  # stale region, no -wal on disk
        _prepare_private_sqlite_storage(db)
        assert shm.exists(), "open unlinked a sibling's -shm (t3 split-brain)"

    def test_sqlite_rebuilds_stale_shm_without_help(self, tmp_path: Path):
        db = self._wal_db(tmp_path)
        shm = db.with_name(db.name + "-shm")
        shm.write_bytes(b"\x00" * 32768)
        db.with_name(db.name + "-wal").write_bytes(b"")  # 0-byte wal boundary
        conn = sqlite3.connect(db, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("INSERT INTO t VALUES(999)")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 21
        conn.close()
        check = sqlite3.connect(db)
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        check.close()


class TestRetryWorthyIngestError:
    """Predicate contract: lock contention retries; corruption never does."""

    @staticmethod
    def _engine_predicate():
        from hermes_trove import engine as engine_module

        # Unbound method: call as pred(None, exc) — self unused by predicate.
        unbound = engine_module.TROVEEngine._is_retry_worthy_ingest_error
        assert callable(unbound)
        return lambda exc: unbound(None, exc)

    def test_locked_errors_retry(self):
        pred = self._engine_predicate()
        assert pred(sqlite3.OperationalError("database is locked")) is True
        assert pred(sqlite3.OperationalError("database table is locked")) is True

    def test_corruption_never_retries(self):
        pred = self._engine_predicate()
        assert pred(sqlite3.DatabaseError("database disk image is malformed")) is False
        assert pred(sqlite3.OperationalError("disk I/O error")) is False
        assert pred(sqlite3.OperationalError("no such table: foo")) is False
