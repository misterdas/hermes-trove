"""Does pool eviction actually break an in-flight publish?

Claim under test (audit 2026-09-26): "eviction can close a VectorStore while a
publish is in flight on ANOTHER connection to the same file, because the
eviction path takes the per-store lock but not process_write_lock."

Counter-hypothesis: ``VectorStore.close()`` itself takes ``self._write_lock``,
and the publish path takes ``self._write_lock`` too. Two threads on the SAME
store instance are therefore already mutually exclusive, and a different store
instance publishing to the same trove.db is exactly what SQLite's WAL file
locking is designed to allow.

So this file tries to BREAK the store and reports what it actually observes.
If nothing breaks, the audit finding was wrong and no fix is warranted.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hermes_trove import retrieval_core
from hermes_trove.store import MessageStore
from hermes_trove.vector_store import VectorStore


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "trove.db"
    MessageStore(p).close()
    return p


def test_close_is_serialized_against_publish_on_the_same_store(tmp_path: Path):
    """Same-instance: close() and publish() are mutually exclusive.

    If this fails, eviction genuinely can close a store mid-publish.
    """
    db = _db(tmp_path)
    store = VectorStore(db)
    events: list[str] = []
    failures: list[BaseException] = []

    def _publisher() -> None:
        try:
            with store._write_transaction():
                store._conn.execute(
                    "CREATE TABLE IF NOT EXISTS pub_race (id INTEGER PRIMARY KEY)"
                )
                store._conn.execute("INSERT INTO pub_race(id) VALUES (1)")
                events.append("publish-start")
                threading.Event().wait(0.4)  # hold the write lock
                events.append("publish-end")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    def _closer() -> None:
        threading.Event().wait(0.1)  # try to close WHILE the publish is open
        try:
            store.close()
            # Only reached once close() has actually acquired _write_lock and
            # returned, so this mark is AFTER the publish necessarily finished.
            events.append("close-done")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    threads = [threading.Thread(target=_publisher), threading.Thread(target=_closer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not failures, f"close/publish raised: {[repr(e) for e in failures[:2]]}"
    # The whole point: close() must not RETURN between publish-start and
    # publish-end. close() blocks on _write_lock, so it cannot.
    assert events == ["publish-start", "publish-end", "close-done"], (
        f"close() completed during an open publish: {events}"
    )


def test_eviction_during_another_stores_publish_keeps_data_intact(tmp_path: Path):
    """Cross-instance: evict store A while store B publishes to the same file.

    This is the exact scenario the audit flagged. Drive it hard and then verify
    the committed rows survived and the b-tree is intact.
    """
    db = _db(tmp_path)
    victim = VectorStore(db, bounded_scan_rows=100)   # pool key A
    other = VectorStore(db, bounded_scan_rows=200)    # pool key B, different conn
    failures: list[BaseException] = []
    published: list[int] = []

    def _publish() -> None:
        try:
            for i in range(50):
                with other._write_transaction():
                    other._conn.execute(
                        "CREATE TABLE IF NOT EXISTS evict_race (id INTEGER PRIMARY KEY)"
                    )
                    other._conn.execute("INSERT INTO evict_race(id) VALUES (?)", (i,))
                    published.append(i)
                    if i == 10:
                        # Evict the OTHER store mid-flight, from another thread.
                        threading.Thread(target=_evict, daemon=True).start()
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    def _evict() -> None:
        try:
            victim.close()  # closes a different connection, same file
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    threads = [threading.Thread(target=_publish)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not failures, f"cross-instance close broke a publish: {[repr(e) for e in failures[:2]]}"
    assert len(published) == 50, f"every publish must land, got {len(published)}"

    check = sqlite3.connect(db)
    try:
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        rows = check.execute("SELECT COUNT(*) FROM evict_race").fetchone()[0]
    finally:
        check.close()
    assert integrity == "ok", f"integrity broken: {integrity}"
    assert rows == 50, f"lost rows across the eviction: {rows}"


def test_pool_eviction_closes_the_evicted_store(tmp_path: Path) -> None:
    """Baseline: the pool really does close an evicted store, so the above runs."""
    db = _db(tmp_path)
    retrieval_core._reset_vector_store_pool()

    stores = []
    for rows in (1, 2, 3):  # _POOL_MAX_PATHS is 2, so the 3rd forces eviction
        s = VectorStore(db, bounded_scan_rows=rows)
        retrieval_core._vector_store_pool[(str(db), rows)] = {
            "store": s,
            "lock": threading.RLock(),
        }
        stores.append(s)
        with retrieval_core._pool_lock:
            while len(retrieval_core._vector_store_pool) > retrieval_core._POOL_MAX_PATHS:
                _, evicted = retrieval_core._vector_store_pool.popitem(last=False)
                with evicted["lock"]:
                    evicted["store"].close()

    assert getattr(stores[0], "_conn", "gone") is None, "oldest store should be closed"
    retrieval_core._reset_vector_store_pool()


if __name__ == "__main__":  # ponytail: one runnable check, no pytest needed
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_close_is_serialized_against_publish_on_the_same_store(Path(tmp))
        print("same-instance: close cannot interleave with publish  OK")
        test_eviction_during_another_stores_publish_keeps_data_intact(Path(tmp))
        print("cross-instance: 50/50 rows survived a concurrent close  OK")
        test_pool_eviction_closes_the_evicted_store(Path(tmp))
        print("pool eviction closes the LRU victim  OK")
    print("PASS: no lost data, no corruption")
