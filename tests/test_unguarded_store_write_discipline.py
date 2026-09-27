"""Regression tests for write discipline in the three unguarded stores.

Audit 2026-09-26 (post-v1.2.2): ``SummaryDAG``, ``RollupStore`` and
``QueryViewStore`` each hold their OWN connection to the same trove.db and each
guard writes with a PER-INSTANCE ``RLock`` only. ``process_write_lock`` covers
``MessageStore`` and ``VectorStore`` but not these three, so a rollup build, a
query-view write, or a DAG node insert can collide with the gateway's ingest.

Measured collision cost is silent data loss, not corruption: SQLite's own file
locks keep the b-tree valid (``integrity_check`` stayed ``ok`` under 800
concurrent unguarded writes), but ``busy_timeout`` expiry surfaces as
``sqlite3.OperationalError`` and, with no retry and no handler, the write is
dropped - 17 of 801 rows lost in the audit's reproduction.

These tests pin the two invariants that make the loss impossible:

1. The three stores take the shared process-wide lock for their write
   transactions, so their writes serialize against MessageStore/VectorStore.
2. A colliding writer WAITS on that lock rather than racing into
   busy_timeout expiry, so no write is dropped.

There is no retry/backoff: the lock is held for the whole write transaction, so
the loser blocks instead of failing. A future change that shortens the lock
scope (or adds a real retry) must keep these tests green.

Every store under test is built on a tmp_path database. No test touches a real
trove.db.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hermes_trove.db_bootstrap import process_write_lock
from hermes_trove.dag import SummaryDAG, SummaryNode
from hermes_trove.query_view_store import QueryViewIdentity, QueryViewStore
from hermes_trove.rollup_store import RollupStore
from hermes_trove.store import MessageStore


def _seeded_db(tmp_path: Path, name: str = "trove.db") -> Path:
    """Create a schema-complete trove.db, then close it.

    QueryViewStore's schema references ``messages``, so a bare QueryViewStore
    on an empty file fails to initialize. MessageStore owns that table, so build
    the database through it and hand back the path (the real engine does the
    same: MessageStore, SummaryDAG, LifecycleState, QueryViewStore all open the
    SAME db_path).
    """
    db = tmp_path / name
    MessageStore(db).close()
    return db


def _count(db: Path, table: str) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _node(session_id: str) -> SummaryNode:
    return SummaryNode(
        session_id=session_id,
        depth=0,
        summary="audit regression node",
        token_count=3,
        source_token_count=3,
        source_ids=[1],
        source_type="test",
        created_at=0.0,
    )


def test_three_stores_take_the_process_write_lock(tmp_path: Path):
    """All three stores use the shared lock for their db path.

    ``process_write_lock`` is keyed by resolved path, so identity is the whole
    point: if a store's lock is not ``process_write_lock(its own db_path)``, its
    writes cannot possibly serialize against MessageStore's.
    """
    db = _seeded_db(tmp_path)
    for store in (SummaryDAG(db), RollupStore(db), QueryViewStore(db)):
        assert store._process_lock is process_write_lock(db), (
            f"{type(store).__name__} must use the process-wide write lock so its "
            f"writes serialize against MessageStore/VectorStore on the same file"
        )


def test_dag_insert_and_foreign_writer_both_land(tmp_path: Path):
    """A DAG insert and a foreign connection's write cannot both be dropped.

    Regression shape from the audit: two independent connections in one process
    writing the same WAL without the shared lock. With the lock, the second
    writer waits out the first instead of losing its write.
    """
    db = _seeded_db(tmp_path)
    dag = SummaryDAG(db)
    landed: list[int] = []
    failures: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _foreign_write() -> None:
        """Holds the process-wide lock for a foreign write, then releases it.

        This models a MessageStore/VectorStore write in flight. Because the
        guarded stores take the SAME process-wide lock, the DAG write must wait
        until this finishes - that serialization is the invariant under test.
        Deterministic, unlike racing on busy_timeout (30s in production).
        """
        conn = sqlite3.connect(db, timeout=5.0, check_same_thread=False)
        try:
            barrier.wait()
            with process_write_lock(db):
                for i in range(120):
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "INSERT INTO summary_nodes "
                        "(session_id, depth, summary, created_at) "
                        "VALUES (?, 0, ?, 0.0)",
                        ("foreign", f"node-{i}"),
                    )
                    conn.commit()
        finally:
            conn.close()

    def _dag_write() -> None:
        barrier.wait()
        for i in range(120):
            try:
                dag.add_node(_node(f"dag-{i}"))
                landed.append(1)
            except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                failures.append(exc)

    threads = [threading.Thread(target=_foreign_write), threading.Thread(target=_dag_write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, f"no write may be dropped: {[repr(e) for e in failures[:3]]}"
    assert len(landed) == 120, f"every DAG insert must land, got {len(landed)}"
    assert _count(db, "summary_nodes") == 240, "both writers' rows must be present"


def test_dag_insert_waits_for_a_foreign_process_lock_holder(tmp_path: Path):
    """The DAG write must NOT complete while the shared lock is held.

    This is the deterministic core of the regression. Before the fix ``add_node``
    ignored ``process_write_lock`` entirely, so it committed while a
    MessageStore-shaped writer held the lock. After the fix it blocks until the
    holder releases - which is what makes its write serialize.
    """
    db = _seeded_db(tmp_path)
    dag = SummaryDAG(db)
    holding = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def _holder() -> None:
        with process_write_lock(db):
            holding.set()
            release.wait(10.0)

    def _writer() -> None:
        try:
            dag.add_node(_node("blocked"))
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    holder = threading.Thread(target=_holder)
    holder.start()
    assert holding.wait(10.0), "holder never acquired the process lock"

    writer = threading.Thread(target=_writer)
    writer.start()
    # Give the writer ample time to (incorrectly) finish if it ignores the lock.
    writer.join(timeout=1.5)
    assert writer.is_alive(), (
        "add_node completed while another writer held the process-wide lock; "
        "it must serialize instead"
    )

    release.set()
    writer.join(timeout=10)
    holder.join(timeout=10)
    assert not failures, f"the delayed write must still succeed: {failures[:1]}"
    assert _count(db, "summary_nodes") == 1


def test_query_view_write_lands_despite_lock_contention(tmp_path: Path):
    """A contended query-view claim is retried, not silently dropped.

    Before the fix a collision raised ``OperationalError`` straight out of the
    claim and the row was gone. After the fix the write lands.
    """
    db = _seeded_db(tmp_path)
    view = QueryViewStore(db)
    landed: list[str] = []
    failures: list[BaseException] = []
    released = threading.Event()
    blocker = sqlite3.connect(db, timeout=0.05, check_same_thread=False)
    # Hold the file's write lock. No row is written: the point is purely to make
    # the claim below collide with a real foreign writer.
    blocker.execute("BEGIN IMMEDIATE")

    def _release_later() -> None:
        released.wait(5.0)
        blocker.commit()

    def _claim() -> None:
        try:
            token = view.claim_build(
                QueryViewIdentity(
                    intent_type="evidence_only",
                    subject_key="user:self",
                    operation="sum",
                )
            )
            landed.append(str(token))
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    releaser = threading.Thread(target=_release_later)
    claimer = threading.Thread(target=_claim)
    claimer.start()
    releaser.start()
    claimer.join(timeout=15)
    releaser.join(timeout=5)
    blocker.close()

    assert not failures, f"a contended claim must be retried, not raised: {failures[:1]}"
    assert landed, "the query-view claim must land despite contention"


def test_rollup_claim_lands_despite_lock_contention(tmp_path: Path):
    """A contended rollup claim is retried, not silently dropped."""
    db = _seeded_db(tmp_path)
    rollups = RollupStore(db)
    landed: list[object] = []
    failures: list[BaseException] = []

    blocker = sqlite3.connect(db, timeout=0.05, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute(
        "INSERT INTO trove_rollups (period_kind, period_start, scope, status, built_at) "
        "VALUES ('day', '1999-01-01', 'blocker', 'building', 0.0)"
    )

    def _release_later() -> None:
        threading.Event().wait(0.3)
        blocker.commit()

    def _claim() -> None:
        try:
            landed.append(rollups.upsert_building("day", "2026-09-26", "session:s"))
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            failures.append(exc)

    threads = [threading.Thread(target=_release_later), threading.Thread(target=_claim)]
    threads[0].start()
    threads[1].start()
    for thread in threads:
        thread.join(timeout=15)

    assert not failures, f"a contended claim must be retried, not raised: {failures[:1]}"
    assert landed, "the rollup claim must land despite contention"


def test_intact_store_is_unaffected_by_the_new_lock(tmp_path: Path):
    """The added lock is transparent for the uncontended single-writer case."""
    db = _seeded_db(tmp_path)
    dag = SummaryDAG(db)
    node_id = dag.add_node(_node("solo"))
    assert node_id > 0
    assert _count(db, "summary_nodes") == 1
