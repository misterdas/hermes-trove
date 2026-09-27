"""Every store that writes to trove.db must serialize on ONE process-wide lock.

A per-instance lock only guards that instance's connection. When several
stores hold their own connections on the same db_path, a write transaction in
one can collide with another's: the loser waits out busy_timeout (30s default)
and then raises OperationalError with the write lost. SQLite's own file locks
coordinate *processes*, never connections inside one process, so the shared
lock in ``db_bootstrap.process_write_lock`` is the only thing that serializes
them. This test fails the moment a store regresses to a private RLock.
"""
from __future__ import annotations

import threading


from hermes_trove.assertion_store import AssertionStore
from hermes_trove.dag import SummaryDAG
from hermes_trove.db_bootstrap import process_write_lock
from hermes_trove.lifecycle_state import LifecycleStateStore
from hermes_trove.query_view_store import QueryViewStore
from hermes_trove.rollup_store import RollupStore
from hermes_trove.store import MessageStore
from hermes_trove.trajectory_store import CorpusIdentity, TrajectoryStore
from hermes_trove.vector_store import VectorStore


def _identity() -> CorpusIdentity:
    return CorpusIdentity("dataset", "rev1", "commit1", "tier1", "domain")


def _build_stores(db_path):
    return {
        "MessageStore": (MessageStore(str(db_path)), "_process_lock"),
        "SummaryDAG": (SummaryDAG(str(db_path)), "_process_lock"),
        "RollupStore": (RollupStore(str(db_path)), "_process_lock"),
        "QueryViewStore": (QueryViewStore(str(db_path)), "_process_lock"),
        "AssertionStore": (AssertionStore(str(db_path)), "_write_lock"),
        "LifecycleStateStore": (LifecycleStateStore(str(db_path)), "_lock"),
        "TrajectoryStore": (
            TrajectoryStore(
                str(db_path), identity=_identity(), asset_root=db_path.parent / "assets"
            ),
            "_lock",
        ),
    }


def test_all_stores_share_one_process_write_lock(tmp_path):
    db_path = tmp_path / "trove.db"
    canonical = process_write_lock(str(db_path))

    offenders = []
    stores = _build_stores(db_path)
    for name, (store, attr) in stores.items():
        lock = getattr(store, attr, None)
        if lock is not canonical:
            offenders.append(f"{name}.{attr}")

    assert not offenders, (
        "these stores write trove.db on a private lock, so their write "
        "transactions cannot serialize against the gateway's ingest: "
        + ", ".join(offenders)
    )


def test_vector_store_write_transaction_takes_the_shared_lock(tmp_path):
    """VectorStore acquires the canonical lock by path inside its txn."""
    db_path = tmp_path / "trove.db"
    store = VectorStore(str(db_path))
    canonical = process_write_lock(str(db_path))

    entered = threading.Event()

    def writer() -> None:
        with store._write_transaction():
            entered.set()

    with canonical:  # hold it: the writer must NOT get in
        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        assert not entered.wait(0.5), (
            "VectorStore._write_transaction entered while the process-wide "
            "write lock was held - it is not serializing on the shared lock"
        )
    thread.join(timeout=5.0)
    assert entered.is_set(), "writer never proceeded after the lock was released"
