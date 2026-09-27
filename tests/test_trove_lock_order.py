"""Cross-store lock-order regression: AB-BA deadlock must be impossible.

Before this fix the four trove.db writers nested their two locks in different
orders (MessageStore/DAG/Rollup took the per-instance lock first,
QueryViewStore took the process-wide lock first). Mixed order between two locks
held for different durations is the classic AB-BA deadlock shape: store A holds
its own lock and waits for the process lock while store B holds the process lock
and waits for its own.

These tests pin the invariant two ways:

1. Every store declares the SAME order: per-instance lock, then process lock.
2. A query-view write and a DAG write actually interleave to completion rather
   than wedging, when driven from two threads on one database file.

Both use tmp_path databases. No real trove.db is touched.
"""
from __future__ import annotations

import threading
from pathlib import Path

from hermes_trove.dag import SummaryDAG, SummaryNode
from hermes_trove.db_bootstrap import process_write_lock
from hermes_trove.query_view_store import QueryViewIdentity, QueryViewStore
from hermes_trove.rollup_store import RollupStore
from hermes_trove.store import MessageStore


def _seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "trove.db"
    MessageStore(db).close()
    return db


def _node(session_id: str) -> SummaryNode:
    return SummaryNode(
        session_id=session_id, depth=0, summary="lock order", token_count=1,
        source_token_count=1, source_ids=[1], source_type="test", created_at=0.0,
    )


def test_every_store_takes_per_instance_lock_before_process_lock(tmp_path: Path):
    """All four trove.db writers declare the SAME lock order: own, then process.

    A runtime probe cannot detect this: both locks are RLocks, so they re-enter
    in either order and no assertion on acquire/release behaviour can tell a
    consistent order from a mixed one. The defect only shows up as a real AB-BA
    deadlock when two stores hold their two locks in OPPOSITE order at the same
    time, which needs a third lock to interleave and is not reliably
    reproducible in a unit test.

    So pin the convention where it actually lives: the source. ``_write_lock``
    must appear before ``_process_lock`` in each writer's transaction guard.
    """
    import inspect

    from hermes_trove import dag as dag_mod
    from hermes_trove import query_view_store as qvs_mod
    from hermes_trove import rollup_store as rollup_mod
    from hermes_trove import store as store_mod

    guards = {
        "MessageStore.write_guard": inspect.getsource(store_mod.MessageStore.write_guard),
        "SummaryDAG.write_guard": inspect.getsource(dag_mod.SummaryDAG.write_guard),
        "RollupStore._write_transaction": inspect.getsource(
            rollup_mod.RollupStore._write_transaction
        ),
        "QueryViewStore._write_transaction": inspect.getsource(
            qvs_mod.QueryViewStore._write_transaction
        ),
    }
    for name, src in guards.items():
        # MessageStore resolves the process lock to a local (``lock = ...``)
        # then enters it; the others name it inline. Match both shapes, and
        # keep only the ``with`` lines so docstrings/comments cannot skew it.
        body = "\n".join(
            line.strip()
            for line in src.splitlines()
            if line.strip().startswith(("with ", "lock ="))
        )
        assert body, f"{name}: no lock acquisition found - update this pin"
        takes_process = "_process_lock" in body or "with lock" in body
        assert takes_process, f"{name} must take the process-wide write lock"
        own_pos = min(
            (pos for pos in (body.find("_write_lock"), body.find("_db_lock")) if pos != -1),
            default=-1,
        )
        process_pos = min(
            (pos for pos in (body.find("_process_lock"), body.find("with lock")) if pos != -1),
            default=-1,
        )
        if own_pos != -1:
            assert own_pos < process_pos, (
                f"{name} nests the locks out of order: per-instance lock at "
                f"{own_pos}, process lock at {process_pos}. Own lock must come first."
            )

    db = _seeded_db(tmp_path)
    for store in (MessageStore(db), SummaryDAG(db), RollupStore(db), QueryViewStore(db)):
        assert store._process_lock is process_write_lock(db), (
            f"{type(store).__name__} must share the process-wide lock for its db path"
        )


def test_query_view_and_dag_writes_interleave_without_deadlock(tmp_path: Path):
    """Two stores writing the same file from two threads both complete.

    Drives real contention on purpose: a query-view claim loop and a DAG insert
    loop, alternating, with a hard join timeout. A mixed lock order would wedge
    here; a consistent one drains.
    """
    db = _seeded_db(tmp_path)
    view = QueryViewStore(db)
    dag = SummaryDAG(db)
    errors: list[BaseException] = []
    done: list[str] = []

    def _views() -> None:
        try:
            for i in range(40):
                view.claim_build(
                    QueryViewIdentity(
                        intent_type="evidence_only",
                        subject_key=f"user:{i}",
                        operation="sum",
                    )
                )
            done.append("views")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    def _nodes() -> None:
        try:
            for i in range(40):
                dag.add_node(_node(f"lock-{i}"))
            done.append("nodes")
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_views), threading.Thread(target=_nodes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        # A deadlock shows up here as a thread that never finishes.
        thread.join(timeout=30)

    assert all(not t.is_alive() for t in threads), "a writer deadlocked"
    assert not errors, f"writers must not error: {[repr(e) for e in errors[:2]]}"
    # Completion ORDER is scheduling-dependent; only membership is meaningful.
    assert sorted(done) == ["nodes", "views"], f"both writers must finish, got {done}"

    view.close()
    dag.close()


def test_nested_query_view_transaction_still_uses_savepoints(tmp_path: Path):
    """Reentrancy survives the lock-order change.

    Both locks are RLocks, so re-entering ``_write_transaction`` nests cleanly
    and the savepoint depth logic must still roll back only inner work.
    """
    db = _seeded_db(tmp_path)
    view = QueryViewStore(db)
    try:
        with view._write_transaction():
            view._conn.execute("CREATE TABLE order_test (id INTEGER PRIMARY KEY)")
            view._conn.execute("INSERT INTO order_test(id) VALUES (1)")
            with view._write_transaction():
                view._conn.execute("INSERT INTO order_test(id) VALUES (2)")
        rows = view._conn.execute("SELECT id FROM order_test ORDER BY id").fetchall()
        assert [r[0] for r in rows] == [1, 2]
    finally:
        view.close()
