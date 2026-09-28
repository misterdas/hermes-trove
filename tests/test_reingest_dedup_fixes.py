"""Regression checks for the gateway-restart re-ingest fixes.

Three fixes are covered:

1. ``ReconcileMixin._front_anchored_cursor_for_store_head`` and
   ``_membership_cursor_for_store_head`` -- the cursors that stop a restart
   from re-ingesting the live conversation.  Duplicates land interleaved, so
   membership (order-blind) is what actually works; the ordered front matcher
   only helps when the replay starts at the head of stored history.
2. ``db_bootstrap.ensure_message_identity_ts_fallback_index`` -- a NULL
   ``observed_at`` is distinct from every other NULL in the primary index, so
   timestamp-less replay rows bypassed uniqueness entirely.
3. ``command._message_duplicate_cleanup_rows`` -- the doctor apply must catch
   timestamp-less duplicates while leaving legitimate short repeats alone.
"""
import json
import pathlib
import sqlite3
import sys

from hermes_trove.config import TROVEConfig
from hermes_trove.engine import TROVEEngine


def _load_command_symbols():
    """Import command.py's dup helpers without the package's relative imports."""
    src = pathlib.Path(__file__).resolve().parents[1] / "command.py"
    text = src.read_text()
    start = text.index("_MESSAGE_DUP_IDENTITY = (")
    end = text.index("def _doctor_duplicate_text")
    ns: dict = {}
    exec(compile(text[start:end], "<command-extract>", "exec"), ns)
    return ns


def _engine(tmp_path):
    return TROVEEngine(
        config=TROVEConfig(), hermes_home=str(tmp_path / "hermes"),
    )


def _identity(role, content, tool_call_id="", tool_calls=""):
    return {"role": role, "content": content, "tool_call_id": tool_call_id,
            "tool_calls": tool_calls}


def _stored(engine, messages):
    """Stored-row dicts, as ``get_session_messages`` would return them."""
    return [dict(m) for m in messages]


# --- 1. front-anchored cursor -------------------------------------------------


def _engine(tmp_path):
    return TROVEEngine(
        config=TROVEConfig(), hermes_home=str(tmp_path / "hermes"),
    )


def _identity(role, content, tool_call_id="", tool_calls=""):
    return {"role": role, "content": content, "tool_call_id": tool_call_id,
            "tool_calls": tool_calls}


def _stored(engine, messages):
    """Stored-row dicts, as ``get_session_messages`` would return them."""
    return [dict(m) for m in messages]


# --- 1. front-anchored cursor -------------------------------------------------


def _replayed(conversation, times=2):
    """Stored history shaped like the real damage: the conversation re-sent."""
    return [dict(m) for m in conversation * times]


def test_front_anchored_cursor_matches_mid_conversation_replay(tmp_path):
    """The stored history IS the live conversation, already re-sent once.

    A tail-suffix matcher cannot see this (the tail is part of the copy), so
    the cursor has to come from the front -- and the ordered-repeat proof
    confirms the prefix is genuinely duplicated, not merely coincidental.
    """
    engine = _engine(tmp_path)
    conversation = [
        _identity("user", f"m{i}") for i in range(10)
    ]
    cursor = engine._front_anchored_cursor_for_store_head(
        conversation, _replayed(conversation),
    )
    assert cursor == len(conversation)


def test_front_anchored_cursor_requires_repeat_evidence(tmp_path):
    """A front match with NO duplicate behind it must not claim a skip.

    Stored history can legitimately start with what the host just sent (a
    rebound replaying a prefix about to be extended).  Skipping there would
    drop the new rows that follow, so the cursor stays undecided.
    """
    engine = _engine(tmp_path)
    conversation = [_identity("user", f"m{i}") for i in range(10)]
    assert engine._front_anchored_cursor_for_store_head(
        conversation, _replayed(conversation, times=1),
    ) is None


def test_front_anchored_cursor_stops_at_first_new_message(tmp_path):
    """A genuinely new tail message must not be skipped."""
    engine = _engine(tmp_path)
    stored = [_identity("user", "a"), _identity("assistant", "b")]
    incoming = [_identity("user", "a"), _identity("assistant", "b"),
                _identity("user", "brand new")]
    # No repeat evidence in a two-row store -> undecided, never a wrong skip.
    assert engine._front_anchored_cursor_for_store_head(
        incoming, _replayed(stored),
    ) is None


def test_front_anchored_cursor_preserves_legitimate_repeats(tmp_path):
    """"ok" twice in a row is two real messages, not a replay.

    Order-sensitivity is what keeps this: the second "ok" binds to the second
    stored "ok" because alignment is positional, not set-membership.
    """
    engine = _engine(tmp_path)
    history = [
        _identity("user", "ok"),
        _identity("user", "ok"),
        _identity("assistant", "sure"),
    ]
    long_history = history * 4
    cursor = engine._front_anchored_cursor_for_store_head(
        [dict(m) for m in history], _replayed(history, times=4),
    )
    assert cursor == 3
    assert len(long_history) == 12


def test_front_anchored_cursor_returns_none_when_no_overlap(tmp_path):
    engine = _engine(tmp_path)
    assert engine._front_anchored_cursor_for_store_head(
        [_identity("user", "totally different")],
        _stored(engine, [_identity("assistant", "unrelated")]),
    ) is None


# --- 1b. membership cursor (the real fix) -------------------------------------


def _interleaved(conversation, times=2):
    """Duplicates woven between originals -- the shape actually observed."""
    out = []
    for _ in range(times):
        for i, m in enumerate(conversation):
            out.append(dict(m))
            if i % 2 == 0:
                out.append(dict(m))
    return out


def test_membership_cursor_matches_interleaved_replay(tmp_path):
    """Re-ingest that no ordered matcher can see still advances the cursor."""
    engine = _engine(tmp_path)
    conversation = [_identity("user", f"m{i}") for i in range(10)]
    cursor = engine._membership_cursor_for_store_head(
        conversation, _interleaved(conversation),
    )
    assert cursor == len(conversation)


def test_membership_cursor_stops_before_genuinely_new_message(tmp_path):
    """The new tail must survive: it consumes no stored copy, so the walk stops."""
    engine = _engine(tmp_path)
    conversation = [_identity("user", f"m{i}") for i in range(10)]
    incoming = conversation + [_identity("assistant", "brand new")]
    cursor = engine._membership_cursor_for_store_head(
        incoming, _interleaved(conversation),
    )
    assert cursor == len(conversation)


def test_membership_cursor_requires_surplus_evidence(tmp_path):
    """A healthy store must be left alone.

    Without this gate membership would fire on any store that happens to hold
    the incoming window, and would be free to shrink the cursor on a healthy
    session.  The surplus proof can only pass on a store already re-ingesting.
    """
    engine = _engine(tmp_path)
    conversation = [_identity("user", f"m{i}") for i in range(10)]
    assert engine._membership_cursor_for_store_head(
        conversation, _stored(engine, conversation),
    ) is None


def test_membership_cursor_never_exceeds_verified_membership(tmp_path):
    """The safety property: a skipped message is always verified stored.

    Walks the same store through the matcher and through an independent count;
    the matcher may only ever return a cursor at or below the count that proves
    each skipped row has an unconsumed stored copy.  That is what makes an
    over-eager cursor re-ingest rather than silently drop.
    """
    engine = _engine(tmp_path)
    cases = [
        ([_identity("user", "a")], [_identity("user", "a"), _identity("user", "b")]),
        ([_identity("user", "ok")], [_identity("user", "ok"), _identity("user", "ok")]),
        ([], [_identity("user", "only new")]),
        ([_identity("user", f"m{i}") for i in range(8)],
         [_identity("user", f"m{i}") for i in range(4)] + [_identity("user", "x")]),
    ]
    for stored, incoming in cases:
        stored = _interleaved(stored) if stored else []
        cursor = engine._membership_cursor_for_store_head(incoming, stored)
        pool = {}
        for row in stored:
            key = (row["role"], row["content"], row.get("tool_call_id") or "",
                   row.get("tool_calls") or "")
            pool[key] = pool.get(key, 0) + 1
        safe = 0
        for msg in incoming:
            key = (msg["role"], msg["content"], msg.get("tool_call_id") or "",
                   msg.get("tool_calls") or "")
            if pool.get(key, 0) <= 0:
                break
            pool[key] -= 1
            safe += 1
        assert cursor is None or cursor <= safe


# --- 2. NULL observed_at identity index --------------------------------------


def _indexed_conn() -> sqlite3.Connection:
    from hermes_trove.db_bootstrap import (
        ensure_message_identity_index,
        ensure_message_identity_ts_fallback_index,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, ingested_at REAL,
            observed_at REAL)"""
    )
    ensure_message_identity_index(conn)
    ensure_message_identity_ts_fallback_index(conn)
    return conn


_INSERT = (
    "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, "
    "ingested_at, observed_at) VALUES (?,?,?,?,?,?,?)"
)


def test_ts_fallback_index_rejects_same_second_null_timestamp_replay():
    """Without the fallback index a NULL observed_at can never collide."""
    conn = _indexed_conn()
    conn.execute(_INSERT, ("s", "user", "ok", None, None, 100.0, None))
    with pytest_raises_integrity():
        conn.execute(_INSERT, ("s", "user", "ok", None, None, 100.0, None))


def test_ts_fallback_index_keeps_different_second_repeat():
    """A real repeat one second later must survive (lossless)."""
    conn = _indexed_conn()
    conn.execute(_INSERT, ("s", "user", "ok", None, None, 100.0, None))
    conn.execute(_INSERT, ("s", "user", "ok", None, None, 101.0, None))


def test_ts_fallback_index_does_not_disturb_timestamped_rows():
    conn = _indexed_conn()
    conn.execute(_INSERT, ("s", "user", "x", None, None, 102.0, 500.0))
    with pytest_raises_integrity():
        conn.execute(_INSERT, ("s", "user", "x", None, None, 103.0, 500.0))


def pytest_raises_integrity():
    import pytest

    return pytest.raises(sqlite3.IntegrityError)


# --- 3. doctor duplicate apply -----------------------------------------------


def _dup_conn(rows):
    """rows: list of (session_id, role, content, observed_at)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, ingested_at REAL,
            observed_at REAL)"""
    )
    for session_id, role, content, observed_at in rows:
        conn.execute(_INSERT, (session_id, role, content, None, None, 1.0, observed_at))
    return conn


def _cleanup_rows(conn):
    ns = _load_command_symbols()
    return ns["_message_duplicate_cleanup_rows"](conn)


def _node_conn(rows, source_ids):
    """_dup_conn plus a summary_nodes table holding one node."""
    conn = _dup_conn(rows)
    conn.execute(
        """CREATE TABLE summary_nodes (
            node_id INTEGER PRIMARY KEY, session_id TEXT, depth INTEGER,
            summary TEXT, source_ids TEXT, source_type TEXT)"""
    )
    conn.execute(
        "INSERT INTO summary_nodes (node_id, session_id, depth, summary, "
        "source_ids, source_type) VALUES (1, 's', 0, 'sum', ?, 'messages')",
        (json.dumps(source_ids),),
    )
    return conn


def _repoint(conn, pairs):
    src = pathlib.Path(__file__).resolve().parents[1] / "command.py"
    text = src.read_text()
    start = text.index("def _repoint_summary_node_source_ids(")
    end = text.index("def _dangling_non_repointable_refs(")
    ns: dict = {"json": json}
    exec(compile(text[start:end], "<command-extract>", "exec"), ns)
    conn.execute("DROP TABLE IF EXISTS temp.trove_dedup_map")
    conn.execute(
        "CREATE TEMP TABLE trove_dedup_map (victim INTEGER PRIMARY KEY, keeper INTEGER)"
    )
    conn.executemany("INSERT INTO temp.trove_dedup_map VALUES (?, ?)", pairs)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(summary_nodes)")}
    moved = ns["_repoint_summary_node_source_ids"](conn, cols)
    conn.execute("DROP TABLE IF EXISTS temp.trove_dedup_map")
    return moved


def test_repoint_moves_summary_refs_off_victims():
    """A summary node citing a re-ingested row must follow it to the keeper.

    Without this the apply refuses the whole cleanup: 191 refs on two nodes
    blocked 21,295 rows on the live DB, every one with a surviving keeper.
    Rows 1-10 are the originals; 11-20 are the re-ingest, so those are victims.
    """
    conversation = [("s", "user", f"m{i}", None) for i in range(10)]
    conn = _node_conn(conversation + conversation, [1, 11, 12, 20])
    pairs = _cleanup_rows(conn)
    victims = {v for v, _ in pairs}
    assert 11 in victims  # fixture sanity: the re-ingest rows are the victims
    moved = _repoint(conn, pairs)
    after = json.loads(
        conn.execute("SELECT source_ids FROM summary_nodes WHERE node_id=1").fetchone()[0]
    )
    assert moved == 3
    assert not (set(after) & victims)
    assert len(after) == 4  # no source lost


def test_repoint_preserves_unrelated_source_ids():
    """Only ids in the dedup map move; everything else is byte-identical."""
    conversation = [("s", "user", f"m{i}", None) for i in range(10)]
    conn = _node_conn(conversation + conversation, [999, 1, 11, 888])
    pairs = _cleanup_rows(conn)
    _repoint(conn, pairs)
    after = json.loads(
        conn.execute("SELECT source_ids FROM summary_nodes WHERE node_id=1").fetchone()[0]
    )
    assert after[0] == 999 and after[3] == 888
    assert after[1] == 1
    assert after[2] == 1  # victim 11 -> its keeper


def test_repoint_target_has_identical_content():
    """A repointed ref must land on byte-identical content, or it is data loss."""
    conversation = [("s", "assistant", f"payload{i}", None) for i in range(8)]
    conn = _node_conn(conversation + conversation, [9])
    pairs = _cleanup_rows(conn)
    victim, keeper = next((v, k) for v, k in pairs if v == 9)
    _repoint(conn, pairs)
    after = json.loads(
        conn.execute("SELECT source_ids FROM summary_nodes WHERE node_id=1").fetchone()[0]
    )
    assert after == [keeper]
    a = conn.execute("SELECT content FROM messages WHERE store_id=?", (victim,)).fetchone()[0]
    b = conn.execute("SELECT content FROM messages WHERE store_id=?", (keeper,)).fetchone()[0]
    assert a == b


def test_repoint_is_noop_without_dedup_map():
    conn = _node_conn([("s", "user", "x", None)], [1])
    assert _repoint(conn, []) == 0


def test_cleanup_removes_ordered_replay_with_null_timestamps():
    """The exact damage shape: NULL observed_at, whole window re-sent."""
    conversation = [("s", "user", f"m{i}", None) for i in range(10)]
    conn = _dup_conn(conversation + conversation)
    pairs = _cleanup_rows(conn)
    assert len(pairs) == 10
    for victim, keeper in pairs:
        assert victim > keeper  # earliest kept


def test_cleanup_keeps_legitimate_repeats():
    """A short repeat of identical content is a real conversation, not a replay."""
    rows = [("s", "user", "ok", None), ("s", "user", "ok", None),
            ("s", "assistant", "sure", None)]
    assert _cleanup_rows(_dup_conn(rows)) == []


def test_cleanup_is_noop_on_clean_session():
    rows = [("s", "user", f"m{i}", None) for i in range(20)]
    assert _cleanup_rows(_dup_conn(rows)) == []


def test_cleanup_victim_is_byte_identical_to_keeper():
    conversation = [("s", "assistant", f"payload{i}", None) for i in range(8)]
    conn = _dup_conn(conversation + conversation)
    for victim, keeper in _cleanup_rows(conn):
        a = conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id "
            "FROM messages WHERE store_id=?", (victim,)).fetchone()
        b = conn.execute(
            "SELECT role, content, tool_call_id, tool_calls, session_id "
            "FROM messages WHERE store_id=?", (keeper,)).fetchone()
        assert a == b


def test_cleanup_does_not_chain_keepers_into_victims():
    conversation = [("s", "user", f"m{i}", None) for i in range(10)]
    pairs = _cleanup_rows(_dup_conn(conversation + conversation + conversation))
    victims = {v for v, _ in pairs}
    assert not ({k for _, k in pairs} & victims)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
