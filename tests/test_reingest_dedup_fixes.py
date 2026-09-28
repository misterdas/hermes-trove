"""Regression checks for the gateway-restart re-ingest cursor.

Covers ``ReconcileMixin._membership_cursor_for_store_head`` -- the order-blind,
one-for-one match that stops the compounding re-ingest loop.

Why this matcher and not an ordered one: duplicate rows land *interleaved* with
the originals, so the replay window is neither a suffix nor a prefix of stored
history and every ordered matcher extends zero rows.  The tail matcher then
settles on cursor=1 and re-ingests nearly the whole window on every restart.
"""
from hermes_trove.config import TROVEConfig
from hermes_trove.engine import TROVEEngine


# --- membership cursor -------------------------------------------------------


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
