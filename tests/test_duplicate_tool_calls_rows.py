"""Minimal check for the duplicate_tool_calls_rows fix.

Asserts the new scan key flags byte-identical tool_calls blobs (which the
content-only repetitive-assistant check structurally cannot see) and stays
empty when every tool_calls blob is unique.
"""
import sqlite3

from hermes_trove.ingest_protection import scan_sqlite_payload_risks


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT,
            source TEXT,
            role TEXT,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL,
            token_estimate INTEGER,
            pinned INTEGER
        )
        """
    )
    return conn


def _add(conn, store_id, tool_calls, content="short prose", role="assistant"):
    conn.execute(
        """INSERT INTO messages
           (store_id, session_id, source, role, content, tool_calls, timestamp)
           VALUES (?, 'sess', 'desktop', ?, ?, ?, ?)""",
        (store_id, role, content, tool_calls, float(store_id)),
    )
    conn.commit()


def test_flags_repeated_identical_tool_calls():
    conn = _conn()
    blob = '[{"id": "a", "function": {"arguments": "%s"}}]' % ("x" * 900)
    for store_id in (1, 2, 3):
        _add(conn, store_id, blob)

    rows = scan_sqlite_payload_risks(conn)["duplicate_tool_calls_rows"]

    assert rows, "identical tool_calls must be flagged"
    top = rows[0]
    assert top["occurrences"] == 3, top
    assert top["tool_calls_len"] == len(blob), top
    assert top["wasted_bytes"] == len(blob) * 2, top
    assert top["suspicious_category"] == "duplicate_tool_calls", top
    # The content-only check still cannot see this -- that is the bug.
    assert scan_sqlite_payload_risks(conn)["suspicious_repetitive_assistant_rows"] == []


def test_stays_empty_when_blobs_unique():
    conn = _conn()
    for store_id in range(1, 6):
        _add(conn, store_id, '[{"id": "%d"}]' % store_id)

    assert scan_sqlite_payload_risks(conn)["duplicate_tool_calls_rows"] == []


def test_ranks_by_wasted_bytes_not_row_count():
    conn = _conn()
    # 20 small copies vs 3 large copies: wasted bytes picks the big one.
    for store_id in range(1, 21):
        _add(conn, store_id, '[{"id": "small-%d"}]' % store_id)
    big = '[{"id": "big", "pad": "%s"}]' % ("y" * 5000)
    for store_id in range(101, 104):
        _add(conn, store_id, big)

    rows = scan_sqlite_payload_risks(conn)["duplicate_tool_calls_rows"]

    assert rows[0]["wasted_bytes"] >= rows[-1]["wasted_bytes"], rows
    assert rows[0]["tool_calls_len"] == len(big), rows[0]


def test_ignores_empty_and_null_tool_calls():
    conn = _conn()
    _add(conn, 1, None, content="a")
    _add(conn, 2, "", content="b")
    _add(conn, 3, "", content="c")

    assert scan_sqlite_payload_risks(conn)["duplicate_tool_calls_rows"] == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all checks passed")
