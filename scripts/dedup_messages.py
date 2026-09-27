#!/usr/bin/env python3
"""Collapse re-ingested duplicate message rows in a TROVE store.

An ingest failure used to leave ``_ingest_cursor`` at its pre-batch
position, so the next turn re-sent the whole session history and every
already-persisted message was written again under a fresh ``store_id``.
The unique identity index cannot catch that on its own: SQLite treats
NULLs as distinct, and ``observed_at`` is NULL for ~75% of rows (the
host supplies no message timestamp), so the re-copies never collide.
This is the operator tool for cleaning up what that produced.

Rows are grouped by message identity
``(session_id, role, observed_at, tool_call_id, content, tool_calls)``
with NULL observed_at folded together (COALESCE -1), the earliest
``store_id`` in each group is kept, and the rest are removed. Chunk
references are re-pointed onto the surviving row FIRST, so recall
keeps working. Content is byte-identical across a group by
construction, so nothing unique is lost — only re-ingested copies.

Destructive. Back up the database first. Dry run by default; pass
``--apply`` to write.

    python3 scripts/dedup_messages.py                       # dry run
    python3 scripts/dedup_messages.py --apply --vacuum      # write + reclaim
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

# NULL observed_at must group together, not float apart: a plain column
# list would make every NULL its own identity cluster.
IDENTITY = (
    "session_id, role, COALESCE(observed_at,-1), "
    "COALESCE(tool_call_id,''), COALESCE(content,''), COALESCE(tool_calls,'')"
)


def _default_db() -> Path:
    import os

    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "trove.db"


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=_default_db(),
        help="path to trove.db (default: $HERMES_HOME/trove.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete rows (default: dry run, no writes)",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="VACUUM after applying, to reclaim disk (implies --apply)",
    )
    args = parser.parse_args(argv)
    apply = args.apply or args.vacuum

    if not args.db.exists():
        print(f"no such database: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    before = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"database: {args.db}")
    print(f"rows before: {before} ({'APPLY' if apply else 'DRY RUN'})")

    # One pass: materialize victim -> keeper so the re-point and the delete
    # read from the same stable snapshot.
    conn.execute("DROP TABLE IF EXISTS temp.dedup_map")
    conn.execute(
        f"""
        CREATE TEMP TABLE dedup_map AS
        SELECT store_id AS victim, keep AS keeper FROM (
            SELECT store_id,
                   MIN(store_id) OVER (PARTITION BY {IDENTITY}) AS keep,
                   ROW_NUMBER() OVER (PARTITION BY {IDENTITY} ORDER BY store_id) AS rn
            FROM messages
        ) WHERE rn > 1
        """
    )
    victims = conn.execute("SELECT COUNT(*) FROM dedup_map").fetchone()[0]
    print(f"duplicate rows to remove: {victims}")

    if victims == 0:
        print("nothing to do")
        conn.close()
        return 0

    # Sanity: never delete a keeper, never re-point at a missing row.
    assert conn.execute(
        "SELECT COUNT(*) FROM dedup_map WHERE victim = keeper"
    ).fetchone()[0] == 0, "victim == keeper"
    assert conn.execute(
        "SELECT COUNT(*) FROM dedup_map d LEFT JOIN messages m "
        "ON m.store_id = d.keeper WHERE m.store_id IS NULL"
    ).fetchone()[0] == 0, "keeper missing"

    referenced = 0
    if _has_table(conn, "trove_chunk_meta"):
        referenced = conn.execute(
            "SELECT COUNT(*) FROM trove_chunk_meta WHERE store_id IN "
            "(SELECT victim FROM dedup_map)"
        ).fetchone()[0]
    print(f"chunk references to re-point: {referenced}")

    if not apply:
        print("dry run: no writes performed (pass --apply)")
        conn.close()
        return 0

    # chunk_id derives from identity_hash, not store_id, so only store_id
    # moves; the chunk text itself stays valid against the keeper.
    if _has_table(conn, "trove_chunk_meta"):
        cur = conn.execute(
            """
            UPDATE trove_chunk_meta
            SET store_id = (SELECT d.keeper FROM dedup_map d
                            WHERE d.victim = trove_chunk_meta.store_id)
            WHERE store_id IN (SELECT victim FROM dedup_map)
            """
        )
        print(f"chunk_meta rows re-pointed: {cur.rowcount}")

    conn.execute("DELETE FROM messages WHERE store_id IN (SELECT victim FROM dedup_map)")

    after = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"rows after: {after} (removed {before - after})")
    conn.commit()

    t0 = time.time()
    print(f"quick_check: {conn.execute('PRAGMA quick_check').fetchone()[0]} ({time.time() - t0:.1f}s)")
    if _has_table(conn, "trove_chunk_meta"):
        print(
            "orphan chunk refs: "
            f"{conn.execute('SELECT COUNT(*) FROM trove_chunk_meta c LEFT JOIN messages m ON m.store_id = c.store_id WHERE m.store_id IS NULL').fetchone()[0]}"
        )
    if _has_table(conn, "messages_fts"):
        fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        print(f"fts rows: {fts} (messages: {after})")
        if fts != after:
            print("WARNING: FTS and messages disagree — run trove doctor", file=sys.stderr)

    if args.vacuum:
        conn.execute("VACUUM")
        print("vacuumed")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
