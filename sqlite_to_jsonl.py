#!/usr/bin/env python3
"""One-shot migration: SQLite store → legacy single-file JSONL.

Reads every row from ``chatable.sqlite3`` (or the path given via --input) and
writes them to ``chatable.jsonl`` (or --output) as plain JSONL lines (the
same node-dict format ``JsonlTreeStore`` understands).  No store internals
are used: on next launch the store auto-migrates the single file into
per-conversation files under ``trees/``.

Run:
    python sqlite_to_jsonl.py
    python sqlite_to_jsonl.py --input ~/.chatable/chatable.sqlite3
    python sqlite_to_jsonl.py --input ~/.chatable/chatable.sqlite3 --output ~/.chatable/chatable.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from models import Node

ROOT_ID = "root"
DEFAULT_DATA_DIR = os.environ.get(
    "CHATTABLE_HOME",
    os.path.join(os.path.expanduser("~"), ".chatable"),
)


def _row_to_node(row: sqlite3.Row) -> Node:
    usage = json.loads(row["usage_json"]) if row["usage_json"] else None
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else None
    return Node(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        parent_id=row["parent_id"],
        created_at=row["created_at"],
        usage=usage,
        pinned=bool(row["pinned"]),
        token_count=int(row["token_count"] or 0),
        title=row["title"] or "",
        summary=row["summary"] or "",
        metadata=metadata,
    )


def migrate(sqlite_path: str, jsonl_path: str) -> int:
    """Write every row from *sqlite_path* as legacy-format JSONL at *jsonl_path*.

    Returns the number of nodes written.  The store's startup auto-migration
    takes it from there (single file → per-conversation files under trees/).
    """
    if not os.path.isfile(sqlite_path):
        print(f"Error: SQLite file not found: {sqlite_path}", file=sys.stderr)
        return 0

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM nodes ORDER BY created_at").fetchall()
    conn.close()

    if not rows:
        print("SQLite store is empty — nothing to migrate.")
        return 0

    out_dir = os.path.dirname(os.path.abspath(jsonl_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            node = _row_to_node(row)
            if node.id == ROOT_ID:
                continue
            fh.write(json.dumps(
                node.to_dict(), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ) + "\n")
            count += 1

    print(f"Migrated {count} nodes → {jsonl_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sqlite_to_jsonl",
        description="Migrate chatable history from SQLite to JSONL.",
    )
    parser.add_argument(
        "--input", "-i", metavar="PATH",
        default=os.path.join(DEFAULT_DATA_DIR, "chatable.sqlite3"),
        help="Path to the SQLite file (default: ~/.chatable/chatable.sqlite3).",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        default=os.path.join(DEFAULT_DATA_DIR, "chatable.jsonl"),
        help="Path for the new JSONL file (default: ~/.chatable/chatable.jsonl).",
    )
    args = parser.parse_args()

    count = migrate(
        os.path.abspath(os.path.expanduser(args.input)),
        os.path.abspath(os.path.expanduser(args.output)),
    )
    if count:
        print("Done. The web UI will load the JSONL file automatically on next launch.")


if __name__ == "__main__":
    main()
