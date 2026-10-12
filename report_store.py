"""Survey reports (综述) — one JSON file per conversation.

A report is a standalone document generated from a whole conversation tree, so
it lives outside the tree: ``<data_dir>/reports/<trunk_id>.json``. Generating
again overwrites it. Plain functions over a directory rather than a class, so
the caller can derive the directory from the active store on every call and a
storage switch (``ChatService.rebind_store``) needs no extra rebinding.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

REPORTS_DIRNAME = "reports"

_SAFE_ID = re.compile(r"^[\w-]+$")


def reports_dir(data_dir: str) -> str:
    return os.path.join(data_dir, REPORTS_DIRNAME)


def _path(directory: str, trunk_id: str) -> str:
    # Trunk ids are uuid hex slices; anything else must not become a path.
    if not _SAFE_ID.match(trunk_id or ""):
        raise ValueError(f"invalid trunk id: {trunk_id!r}")
    return os.path.join(directory, f"{trunk_id}.json")


def get(directory: str, trunk_id: str) -> Optional[dict]:
    """The saved report for a conversation, or None (missing or unreadable)."""
    try:
        with open(_path(directory, trunk_id), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save(directory: str, report: dict) -> None:
    """Atomically write ``report`` (temp file + ``os.replace``)."""
    path = _path(directory, report.get("trunk_id", ""))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, sort_keys=True, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def delete(directory: str, trunk_id: str) -> bool:
    """Remove a conversation's report. Returns whether a file was removed."""
    try:
        os.unlink(_path(directory, trunk_id))
    except (OSError, ValueError):
        return False
    return True
