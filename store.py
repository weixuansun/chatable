"""Tree store — now backed by JSONL for cloud-sync safety.

This module re-exports ``JsonlTreeStore`` as ``TreeStore`` so existing imports
(``from store import TreeStore``) continue to work without changes.
"""
from __future__ import annotations

from jsonl_store import (  # noqa: F401 — re-export with old names
    JsonlTreeStore as TreeStore,
    ROOT_ID,
    NodeNotFoundError,
    resolve_db_path,
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    DB_FILENAME,
    SCHEMA_VERSION,
    NODE_COLUMN_DEFS,
)
