"""JSONL-backed bookmark store — cloud-sync friendly.

Bookmarks are stored in a separate JSONL file next to the chat history so they
can be refreshed, deleted, and listed independently of the conversation tree.
"""
from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Optional

from models import Bookmark

DEFAULT_BOOKMARKS_FILENAME = "bookmarks.jsonl"


class BookmarkNotFoundError(KeyError):
    """Raised when a requested bookmark id is not in the store."""


def resolve_bookmarks_path(data_dir: Optional[str] = None,
                           db_path: Optional[str] = None) -> str:
    """Resolve the bookmarks JSONL path.

    Precedence:
      1. ``db_path`` argument if it ends with ``.jsonl``
      2. ``data_dir`` argument → ``<data_dir>/bookmarks.jsonl``
      3. ``CHATTABLE_BOOKMARKS_FILE`` env var
      4. Parent folder of ``CHATTABLE_DB`` → ``<dir>/bookmarks.jsonl``
      5. ``~/.chatable/bookmarks.jsonl``
    """
    if db_path and db_path.endswith(".jsonl"):
        return os.path.abspath(os.path.expanduser(db_path))
    if data_dir:
        return os.path.join(
            os.path.abspath(os.path.expanduser(data_dir)),
            DEFAULT_BOOKMARKS_FILENAME,
        )
    env_path = os.environ.get("CHATTABLE_BOOKMARKS_FILE")
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))
    chat_db = os.environ.get("CHATTABLE_DB")
    if chat_db:
        return os.path.join(
            os.path.dirname(os.path.abspath(os.path.expanduser(chat_db))),
            DEFAULT_BOOKMARKS_FILENAME,
        )
    home = os.environ.get(
        "CHATTABLE_HOME",
        os.path.join(os.path.expanduser("~"), ".chatable"),
    )
    return os.path.join(os.path.abspath(os.path.expanduser(home)), DEFAULT_BOOKMARKS_FILENAME)


class BookmarkStore:
    """A JSONL-backed store for bookmarks.

    All bookmarks are held in memory and flushed atomically on every mutation.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or resolve_bookmarks_path()
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = Lock()
        self._bookmarks: dict[str, Bookmark] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            return
        with open(self.db_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [bookmarks] skipping malformed line {lineno} in {self.db_path!r}")
                    continue
                bid = obj.get("id")
                if not bid:
                    continue
                self._bookmarks[bid] = self._dict_to_bookmark(obj)

    def _flush(self) -> None:
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        tmp = os.path.join(db_dir, f".bookmarks.jsonl.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for bm in sorted(self._bookmarks.values(), key=lambda b: b.created_at):
                    fh.write(json.dumps(
                        self._bookmark_to_dict(bm),
                        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                    ))
                    fh.write("\n")
            os.replace(tmp, self.db_path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    @staticmethod
    def _bookmark_to_dict(bm: Bookmark) -> dict:
        return {
            "id": bm.id,
            "url": bm.url,
            "title": bm.title,
            "summary": bm.summary,
            "content": bm.content,
            "created_at": bm.created_at,
            "updated_at": bm.updated_at,
            "metadata": bm.metadata,
        }

    @staticmethod
    def _dict_to_bookmark(d: dict) -> Bookmark:
        return Bookmark(
            id=d.get("id", ""),
            url=d.get("url", ""),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            content=d.get("content", ""),
            created_at=d.get("created_at", 0.0) or time.time(),
            updated_at=d.get("updated_at", 0.0) or time.time(),
            metadata=d.get("metadata"),
        )

    def list(self) -> list[Bookmark]:
        """Return all bookmarks sorted by creation time (newest last)."""
        return sorted(self._bookmarks.values(), key=lambda b: b.created_at)

    def get(self, bid: str) -> Bookmark:
        try:
            return self._bookmarks[bid]
        except KeyError as e:
            raise BookmarkNotFoundError(bid) from e

    def add(self, bookmark: Bookmark) -> Bookmark:
        with self._lock:
            self._bookmarks[bookmark.id] = bookmark
            self._flush()
        return bookmark

    def update(self, bid: str, **fields) -> Bookmark:
        with self._lock:
            bm = self._bookmarks.get(bid)
            if bm is None:
                raise BookmarkNotFoundError(bid)
            for key, value in fields.items():
                if hasattr(bm, key):
                    setattr(bm, key, value)
            bm.updated_at = time.time()
            self._flush()
        return bm

    def delete(self, bid: str) -> bool:
        with self._lock:
            if bid not in self._bookmarks:
                return False
            del self._bookmarks[bid]
            self._flush()
        return True

    def search(self, query: str) -> list[Bookmark]:
        """Case-insensitive search across URL, title, and summary."""
        q = query.casefold()
        return [
            bm for bm in self._bookmarks.values()
            if q in bm.url.casefold()
            or q in bm.title.casefold()
            or q in bm.summary.casefold()
        ]
