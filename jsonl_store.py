"""Per-conversation JSONL tree store — cloud-sync friendly.

Layout under the data directory (the folder holding the legacy DB file)::

    <data_dir>/trees/<trunk_id>.jsonl   # one file per conversation
    <data_dir>/chatable.jsonl           # legacy single file — auto-migrated
                                        # into trees/ on first open, then
                                        # renamed to chatable.jsonl.bak

A *trunk* is a direct child of the synthetic ROOT — i.e. one sidebar
conversation.  Every node belongs to exactly one trunk for its entire life
(the public API has no reparent operation; forking just adds a child inside
the same tree), so a mutation only ever touches one small file, and cloud
sync (iCloud/Dropbox) only ships the conversation that actually changed.
Conflicts shrink from "the whole history" to "one conversation".

Writes are append-only: a mutation appends a fresh, complete line for the
node and the load-time "last occurrence wins" rule supersedes stale lines.
A trunk file is compacted (atomic temp-file + ``os.replace`` full rewrite)
once its stale appended-line count exceeds ``max(200, live nodes)``.  Deletes
rewrite (or remove) the affected file — they are rare, so no tombstones.

The public API is identical to the previous single-file ``JsonlTreeStore`` —
service.py and server.py need no changes.
"""
from __future__ import annotations

import glob
import json
import os
import time
from threading import Lock
from typing import Optional

from models import Node

ROOT_ID = "root"
DEFAULT_DATA_DIR = os.environ.get(
    "CHATTABLE_HOME",
    os.path.join(os.path.expanduser("~"), ".chatable"),
)
DEFAULT_DB_PATH = os.environ.get(
    "CHATTABLE_DB",
    os.path.join(DEFAULT_DATA_DIR, "chatable.jsonl"),
)
DB_FILENAME = "chatable.jsonl"
TREES_DIRNAME = "trees"

# Compact a trunk file once this many stale appended lines have piled up,
# scaled by the live size so large conversations don't rewrite constantly.
_COMPACT_MIN_DIRTY = 200

# Keep backwards-compatible names that server.py imports.
SCHEMA_VERSION = 1
NODE_COLUMN_DEFS: dict[str, str] = {}  # not meaningful for JSONL


class NodeNotFoundError(KeyError):
    """Raised when a requested node_id is not in the store."""


def resolve_db_path(data_dir: Optional[str] = None) -> str:
    """Resolve the (legacy) JSONL file path.

    1. ``data_dir`` argument — if it ends with ``.jsonl``, use it directly;
       otherwise treat it as a folder → ``<folder>/chatable.jsonl``.
    2. ``CHATTABLE_DB`` / ``CHATTABLE_HOME`` env vars
    3. ``~/.chatable/chatable.jsonl``

    The store keeps its per-conversation files in ``trees/`` next to this
    path; the path itself is only read once, for auto-migration.
    """
    if data_dir:
        path = os.path.abspath(os.path.expanduser(data_dir))
        if path.endswith(".jsonl"):
            return path
        return os.path.join(path, DB_FILENAME)
    return DEFAULT_DB_PATH


class JsonlTreeStore:
    """A TreeStore-compatible store backed by per-conversation JSONL files.

    All nodes are held in memory; every mutation appends the node's fresh
    line to its trunk's file (last-occurrence-wins on load).  Full rewrites
    (compaction, branch deletes, startup repairs) use an atomic temp-file +
    ``os.replace``, so sync agents and readers never see a partial file.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.trees_dir = os.path.join(db_dir, TREES_DIRNAME)
        os.makedirs(self.trees_dir, exist_ok=True)
        self._lock = Lock()
        # _nodes: id → Node  (the synthetic ROOT is never stored)
        self._nodes: dict[str, Node] = {}
        # trunk_id → ordered dict of member ids (used as an insertion-ordered set)
        self._members: dict[str, dict[str, None]] = {}
        # nid → stem of the file the node was last loaded from / written to
        self._file_of: dict[str, str] = {}
        # trunk_id → appended lines since the last full rewrite of that file
        self._dirty: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Read every ``trees/*.jsonl`` file into ``_nodes``.

        Files are read oldest-mtime first so that when the same id appears in
        several files (sync conflict copies) the most recently modified file
        wins — same "last occurrence wins" rule as duplicate lines within a
        file.  Malformed lines are skipped (a crash may leave a torn tail).
        """
        self._maybe_migrate_legacy()
        files = glob.glob(os.path.join(self.trees_dir, "*.jsonl"))
        files.sort(key=lambda p: (os.path.getmtime(p), p))
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"  [jsonl] skipping malformed line {lineno} in {path!r}")
                        continue
                    nid = obj.get("id")
                    if not nid or nid == ROOT_ID:
                        continue
                    self._nodes[nid] = self._dict_to_node(obj)
                    self._file_of[nid] = stem
        orphans = self._repair_orphans(self._nodes)
        if orphans:
            print(f"  [jsonl] {orphans} orphan node(s) re-parented to root")
        self._rebuild_members()
        # One-time startup fix-up: rewrite trunk files whose on-disk contents
        # don't match the computed trunks (migration leftovers, orphans that
        # became new trunks, sync conflict copies like "<id> 2.jsonl").  After
        # the rewrite the canonical file is the newest, so it wins next load.
        bad = {
            self._trunk_of(nid)
            for nid in self._nodes
            if self._file_of.get(nid) != self._trunk_of(nid)
        }
        if bad:
            print(f"  [jsonl] rewriting {len(bad)} tree file(s) to repair placement")
            for trunk in sorted(bad):
                self._rewrite_trunk(trunk)

    def _maybe_migrate_legacy(self) -> None:
        """Split a legacy single-file ``chatable.jsonl`` into per-trunk files.

        Runs at most once: if ``trees/`` already holds any ``.jsonl`` the
        legacy file is considered retired and left untouched.
        """
        legacy = os.path.abspath(self.db_path)
        if not os.path.isfile(legacy):
            return
        if glob.glob(os.path.join(self.trees_dir, "*.jsonl")):
            return
        nodes: dict[str, Node] = {}
        with open(legacy, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nid = obj.get("id")
                if not nid or nid == ROOT_ID:
                    continue
                nodes[nid] = self._dict_to_node(obj)
        backup = legacy + ".bak"
        if not nodes:
            os.rename(legacy, backup if not os.path.exists(backup)
                      else f"{legacy}.{int(time.time())}.bak")
            return
        self._repair_orphans(nodes)
        # _trunk_of reads self._nodes; borrow it for the grouping pass, then
        # reset so the normal scan below loads from the files we just wrote.
        self._nodes = nodes
        groups: dict[str, list[Node]] = {}
        for nid, node in nodes.items():
            groups.setdefault(self._trunk_of(nid), []).append(node)
        self._nodes = {}
        for trunk, members in groups.items():
            self._write_file(self._trunk_path(trunk), members)
        if os.path.exists(backup):
            backup = f"{legacy}.{int(time.time())}.bak"
        os.rename(legacy, backup)
        print(f"  [jsonl] migrated {len(nodes)} node(s) into {len(groups)} tree "
              f"file(s) under {self.trees_dir!r}; legacy file saved as {backup!r}")

    @staticmethod
    def _repair_orphans(nodes: dict[str, Node]) -> int:
        """Re-parent nodes whose parent is missing to root. Returns the count."""
        orphans = 0
        for node in nodes.values():
            pid = node.parent_id
            if pid and pid != ROOT_ID and pid not in nodes:
                node.parent_id = ROOT_ID
                orphans += 1
        return orphans

    def _rebuild_members(self) -> None:
        self._members = {}
        for nid in self._nodes:
            self._members.setdefault(self._trunk_of(nid), {})[nid] = None

    def _trunk_of(self, nid: str) -> str:
        """The top-level ancestor of *nid* below root (its conversation file).

        After orphan repair every parent chain ends at root; the visited-set
        guard keeps a (sync-induced) cycle from looping forever by treating
        the node where the cycle is detected as its own trunk.
        """
        cur = self._nodes[nid]
        seen = {nid}
        while cur.parent_id and cur.parent_id != ROOT_ID:
            pid = cur.parent_id
            if pid in seen or pid not in self._nodes:
                return cur.id
            cur = self._nodes[pid]
            seen.add(pid)
        return cur.id

    def _trunk_path(self, trunk: str) -> str:
        return os.path.join(self.trees_dir, f"{trunk}.jsonl")

    @staticmethod
    def _node_line(node: Node) -> str:
        return json.dumps(
            JsonlTreeStore._node_to_dict(node),
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"

    @staticmethod
    def _write_file(path: str, nodes) -> None:
        """Atomically write *nodes* as JSONL (temp file + ``os.replace``)."""
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for node in nodes:
                    fh.write(JsonlTreeStore._node_line(node))
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _rewrite_trunk(self, trunk: str) -> None:
        """Atomically rewrite one trunk file from memory (or remove it)."""
        members = self._members.get(trunk)
        path = self._trunk_path(trunk)
        if not members:
            if os.path.exists(path):
                os.unlink(path)
            self._dirty.pop(trunk, None)
            return
        self._write_file(path, [self._nodes[nid] for nid in members])
        for nid in members:
            self._file_of[nid] = trunk
        self._dirty[trunk] = 0

    def _append(self, node: Node) -> None:
        """Append *node*'s fresh line to its trunk file; compact when stale."""
        trunk = self._trunk_of(node.id)
        with open(self._trunk_path(trunk), "a", encoding="utf-8") as fh:
            fh.write(self._node_line(node))
        self._file_of[node.id] = trunk
        dirty = self._dirty.get(trunk, 0) + 1
        self._dirty[trunk] = dirty
        if dirty > max(_COMPACT_MIN_DIRTY, len(self._members.get(trunk, ()))):
            self._rewrite_trunk(trunk)

    # ------------------------------------------------------------------ #
    # serialization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_to_dict(node: Node) -> dict:
        return {
            "id": node.id,
            "role": node.role,
            "content": node.content,
            "parent_id": node.parent_id,
            "created_at": node.created_at,
            "usage": node.usage,
            "pinned": node.pinned,
            "token_count": node.token_count,
            "title": node.title,
            "summary": node.summary,
            "metadata": node.metadata,
        }

    @staticmethod
    def _dict_to_node(d: dict) -> Node:
        return Node(
            id=d.get("id", ""),
            role=d.get("role", "user"),
            content=d.get("content", ""),
            parent_id=d.get("parent_id"),
            created_at=d.get("created_at", time.time()),
            usage=d.get("usage"),
            pinned=bool(d.get("pinned", False)),
            token_count=int(d.get("token_count", 0)),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            metadata=d.get("metadata"),
        )

    # ------------------------------------------------------------------ #
    # public API (identical to TreeStore)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _root() -> Node:
        return Node(
            id=ROOT_ID, role="user", content="",
            parent_id=None, created_at=0.0,
        )

    def get(self, nid: str) -> Node:
        if nid == ROOT_ID:
            return self._root()
        try:
            node = self._nodes[nid]
            # Return an independent copy so callers can't accidentally mutate
            # the store through a reference (matching old SQLite behaviour where
            # every row fetch produced a new object).
            return Node(
                id=node.id,
                role=node.role,
                content=node.content,
                parent_id=node.parent_id,
                created_at=node.created_at,
                usage=dict(node.usage) if node.usage else None,
                pinned=node.pinned,
                token_count=node.token_count,
                title=node.title,
                summary=node.summary,
                metadata=dict(node.metadata) if node.metadata else None,
            )
        except KeyError as e:
            raise NodeNotFoundError(nid) from e

    def exists(self, nid: str) -> bool:
        if nid == ROOT_ID:
            return True
        return nid in self._nodes

    def _lookup(self, nid: str) -> Node:
        node = self._nodes.get(nid)
        if node is None:
            raise NodeNotFoundError(nid)
        return node

    def add(self, node: Node) -> None:
        if node.id == ROOT_ID:
            return
        with self._lock:
            is_new = node.id not in self._nodes
            self._nodes[node.id] = node
            if is_new:
                trunk = self._trunk_of(node.id)
                self._members.setdefault(trunk, {})[node.id] = None
            self._append(node)

    def append_content(self, nid: str, delta: str) -> None:
        if nid == ROOT_ID:
            return
        with self._lock:
            node = self._lookup(nid)
            node.content += delta
            node.token_count = max(1, len(node.content) // 4)
            self._append(node)

    def set_usage(self, nid: str, usage: dict) -> None:
        if nid == ROOT_ID:
            return
        with self._lock:
            node = self._lookup(nid)
            node.usage = usage
            self._append(node)

    def set_metadata(self, nid: str, metadata: dict) -> None:
        if nid == ROOT_ID:
            return
        with self._lock:
            node = self._lookup(nid)
            node.metadata = metadata
            self._append(node)

    def finalize_node(self, nid: str, content: str,
                      usage: Optional[dict], metadata: Optional[dict]) -> None:
        """Append *content* and set usage+metadata in a single locked write.

        Replaces the append_content + set_usage + set_metadata trio at the
        end of every turn (3 appends) with one.
        """
        if nid == ROOT_ID:
            return
        with self._lock:
            node = self._lookup(nid)
            if content:
                node.content += content
                node.token_count = max(1, len(node.content) // 4)
            node.usage = usage
            node.metadata = metadata
            self._append(node)

    def set_title(self, nid: str, title: str) -> Node:
        if nid == ROOT_ID:
            raise ValueError("root cannot be titled")
        with self._lock:
            node = self._lookup(nid)
            node.title = title.strip()
            self._append(node)
        return self.get(nid)

    def update_metadata(self, nid: str, updates: dict) -> Node:
        if nid == ROOT_ID:
            raise ValueError("root cannot store metadata")
        with self._lock:
            node = self._lookup(nid)
            md = dict(node.metadata or {})
            md.update(updates)
            node.metadata = md
            self._append(node)
        return self.get(nid)

    def set_tags(self, nid: str, tags: list[str]) -> Node:
        clean: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            value = tag.strip()
            key = value.casefold()
            if value and key not in seen:
                clean.append(value)
                seen.add(key)
        return self.update_metadata(nid, {"tags": clean})

    def add_tags(self, nid: str, tags: list[str]) -> Node:
        node = self.get(nid)
        current = []
        if isinstance(node.metadata, dict) and isinstance(node.metadata.get("tags"), list):
            current = [str(tag) for tag in node.metadata["tags"]]
        return self.set_tags(nid, current + tags)

    def remove_tags(self, nid: str, tags: list[str]) -> Node:
        node = self.get(nid)
        current = []
        if isinstance(node.metadata, dict) and isinstance(node.metadata.get("tags"), list):
            current = [str(tag) for tag in node.metadata["tags"]]
        remove = {tag.casefold() for tag in tags}
        return self.set_tags(nid, [tag for tag in current if tag.casefold() not in remove])

    def delete_subtree(self, nid: str) -> int:
        """Delete a node and all its descendants. Returns the count removed."""
        if nid == ROOT_ID:
            raise ValueError("root cannot be deleted")
        with self._lock:
            if nid not in self._nodes:
                return 0
            trunk = self._trunk_of(nid)
            members = self._members.get(trunk, {})
            # Build the children index over this trunk only — O(trunk size)
            # instead of scanning every node once per BFS level.
            children: dict[str, list[str]] = {}
            for mid in members:
                pid = self._nodes[mid].parent_id
                if pid:
                    children.setdefault(pid, []).append(mid)
            to_delete: set[str] = set()
            stack = [nid]
            while stack:
                cur = stack.pop()
                if cur in to_delete:
                    continue
                to_delete.add(cur)
                stack.extend(children.get(cur, ()))
            for i in to_delete:
                self._nodes.pop(i, None)
                self._file_of.pop(i, None)
                members.pop(i, None)
            if nid == trunk:
                # Whole conversation gone — drop its file outright.
                path = self._trunk_path(trunk)
                if os.path.exists(path):
                    os.unlink(path)
                self._members.pop(trunk, None)
                self._dirty.pop(trunk, None)
            else:
                # Rare enough that a full atomic rewrite beats tombstones.
                self._rewrite_trunk(trunk)
            return len(to_delete)

    def set_pinned(self, nid: str, pinned: bool) -> Node:
        if nid == ROOT_ID:
            raise ValueError("root cannot be pinned")
        with self._lock:
            node = self._lookup(nid)
            node.pinned = pinned
            self._append(node)
        return self.get(nid)

    def pinned_nodes(self) -> list[Node]:
        return sorted(
            [n for n in self._nodes.values() if n.pinned],
            key=lambda n: n.created_at,
        )

    def path_to_root(self, nid: str) -> list[Node]:
        """Return root→nid order, EXCLUDING the synthetic ROOT node."""
        out: list[Node] = []
        cur = self.get(nid)
        while cur is not None and cur.parent_id is not None:
            out.append(cur)
            cur = self.get(cur.parent_id) if self.exists(cur.parent_id) else None
        out.reverse()
        return out

    def all_nodes(self) -> list[Node]:
        return sorted(self._nodes.values(), key=lambda n: n.created_at)

    def children_map(self) -> dict[str, list[str]]:
        created: dict[str, float] = {nid: n.created_at for nid, n in self._nodes.items()}
        m: dict[str, list[str]] = {}
        for nid, node in self._nodes.items():
            if node.parent_id is not None:
                m.setdefault(node.parent_id, []).append(nid)
        for k in m:
            # Newest children first so the sidebar tree shows recent turns at the top.
            m[k].sort(key=lambda i: created.get(i, 0), reverse=True)
        return m

    def node_columns(self) -> set[str]:
        """Backwards-compat: all fields are always present."""
        return set(JsonlTreeStore._node_to_dict(Node.new("user", "", ROOT_ID)).keys())

    def schema_info(self) -> dict:
        files = glob.glob(os.path.join(self.trees_dir, "*.jsonl"))
        return {
            "db_path": self.db_path,
            "trees_dir": self.trees_dir,
            "tree_files": len(files),
            "nodes": len(self._nodes),
            "schema_version": 1,
            "expected_schema_version": 1,
            "columns": sorted(self.node_columns()),
            "missing_columns": [],
            "ok": True,
        }
