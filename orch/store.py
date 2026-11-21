"""Orchestrator DAG store: one append-only JSONL file per project.

A *project* is a workspace dir; its nodes form a DAG of snapshots (the root
node is the initial workspace snapshot). Runs are the live instructions that
produce child nodes. Records: {"kind": "node"|"run"|"meta", ...} — appended
on every mutation (last occurrence wins on load), so a crash mid-run never
corrupts history.
"""
from __future__ import annotations

import json
import os
import threading
import time
from uuid import uuid4


def new_id() -> str:
    return uuid4().hex[:12]


class OrchStore:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self.nodes: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.meta: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = obj.get("kind")
                    if kind == "node":
                        self.nodes[obj["id"]] = obj
                    elif kind == "run":
                        self.runs[obj["id"]] = obj
                    elif kind == "meta":
                        self.meta.update(obj.get("data") or {})
        except FileNotFoundError:
            pass

    def _append(self, obj: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

    def add_node(self, node: dict) -> None:
        with self._lock:
            self.nodes[node["id"]] = node
            self._append({**node, "kind": "node"})

    def update_node(self, node_id: str, **fields) -> dict:
        with self._lock:
            node = {**self.nodes[node_id], **fields}
            self.nodes[node_id] = node
            self._append({**node, "kind": "node"})
            return node

    def add_run(self, run: dict) -> None:
        with self._lock:
            self.runs[run["id"]] = run
            self._append({**run, "kind": "run"})

    def update_run(self, run_id: str, **fields) -> dict:
        with self._lock:
            run = {**self.runs[run_id], **fields}
            self.runs[run_id] = run
            self._append({**run, "kind": "run"})
            return run

    def set_meta(self, **fields) -> None:
        with self._lock:
            self.meta.update(fields)
            self._append({"kind": "meta", "data": fields})

    def root_node(self) -> dict | None:
        for n in self.nodes.values():
            if n.get("root"):
                return n
        return None

    def children_of(self, node_id: str) -> list[dict]:
        return [n for n in self.nodes.values() if node_id in (n.get("parents") or [])]

    def path_to_root(self, node_id: str) -> list[dict]:
        """Ancestor chain root→node following the first parent."""
        out: list[dict] = []
        cur = self.nodes.get(node_id)
        seen: set[str] = set()
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            out.append(cur)
            parents = cur.get("parents") or []
            cur = self.nodes.get(parents[0]) if parents else None
        out.reverse()
        return out

    def ancestors_of(self, node_id: str) -> list[dict]:
        """Every ancestor of the node, walking ALL parents (merge DAG),
        deduplicated, oldest first by creation time."""
        seen: set[str] = set()
        out: list[dict] = []
        stack = [node_id]
        while stack:
            cur_id = stack.pop()
            node = self.nodes.get(cur_id)
            if node is None or cur_id in seen:
                continue
            seen.add(cur_id)
            out.append(node)
            stack.extend(node.get("parents") or [])
        out.sort(key=lambda n: n.get("created_at", 0))
        return out
