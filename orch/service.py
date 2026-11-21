"""OrchService: orchestrator state, run queue, adopt/compare, SSE event log."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Iterator, Optional

from agent_tools import do_run_shell
from orch.context import PROMPT_TEMPLATE, detect_test_cmd, parse_conclusion, write_context
from orch.executors import RUN_TIMEOUT_S, default_executors
from orch.store import OrchStore, new_id
from orch.worktree import (
    commit_worktree_tree,
    facts_files,
    merge_base,
    merge_into_worktree,
    worktree_create,
    worktree_remove,
    worktree_snapshot,
)
from workspace import Workspace, WorkspaceError

TEST_TIMEOUT_S = int(os.environ.get("CHATTABLE_ORCH_TEST_TIMEOUT_S", "300"))
_EVENT_CAP_PER_RUN = 300


def _parse_limits() -> dict[str, int]:
    raw = os.environ.get("CHATTABLE_ORCH_LIMITS", "")
    limits = {"kimi": 2, "claude": 2, "codex": 2, "builtin": 3}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                limits[k.strip()] = max(1, int(v))
            except ValueError:
                pass
    return limits


class OrchService:
    def __init__(self, chat_service, executors: Optional[dict] = None) -> None:
        self.chat = chat_service
        self.workspace = chat_service.workspace()
        self.shadow = chat_service.shadow_git()
        self.data_dir = chat_service.data_dir()
        self.project_id = hashlib.sha1(self.workspace.root.encode()).hexdigest()[:12]
        self.store = OrchStore(os.path.join(self.data_dir, "orch",
                                            f"{self.project_id}.jsonl"))
        self.runs_dir = os.path.join(self.data_dir, "orch", "runs")
        self.executors = executors or default_executors(chat_service)
        self._limits = _parse_limits()
        self._events: list[dict] = []
        self._cond = threading.Condition()
        self._procs: dict = {}
        self._cancelled: set[str] = set()
        self._ensure_root()
        threading.Thread(target=self._queue_loop, daemon=True).start()

    # ------------------------------------------------------------------ #
    # state / SSE
    # ------------------------------------------------------------------ #

    def _ensure_root(self) -> dict:
        root = self.store.root_node()
        if root:
            return root
        commit = self.shadow.snapshot("orchestrator root", allow_empty=True)
        node = {
            "id": new_id(), "root": True, "snapshot": commit,
            "instruction": None, "parents": [], "agent": None, "model": None,
            "status": "ok", "adopted": False, "conclusion": {},
            "facts": {}, "session_id": None, "events": [],
            "created_at": time.time(),
        }
        self.store.add_node(node)
        self._emit({"type": "node_created", "node": node})
        return node

    def _emit(self, ev: dict) -> None:
        with self._cond:
            self._events.append(ev)
            self._cond.notify_all()

    def subscribe(self, from_index: int = 0, timeout: float = 15.0) -> Iterator[dict]:
        """Replay + follow the orchestrator event log (SSE)."""
        i = max(0, from_index)
        while True:
            with self._cond:
                if i >= len(self._events):
                    self._cond.wait(timeout=timeout)
                idle = i >= len(self._events)
                batch = [] if idle else self._events[i:]
            if idle:
                yield {"type": "_ping"}
                continue
            for ev in batch:
                yield ev
            i += len(batch)

    def state(self) -> dict:
        return {
            "project": {
                "id": self.project_id,
                "workspace": self.workspace.root,
                "test_cmd": self.store.meta.get("test_cmd"),
                "root": (self.store.root_node() or {}).get("id"),
            },
            "nodes": sorted(self.store.nodes.values(),
                            key=lambda n: n.get("created_at", 0)),
            "runs": sorted(self.store.runs.values(),
                           key=lambda r: r.get("created_at", 0)),
            "executors": {
                eid: {"available": ok, "reason": reason}
                for eid, ex in self.executors.items()
                for (ok, reason) in [ex.available()]
            },
            "limits": self._limits,
        }

    def node_detail(self, node_id: str) -> dict:
        return self.store.nodes[node_id]  # KeyError → 404

    # ------------------------------------------------------------------ #
    # submit / cancel
    # ------------------------------------------------------------------ #

    def submit_runs(self, from_node: str, items: list[dict]) -> list[str]:
        if from_node not in self.store.nodes:
            raise KeyError(from_node)
        created: list[str] = []
        for item in items:
            instruction = (item.get("instruction") or "").strip()
            if not instruction:
                continue
            agents = item.get("agents") or ["kimi"]
            for agent in agents:
                if agent not in self.executors:
                    raise ValueError(f"unknown executor: {agent}")
                run = {
                    "id": new_id(), "from_node": from_node,
                    "instruction": instruction, "agent": agent,
                    "after": item.get("after"),
                    "status": "queued", "created_at": time.time(),
                    "started_at": None, "finished_at": None,
                    "node_id": None, "events": [], "error": "",
                }
                self.store.add_run(run)
                created.append(run["id"])
        with self._cond:
            self._cond.notify_all()
        return created

    def cancel_run(self, run_id: str) -> bool:
        run = self.store.runs.get(run_id)
        if run is None:
            return False
        if run["status"] == "queued":
            self.store.update_run(run_id, status="cancelled",
                                  finished_at=time.time())
            return True
        if run["status"] == "running":
            self._cancelled.add(run_id)
            proc = self._procs.get(run_id)
            if proc is not None:
                try:
                    import os as _os
                    import signal as _signal
                    _os.killpg(_os.getpgid(proc.pid), _signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            return True
        return False

    # ------------------------------------------------------------------ #
    # queue
    # ------------------------------------------------------------------ #

    def _running_count(self, agent: str) -> int:
        return sum(1 for r in self.store.runs.values()
                   if r["status"] == "running" and r["agent"] == agent)

    def _pick_next(self) -> Optional[dict]:
        queued = sorted((r for r in self.store.runs.values()
                         if r["status"] == "queued"),
                        key=lambda r: r.get("created_at", 0))
        for run in queued:
            dep = run.get("after")
            if dep:
                dep_run = self.store.runs.get(dep)
                if dep_run is None or dep_run["status"] in ("queued", "running"):
                    continue  # still waiting for its dependency
                if dep_run["status"] != "ok" or not dep_run.get("node_id"):
                    # The dependency failed/was cancelled: cascade-cancel.
                    self.store.update_run(run["id"], status="cancelled",
                                          finished_at=time.time(),
                                          error=f"dependency {dep} did not succeed")
                    self._emit({"type": "run_finished", "run_id": run["id"],
                                "status": "cancelled", "node_id": None})
                    continue
                # Rebase onto the dependency's resulting node.
                self.store.update_run(run["id"], from_node=dep_run["node_id"])
                run = self.store.runs[run["id"]]
            if self._running_count(run["agent"]) < self._limits.get(run["agent"], 2):
                return run
        return None

    def _queue_loop(self) -> None:
        while True:
            run = self._pick_next()
            if run is None:
                with self._cond:
                    self._cond.wait(timeout=5.0)
                continue
            self.store.update_run(run["id"], status="running",
                                  started_at=time.time())
            self._emit({"type": "run_started", "run_id": run["id"],
                        "agent": run["agent"], "instruction": run["instruction"]})
            threading.Thread(target=self._run_one, args=(dict(run),),
                             daemon=True).start()

    # ------------------------------------------------------------------ #
    # run execution
    # ------------------------------------------------------------------ #

    def _run_one(self, run: dict) -> None:
        run_id = run["id"]
        parent = self.store.nodes[run["from_node"]]
        wt = os.path.join(self.runs_dir, run_id, "wt")
        t0 = time.monotonic()
        events: list[dict] = []
        result = {"final_text": "", "exit_code": -1, "session_id": None,
                  "usage": {}, "killed": None}
        error = ""
        test_cmd: Optional[str] = None
        try:
            worktree_create(self.shadow, parent["snapshot"], wt)
            test_cmd = self.store.meta.get("test_cmd") or detect_test_cmd(wt)
            if test_cmd and not self.store.meta.get("test_cmd"):
                self.store.set_meta(test_cmd=test_cmd)
            write_context(self.store, parent, run["instruction"], test_cmd, wt,
                          merge=run.get("merge"),
                          parent_diff=self._parent_diff(parent))
            prompt = PROMPT_TEMPLATE.format(instruction=run["instruction"])
            executor = self.executors[run["agent"]]
            ok, reason = executor.available()
            if not ok:
                raise RuntimeError(reason)
            resume_id = (parent.get("session_id")
                         if parent.get("agent") == run["agent"] else None)

            def should_stop() -> bool:
                return run_id in self._cancelled

            def proc_reg(p) -> None:
                if p is None:
                    self._procs.pop(run_id, None)
                else:
                    self._procs[run_id] = p

            for ev in executor.run(wt, prompt, session_id=resume_id,
                                   should_stop=should_stop, proc_reg=proc_reg,
                                   timeout=RUN_TIMEOUT_S):
                events.append(ev)
                self._emit({"type": "run_event", "run_id": run_id, "event": ev})
                if ev.get("type") == "result":
                    result = ev
        except Exception as e:  # noqa: BLE001 - a failed run still produces a node
            error = f"{type(e).__name__}: {e}"
        finally:
            cancelled = run_id in self._cancelled
            self._cancelled.discard(run_id)
            self._procs.pop(run_id, None)

        # Snapshot whatever state the run left (failed/cancelled included).
        end_commit = None
        if os.path.isdir(wt):
            try:
                end_commit = worktree_snapshot(self.shadow, wt,
                                               f"orch run {run_id}")
            except WorkspaceError:
                end_commit = None
        if end_commit is None:
            end_commit = parent["snapshot"]
        files = facts_files(self.shadow, parent["snapshot"], end_commit)

        status = ("cancelled" if cancelled
                  else "ok" if result.get("exit_code") == 0 and not error else "failed")
        facts = {
            "files": files,
            "exit_code": result.get("exit_code"),
            "duration_s": round(time.monotonic() - t0, 1),
            "usage": result.get("usage") or {},
        }
        if error:
            facts["error"] = error
        if result.get("killed"):
            facts["killed"] = result["killed"]

        # Post-run test (green-check only; artifacts are excluded from snapshots).
        if status == "ok" and test_cmd and os.path.isdir(wt):
            test_res = do_run_shell(Workspace(wt), test_cmd, timeout=TEST_TIMEOUT_S)
            tail = test_res.output.strip().splitlines()
            facts["test"] = {"cmd": test_cmd, "ok": test_res.ok,
                             "summary": "\n".join(tail[-5:])[:800]}
            if not test_res.ok:
                status = "failed"
                facts["error"] = "test command failed"

        conclusion = parse_conclusion(result.get("final_text") or error or "")
        node = {
            "id": new_id(), "snapshot": end_commit,
            "instruction": run["instruction"], "parents": [parent["id"]],
            "agent": run["agent"], "model": None,
            "status": status, "adopted": False,
            "conclusion": conclusion, "facts": facts,
            "session_id": result.get("session_id"),
            "events": events[-_EVENT_CAP_PER_RUN:],
            "created_at": time.time(),
        }
        self.store.add_node(node)
        self.store.update_run(run_id, status=status, node_id=node["id"],
                              finished_at=time.time(), error=error,
                              events=events[-_EVENT_CAP_PER_RUN:])
        self._emit({"type": "run_finished", "run_id": run_id,
                    "status": status, "node_id": node["id"]})
        self._emit({"type": "node_created", "node": node})
        worktree_remove(self.shadow, wt)
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # compare / adopt
    # ------------------------------------------------------------------ #

    def _parent_diff(self, node: dict) -> str:
        """The diff that produced a node (vs its first parent's snapshot)."""
        parents = node.get("parents") or []
        if not parents or node.get("root"):
            return ""
        base = self.store.nodes.get(parents[0])
        if not base or not base.get("snapshot") or not node.get("snapshot"):
            return ""
        try:
            return self.shadow.diff(node["snapshot"], base["snapshot"])
        except WorkspaceError:
            return ""

    def project_info(self) -> dict:
        return {"id": self.project_id, "workspace": self.workspace.root,
                "notes": self.store.meta.get("notes", ""),
                "test_cmd": self.store.meta.get("test_cmd"),
                "detected_test_cmd": self.store.meta.get("detected_test_cmd")}

    def update_project(self, notes: Optional[str], test_cmd: Optional[str]) -> dict:
        fields = {}
        if notes is not None:
            fields["notes"] = notes
        if test_cmd is not None:
            fields["test_cmd"] = test_cmd
        if fields:
            self.store.set_meta(**fields)
        return self.project_info()

    def compare(self, a_id: str, b_id: str) -> dict:
        b = self.store.nodes.get(b_id)
        if a_id == "workspace":
            ca = self.shadow.snapshot("current workspace state", allow_empty=True)
            if not ca:
                return {"ok": False, "reason": "workspace snapshots unavailable"}
        else:
            a = self.store.nodes.get(a_id)
            ca = a["snapshot"] if a else None
        if not ca or not b:
            return {"ok": False, "reason": "unknown node id"}
        cb = b["snapshot"]
        return {"ok": True, "a_commit": ca, "b_commit": cb,
                "diff": self.shadow.diff(cb, ca),
                "files": self.shadow.diff_numstat(ca, cb)}

    # ------------------------------------------------------------------ #
    # merge (DAG)
    # ------------------------------------------------------------------ #

    def merge(self, node_ids: list[str], agent: str, run_tests: bool = True) -> dict:
        """3-way merge of ≥2 nodes' snapshots in a throwaway worktree.

        Clean + green → merge node M. Conflict or red tests → M with status
        conflict/tests_failed (markers committed) and an auto-dispatched
        resolver run from M with the full merge context.
        """
        nodes = [self.store.nodes.get(i) for i in node_ids]
        if len(nodes) < 2 or any(n is None for n in nodes):
            raise ValueError("merge needs at least two existing nodes")
        if agent not in self.executors:
            raise ValueError(f"unknown executor: {agent}")
        commits = [n["snapshot"] for n in nodes]
        instruction = "Merge " + " + ".join(f"#{i[:6]}" for i in node_ids)
        self._emit({"type": "merge_started", "nodes": node_ids,
                    "instruction": instruction})
        wt = os.path.join(self.runs_dir, f"merge-{new_id()}", "wt")
        base = None
        conflict_files: list[str] = []
        test_fact = None
        try:
            merged = merge_into_worktree(self.shadow, commits, wt)
            base = merged["base"]
            conflict_files = merged["conflict_files"]
            status = "conflict" if conflict_files else "ok"
            if run_tests and not conflict_files:
                test_cmd = (self.store.meta.get("test_cmd")
                            or detect_test_cmd(wt))
                if test_cmd:
                    test_res = do_run_shell(Workspace(wt), test_cmd,
                                            timeout=TEST_TIMEOUT_S)
                    tail = test_res.output.strip().splitlines()
                    test_fact = {"cmd": test_cmd, "ok": test_res.ok,
                                 "summary": "\n".join(tail[-5:])[:800]}
                    if not test_res.ok:
                        status = "tests_failed"
            merge_commit = commit_worktree_tree(
                self.shadow, wt, commits,
                f"orchestrator merge of {len(commits)} nodes")
        finally:
            worktree_remove(self.shadow, wt)

        facts = {"files": facts_files(self.shadow, base, merge_commit)
                 if base else [],
                 "exit_code": 0, "duration_s": 0.0, "usage": {}}
        if conflict_files:
            facts["conflict_files"] = conflict_files
        if test_fact:
            facts["test"] = test_fact
        node = {
            "id": new_id(), "snapshot": merge_commit,
            "instruction": instruction, "parents": list(node_ids),
            "agent": "merge", "model": None,
            "status": status, "adopted": False,
            "conclusion": {
                "done": f"Merged {len(nodes)} branches" +
                        (" with conflicts" if status == "conflict"
                         else " cleanly" if status == "ok"
                         else ", but tests fail"),
                "decisions": "", "not_done": "" if status == "ok" else status,
                "next": "" if status == "ok" else "Resolve via the dispatched run",
            },
            "facts": facts, "session_id": None, "events": [],
            "created_at": time.time(),
        }
        self.store.add_node(node)
        self._emit({"type": "merge_finished", "node_id": node["id"],
                    "status": status})
        self._emit({"type": "node_created", "node": node})

        run_id = None
        if status in ("conflict", "tests_failed"):
            run_id = self._dispatch_merge_resolver(node, status, conflict_files,
                                                   test_fact, agent)
        return {"ok": True, "node_id": node["id"], "status": status,
                "run_id": run_id}

    def _dispatch_merge_resolver(self, node: dict, status: str,
                                 conflict_files: list[str],
                                 test_fact: Optional[dict], agent: str) -> str:
        """Auto-dispatch a run that resolves the merge (conflicts or tests)."""
        if status == "conflict":
            instruction = ("Resolve the merge conflicts in the marked files, "
                           "keeping the intent of every parent change. "
                           "Run the tests afterwards if a test command is known.")
        else:
            instruction = ("The merged result fails the project tests. Make the "
                           "tests pass while keeping the intent of every parent change.")
        merge_ctx = {
            "parents": [self.store.nodes[p] for p in node["parents"]],
            "conflict_files": conflict_files,
            "test_output": (test_fact or {}).get("summary", ""),
        }
        run = {
            "id": new_id(), "from_node": node["id"],
            "instruction": instruction, "agent": agent, "after": None,
            "status": "queued", "created_at": time.time(),
            "started_at": None, "finished_at": None,
            "node_id": None, "events": [], "error": "",
            "merge": merge_ctx,
        }
        self.store.add_run(run)
        with self._cond:
            self._cond.notify_all()
        return run["id"]

    def adopt(self, node_id: str) -> dict:
        node = self.store.nodes.get(node_id)
        if not node:
            raise KeyError(node_id)
        previous = self.shadow.snapshot(
            f"workspace before adopting orch node {node_id[:6]}", allow_empty=True)
        self.shadow.restore(node["snapshot"])
        self.store.update_node(node_id, adopted=True, adopted_at=time.time(),
                               previous_commit=previous)
        files = (self.shadow.diff_numstat(previous, node["snapshot"])
                 if previous else [])
        return {"ok": True, "adopted_commit": node["snapshot"],
                "previous_commit": previous, "files": files}

    def undo_adopt(self, previous_commit: str) -> dict:
        self.shadow.snapshot("workspace before undo-adopt", allow_empty=True)
        self.shadow.restore(previous_commit)
        return {"ok": True, "restored": previous_commit}
