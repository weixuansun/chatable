"""OrchService: orchestrator state, run queue, adopt/compare, SSE event log."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Iterator, Optional

from orch.chat_runs import ChatRunsMixin, _args_summary
from orch.context import (PROMPT_TEMPLATE, build_chat_context, detect_test_cmd,
                          parse_conclusion, write_chat_context, write_context)
from orch.executors import RUN_TIMEOUT_S, default_executors
from orch.store import OrchStore, new_id
from orch.testrun import compare_to_baseline, run_test_cmd
from orch.worktree import (
    commit_worktree_tree,
    facts_files,
    merge_base,
    merge_into_worktree,
    worktree_create,
    worktree_remove,
    worktree_snapshot,
)
from workspace import WorkspaceError, adopt_preview_files

TEST_TIMEOUT_S = int(os.environ.get("CHATTABLE_ORCH_TEST_TIMEOUT_S", "300"))
_EVENT_CAP_PER_RUN = 300
_STEP_CAP_PER_RUN = 200
_STEP_OUTPUT_CAP = 300

log = logging.getLogger(__name__)


def _record_step(steps: list[dict], ev: dict) -> bool:
    """Fold one executor event into the compact, persisted step trail
    (tool name, args summary, time, ok/error). True when a step closed."""
    typ = ev.get("type")
    if typ == "tool_call":
        name = str(ev.get("name") or "tool")
        args = ev.get("arguments") if isinstance(ev.get("arguments"), dict) else {}
        steps.append({"id": ev.get("id"), "name": name,
                      "args_summary": _args_summary(name, args),
                      "at": ev.get("at") or time.time(), "ok": None})
        if len(steps) > _STEP_CAP_PER_RUN:
            del steps[:-_STEP_CAP_PER_RUN]
        return False
    if typ == "tool_result":
        cid = ev.get("id")
        match = None
        if cid is not None:
            for s in reversed(steps):
                if s.get("ok") is None and s.get("id") == cid:
                    match = s
                    break
        if match is None:
            for s in steps:
                if s.get("ok") is None:
                    match = s
                    break
        if match is None:
            return False
        match["ok"] = ev.get("ok") is not False
        output = str(ev.get("output") or "")
        if output:
            match["output"] = output[:_STEP_OUTPUT_CAP]
        return True
    return False


def _live_worktree_files(wt: str, base: str) -> list[dict]:
    """Files a running agent has changed so far, read-only.

    Polled every couple of seconds by the UI, so it must not commit or take
    the worktree's index lock (that would race the agent and the run's final
    snapshot): working tree vs ``base`` via ``git diff`` with optional locks
    off, plus untracked files.
    """
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git(*args: str) -> str:
        r = subprocess.run(["git", "-C", wt, *args], capture_output=True,
                           text=True, env=env, timeout=10)
        return r.stdout if r.returncode == 0 else ""

    try:
        status = {}
        for line in git("diff", "--no-renames", "--name-status", base).splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status[parts[-1]] = parts[0][:1]
        files = []
        for line in git("diff", "--no-renames", "--numstat", base).splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            add, rem, path = parts
            files.append({"path": path, "status": status.get(path, "M"),
                          "added": int(add) if add.isdigit() else 0,
                          "removed": int(rem) if rem.isdigit() else 0})
        for path in git("ls-files", "--others", "--exclude-standard").splitlines():
            if not path or path.startswith(".chatable/"):
                continue
            try:
                with open(os.path.join(wt, path), "rb") as fh:
                    added = fh.read(2_000_000).count(b"\n")
            except OSError:
                added = 0
            files.append({"path": path, "status": "A", "added": added, "removed": 0})
        return files
    except Exception:  # noqa: BLE001 - detail is best-effort
        return []


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


class OrchService(ChatRunsMixin):
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
        # Test results cached per snapshot commit — one test run serves as
        # the baseline for every sibling branching off that snapshot.
        self._test_cache: dict[str, dict] = {}
        # CHA-5 chat-native runs: run id → live chat turn; assistant id → run id.
        self._chat_turns: dict = {}
        self._chat_run_by_asst: dict[str, str] = {}
        # baseline-*/merge-* scratch dirs in use by this process (the sweep
        # must not delete them out from under a concurrent merge).
        self._scratch_in_use: set[str] = set()
        self._ensure_root()
        self._recover_chat_runs()
        self.sweep_run_dirs()
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

    def busy(self) -> bool:
        """True while any orchestrator run is queued or running."""
        return any(r.get("status") in ("queued", "running")
                   for r in self.store.runs.values())

    def rebind(self) -> None:
        """Follow the chat service onto its current data folder.

        The service's store may be swapped after construction (``--data-dir``
        at launch, or the storage folder in settings); without this the
        orchestrator kept writing runs, snapshots and its node store under the
        folder it was created with. Refuses while runs are queued/running.
        """
        with self._cond:
            if self.busy():
                raise RuntimeError("cannot switch storage while agent runs are queued or running")
            data_dir = self.chat.data_dir()
            workspace = self.chat.workspace()
            if data_dir == self.data_dir and workspace.root == self.workspace.root:
                return
            self.workspace = workspace
            self.shadow = self.chat.shadow_git()
            self.data_dir = data_dir
            self.project_id = hashlib.sha1(self.workspace.root.encode()).hexdigest()[:12]
            self.store = OrchStore(os.path.join(self.data_dir, "orch",
                                                f"{self.project_id}.jsonl"))
            self.runs_dir = os.path.join(self.data_dir, "orch", "runs")
            self._test_cache.clear()
            self._chat_turns.clear()
            self._chat_run_by_asst.clear()
            self._ensure_root()
            self._recover_chat_runs()
            # Under the lock: the queue loop can't start a run (and create
            # its dir) between the sweep's active-run read and its deletes.
            self.sweep_run_dirs()
            self._cond.notify_all()

    def _recover_chat_runs(self) -> None:
        """Chat runs interrupted by a server restart: mark them failed so
        their nodes don't spin forever (queued ones simply run again)."""
        for run in list(self.store.runs.values()):
            if not run.get("chat") or run.get("status") != "running":
                continue
            self.store.update_run(run["id"], status="failed", finished_at=time.time(),
                                  error="server restarted during the run")
            aid = run["chat"].get("assistant_id")
            try:
                if aid and self.chat.store.exists(aid):
                    self.chat.store.finalize_node(aid, "运行被中断（服务重启）。", {}, {
                        "agent": True, "orch_run": True, "executor": run.get("agent"),
                        "run_id": run["id"], "run_status": "failed",
                        "ws_commit": run["chat"].get("base_commit"),
                        "ws_parent_commit": run["chat"].get("base_commit"),
                        "files_changed": 0, "stop_reason": "error"})
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # run dir cleanup
    # ------------------------------------------------------------------ #

    def _cleanup_run_dir(self, name: str) -> None:
        """Delete ``<runs_dir>/<name>`` entirely: the worktree (with its git
        worktree metadata) plus any scratch the executor left there (the
        builtin executor keeps its own shadow repo/chat log inside). Only
        call this once the run's snapshot is committed to the shadow repo —
        the snapshot is the result, the dir is scratch. Best-effort."""
        run_dir = os.path.join(self.runs_dir, name)
        if os.path.dirname(os.path.normpath(run_dir)) != os.path.normpath(self.runs_dir):
            return
        try:
            worktree_remove(self.shadow, os.path.join(run_dir, "wt"))
        except Exception as e:  # noqa: BLE001 - cleanup is best-effort
            log.warning("worktree remove failed for %s: %s", name, e)
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except OSError as e:
            log.warning("run dir cleanup failed for %s: %s", name, e)

    def _active_run_ids(self) -> set[str]:
        """Ids of runs queued/running in ANY project store under this data
        dir (runs_dir is shared by every project)."""
        ids: set[str] = set()
        orch_dir = os.path.dirname(self.store.path)
        try:
            paths = [os.path.join(orch_dir, f) for f in os.listdir(orch_dir)
                     if f.endswith(".jsonl")]
        except OSError:
            paths = []
        for path in paths:
            try:
                store = (self.store if os.path.abspath(path) == os.path.abspath(self.store.path)
                         else OrchStore(path))
                for run in store.runs.values():
                    if run.get("status") in ("queued", "running"):
                        ids.add(run["id"])
            except Exception as e:  # noqa: BLE001 - one bad store must not stop the sweep
                log.warning("could not read orch store %s: %s", path, e)
        return ids

    def sweep_run_dirs(self) -> None:
        """Startup/rebind sweep: drop run dirs no live run can still need.

        A dir is kept only while some project's store marks that run
        queued/running (a queued run recreates the dir when it starts;
        baseline-*/merge-* scratch dirs are never in use at sweep time).
        Afterwards ``git worktree prune`` drops the shadow repo's stale
        worktree metadata. Never touches objects/refs — node snapshots must
        survive. Best-effort: errors are logged, never raised."""
        try:
            active = self._active_run_ids()
        except Exception as e:  # noqa: BLE001
            log.warning("run dir sweep: could not list active runs: %s", e)
            return
        try:
            names = os.listdir(self.runs_dir)
        except OSError:
            names = []
        for name in names:
            if name in active or name in self._scratch_in_use:
                continue
            if not os.path.isdir(os.path.join(self.runs_dir, name)):
                continue
            self._cleanup_run_dir(name)
        try:
            subprocess.run(["git", f"--git-dir={self.shadow.git_dir}",
                            "worktree", "prune"],
                           capture_output=True, timeout=60)
        except Exception as e:  # noqa: BLE001
            log.warning("git worktree prune failed: %s", e)

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

    def _run_files(self, base: str, end: str) -> list[dict]:
        """Changed files base→end with git status (A/M/D…) + line counts."""
        try:
            return [{"path": f["path"], "status": f.get("status", "M"),
                     "added": f["additions"], "removed": f["deletions"]}
                    for f in self.shadow.diff_numstat(base, end)]
        except Exception:  # noqa: BLE001 - detail is best-effort
            return []

    def run_detail(self, run_id: str) -> dict:
        """One run, queued/running/finished: timing, step trail, changed
        files (snapshot diff), failure reason. KeyError → 404."""
        run = self.store.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        status = run.get("status") or "queued"
        started = run.get("started_at")
        finished = run.get("finished_at")
        duration = round((finished or time.time()) - started, 1) if started else None
        chat = run.get("chat") or {}
        user_id = chat.get("user_id")
        assistant_id = chat.get("assistant_id")
        node_id = run.get("node_id")
        base, end = None, None
        if chat:
            base = chat.get("base_commit")
            if assistant_id and self.chat.store.exists(assistant_id):
                md = self.chat.store.get(assistant_id).metadata or {}
                end = md.get("ws_commit")
        else:
            parent = self.store.nodes.get(run.get("from_node") or "")
            base = (parent or {}).get("snapshot")
            node = self.store.nodes.get(node_id or "")
            end = (node or {}).get("snapshot")
        files: list[dict] = []
        if base:
            if end and end != base:
                files = self._run_files(base, end)
            elif status == "running":
                # Mid-run: snapshot whatever the worktree currently holds.
                wt = os.path.join(self.runs_dir, run_id, "wt")
                if os.path.isdir(wt):
                    files = _live_worktree_files(wt, base)
        steps = run.get("steps")
        if steps is None:  # runs recorded before the step trail existed
            steps = []
            for ev in run.get("events") or []:
                _record_step(steps, ev)
        steps_done = run.get("steps_done")
        if steps_done is None:
            steps_done = sum(1 for s in steps if s.get("ok") is not None)
        current = next((s for s in reversed(steps) if s.get("ok") is None), None)
        if current is None and steps:
            current = steps[-1]
        out = {
            "id": run["id"], "agent": run.get("agent"), "status": status,
            "instruction": run.get("instruction") or "",
            "created_at": run.get("created_at"), "started_at": started,
            "finished_at": finished, "duration_s": duration,
            "user_id": user_id, "assistant_id": assistant_id, "node_id": node_id,
            "steps_done": steps_done,
            "current_step": ({k: current.get(k) for k in ("name", "args_summary", "ok")}
                             if current else None),
            "steps": steps[-_STEP_CAP_PER_RUN:],
            "files": files,
            "error": run.get("error") or "",
        }
        if status in ("failed", "cancelled"):
            tail = [s["output"] for s in steps if s.get("ok") is False and s.get("output")]
            if not tail:
                tail = [s["output"] for s in steps if s.get("output")]
            out["output_tail"] = tail[-3:]
        return out

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
                    "steps": [], "steps_done": 0,
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
            if run.get("chat"):
                base = run["chat"]["base_commit"]
                self._chat_run_finished(run, status="cancelled", end_commit=base, base=base,
                                        files=[], facts={}, conclusion={}, final_text="",
                                        session_id=None, usage={}, error="")
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
            if run.get("chat"):
                try:
                    self._chat_run_started(run)
                except Exception:  # noqa: BLE001
                    pass
            threading.Thread(target=self._run_one, args=(dict(run),),
                             daemon=True).start()

    # ------------------------------------------------------------------ #
    # run execution
    # ------------------------------------------------------------------ #

    def _run_one(self, run: dict) -> None:
        run_id = run["id"]
        chat = run.get("chat")
        if chat:
            parent = {"id": None, "snapshot": chat["base_commit"]}
        else:
            parent = self.store.nodes[run["from_node"]]
        chat_state: dict = {}
        wt = os.path.join(self.runs_dir, run_id, "wt")
        t0 = time.monotonic()
        events: list[dict] = []
        steps: list[dict] = []
        steps_done = 0
        steps_saved_at = 0.0
        result = {"final_text": "", "exit_code": -1, "session_id": None,
                  "usage": {}, "killed": None}
        error = ""
        test_cmd: Optional[str] = None
        try:
            worktree_create(self.shadow, parent["snapshot"], wt)
            test_cmd = self.store.meta.get("test_cmd") or detect_test_cmd(wt)
            if test_cmd and not self.store.meta.get("test_cmd"):
                self.store.set_meta(test_cmd=test_cmd)
            if chat:
                write_chat_context(wt, build_chat_context(
                    self._chat_path_nodes(chat["parent_id"]), run["instruction"],
                    test_cmd, wt, notes=self.store.meta.get("notes", ""),
                    parent_diff=self._chat_parent_diff(chat["parent_id"]),
                    extra=chat.get("extra_context", "")))
            else:
                write_context(self.store, parent, run["instruction"], test_cmd, wt,
                              merge=run.get("merge"),
                              parent_diff=self._parent_diff(parent))
            prompt = PROMPT_TEMPLATE.format(instruction=run["instruction"])
            executor = self.executors[run["agent"]]
            ok, reason = executor.available()
            if not ok:
                raise RuntimeError(reason)
            if chat:
                resume_id = self._chat_resume_session(chat["parent_id"], run["agent"])
            else:
                resume_id = (parent.get("session_id")
                             if parent.get("agent") == run["agent"] else None)

            def should_stop() -> bool:
                if chat and self._chat_should_stop(run):
                    self._cancelled.add(run_id)
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
                if chat:
                    self._chat_event(run, ev, chat_state)
                if _record_step(steps, ev):
                    steps_done += 1
                # Persist the compact step trail as the run progresses (the
                # full event log only lands at the end), throttled to ~1/s.
                if ev.get("type") in ("tool_call", "tool_result"):
                    now = time.monotonic()
                    if ev["type"] == "tool_result" or now - steps_saved_at > 1.0:
                        steps_saved_at = now
                        try:
                            self.store.update_run(run_id, steps=list(steps),
                                                  steps_done=steps_done)
                        except Exception:  # noqa: BLE001 - detail is best-effort
                            pass
                if ev.get("type") == "result":
                    result = ev
        except Exception as e:  # noqa: BLE001 - a failed run still produces a node
            error = f"{type(e).__name__}: {e}"
        finally:
            cancelled = run_id in self._cancelled
            self._cancelled.discard(run_id)
            self._procs.pop(run_id, None)

        # Snapshot whatever state the run left (failed/cancelled included) —
        # before the finally below wipes the worktree and the run dir.
        try:
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

            # Post-run test, RELATIVE to the parent snapshot's baseline: a run is
            # red only when it introduces NEW failures — pre-existing failures
            # (e.g. a bug being fixed on a sibling branch) don't count.
            if status == "ok" and test_cmd and os.path.isdir(wt):
                if chat:
                    self._chat_notice(run, f"运行测试：{test_cmd}")
                fact = self._build_test_fact(wt, test_cmd, parent["snapshot"])
                facts["test"] = fact
                if fact["ok"] is False:
                    status = "failed"
                    facts["error"] = "new test failures vs the parent baseline"

            conclusion = parse_conclusion(result.get("final_text") or error or "")
            if chat:
                self.store.update_run(run_id, status=status, node_id=chat["assistant_id"],
                                      finished_at=time.time(), error=error,
                                      events=events[-_EVENT_CAP_PER_RUN:],
                                      steps=list(steps), steps_done=steps_done)
                try:
                    self._chat_run_finished(
                        run, status=status, end_commit=end_commit, base=parent["snapshot"],
                        files=files, facts=facts, conclusion=conclusion,
                        final_text=result.get("final_text") or "",
                        session_id=result.get("session_id"),
                        usage=result.get("usage") or {}, error=error)
                finally:
                    self._emit({"type": "run_finished", "run_id": run_id,
                                "status": status, "node_id": None})
                return
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
                                  events=events[-_EVENT_CAP_PER_RUN:],
                                  steps=list(steps), steps_done=steps_done)
            self._emit({"type": "run_finished", "run_id": run_id,
                        "status": status, "node_id": node["id"]})
            self._emit({"type": "node_created", "node": node})
        finally:
            self._cleanup_run_dir(run_id)
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

    def baseline_test_result(self, commit: str, test_cmd: str) -> dict:
        """The test result for a snapshot, computed lazily once and reused
        as the baseline for every run merging from / branching off it."""
        cached = self._test_cache.get(commit)
        if cached and cached.get("cmd") == test_cmd:
            return cached
        name = f"baseline-{new_id()}"
        self._scratch_in_use.add(name)
        wt = os.path.join(self.runs_dir, name, "wt")
        worktree_create(self.shadow, commit, wt)
        try:
            result = run_test_cmd(wt, test_cmd, TEST_TIMEOUT_S)
        finally:
            self._cleanup_run_dir(name)
            self._scratch_in_use.discard(name)
        self._test_cache[commit] = result
        return result

    def _build_test_fact(self, wt: str, test_cmd: str,
                         baseline_commit: Optional[str]) -> dict:
        """Run tests in *wt* and shape facts.test, baseline-relative."""
        result = run_test_cmd(wt, test_cmd, TEST_TIMEOUT_S)
        if not result.get("exit_ok") and not result["passed"] and not result["failed"]:
            # Non-zero exit with nothing parsed: the runner itself could not
            # run (command missing, no tests collected, import error before
            # collection). Report that honestly instead of a vacuous pass.
            return {
                "cmd": test_cmd, "ran": False, "ok": None,
                "passed": 0, "failed": 0, "failed_ids": [],
                "baseline": {"passed": 0, "failed": 0},
                "new_failures": [], "fixed": [],
                "summary": result["summary"],
            }
        baseline = (self.baseline_test_result(baseline_commit, test_cmd)
                    if baseline_commit else
                    {"passed": 0, "failed": 0, "failed_ids": []})
        delta = compare_to_baseline(result, baseline)
        return {
            "cmd": test_cmd,
            "ran": True,
            "ok": not delta["new_failures"],
            "passed": result["passed"],
            "failed": result["failed"],
            "failed_ids": result["failed_ids"],
            "baseline": {"passed": baseline.get("passed", 0),
                         "failed": baseline.get("failed", 0)},
            "new_failures": delta["new_failures"],
            "fixed": delta["fixed"],
            "summary": result["summary"],
        }

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
        t0 = time.monotonic()
        name = f"merge-{new_id()}"
        self._scratch_in_use.add(name)
        wt = os.path.join(self.runs_dir, name, "wt")
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
                    # Baseline = the merge base's result: red only on NEW failures.
                    test_fact = self._build_test_fact(wt, test_cmd, base)
                    if test_fact["ok"] is False:
                        status = "tests_failed"
            merge_commit = commit_worktree_tree(
                self.shadow, wt, commits,
                f"orchestrator merge of {len(commits)} nodes")
        finally:
            self._cleanup_run_dir(name)
            self._scratch_in_use.discard(name)

        instr_lines = ["- " + " ".join((n.get("instruction") or "?").split())[:80]
                       for n in nodes]
        suffix = (" with conflicts" if status == "conflict"
                  else ", but tests fail" if status == "tests_failed"
                  else " cleanly")
        facts = {"files": facts_files(self.shadow, base, merge_commit)
                 if base else [],
                 "exit_code": 0,
                 "duration_s": round(time.monotonic() - t0, 1), "usage": {}}
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
                "done": f"Merged {len(nodes)} branches{suffix}:\n" + "\n".join(instr_lines),
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
            "steps": [], "steps_done": 0,
            "merge": merge_ctx,
        }
        self.store.add_run(run)
        with self._cond:
            self._cond.notify_all()
        return run["id"]

    def adopt_preview(self, node_id: str) -> dict:
        """Preview of adopting a node's snapshot into the CURRENT workspace:
        per-file changes with will_delete/local_edit flags and counts (see
        ``workspace.adopt_preview_files``). ``base_commit`` is the snapshot of
        the node's first parent (the run's starting point); null for the root
        node — then no file can be flagged ``local_edit``."""
        node = self.store.nodes.get(node_id)
        if not node:
            raise KeyError(node_id)
        current = self.shadow.snapshot("current workspace state", allow_empty=True)
        if not current:
            return {"ok": False, "reason": "workspace snapshots unavailable"}
        parents = node.get("parents") or []
        parent = self.store.nodes.get(parents[0]) if parents else None
        base = (parent or {}).get("snapshot")
        target = node["snapshot"]
        return {"ok": True, "node_id": node_id, "workspace_commit": current,
                "target_commit": target, "base_commit": base,
                "diff": self.shadow.diff(target, current),
                **adopt_preview_files(self.shadow, current, target, base)}

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
