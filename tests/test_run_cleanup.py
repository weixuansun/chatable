"""Run dir cleanup: end-of-run wipe (ok/failed/cancelled/crashed), the
node-delete hook, and the startup sweep over the shared runs dir."""
import os
import subprocess
import threading
import time

import pytest

from orch.service import OrchService
from orch.store import OrchStore
from orch.worktree import worktree_create
from service import ChatService
from store import ROOT_ID, TreeStore

from test_agent_loop import git_required
from test_chat_runs import WriterExecutor, _wait
from test_orch import ChatOrchWait, FakeExecutor


class BlockingExecutor(FakeExecutor):
    """Holds the run open until released (or stopped via should_stop)."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        self.started.set()
        while not self.release.is_set():
            if should_stop():
                yield {"type": "result", "final_text": "", "exit_code": -9,
                       "session_id": None, "usage": {}, "killed": "cancelled"}
                return
            time.sleep(0.02)
        with open(os.path.join(workdir, "fake.txt"), "w") as fh:
            fh.write("done\n")
        yield {"type": "result", "final_text": "Done.", "exit_code": 0,
               "session_id": None, "usage": {}}


def _svc(tmp_path, monkeypatch, executors):
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CHATTABLE_ORCH_LIMITS", "a=2,fake=2,builtin=3")
    monkeypatch.setenv("CHATTABLE_RETAIN_FINISHED_TURN_S", "60")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    orch = OrchService(chat, executors=executors)
    chat.run_stop_hook = orch.stop_chat_run
    chat.subtree_deleted_hook = orch.chat_nodes_deleted
    os.makedirs(chat.workspace().root, exist_ok=True)
    with open(os.path.join(chat.workspace().root, "base.txt"), "w") as fh:
        fh.write("base\n")
    return chat, orch


def _wait_gone(path, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not os.path.exists(path):
            return
        time.sleep(0.05)
    raise AssertionError(f"still exists: {path}")


def _wait_dir(path, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.isdir(path):
            return
        time.sleep(0.02)
    raise AssertionError(f"never appeared: {path}")


# ------------------------------------------------------------- end of run

@git_required
def test_successful_run_removes_run_dir_but_keeps_snapshot(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    (r,) = orch.start_chat_runs("make a.txt", None, ["a"])
    _wait(chat, r["assistant_id"])
    _wait_gone(os.path.join(orch.runs_dir, r["run_id"]))
    # The result snapshot still reads back fine (diff / file list).
    d = chat.node_diff(r["assistant_id"])
    assert [f["path"] for f in d["files"]] == ["a.txt"]


@git_required
def test_failed_run_removes_run_dir(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"fake": FakeExecutor(exit_code=1)})
    root = orch.store.root_node()
    run_id = orch.submit_runs(root["id"], [{"instruction": "doomed",
                                            "agents": ["fake"]}])[0]
    run = ChatOrchWait(orch).wait(run_id)
    assert run["status"] == "failed"
    _wait_gone(os.path.join(orch.runs_dir, run_id))


@git_required
def test_cancelled_run_removes_run_dir(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt", sleep=5)})
    (r,) = orch.start_chat_runs("slow", None, ["a"])
    deadline = time.monotonic() + 5
    while (chat.store.get(r["assistant_id"]).metadata or {}).get("run_status") != "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert chat.request_stop(r["assistant_id"]) is True
    _ev, md = _wait(chat, r["assistant_id"])
    assert md["run_status"] == "cancelled"
    _wait_gone(os.path.join(orch.runs_dir, r["run_id"]))


@git_required
def test_post_run_failure_still_cleans_up(tmp_path, monkeypatch):
    """A crash in the end phase (after the snapshot) must not leak the dir."""
    chat, orch = _svc(tmp_path, monkeypatch, {"fake": FakeExecutor()})

    def boom(*args, **kwargs):
        raise RuntimeError("store on fire")

    monkeypatch.setattr("orch.service.facts_files", boom)
    root = orch.store.root_node()
    run_id = orch.submit_runs(root["id"], [{"instruction": "x",
                                            "agents": ["fake"]}])[0]
    _wait_gone(os.path.join(orch.runs_dir, run_id))


@git_required
def test_sweep_keeps_a_running_runs_worktree(tmp_path, monkeypatch):
    ex = BlockingExecutor()
    chat, orch = _svc(tmp_path, monkeypatch, {"a": ex})
    (r,) = orch.start_chat_runs("slow", None, ["a"])
    assert ex.started.wait(10)
    wt = os.path.join(orch.runs_dir, r["run_id"], "wt")
    _wait_dir(wt)
    orch.sweep_run_dirs()
    assert os.path.isdir(wt), "sweep must not touch a running run's worktree"
    ex.release.set()
    _wait(chat, r["assistant_id"])
    _wait_gone(os.path.join(orch.runs_dir, r["run_id"]))


# -------------------------------------------------------------- node delete

@git_required
def test_delete_node_removes_finished_run_dir(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    (r,) = orch.start_chat_runs("make a.txt", None, ["a"])
    _wait(chat, r["assistant_id"])
    run_dir = os.path.join(orch.runs_dir, r["run_id"])
    _wait_gone(run_dir)
    # A leftover from a crash / an older version: deleting the node clears it.
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "junk.txt"), "w") as fh:
        fh.write("leftover\n")
    assert chat.delete_subtree(r["user_id"]) > 0
    assert not os.path.exists(run_dir)


@git_required
def test_delete_node_during_run_cancels_and_cleans_at_run_end(tmp_path, monkeypatch):
    ex = BlockingExecutor()
    chat, orch = _svc(tmp_path, monkeypatch, {"a": ex})
    (r,) = orch.start_chat_runs("slow", None, ["a"])
    assert ex.started.wait(10)
    run_dir = os.path.join(orch.runs_dir, r["run_id"])
    _wait_dir(os.path.join(run_dir, "wt"))
    assert chat.delete_subtree(r["user_id"]) > 0
    assert os.path.isdir(run_dir), "the dir must not be pulled from under a live run"
    # The run winds down as cancelled and the end-of-run cleanup wipes the dir.
    run = ChatOrchWait(orch).wait(r["run_id"])
    assert run["status"] == "cancelled"
    _wait_gone(run_dir)


# ------------------------------------------------------------ startup sweep

@git_required
def test_startup_sweep(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    (r,) = orch.start_chat_runs("make a.txt", None, ["a"])
    _wait(chat, r["assistant_id"])
    asst_md = chat.store.get(r["assistant_id"]).metadata

    # Leftovers: a crashed run's real worktree, baseline/merge scratch dirs.
    crashed = os.path.join(orch.runs_dir, "deadbeef0001")
    worktree_create(orch.shadow, asst_md["ws_commit"], os.path.join(crashed, "wt"))
    os.makedirs(os.path.join(orch.runs_dir, "baseline-x", "wt"))
    os.makedirs(os.path.join(orch.runs_dir, "merge-y", "wt"))
    # A chat run interrupted by a kill/restart: still "running" in our store —
    # _recover_chat_runs flips it to failed, then the sweep removes its dir.
    orch.store.add_run({
        "id": "deadchat1234", "from_node": None, "instruction": "x",
        "agent": "a", "after": None, "status": "running",
        "created_at": time.time(), "started_at": time.time(),
        "finished_at": None, "node_id": None, "events": [], "error": "",
        "steps": [], "steps_done": 0,
        "chat": {"user_id": "gone-u", "assistant_id": "gone-a",
                 "parent_id": ROOT_ID, "base_commit": asst_md["ws_commit"],
                 "extra_context": ""},
    })
    os.makedirs(os.path.join(orch.runs_dir, "deadchat1234", "wt"))
    # Another project sharing runs_dir, with runs still queued/running.
    other = OrchStore(os.path.join(chat.data_dir(), "orch", "otherproject0.jsonl"))
    for rid, status in (("otherrunning1", "running"), ("otherqueued1", "queued")):
        other.add_run({"id": rid, "from_node": None, "instruction": "x",
                       "agent": "a", "status": status, "created_at": time.time()})
        os.makedirs(os.path.join(orch.runs_dir, rid, "wt"))

    # Reconstruct the service: _recover_chat_runs, then the sweep.
    orch2 = OrchService(chat, executors={"a": WriterExecutor("a.txt")})
    for gone in (crashed, os.path.join(orch.runs_dir, "baseline-x"),
                 os.path.join(orch.runs_dir, "merge-y"),
                 os.path.join(orch.runs_dir, "deadchat1234")):
        assert not os.path.exists(gone), gone
    for kept in ("otherrunning1", "otherqueued1"):
        assert os.path.isdir(os.path.join(orch.runs_dir, kept)), kept
    # The crashed worktree's metadata was pruned from the shadow repo.
    out = subprocess.run(["git", f"--git-dir={orch.shadow.git_dir}",
                          "worktree", "list", "--porcelain"],
                         capture_output=True, text=True).stdout
    assert "deadbeef0001" not in out
    # Every snapshot commit referenced by orch nodes / chat nodes survives.
    commits = {n["snapshot"] for n in orch2.store.nodes.values() if n.get("snapshot")}
    for n in chat.store.all_nodes():
        md = n.metadata if isinstance(n.metadata, dict) else {}
        commits.update(c for c in (md.get("ws_commit"), md.get("ws_parent_commit")) if c)
    assert commits
    for commit in commits:
        proc = subprocess.run(["git", f"--git-dir={orch.shadow.git_dir}",
                               "cat-file", "-e", commit],
                              capture_output=True)
        assert proc.returncode == 0, f"snapshot commit lost: {commit}"


@git_required
def test_sweep_skips_scratch_dirs_in_use(tmp_path, monkeypatch):
    """A merge/baseline scratch dir this process is using survives a sweep
    (e.g. a rebind sweep racing a merge); a stale one is removed."""
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    live = os.path.join(orch.runs_dir, "merge-live", "wt")
    stale = os.path.join(orch.runs_dir, "merge-stale", "wt")
    os.makedirs(live)
    os.makedirs(stale)
    orch._scratch_in_use.add("merge-live")
    orch.sweep_run_dirs()
    assert os.path.isdir(live)
    assert not os.path.exists(os.path.dirname(stale))
