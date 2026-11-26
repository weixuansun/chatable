"""Run dir cleanup: end-of-run wipe (ok/failed/cancelled/crashed), the
node-delete hook, and the startup sweep over the shared runs dir."""
import hashlib
import os
import subprocess
import threading
import time

import pytest

from models import Node
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
    # Another project sharing runs_dir, with stale queued/running runs.
    # Recovery treats those as interrupted too (they are not live here).
    other = OrchStore(os.path.join(chat.data_dir(), "orch", "otherproject0.jsonl"))
    for rid, status in (("otherrunning1", "running"), ("otherqueued1", "queued")):
        other.add_run({"id": rid, "from_node": None, "instruction": "x",
                       "agent": "a", "status": status, "created_at": time.time()})
        os.makedirs(os.path.join(orch.runs_dir, rid, "wt"))

    # Reconstruct the service: _recover_chat_runs, then the sweep.
    orch2 = OrchService(chat, executors={"a": WriterExecutor("a.txt")})
    for gone in (crashed, os.path.join(orch.runs_dir, "baseline-x"),
                 os.path.join(orch.runs_dir, "merge-y"),
                 os.path.join(orch.runs_dir, "deadchat1234"),
                 os.path.join(orch.runs_dir, "otherrunning1"),
                 os.path.join(orch.runs_dir, "otherqueued1")):
        assert not os.path.exists(gone), gone
    # Other-project queued/running runs were not live in this process, so
    # recovery marks them failed (and the sweep then drops their dirs).
    other2 = OrchStore(os.path.join(chat.data_dir(), "orch", "otherproject0.jsonl"))
    assert other2.runs["otherrunning1"]["status"] == "failed"
    assert other2.runs["otherqueued1"]["status"] == "failed"
    assert other2.runs["otherrunning1"]["error"] == "interrupted: server restarted"
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



# ------------------------------------------------ interrupted-run recovery

def _planted(tmp_path, monkeypatch):
    """A chat service and an on-disk orch store with no queue loop yet."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CHATTABLE_ORCH_LIMITS", "a=2,fake=2")
    monkeypatch.setenv("CHATTABLE_RETAIN_FINISHED_TURN_S", "60")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    os.makedirs(chat.workspace().root, exist_ok=True)
    with open(os.path.join(chat.workspace().root, "base.txt"), "w") as fh:
        fh.write("base\n")
    snap = chat.shadow_git().snapshot("test root", allow_empty=True)
    pid = hashlib.sha1(chat.workspace().root.encode()).hexdigest()[:12]
    store = OrchStore(os.path.join(chat.data_dir(), "orch", f"{pid}.jsonl"))
    root = {"id": "rootplant000", "root": True, "snapshot": snap,
            "instruction": None, "parents": [], "agent": None,
            "status": "ok", "adopted": False, "conclusion": {}, "facts": {},
            "created_at": time.time()}
    store.add_node(root)
    return chat, store, snap, root


@git_required
def test_interrupted_orch_run_marked_failed_and_dir_removed(tmp_path, monkeypatch):
    """An orchestrator-view (non-chat) run left 'running' or 'queued' after a
    crash is marked failed on the next OrchService start; its dir is swept
    and any linked DAG node is flipped from running to failed."""
    chat, store, snap, root = _planted(tmp_path, monkeypatch)
    running, queued, node_id = "orchrunning01", "orchqueued001", "orchnode00001"
    store.add_node({
        "id": node_id, "snapshot": snap, "instruction": "half done",
        "parents": [root["id"]], "agent": "a", "status": "running",
        "adopted": False, "conclusion": {}, "facts": {},
        "created_at": time.time(),
    })
    store.add_run({
        "id": running, "from_node": root["id"], "instruction": "half done",
        "agent": "a", "after": None, "status": "running",
        "created_at": time.time(), "started_at": time.time(),
        "finished_at": None, "node_id": node_id, "events": [], "error": "",
        "steps": [], "steps_done": 0,
    })
    store.add_run({
        "id": queued, "from_node": root["id"], "instruction": "never started",
        "agent": "a", "after": None, "status": "queued",
        "created_at": time.time(), "started_at": None,
        "finished_at": None, "node_id": None, "events": [], "error": "",
        "steps": [], "steps_done": 0,
    })
    runs_dir = os.path.join(chat.data_dir(), "orch", "runs")
    worktree_create(chat.shadow_git(), snap, os.path.join(runs_dir, running, "wt"))
    os.makedirs(os.path.join(runs_dir, queued, "wt"))

    orch = OrchService(chat, executors={"a": WriterExecutor("a.txt")})
    for rid in (running, queued):
        run = orch.store.runs[rid]
        assert run["status"] == "failed", run
        assert run["error"] == "interrupted: server restarted"
        assert run.get("finished_at")
        assert not os.path.exists(os.path.join(orch.runs_dir, rid))
    node = orch.store.nodes[node_id]
    assert node["status"] == "failed"
    assert node["facts"]["error"] == "interrupted: server restarted"
    proc = subprocess.run(["git", f"--git-dir={orch.shadow.git_dir}",
                           "cat-file", "-e", snap], capture_output=True)
    assert proc.returncode == 0
    (new_id,) = orch.submit_runs(root["id"], [{"instruction": "write a.txt",
                                               "agents": ["a"]}])
    done = ChatOrchWait(orch).wait(new_id)
    assert done["status"] == "ok"
    new_node = orch.store.nodes[done["node_id"]]
    assert new_node["status"] == "ok" and new_node["snapshot"]


@git_required
def test_interrupted_chat_run_queued_or_running_is_finalized(tmp_path, monkeypatch):
    """Chat-path runs left queued or running (server killed) finalize their
    assistant node as failed and do not auto-resume after restart."""
    chat, store, snap, root = _planted(tmp_path, monkeypatch)
    runs_dir = os.path.join(chat.data_dir(), "orch", "runs")
    planted = []
    for rid, status in (("deadchatrun01", "running"), ("deadchatqueue1", "queued")):
        user = Node.new(role="user", content=f"do {status}", parent_id=ROOT_ID)
        user.metadata = {"orch_run": True, "executor": "a", "ws_commit": snap}
        chat.store.add(user)
        asst = Node.new(role="assistant", content="", parent_id=user.id)
        asst.metadata = {"agent": True, "orch_run": True, "executor": "a",
                         "run_id": rid, "run_status": status,
                         "ws_parent_commit": snap}
        chat.store.add(asst)
        store.add_run({
            "id": rid, "from_node": None, "instruction": f"do {status}",
            "agent": "a", "after": None, "status": status,
            "created_at": time.time(), "started_at": time.time() if status == "running" else None,
            "finished_at": None, "node_id": None, "events": [], "error": "",
            "steps": [], "steps_done": 0,
            "chat": {"user_id": user.id, "assistant_id": asst.id,
                     "parent_id": ROOT_ID, "base_commit": snap, "extra_context": ""},
        })
        os.makedirs(os.path.join(runs_dir, rid, "wt"))
        planted.append((rid, asst.id))

    orch = OrchService(chat, executors={"a": WriterExecutor("a.txt")})
    for rid, aid in planted:
        run = orch.store.runs[rid]
        assert run["status"] == "failed", run
        assert run["error"] == "interrupted: server restarted"
        md = chat.store.get(aid).metadata or {}
        assert md.get("run_status") == "failed"
        assert md.get("stop_reason") == "error"
        assert "运行被中断" in (chat.store.get(aid).content or "")
        assert not os.path.exists(os.path.join(orch.runs_dir, rid))
    # A new chat run still works after recovery.
    (fresh,) = orch.start_chat_runs("write a.txt", None, ["a"])
    _wait(chat, fresh["assistant_id"])
    assert orch.store.runs[fresh["run_id"]]["status"] == "ok"
    assert (chat.store.get(fresh["assistant_id"]).metadata or {}).get("run_status") == "ok"


@git_required
def test_recover_leaves_live_run_alone(tmp_path, monkeypatch):
    """A run this process is actually executing is not marked failed."""
    chat, orch = _svc(tmp_path, monkeypatch, {"a": BlockingExecutor()})
    block = orch.executors["a"]
    (r,) = orch.start_chat_runs("hold", None, ["a"])
    assert block.started.wait(5)
    assert orch.store.runs[r["run_id"]]["status"] == "running"
    wt = os.path.join(orch.runs_dir, r["run_id"], "wt")
    _wait_dir(wt)
    orch._recover_interrupted_runs()
    orch.sweep_run_dirs()
    assert orch.store.runs[r["run_id"]]["status"] == "running"
    assert os.path.isdir(wt)
    assert (chat.store.get(r["assistant_id"]).metadata or {}).get("run_status") == "running"
    block.release.set()
    _wait(chat, r["assistant_id"])
    assert orch.store.runs[r["run_id"]]["status"] == "ok"
