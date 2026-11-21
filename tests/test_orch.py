"""Orchestrator core: store, worktree lifecycle, context file, conclusion
parsing, run queue (limits, dependency, cancel) and end-to-end runs with a
fake executor."""
import os
import time

import pytest

from orch.context import detect_test_cmd, parse_conclusion, write_context
from orch.service import OrchService
from orch.store import OrchStore
from orch.worktree import facts_files, worktree_create, worktree_remove, worktree_snapshot

import service as service_mod
from service import ChatService
from store import TreeStore
from workspace import Workspace

from test_agent_loop import git_required


class FakeExecutor:
    """Tiny in-process executor: writes fake.txt and emits JSON-ish events."""

    def __init__(self, sleep=0.0, exit_code=0):
        self.sleep = sleep
        self.exit_code = exit_code

    def available(self):
        return (True, "")

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        yield {"type": "tool_call", "name": "write_file",
               "arguments": {"path": "fake.txt"}}
        if self.sleep:
            deadline = time.monotonic() + self.sleep
            while time.monotonic() < deadline:
                if should_stop():
                    yield {"type": "result", "final_text": "", "exit_code": -9,
                           "session_id": None, "usage": {}, "killed": "cancelled"}
                    return
                time.sleep(0.02)
        with open(os.path.join(workdir, "fake.txt"), "w") as fh:
            fh.write(f"from run in {os.path.basename(workdir)}\n")
        yield {"type": "tool_result", "id": "c1", "output": "wrote fake.txt"}
        yield {
            "type": "result",
            "final_text": "Wrote fake.txt.\n\n## Conclusion\n### Done\n"
                          "Wrote fake.txt.\n### Key decisions\nNone.\n"
                          "### Not done\nNothing.\n### Next steps\nShip it.",
            "exit_code": self.exit_code,
            "session_id": "fake-session-1",
            "usage": {"input": 1, "output": 1},
        }


def _orch(tmp_path, monkeypatch, fake_exec=None):
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CHATTABLE_ORCH_LIMITS", "fake=1,builtin=3")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    svc = OrchService(chat, executors={"fake": fake_exec or FakeExecutor()})
    return svc, tmp_path


# ---------------------------------------------------------------------- store

def test_store_append_reload_and_path(tmp_path):
    path = str(tmp_path / "orch" / "p.jsonl")
    store = OrchStore(path)
    root = {"id": "root1", "root": True, "snapshot": "c0", "parents": [],
            "created_at": 1.0}
    child = {"id": "n2", "snapshot": "c1", "parents": ["root1"],
             "instruction": "do x", "created_at": 2.0}
    store.add_node(root)
    store.add_node(child)
    store.add_run({"id": "r1", "from_node": "root1", "instruction": "do x",
                   "agent": "fake", "status": "queued", "created_at": 3.0})
    store.update_run("r1", status="ok", node_id="n2")
    store2 = OrchStore(path)  # reload: last occurrence wins
    assert store2.runs["r1"]["status"] == "ok"
    assert [n["id"] for n in store2.path_to_root("n2")] == ["root1", "n2"]
    assert [c["id"] for c in store2.children_of("root1")] == ["n2"]


# ------------------------------------------------------------------- worktree

@git_required
def test_worktree_create_snapshot_facts_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    shadow = chat.shadow_git()
    ws = Workspace(str(tmp_path / "ws"))
    with open(ws.root + "/a.txt", "w") as fh:
        fh.write("v1\n")
    c1 = shadow.snapshot("base", allow_empty=True)

    wt = str(tmp_path / "runs" / "r1" / "wt")
    worktree_create(shadow, c1, wt)
    assert open(os.path.join(wt, "a.txt")).read() == "v1\n"

    # Change the worktree only; snapshot back into the shadow repo.
    with open(os.path.join(wt, "a.txt"), "w") as fh:
        fh.write("v2\n")
    os.makedirs(os.path.join(wt, ".chatable"))
    with open(os.path.join(wt, ".chatable", "CONTEXT.md"), "w") as fh:
        fh.write("handoff")
    c2 = worktree_snapshot(shadow, wt, "run r1")
    assert c2 and c2 != c1
    files = facts_files(shadow, c1, c2)
    assert files == [{"path": "a.txt", "added": 1, "removed": 1}]
    # .chatable/ was never snapshotted.
    tracked = shadow._git("ls-tree", "-r", "--name-only", c2).stdout
    assert ".chatable" not in tracked
    # The main workspace is untouched by all of this.
    assert open(ws.root + "/a.txt").read() == "v1\n"

    worktree_remove(shadow, wt)
    assert not os.path.exists(wt)


# ------------------------------------------------------------------- context

def test_detect_test_cmd(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    assert detect_test_cmd(str(tmp_path)) == "npm test"
    (tmp_path / "package.json").unlink()
    (tmp_path / "conftest.py").write_text("")
    assert detect_test_cmd(str(tmp_path)) == "pytest -q"


def test_context_file_layers(tmp_path):
    store = OrchStore(str(tmp_path / "p.jsonl"))
    store.add_node({"id": "root", "root": True, "snapshot": "c0", "parents": []})
    store.add_node({
        "id": "n1", "snapshot": "c1", "parents": ["root"],
        "instruction": "add counts_by",
        "conclusion": {"done": "Implemented counts_by with Counter.",
                       "decisions": "Used approach A.", "not_done": "",
                       "next": "Add docs."},
        "facts": {"files": [{"path": "more.py", "added": 12, "removed": 0}]},
        "created_at": 1.0,
    })
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "AGENTS.md").write_text("Always run pytest -q before finishing.")
    path = write_context(store, store.nodes["n1"], "improve the docs",
                         "pytest -q", str(wt))
    content = open(path).read()
    assert "## Project" in content and "Always run pytest -q" in content
    assert "pytest -q" in content
    assert "## Path so far" in content
    assert "add counts_by" in content
    assert "Implemented counts_by with Counter." in content
    assert "more.py (+12/-0)" in content
    assert "## Task" in content and "improve the docs" in content


def test_parse_conclusion_variants():
    full = parse_conclusion(
        "Some work.\n\n## Conclusion\n### Done\nWrote a.py and tests.\n"
        "### Key decisions\nUsed Counter.\n### Not done\nDocs.\n"
        "### Next steps\nWrite docs.")
    assert full["done"] == "Wrote a.py and tests."
    assert full["decisions"] == "Used Counter."
    assert full["not_done"] == "Docs."
    assert full["next"] == "Write docs."

    bold = parse_conclusion("## Conclusion\n**Done**: did the fix\n**Next steps**: none")
    assert bold["done"] == "did the fix"
    assert bold["next"] == "none"

    assert parse_conclusion("no section at all")["done"] == "no section at all"
    assert parse_conclusion("")["done"] == ""


# ------------------------------------------------------------ runs and queue

def test_run_lifecycle_with_fake_executor(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    ids = svc.submit_runs(root["id"], [{"instruction": "make fake.txt",
                                        "agents": ["fake"]}])
    run = ChatOrchWait(svc).wait(ids[0])
    assert run["status"] == "ok"
    node = svc.store.nodes[run["node_id"]]
    assert node["parents"] == [root["id"]]
    assert node["agent"] == "fake"
    assert node["conclusion"]["done"] == "Wrote fake.txt."
    assert node["session_id"] == "fake-session-1"
    assert node["snapshot"] != root["snapshot"]
    assert {f["path"] for f in node["facts"]["files"]} == {"fake.txt"}
    assert node["facts"]["exit_code"] == 0
    assert node["events"], "run transcript persisted on the node"


def test_queue_respects_per_agent_limit(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch, FakeExecutor(sleep=0.4))
    root = svc.store.root_node()
    ids = svc.submit_runs(root["id"], [
        {"instruction": "job one", "agents": ["fake"]},
        {"instruction": "job two", "agents": ["fake"]},
    ])
    time.sleep(0.15)  # fake=1: the second must still be queued
    statuses = {i: svc.store.runs[i]["status"] for i in ids}
    assert statuses[ids[0]] == "running"
    assert statuses[ids[1]] == "queued"
    w = ChatOrchWait(svc)
    assert w.wait(ids[0])["status"] == "ok"
    assert w.wait(ids[1])["status"] == "ok"


def test_dependency_rebases_onto_result_node(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    first = svc.submit_runs(root["id"], [{"instruction": "step one",
                                          "agents": ["fake"]}])[0]
    second = svc.submit_runs(root["id"], [{"instruction": "step two",
                                           "agents": ["fake"], "after": first}])[0]
    w = ChatOrchWait(svc)
    run1 = w.wait(first)
    run2 = w.wait(second)
    assert run2["status"] == "ok"
    # The dependent run started from the first run's resulting node.
    assert run2["from_node"] == run1["node_id"]
    node2 = svc.store.nodes[run2["node_id"]]
    assert node2["parents"] == [run1["node_id"]]


def test_cancel_queued_run(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch, FakeExecutor(sleep=1.0))
    root = svc.store.root_node()
    ids = svc.submit_runs(root["id"], [
        {"instruction": "job one", "agents": ["fake"]},
        {"instruction": "job two", "agents": ["fake"]},
    ])
    time.sleep(0.2)
    assert svc.cancel_run(ids[1]) is True
    assert svc.store.runs[ids[1]]["status"] == "cancelled"
    assert svc.cancel_run("nope") is False


def test_failed_exit_produces_failed_node(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch, FakeExecutor(exit_code=1))
    root = svc.store.root_node()
    ids = svc.submit_runs(root["id"], [{"instruction": "doomed",
                                        "agents": ["fake"]}])
    run = ChatOrchWait(svc).wait(ids[0])
    assert run["status"] == "failed"
    node = svc.store.nodes[run["node_id"]]
    assert node["status"] == "failed"
    assert node["facts"]["exit_code"] == 1


def test_unknown_executor_rejected(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    with pytest.raises(ValueError, match="unknown executor"):
        svc.submit_runs(root["id"], [{"instruction": "x", "agents": ["nope"]}])
    with pytest.raises(KeyError):
        svc.submit_runs("no-such-node", [{"instruction": "x", "agents": ["fake"]}])


class ChatOrchWait:
    def __init__(self, svc):
        self.svc = svc

    def wait(self, run_id, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.svc.store.runs[run_id]
            if run["status"] not in ("queued", "running"):
                return run
            time.sleep(0.05)
        raise AssertionError(f"run {run_id} did not finish: {run['status']}")


@git_required
def test_adopt_and_undo(tmp_path, monkeypatch):
    svc, tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    run = ChatOrchWait(svc).wait(svc.submit_runs(
        root["id"], [{"instruction": "make fake.txt", "agents": ["fake"]}])[0])
    res = svc.adopt(run["node_id"])
    assert res["ok"]
    ws_file = os.path.join(svc.workspace.root, "fake.txt")
    assert os.path.exists(ws_file)
    node = svc.store.nodes[run["node_id"]]
    assert node["adopted"] is True
    undo = svc.undo_adopt(res["previous_commit"])
    assert undo["ok"]
    assert not os.path.exists(ws_file)
