"""CHA-5: agent runs inside the conversation tree (unified mode)."""
import os
import time

import pytest

from orch.context import build_chat_context
from orch.service import OrchService
from service import ChatService
from store import ROOT_ID, TreeStore

from test_agent_loop import git_required
from test_orch import FakeExecutor


class WriterExecutor(FakeExecutor):
    """Writes a given file (so parallel runs touch different files)."""

    def __init__(self, fname, content="x\n", **kw):
        super().__init__(**kw)
        self.fname, self.content = fname, content
        self.prompts = []

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        self.prompts.append((workdir, session_id,
                             open(os.path.join(workdir, ".chatable", "CONTEXT.md")).read()))
        yield {"type": "tool_call", "id": "c1", "name": "write_file",
               "arguments": {"path": self.fname}}
        if self.sleep:
            deadline = time.monotonic() + self.sleep
            while time.monotonic() < deadline:
                if should_stop():
                    yield {"type": "result", "final_text": "", "exit_code": -9,
                           "session_id": None, "usage": {}, "killed": "cancelled"}
                    return
                time.sleep(0.02)
        with open(os.path.join(workdir, self.fname), "w") as fh:
            fh.write(self.content)
        yield {"type": "tool_result", "id": "c1", "output": f"wrote {self.fname}"}
        yield {"type": "result", "final_text": f"Wrote {self.fname}.\n\n## Conclusion\n"
               f"### Done\nWrote {self.fname}.", "exit_code": self.exit_code,
               "session_id": f"sess-{self.fname}", "usage": {}}


def _svc(tmp_path, monkeypatch, executors):
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CHATTABLE_ORCH_LIMITS", "a=2,b=2,c=2")
    monkeypatch.setenv("CHATTABLE_RETAIN_FINISHED_TURN_S", "60")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    orch = OrchService(chat, executors=executors)
    chat.run_stop_hook = orch.stop_chat_run
    os.makedirs(chat.workspace().root, exist_ok=True)
    with open(os.path.join(chat.workspace().root, "base.txt"), "w") as fh:
        fh.write("base\n")
    return chat, orch


def _wait(chat, aid, timeout=20):
    events = list(chat.subscribe(aid, 0))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        md = chat.store.get(aid).metadata or {}
        if md.get("run_status") not in (None, "queued", "running"):
            return events, md
        time.sleep(0.05)
    raise AssertionError("run did not finish")


@git_required
def test_run_creates_tree_nodes_steps_and_snapshot(tmp_path, monkeypatch):
    ex = WriterExecutor("a.txt")
    chat, orch = _svc(tmp_path, monkeypatch, {"a": ex})
    (r,) = orch.start_chat_runs("make a.txt", None, ["a"])
    events, md = _wait(chat, r["assistant_id"])
    user = chat.store.get(r["user_id"])
    asst = chat.store.get(r["assistant_id"])
    assert user.parent_id == ROOT_ID and user.content == "make a.txt"
    assert user.metadata["orch_run"] and user.metadata["executor"] == "a"
    assert asst.content.startswith("Wrote a.txt")
    assert md["run_status"] == "ok" and md["files_changed"] == 1
    assert md["ws_commit"] != md["ws_parent_commit"]
    assert md["session_id"] == "sess-a.txt"
    # Steps streamed + persisted like agent steps (chat view only).
    types = [e["type"] for e in events]
    assert "step_start" in types and "step_done" in types and types[-1] == "done"
    steps = [n for n in chat.store.all_nodes() if n.role == "step"]
    assert len(steps) == 1 and steps[0].parent_id == asst.id
    assert steps[0].metadata["status"] == "ok"
    # The real workspace is untouched until adopt; the node actions work.
    ws = chat.workspace().root
    assert not os.path.exists(os.path.join(ws, "a.txt"))
    d = chat.node_diff(asst.id)
    assert [f["path"] for f in d["files"]] == ["a.txt"]
    chat.adopt_snapshot(asst.id)
    assert open(os.path.join(ws, "a.txt")).read() == "x\n"
    # Tree / path payloads carry the run fields for the map + chat.
    msgs = chat.path_messages(asst.id)
    assert msgs[1]["run_status"] == "ok" and msgs[1]["executor"] == "a"


@git_required
def test_parallel_runs_from_one_node_then_merge(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch,
                      {"a": WriterExecutor("a.txt"), "b": WriterExecutor("b.txt"),
                       "c": WriterExecutor("c.txt")})
    (r0,) = orch.start_chat_runs("setup", None, ["c"])
    _wait(chat, r0["assistant_id"])
    p = r0["assistant_id"]
    (ra,) = orch.start_chat_runs("do A", p, ["a"])
    (rb,) = orch.start_chat_runs("do B", p, ["b"])
    _wait(chat, ra["assistant_id"]); _wait(chat, rb["assistant_id"])
    assert chat.store.get(ra["user_id"]).parent_id == p
    assert chat.store.get(rb["user_id"]).parent_id == p
    m = orch.start_chat_merge([ra["assistant_id"], rb["assistant_id"]], "a")
    assert m["parent_id"] == p  # merge node hangs off the common ancestor
    _ev, md = _wait(chat, m["assistant_id"])
    assert md["run_status"] == "ok" and md["orch_merge"]
    files = sorted(f["path"] for f in chat.node_diff(m["assistant_id"])["files"])
    assert files == ["a.txt", "b.txt"]
    content = chat.store.get(m["assistant_id"]).content
    assert "do A" in content and "do B" in content  # both branches in context
    with pytest.raises(ValueError):
        orch.start_chat_merge([ra["assistant_id"]], "a")


@git_required
def test_merge_under_a_plain_chat_node_without_snapshot(tmp_path, monkeypatch):
    """Runs dispatched from an ordinary chat reply (no ws_commit of its own):
    the merge still gets a base (git merge-base) so diff / files changed work."""
    from models import Node
    chat, orch = _svc(tmp_path, monkeypatch,
                      {"a": WriterExecutor("a.txt"), "b": WriterExecutor("b.txt")})
    u = Node.new(role="user", content="hi", parent_id=ROOT_ID)
    chat.store.add(u)
    r = Node.new(role="assistant", content="hello", parent_id=u.id)
    chat.store.add(r)
    (ra,) = orch.start_chat_runs("do A", r.id, ["a"])
    (rb,) = orch.start_chat_runs("do B", r.id, ["b"])
    _wait(chat, ra["assistant_id"]); _wait(chat, rb["assistant_id"])
    m = orch.start_chat_merge([ra["assistant_id"], rb["assistant_id"]], "a")
    assert m["parent_id"] == r.id
    _ev, md = _wait(chat, m["assistant_id"])
    assert md["run_status"] == "ok" and md["ws_parent_commit"]
    assert md["files_changed"] == 2
    files = sorted(f["path"] for f in chat.node_diff(m["assistant_id"])["files"])
    assert files == ["a.txt", "b.txt"]


@git_required
def test_racing_same_instruction_on_two_agents(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch,
                      {"a": WriterExecutor("x.txt", "A\n"), "b": WriterExecutor("x.txt", "B\n")})
    runs = orch.start_chat_runs("write x", None, ["a", "b"])
    assert len(runs) == 2 and {r["agent"] for r in runs} == {"a", "b"}
    for r in runs:
        _wait(chat, r["assistant_id"])
    texts = {r["agent"]: chat.store.get(r["assistant_id"]).metadata["ws_commit"] for r in runs}
    assert texts["a"] != texts["b"]


@git_required
def test_context_comes_from_the_conversation_path_and_resumes_session(tmp_path, monkeypatch):
    ex = WriterExecutor("a.txt")
    chat, orch = _svc(tmp_path, monkeypatch, {"a": ex})
    # A plain chat turn first (stored directly: no model call needed).
    from models import Node
    u = Node.new(role="user", content="我们要做一个计算器", parent_id=ROOT_ID)
    chat.store.add(u)
    a = Node.new(role="assistant", content="好的，先写 add 函数。", parent_id=u.id)
    chat.store.add(a)
    (r1,) = orch.start_chat_runs("first run", a.id, ["a"])
    _wait(chat, r1["assistant_id"])
    (r2,) = orch.start_chat_runs("second run", r1["assistant_id"], ["a"])
    _wait(chat, r2["assistant_id"])
    _wt, session, ctx = ex.prompts[-1]
    assert "我们要做一个计算器" in ctx and "好的，先写 add 函数" in ctx
    assert "first run" in ctx and "Wrote a.txt" in ctx
    assert "## Task\n\nsecond run" in ctx
    assert "+x" in ctx  # diff of the parent node
    assert session == "sess-a.txt"  # same agent → resumed session


@git_required
def test_stop_cancels_a_running_run_and_keeps_a_node(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt", sleep=5)})
    (r,) = orch.start_chat_runs("slow", None, ["a"])
    deadline = time.monotonic() + 5
    while (chat.store.get(r["assistant_id"]).metadata or {}).get("run_status") != "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert chat.request_stop(r["assistant_id"]) is True
    _ev, md = _wait(chat, r["assistant_id"])
    assert md["run_status"] == "cancelled" and md["stop_reason"] == "stopped"
    assert md["ws_commit"]


def test_explicit_agent_false_is_plain_chat_in_an_agent_conversation(tmp_path, monkeypatch):
    """Stale agent_mode metadata on old nodes is ignored: agent=False (the
    default for chat sends) always runs a plain chat turn."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    from models import Node
    u = Node.new(role="user", content="q", parent_id=ROOT_ID)
    u.metadata = {"agent_mode": True}  # legacy conversation-level flag
    chat.store.add(u)
    import service as service_mod
    from unittest.mock import patch

    def fake(messages, tools=None):
        yield ("delta", "hi")
        yield ("done", {"input": 1, "output": 1})
    with patch.object(service_mod, "stream_backend", fake):
        info = chat.start_turn("plain", u.id, agent=False)
        list(chat.subscribe(info["assistant_id"], 0))
    assert info["agent"] is False


def test_build_chat_context_layers(tmp_path):
    from models import Node
    nodes = []
    for i in range(10):
        n = Node.new(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}", parent_id=None)
        nodes.append(n)
    ctx = build_chat_context(nodes, "do it", "pytest -q", str(tmp_path), notes="N",
                             parent_diff="+x", extra="## Merge\nkeep")
    assert "older turns" in ctx and "- User: msg 0" in ctx
    assert "### Assistant\nmsg 9" in ctx and "Test command: `pytest -q`" in ctx
    assert ctx.rstrip().endswith("do it") and "## Merge" in ctx


@git_required
def test_orch_follows_chat_store_rebind(tmp_path, monkeypatch):
    """--data-dir / the settings storage switch rebind the chat service in
    place; the orchestrator must move its store, runs and shadow repo too."""
    monkeypatch.delenv("CHATTABLE_WORKSPACE", raising=False)
    chat = ChatService(TreeStore(str(tmp_path / "old" / "chatable.jsonl")))
    orch = OrchService(chat, executors={"a": WriterExecutor("a.txt")})
    assert orch.data_dir == str(tmp_path / "old")
    chat.rebind_store(str(tmp_path / "new"))
    orch.rebind()
    new = str(tmp_path / "new")
    assert orch.data_dir == new
    assert orch.workspace.root.startswith(os.path.realpath(new))
    assert orch.runs_dir == os.path.join(new, "orch", "runs")
    assert os.path.isdir(os.path.join(new, "orch"))
    assert orch.store.root_node()  # fresh root in the new folder

    os.makedirs(orch.workspace.root, exist_ok=True)
    runs = orch.start_chat_runs("write a", ROOT_ID, ["a"])
    _events, md = _wait(chat, runs[0]["assistant_id"])
    assert md["run_status"] == "ok"
    assert os.path.isdir(os.path.join(new, "orch", "runs"))
    assert not os.path.exists(tmp_path / "old" / "orch" / "runs")


@git_required
def test_orch_rebind_refuses_while_busy(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt", sleep=2)})
    orch.start_chat_runs("write a", ROOT_ID, ["a"])
    assert orch.busy()
    with pytest.raises(RuntimeError):
        orch.rebind()
    for r in list(orch.store.runs.values()):
        orch.cancel_run(r["id"])
