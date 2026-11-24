"""Agent loop: step nodes, snapshots, limits, token gate.

Agent turns are internal to the orchestrator: ``start_turn(agent=True)``
requires ``agent_overrides`` pointing at an isolated Workspace+ShadowGit
(the way orch/executors.py's BuiltinExecutor runs it).

The model backend is faked exactly like tests/test_core.py does: a scripted
generator of ("delta"|"tool_calls"|"done") events patched over
``service.stream_backend``.
"""
import os
import shutil
import json
import threading
import time
from unittest.mock import patch

import pytest

import service as service_mod
from service import ChatService
from store import TreeStore
from tools import ToolResult
from workspace import WorkspaceError

git_required = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """A ChatService with agent mode fully usable (token + auto-approve)."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("CHATTABLE_AGENT_AUTO_APPROVE", "all")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    return ChatService(TreeStore(str(tmp_path / "chat.jsonl")))


def make_fake(script):
    """script: list of rounds; each round is a list of stream events."""
    calls = []

    def fake_stream(messages, tools=None):
        calls.append({"messages": list(messages), "tools": tools})
        round_events = script[min(len(calls) - 1, len(script) - 1)]
        yield from round_events

    return fake_stream, calls


def agent_overrides(svc, auto_approve=True):
    """Isolated workspace+shadow for start_turn(agent=True), the way the
    orchestrator's BuiltinExecutor passes them. Both live under tmp_path:
    the workspace via CHATTABLE_WORKSPACE, the shadow repo under the data dir."""
    return {"workspace": svc.workspace(), "shadow": svc.shadow_git(),
            "auto_approve": auto_approve}


def run_turn(svc, script, content="do it", **kwargs):
    kwargs.setdefault("agent", True)
    if kwargs["agent"]:
        kwargs.setdefault("agent_overrides", agent_overrides(svc))
    fake, calls = make_fake(script)
    with patch.object(service_mod, "stream_backend", fake):
        info = svc.start_turn(content, kwargs.pop("parent_id", None), **kwargs)
        events = list(svc.subscribe(info["assistant_id"], 0))
    return info, events, calls


def step_nodes(svc):
    return [n for n in svc.store.all_nodes() if n.role == "step"]


def tool_call(name, **arguments):
    return [("tool_calls", [{"id": f"c_{name}", "name": name, "arguments": arguments}]),
            ("done", {"input": 1, "output": 1})]


ANSWER = [("delta", "all done"), ("done", {"input": 1, "output": 1})]


# ---------------------------------------------------------------------- basics

def test_agent_turn_requires_overrides(svc):
    """agent=True is orch-internal: it refuses without an isolated
    workspace+shadow in agent_overrides, and persists nothing."""
    for bad in (None, {}, {"workspace": svc.workspace()},
                {"shadow": svc.shadow_git()}):
        with pytest.raises(ValueError, match="agent_overrides"):
            svc.start_turn("do it", None, agent=True, agent_overrides=bad)
    assert svc.store.all_nodes() == []


def test_agent_turn_with_full_overrides_runs(svc):
    info, _events, _calls = run_turn(svc, [ANSWER])
    node = svc.store.get(info["assistant_id"])
    assert node.content == "all done"
    assert node.metadata["agent"] is True
    # …and the turn ran against the overridden workspace, not the main one.
    assert info["agent"] is True


def test_agent_turn_persists_step_nodes_chained(svc):
    info, events, calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="hello\n"),
        tool_call("read_file", path="a.txt"),
        ANSWER,
    ])
    node = svc.store.get(info["assistant_id"])
    assert node.content == "all done"
    assert node.metadata["agent"] is True
    assert node.metadata["tool_messages"]  # native replay still persisted

    steps = step_nodes(svc)
    assert len(steps) == 2
    write_step, read_step = steps
    # Chained: first step under the assistant turn, each step under the last.
    assert write_step.parent_id == info["assistant_id"]
    assert read_step.parent_id == write_step.id
    assert write_step.metadata["tool"] == "write_file"
    assert write_step.metadata["status"] == "ok"
    assert write_step.metadata["args_summary"] == "write_file a.txt"
    assert "diff" in write_step.metadata
    assert read_step.metadata["status"] == "ok"
    assert "hello" in read_step.metadata["result"]

    # The file was really written to the workspace.
    with open(os.path.join(svc.workspace().root, "a.txt")) as fh:
        assert fh.read() == "hello\n"
    # The model got the tool result folded back as a native tool message.
    assert calls[1]["messages"][-1]["role"] == "tool"
    assert "wrote a.txt" in calls[1]["messages"][-1]["content"]
    # SSE: step lifecycle events were emitted.
    types = [e["type"] for e in events]
    assert "step_start" in types and "step_done" in types


def test_step_nodes_visible_in_tree_and_path(svc):
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="x\n"),
        ANSWER,
    ])
    step = step_nodes(svc)[0]

    def roles(tree_dict):
        out = [tree_dict.get("role")]
        for kid in tree_dict.get("children", []):
            out.extend(roles(kid))
        return out

    assert "step" in roles(svc.tree())
    msgs = svc.path_messages(step.id)
    assert [m["role"] for m in msgs] == ["user", "assistant", "step"]
    assert msgs[-1]["tool"] == "write_file"
    detail = svc.node_detail(step.id)
    assert detail["metadata"]["tool"] == "write_file"


def test_path_messages_expands_steps_below_the_assistant_node(svc):
    """Reopening a conversation at the assistant reply must show its steps."""
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("read_file", path="a.txt"),
        ANSWER,
    ])
    steps = step_nodes(svc)
    msgs = svc.path_messages(info["assistant_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "step", "step"]
    assert [m["id"] for m in msgs[2:]] == [s.id for s in steps]
    # The step payload mirrors the live view (status, diff, duration, commit).
    md = msgs[2]["step"]
    assert md["tool"] == "write_file" and md["status"] == "ok"
    assert md["diff"]
    assert not md["ws_commit"]  # node granularity: no per-step snapshot
    assert msgs[1]["ws_commit"]  # …the assistant node carries the snapshot
    assert "duration_ms" in md
    # Selecting a step itself: steps come from the path, with no duplicates.
    msgs2 = svc.path_messages(steps[1].id)
    assert [m["role"] for m in msgs2] == ["user", "assistant", "step", "step"]
    assert [m["id"] for m in msgs2] == [msgs[0]["id"], msgs[1]["id"],
                                        steps[0].id, steps[1].id]


def test_context_excludes_step_nodes_but_stays_valid(svc):
    """Forking from a step: the next request must be a valid OpenAI sequence —
    no role=step messages, and assistant tool_calls answered by tool messages."""
    info, _e, _c = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="x\n"),
        ANSWER,
    ])
    step = step_nodes(svc)[0]
    _info2, _e2, calls2 = run_turn(svc, [ANSWER], content="continue",
                                   parent_id=step.id)
    sent = calls2[0]["messages"]
    assert all(m["role"] != "step" for m in sent)
    for i, m in enumerate(sent):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            following = sent[i + 1:i + 1 + len(ids)]
            assert [t.get("tool_call_id") for t in following] == ids


# ---------------------------------------------------------------------- limits

def test_agent_mode_allows_more_than_10_rounds(svc):
    script = [tool_call("read_file", path="f.txt")] * 12 + [ANSWER]
    _info, _events, calls = run_turn(svc, script)
    assert len(step_nodes(svc)) == 12
    assert len(calls) == 13  # 12 tool rounds + the final answer round


def test_agent_mode_does_not_dedupe_repeated_calls(svc):
    script = [tool_call("read_file", path="same.txt")] * 3 + [ANSWER]
    _info, _events, calls = run_turn(svc, script)
    steps = step_nodes(svc)
    assert len(steps) == 3
    assert all(s.metadata["tool"] == "read_file" for s in steps)
    # Each repeated call really ran against the workspace tool registry.
    assert all("no such file" in s.metadata["result"] or "cannot read" in s.metadata["result"]
               or s.metadata["status"] == "error" for s in steps)
    # And each round's tool result was folded back.
    assert sum(1 for c in calls for m in c["messages"] if m["role"] == "tool") >= 3


def test_normal_mode_still_dedupes_and_caps(tmp_path, monkeypatch):
    """Non-agent turns are untouched: dedupe on, 3 calls/round, 10 rounds."""
    monkeypatch.delenv("CHATTABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATABLE_AUTH_TOKEN", raising=False)
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    ran = []

    def fake_stream(messages, tools=None):
        if tools is not None:
            yield ("tool_calls", [{"id": "c1", "name": "web_search",
                                   "arguments": {"query": "same"}}])
            yield ("done", {"input": 1, "output": 1})
        else:
            yield ("delta", "final")
            yield ("done", {"input": 1, "output": 1})

    def fake_run(name, arguments):
        ran.append(name)
        return ToolResult(call_id="t", name=name, arguments=arguments,
                          ok=True, output="out")

    with patch.object(service_mod, "stream_backend", fake_stream), \
            patch.object(service_mod.TOOL_REGISTRY, "run", fake_run), \
            patch.object(service_mod, "AUTO_TOOL_MAX_ROUNDS", 3):
        info = svc.start_turn("q", None)
        list(svc.subscribe(info["assistant_id"], 0))
    assert ran == ["web_search"]  # identical calls deduped, fuse stopped the loop
    assert step_nodes(svc) == []  # no step nodes outside agent mode


# --------------------------------------------------------------- snapshots

@git_required
def test_turn_takes_one_node_snapshot_not_per_step(svc):
    """Node granularity: steps keep inline diffs but no ws_commit; the turn
    is snapshotted exactly once, onto the assistant node."""
    shadow = svc.shadow_git()
    info, events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("write_file", path="a.txt", content="v2\n"),
        tool_call("write_file", path="b.txt", content="b\n"),
        ANSWER,
    ])
    steps = step_nodes(svc)
    assert len(steps) == 3
    for s in steps:
        assert "ws_commit" not in s.metadata
        assert s.metadata["diff"]  # each step still shows its own change
    assert all(e.get("ws_commit") is None for e in events if e.get("type") == "step_done")
    node = svc.store.get(info["assistant_id"])
    base = svc.store.get(info["user_id"]).metadata["ws_commit"]
    assert node.metadata["ws_commit"] and node.metadata["ws_commit"] != base
    assert node.metadata["ws_parent_commit"] == base
    # Exactly two shadow commits for the turn: the base + the node snapshot.
    log = shadow._git("rev-list", "--count", "HEAD").stdout.strip()
    assert log == "2"

    # Restore/adopt resolve to the node snapshot.
    target = os.path.join(svc.workspace().root, "a.txt")
    open(target, "w").write("scribble\n")
    res = svc.adopt_snapshot(info["assistant_id"])
    assert res["ok"] and res["adopted_commit"] == node.metadata["ws_commit"]
    assert open(target).read() == "v2\n"


@git_required
def test_node_diff_covers_the_whole_turn(svc):
    with open(os.path.join(svc.workspace().root, "a.txt"), "w") as fh:
        fh.write("orig\n")
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("write_file", path="a.txt", content="v2\n"),
        tool_call("run_shell", command="echo from-shell > shell.txt"),
        ANSWER,
    ])
    d = svc.node_diff(info["assistant_id"])
    assert d["ok"]
    assert "-orig" in d["diff"] and "+v2" in d["diff"] and "+v1" not in d["diff"]
    assert "shell.txt" in d["diff"] and "+from-shell" in d["diff"]
    assert sorted(f["path"] for f in d["files"]) == ["a.txt", "shell.txt"]
    # A second turn's diff is relative to ITS start, not the first turn's.
    info2, _e, _c = run_turn(svc, [tool_call("write_file", path="c.txt", content="c\n"),
                                   ANSWER], parent_id=info["assistant_id"])
    d2 = svc.node_diff(info2["assistant_id"])
    assert [f["path"] for f in d2["files"]] == ["c.txt"]


@git_required
def test_old_step_level_snapshots_still_resolve(svc):
    """Conversations saved before node snapshots: the assistant node has no
    ws_commit, only its steps do → fall back to the last step's commit."""
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        ANSWER,
    ])
    shadow = svc.shadow_git()
    node = svc.store.get(info["assistant_id"])
    commit = node.metadata["ws_commit"]
    (step,) = step_nodes(svc)
    # Rewrite into the legacy shape.
    svc.store.update_metadata(step.id, {"ws_commit": commit})
    svc.store.update_metadata(info["assistant_id"],
                              {"ws_commit": None, "ws_parent_commit": None})
    assert not svc.store.get(info["assistant_id"]).metadata.get("ws_commit")
    assert svc.nearest_ws_commit(info["assistant_id"]) == commit
    msgs = svc.path_messages(info["assistant_id"])
    assert msgs[1]["ws_commit"] == commit
    d = svc.node_diff(info["assistant_id"])
    assert "+v1" in d["diff"]
    assert shadow.available


@git_required
def test_crashed_turn_still_snapshots_the_node(svc):
    def boom(messages, tools=None):
        if any(m.get("role") == "tool" for m in messages):
            raise RuntimeError("upstream exploded")
        yield ("tool_calls", [{"id": "c1", "name": "write_file",
                               "arguments": {"path": "a.txt", "content": "x\n"}}])
        yield ("done", {"input": 1, "output": 1})
    with patch.object(service_mod, "stream_backend", boom):
        info = svc.start_turn("go", None, agent=True,
                              agent_overrides=agent_overrides(svc))
        list(svc.subscribe(info["assistant_id"], 0))
    md = svc.store.get(info["assistant_id"]).metadata
    assert md.get("ws_commit")
    d = svc.node_diff(info["assistant_id"])
    assert "+x" in d["diff"]


@git_required
def test_readonly_turn_records_a_base_snapshot(svc):
    """Every agent turn snapshots the pre-turn workspace onto the user node,
    so even a turn with no writes has a base its diff is computed against."""
    info, _events, _calls = run_turn(svc, [ANSWER])
    trunk = svc.store.get(info["user_id"])
    base = trunk.metadata.get("ws_commit")
    assert base
    assert svc.node_diff(info["assistant_id"])["ok"]


# ------------------------------------------------------------- mode + token

def test_agent_mode_available_on_loopback_without_token(tmp_path, monkeypatch):
    """Default bind (127.0.0.1): agent mode needs no access token."""
    monkeypatch.delenv("CHATTABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATTABLE_BIND_HOST", raising=False)
    monkeypatch.delenv("CHATTABLE_WEB_HOST", raising=False)
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    assert svc.agent_status()["available"] is True
    for host in ("localhost", "::1", "[::1]", "127.0.0.2"):
        monkeypatch.setenv("CHATTABLE_BIND_HOST", host)
        assert service_mod.agent_available() is True, host
    for host in ("0.0.0.0", "192.168.1.5", "::", "myhost"):
        monkeypatch.setenv("CHATTABLE_BIND_HOST", host)
        assert service_mod.agent_available() is False, host
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "t")
    assert service_mod.agent_available() is True


def test_agent_mode_requires_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTABLE_BIND_HOST", "0.0.0.0")
    monkeypatch.delenv("CHATTABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATABLE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    with pytest.raises(ValueError, match="CHATTABLE_AUTH_TOKEN"):
        svc.start_turn("do agent things", None, agent=True)
    # Nothing was persisted for the refused turn.
    assert svc.store.all_nodes() == []
    status = svc.agent_status()
    assert status["available"] is False
    assert "CHATTABLE_AUTH_TOKEN" in status["reason"]


def test_agent_status_reports_workspace(svc, tmp_path):
    status = svc.agent_status()
    assert status["available"] is True
    assert status["workspace"] == os.path.realpath(str(tmp_path / "ws"))
    assert status["auto_approve"] == ["all"]
    assert os.path.isdir(status["workspace"])  # created on demand


# -------------------------------------------------- v3: limits, plan, streaming

def test_round_limit_forces_a_final_summary(svc, monkeypatch):
    monkeypatch.setattr(service_mod, "AGENT_MAX_ROUNDS", 2)
    script = [tool_call("list_dir", path="."), tool_call("list_dir", path="."), ANSWER]
    info, events, calls = run_turn(svc, script)
    assert svc.store.get(info["assistant_id"]).content == "all done"
    notices = [e for e in events if e["type"] == "notice"]
    assert notices and "round limit" in notices[0]["text"]
    assert len(calls) == 3  # 2 tool rounds + the wrap-up round…
    assert calls[-1]["tools"] is None  # …which runs without tools


def test_long_turn_shrinks_old_tool_results(svc, monkeypatch):
    monkeypatch.setattr(service_mod, "AGENT_CTX_BUDGET_TOKENS", 300)
    with open(os.path.join(svc.workspace().root, "big.txt"), "w") as fh:
        fh.write("x" * 4000)
    script = [tool_call("read_file", path="big.txt")] * 8 + [ANSWER]
    _info, _events, calls = run_turn(svc, script)
    final = calls[-1]["messages"]
    tool_msgs = [m for m in final if m["role"] == "tool"]
    assert len(tool_msgs) == 8
    # The oldest results are shrunk with a note; the last 6 stay intact.
    assert all("older tool output shrunk" in m["content"] for m in tool_msgs[:2])
    assert not any("older tool output shrunk" in m["content"] for m in tool_msgs[2:])


def test_wrap_up_record_with_changes_and_stop_reason(svc, monkeypatch):
    """The round-limit wrap-up carries the server-computed factual record."""
    monkeypatch.setattr(service_mod, "AGENT_MAX_ROUNDS", 1)
    info, _events, calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        ANSWER,
    ])
    node = svc.store.get(info["assistant_id"])
    assert node.metadata["stop_reason"] == "round_limit"
    assert node.metadata["incomplete"] is True
    nudge = calls[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "Files changed this turn" in nudge["content"]
    assert "a.txt" in nudge["content"]
    assert "+1" in nudge["content"]
    assert "Answer in the language of the user's request" in nudge["content"]
    # The record is English, unlike the old Chinese nudge.
    assert "系统提示" not in nudge["content"]


def test_wrap_up_record_without_changes(svc, monkeypatch):
    monkeypatch.setattr(service_mod, "AGENT_MAX_ROUNDS", 1)
    with open(os.path.join(svc.workspace().root, "x.txt"), "w") as fh:
        fh.write("x\n")
    _i, _e, calls = run_turn(svc, [
        tool_call("read_file", path="x.txt"),
        ANSWER,
    ])
    nudge = calls[1]["messages"][-1]["content"]
    assert "Files changed this turn: NONE." in nudge
    assert "Shell commands this turn: none." in nudge


def test_wrap_up_record_includes_shell_results(svc, monkeypatch):
    monkeypatch.setattr(service_mod, "AGENT_MAX_ROUNDS", 1)
    _i, _e, calls = run_turn(svc, [
        tool_call("run_shell", command="true"),
        ANSWER,
    ])
    nudge = calls[1]["messages"][-1]["content"]
    assert "Shell commands this turn" in nudge
    assert "$ true -> ok" in nudge


def test_parallel_readonly_calls_run_concurrently(svc, monkeypatch):
    """A batch of read-only calls in one round executes concurrently, with
    step nodes and tool messages still in strict call order."""
    import agent_tools
    lock = threading.Lock()
    cur = {"n": 0, "max": 0}
    orig = agent_tools.do_read_file

    def slow_read(ws, path, offset=1, limit=0):
        with lock:
            cur["n"] += 1
            cur["max"] = max(cur["max"], cur["n"])
        time.sleep(0.15)
        try:
            return orig(ws, path, offset, limit)
        finally:
            with lock:
                cur["n"] -= 1

    monkeypatch.setattr(agent_tools, "do_read_file", slow_read)
    for f in ("f1.txt", "f2.txt", "f3.txt"):
        with open(os.path.join(svc.workspace().root, f), "w") as fh:
            fh.write(f + "\n")
    script = [[("tool_calls", [
        {"id": "r1", "name": "read_file", "arguments": {"path": "f1.txt"}},
        {"id": "r2", "name": "read_file", "arguments": {"path": "f2.txt"}},
        {"id": "r3", "name": "read_file", "arguments": {"path": "f3.txt"}},
    ]), ("done", {"input": 1, "output": 1})], ANSWER]
    info, _events, calls = run_turn(svc, script)
    assert cur["max"] > 1, "read-only calls did not run concurrently"
    assert [m["tool_call_id"] for m in calls[1]["messages"]
            if m.get("role") == "tool"] == ["r1", "r2", "r3"]
    steps = step_nodes(svc)
    assert [s.metadata["call_id"] for s in steps] == ["r1", "r2", "r3"]
    # The trajectory stays a chain even though the executions overlapped.
    assert steps[0].parent_id == info["assistant_id"]
    assert [s.parent_id for s in steps[1:]] == [s.id for s in steps[:-1]]


def test_mutating_calls_stay_sequential(svc, monkeypatch):
    """Writes never run concurrently, even between read-only batches."""
    import agent_tools
    lock = threading.Lock()
    cur = {"n": 0, "max": 0}
    orig = agent_tools.do_write_file

    def slow_write(ws, path, content):
        with lock:
            cur["n"] += 1
            cur["max"] = max(cur["max"], cur["n"])
        time.sleep(0.1)
        try:
            return orig(ws, path, content)
        finally:
            with lock:
                cur["n"] -= 1

    monkeypatch.setattr(agent_tools, "do_write_file", slow_write)
    script = [[("tool_calls", [
        {"id": "w1", "name": "write_file",
         "arguments": {"path": "a.txt", "content": "1\n"}},
        {"id": "g1", "name": "grep", "arguments": {"pattern": "x"}},
        {"id": "g2", "name": "grep", "arguments": {"pattern": "y"}},
        {"id": "w2", "name": "write_file",
         "arguments": {"path": "b.txt", "content": "2\n"}},
    ]), ("done", {"input": 1, "output": 1})], ANSWER]
    _info, _events, calls = run_turn(svc, script)
    assert cur["max"] == 1, "mutating calls overlapped"
    assert [m["tool_call_id"] for m in calls[1]["messages"]
            if m.get("role") == "tool"] == ["w1", "g1", "g2", "w2"]
    assert os.path.exists(os.path.join(svc.workspace().root, "a.txt"))
    assert os.path.exists(os.path.join(svc.workspace().root, "b.txt"))


def test_update_plan_streams_and_persists(svc):
    items = [{"content": "do a", "status": "in_progress"},
             {"content": "do b", "status": "pending"}]
    info, events, _calls = run_turn(svc, [
        tool_call("update_plan", items=items),
        ANSWER,
    ])
    plan_events = [e for e in events if e["type"] == "plan"]
    assert plan_events and plan_events[0]["items"][0]["content"] == "do a"
    (step,) = step_nodes(svc)
    assert step.metadata["tool"] == "update_plan"
    assert step.metadata["status"] == "ok"  # read-only: never needs approval
    # Plan steps get a compact summary instead of raw JSON args.
    assert step.metadata["args_summary"] == "update_plan 0/2"
    assert step.content == "update_plan 0/2"
    plan = svc.store.get(info["assistant_id"]).metadata["plan"]
    assert plan[1] == {"content": "do b", "status": "pending"}
    # …and it reaches the UI payloads (path messages + tree summary).
    msgs = svc.path_messages(info["assistant_id"])
    assert msgs[1]["plan"][0]["content"] == "do a"
    assert svc._node_summary_dict(svc.store.get(info["assistant_id"]))["plan"] == {
        "done": 0, "total": 2}


def test_shell_output_streams_as_step_events(svc):
    _info, events, _calls = run_turn(svc, [
        tool_call("run_shell", command="echo streamed-line"),
        ANSWER,
    ])
    outs = [e for e in events if e["type"] == "step_output"]
    assert outs and outs[0]["step_id"]
    assert "streamed-line" in "".join(e["chunk"] for e in outs)


@git_required
def test_compare_and_snapshot_file_reads(svc):
    info1, _e1, _c1 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        ANSWER,
    ])
    info2, _e2, _c2 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v2\n"),
        tool_call("write_file", path="b.txt", content="new\n"),
        ANSWER,
    ], content="second turn", parent_id=info1["assistant_id"])

    cmp = svc.agent_compare(info1["assistant_id"], info2["assistant_id"])
    assert cmp["ok"]
    files = {f["path"]: f for f in cmp["files"]}
    assert files["a.txt"]["status"] == "M"
    assert files["a.txt"]["additions"] == 1 and files["a.txt"]["deletions"] == 1
    assert files["b.txt"]["status"] == "A"
    assert "-v1" in cmp["diff"] and "+v2" in cmp["diff"]

    fl = svc.agent_files(info1["assistant_id"])
    assert "a.txt" in fl["files"] and "b.txt" not in fl["files"]
    fc = svc.agent_file(info1["assistant_id"], "a.txt")
    assert fc["content"] == "v1\n" and fc["truncated"] is False

    with pytest.raises(WorkspaceError):
        svc.agent_file(info1["assistant_id"], "../escape.txt")
    cmp_bad = svc.agent_compare(info1["assistant_id"], "no-such-node")
    assert cmp_bad["ok"] is False


@git_required
def test_adopt_and_undo(svc):
    """Adopt restores the branch's snapshot, records it, and is undoable."""
    info1, _e1, _c1 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        ANSWER,
    ])
    info2, _e2, _c2 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v2\n"),
        ANSWER,
    ], content="second turn", parent_id=info1["assistant_id"])
    target = os.path.join(svc.workspace().root, "a.txt")
    assert open(target).read() == "v2\n"

    # Preview: compare the CURRENT workspace against the target snapshot.
    preview = svc.agent_compare("workspace", info1["assistant_id"])
    assert preview["ok"]
    assert {f["path"] for f in preview["files"]} == {"a.txt"}

    res = svc.adopt_snapshot(info1["assistant_id"])
    assert res["ok"] and res["adopted_commit"] and res["previous_commit"]
    assert {f["path"] for f in res["files"]} == {"a.txt"}
    assert open(target).read() == "v1\n"
    md = svc.store.get(info1["assistant_id"]).metadata
    assert md["adopted_commit"] == res["adopted_commit"]
    assert md["previous_commit"] == res["previous_commit"]

    undo = svc.undo_adopt(res["previous_commit"])
    assert undo["ok"]
    assert open(target).read() == "v2\n"

    # The other branch's files are untouched by all of this.
    assert open(target).read() == svc.agent_file(
        info2["assistant_id"], "a.txt")["content"]


@git_required
def test_compare_marks_binary_files(svc):
    """Binary files appear in the file list flagged binary (no text diff)."""
    info1, _e1, _c1 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        ANSWER,
    ])
    info2, _e2, _c2 = run_turn(svc, [
        tool_call("run_shell", command="printf 'a\\0b\\0c' > blob.dat"),
        ANSWER,
    ], content="second", parent_id=info1["assistant_id"])
    cmp = svc.agent_compare(info1["assistant_id"], info2["assistant_id"])
    files = {f["path"]: f for f in cmp["files"]}
    assert files["blob.dat"]["binary"] is True
    assert files["blob.dat"]["status"] == "A"
    assert files["blob.dat"]["additions"] == 0


def test_fork_from_step_truncates_context(svc):
    """Forking from step k: the replay keeps only calls up to k (a round's
    tool_calls list is split when the fork lands mid-round), the turn's final
    answer is dropped, and a continuation note marks the point."""
    round1 = [("tool_calls", [
        {"id": "c1", "name": "write_file",
         "arguments": {"path": "a.txt", "content": "v1\n"}},
        {"id": "c2", "name": "write_file",
         "arguments": {"path": "b.txt", "content": "v1\n"}},
    ]), ("done", {"input": 1, "output": 1})]
    round2 = [("tool_calls", [
        {"id": "c3", "name": "write_file",
         "arguments": {"path": "a.txt", "content": "v2\n"}},
    ]), ("done", {"input": 1, "output": 1})]
    _info, _e, _c = run_turn(svc, [round1, round2, ANSWER])
    s1, s2, s3 = step_nodes(svc)

    # Fork from step 2: both round-1 calls kept, round 2 and the answer gone.
    _i2, _e2, calls2 = run_turn(svc, [ANSWER], content="redo from step 2",
                                parent_id=s2.id)
    sent = calls2[0]["messages"]
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assistants = [m for m in sent if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistants) == 1
    assert [c["id"] for c in assistants[0]["tool_calls"]] == ["c1", "c2"]
    assert not any("c3" in json.dumps(m) for m in sent)
    assert not any(m.get("role") == "assistant" and m.get("content") == "all done"
                   for m in sent)
    assert any(m.get("role") == "system"
               and "continuing from step 2" in (m.get("content") or "")
               for m in sent)

    # Fork from step 1: the round-1 tool_calls list is split down to just c1.
    _i3, _e3, calls3 = run_turn(svc, [ANSWER], content="redo from step 1",
                                parent_id=s1.id)
    sent3 = calls3[0]["messages"]
    assert [m["tool_call_id"] for m in sent3 if m.get("role") == "tool"] == ["c1"]
    a3 = [m for m in sent3 if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(a3) == 1 and [c["id"] for c in a3[0]["tool_calls"]] == ["c1"]
    assert any("continuing from step 1" in (m.get("content") or "")
               for m in sent3 if m.get("role") == "system")

    # The truncated sequences are valid: every tool_call id is answered by an
    # immediately following tool message.
    for msgs in (sent, sent3):
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                ids = [c["id"] for c in m["tool_calls"]]
                following = msgs[i + 1:i + 1 + len(ids)]
                assert [t.get("tool_call_id") for t in following] == ids


@git_required
def test_followup_continues_the_step_chain_with_full_context(svc):
    """A plain follow-up from an agent reply hangs off the REPLY node (node
    granularity) and keeps the turn's WHOLE context (full replay + final
    answer) — only a mid-turn fork truncates."""
    info1, _e1, _c1 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("read_file", path="a.txt"),
        ANSWER,
    ])
    steps1 = step_nodes(svc)
    _i2, _e2, calls2 = run_turn(svc, [ANSWER], content="continue please",
                                parent_id=info1["assistant_id"])
    u2 = svc.store.get(_i2["user_id"])
    assert u2.parent_id == info1["assistant_id"]  # child of the reply node
    assert steps1  # the turn's steps exist, but are display-only
    sent = calls2[0]["messages"]
    # Full replay kept (both calls) AND the final answer is in context.
    assert [m["tool_call_id"] for m in sent if m.get("role") == "tool"] == [
        "c_write_file", "c_read_file"]
    assert any(m.get("role") == "assistant" and m.get("content") == "all done"
               for m in sent)
    assert not any("continuing from step" in (m.get("content") or "")
                   for m in sent)


@git_required
def test_shell_step_lists_touched_files_without_snapshot(svc):
    with open(os.path.join(svc.workspace().root, "keep.txt"), "w") as fh:
        fh.write("k\n")
    info, events, _calls = run_turn(svc, [
        tool_call("write_file", path="w.txt", content="w\n"),
        tool_call("run_shell", command="echo hi > s.txt && rm keep.txt"),
        ANSWER,
    ])
    _w, sh = step_nodes(svc)
    assert "ws_commit" not in sh.metadata
    touched = {f["path"]: f["status"] for f in sh.metadata["touched_files"]}
    # Only what the COMMAND touched — not the earlier write_file edit.
    assert touched == {"s.txt": "A", "keep.txt": "D"}
    done = [e for e in events if e.get("type") == "step_done" and e["id"] == sh.id][0]
    assert {f["path"] for f in done["touched_files"]} == {"s.txt", "keep.txt"}
    msgs = svc.path_messages(info["assistant_id"])
    assert msgs[-1]["step"]["touched_files"]


@git_required
def test_legacy_followup_under_step_tail_still_works(svc):
    """Old data: follow-ups were stored under the last step. Continuing from
    such a node keeps full context and restores the right files."""
    info1, _e1, _c1 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"), ANSWER])
    (s1,) = step_nodes(svc)
    # A legacy-shaped follow-up: parent = step tail.
    info2, _e2, calls2 = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v2\n"), ANSWER],
        content="more", parent_id=s1.id)
    assert svc.store.get(info2["user_id"]).parent_id == s1.id
    sent = calls2[0]["messages"]
    assert any(m.get("role") == "assistant" and m.get("content") == "all done" for m in sent)
    # Its reply resolves to its own node snapshot (v2), the turn-1 reply to v1.
    target = os.path.join(svc.workspace().root, "a.txt")
    svc.adopt_snapshot(info1["assistant_id"])
    assert open(target).read() == "v1\n"
    svc.adopt_snapshot(info2["assistant_id"])
    assert open(target).read() == "v2\n"
    # path_messages shows turn-1 steps once, then turn 2.
    roles = [m["role"] for m in svc.path_messages(info2["assistant_id"])]
    assert roles == ["user", "assistant", "step", "user", "assistant", "step"]


@git_required
def test_reply_node_status_fields_for_the_map(svc):
    info, _e, _c = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="1\n"),
        tool_call("write_file", path="b.txt", content="2\n"),
        ANSWER,
    ])
    md = svc.store.get(info["assistant_id"]).metadata
    assert md["files_changed"] == 2
    # Tree payload: the reply carries agent status; steps stay in the tree
    # (the UI lifts them out of the sidebar and the map).
    def find(n, nid):
        if n.get("id") == nid:
            return n
        for c in n.get("children", []):
            r = find(c, nid)
            if r:
                return r
    tree = svc.tree()
    reply = find(tree, info["assistant_id"])
    assert reply["agent"] is True and reply["files_changed"] == 2
    assert reply["stop_reason"] is None
