"""Agent loop: step nodes, snapshots, limits, restore-on-send, token gate.

The model backend is faked exactly like tests/test_core.py does: a scripted
generator of ("delta"|"tool_calls"|"done") events patched over
``service.stream_backend``.
"""
import os
import shutil
import json
from unittest.mock import patch

import pytest

import service as service_mod
from service import ChatService
from store import ROOT_ID, TreeStore
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


def run_turn(svc, script, content="do it", **kwargs):
    kwargs.setdefault("agent", True)
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
    assert md["diff"] and md["ws_commit"]
    assert "duration_ms" in md
    # Selecting a step itself: steps come from the path, with no duplicates.
    msgs2 = svc.path_messages(steps[1].id)
    assert [m["role"] for m in msgs2] == ["user", "assistant", "step", "step"]
    assert [m["id"] for m in msgs2] == [msgs[0]["id"], msgs[1]["id"],
                                        steps[0].id, steps[1].id]


def test_first_agent_send_marks_the_conversation(svc):
    """agent:true on a brand-new trunk persists agent_mode on its metadata."""
    info, _events, _calls = run_turn(svc, [ANSWER])
    trunk = svc.store.get(info["user_id"])
    assert trunk.parent_id == ROOT_ID
    assert trunk.metadata["agent_mode"] is True
    # A later plain send in this conversation still runs as agent.
    assert svc._conversation_agent_mode(info["assistant_id"]) is True


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
def test_steps_record_ws_commits_and_restore(svc):
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("write_file", path="a.txt", content="v2\n"),
        ANSWER,
    ])
    steps = step_nodes(svc)
    s1, s2 = steps
    assert s1.metadata["ws_commit"]
    assert s2.metadata["ws_commit"] != s1.metadata["ws_commit"]
    assert s2.metadata["ws_parent_commit"] == s1.metadata["ws_commit"]

    target = os.path.join(svc.workspace().root, "a.txt")
    res = svc.restore_workspace(s1.id)
    assert res["restored"] and res["commit"] == s1.metadata["ws_commit"]
    assert open(target).read() == "v1\n"

    # The assistant node resolves to its nearest step's snapshot.
    res = svc.restore_workspace(info["assistant_id"])
    assert res["commit"] == s2.metadata["ws_commit"]
    assert open(target).read() == "v2\n"

    # Diff endpoint data: stored edit diff for the write step…
    d1 = svc.step_diff(s1.id)
    assert "+v1" in d1["diff"]
    d2 = svc.step_diff(s2.id)
    assert "-v1" in d2["diff"] and "+v2" in d2["diff"]


@git_required
def test_shell_step_snapshot_diff_falls_back_to_git(svc):
    _info, _events, _calls = run_turn(svc, [
        tool_call("run_shell", command="echo from-shell > shell.txt"),
        ANSWER,
    ])
    (step,) = step_nodes(svc)
    assert step.metadata["ws_commit"]
    assert "diff" not in step.metadata  # shell steps have no stored edit diff
    d = svc.step_diff(step.id)
    assert "shell.txt" in d["diff"] and "+from-shell" in d["diff"]


@git_required
def test_send_from_step_restores_workspace_first(svc):
    """Fork in agent mode = time travel: the workspace follows the fork point."""
    info, _events, _calls = run_turn(svc, [
        tool_call("write_file", path="a.txt", content="v1\n"),
        tool_call("write_file", path="a.txt", content="v2\n"),
        ANSWER,
    ])
    s1, _s2 = step_nodes(svc)
    target = os.path.join(svc.workspace().root, "a.txt")
    assert open(target).read() == "v2\n"

    info2, _e2, _c2 = run_turn(svc, [ANSWER], content="try again from v1",
                               parent_id=s1.id)
    assert open(target).read() == "v1\n"
    # The fork's user node hangs off the step node.
    assert svc.store.get(info2["user_id"]).parent_id == s1.id


@git_required
def test_readonly_turn_records_a_base_snapshot(svc):
    """Every agent turn snapshots the pre-turn workspace onto the user node,
    so even a turn with no writes can be restored to."""
    info, _events, _calls = run_turn(svc, [ANSWER])
    trunk = svc.store.get(info["user_id"])
    base = trunk.metadata.get("ws_commit")
    assert base
    res = svc.restore_workspace(info["assistant_id"])
    assert res["restored"] and res["commit"] == base


@git_required
def test_fork_from_readonly_step_restores_pre_turn_state(svc):
    """A read-only step before the first write restores to the turn's base."""
    with open(os.path.join(svc.workspace().root, "existing.txt"), "w") as fh:
        fh.write("orig\n")
    _info, _events, _calls = run_turn(svc, [
        tool_call("read_file", path="existing.txt"),
        tool_call("write_file", path="new.txt", content="n\n"),
        ANSWER,
    ])
    read_step, _write_step = step_nodes(svc)
    assert read_step.metadata["tool"] == "read_file"
    assert "ws_commit" not in read_step.metadata  # read-only steps don't snapshot
    res = svc.restore_workspace(read_step.id)
    assert res["restored"]
    assert not os.path.exists(os.path.join(svc.workspace().root, "new.txt"))
    with open(os.path.join(svc.workspace().root, "existing.txt")) as fh:
        assert fh.read() == "orig\n"


# ------------------------------------------------------------- mode + token

def test_agent_mode_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("CHATTABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATABLE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    with pytest.raises(ValueError, match="requires CHATTABLE_AUTH_TOKEN"):
        svc.start_turn("do agent things", None, agent=True)
    # Nothing was persisted for the refused turn.
    assert svc.store.all_nodes() == []
    status = svc.agent_status()
    assert status["available"] is False
    assert "CHATTABLE_AUTH_TOKEN" in status["reason"]


def test_conversation_level_agent_mode(tmp_path, monkeypatch):
    """The per-conversation toggle lives on the trunk node's metadata."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("CHATTABLE_AGENT_AUTO_APPROVE", "all")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))

    with patch.object(service_mod, "stream_backend", make_fake([ANSWER])[0]):
        info = svc.start_turn("hello", None)  # normal turn
        list(svc.subscribe(info["assistant_id"], 0))

    res = svc.set_agent_mode(info["user_id"], True)
    assert res["agent_mode"] is True and res["trunk_id"] == info["user_id"]

    # Next turn in this conversation runs as agent WITHOUT the per-send flag.
    info2, events2, _calls2 = run_turn(svc, [
        tool_call("write_file", path="m.txt", content="x\n"),
        ANSWER,
    ], parent_id=info["assistant_id"])
    assert info2["agent"] is True
    assert len(step_nodes(svc)) == 1
    assert svc.store.get(info2["user_id"]).metadata["agent"] is True

    # Toggling off returns to normal turns.
    svc.set_agent_mode(info["user_id"], False)
    with patch.object(service_mod, "stream_backend", make_fake([ANSWER])[0]):
        info3 = svc.start_turn("normal again", info2["assistant_id"])
        list(svc.subscribe(info3["assistant_id"], 0))
    assert info3["agent"] is False
    assert len(step_nodes(svc)) == 1  # unchanged


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
