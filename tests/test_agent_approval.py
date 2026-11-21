"""Agent approval flow: pending_approval steps, approve/deny/timeout, always."""
import os
import time
from unittest.mock import patch

import pytest

import service as service_mod
from service import ChatService
from store import TreeStore

from test_agent_loop import ANSWER, make_fake, step_nodes, tool_call


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """Token set, but NO auto-approve: write/shell tools must ask."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "test-token")
    monkeypatch.delenv("CHATTABLE_AGENT_AUTO_APPROVE", raising=False)
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    return ChatService(TreeStore(str(tmp_path / "chat.jsonl")))


def drive(svc, script, on_approval):
    """Run an agent turn, resolving each approval_required event via callback."""
    fake, calls = make_fake(script)
    approvals = []
    with patch.object(service_mod, "stream_backend", fake):
        info = svc.start_turn("do it", None, agent=True)
        for ev in svc.subscribe(info["assistant_id"], 0):
            if ev["type"] == "approval_required":
                approvals.append(ev)
                on_approval(svc, ev)
    return info, approvals, calls


def test_approve_runs_the_tool(svc):
    def approve(svc, ev):
        assert ev["tool"] == "write_file"
        assert "+hello" in ev["preview_diff"]  # diff shown BEFORE the write
        # The step node is persisted as pending while the loop waits.
        assert svc.store.get(ev["step_id"]).metadata["status"] == "pending_approval"
        assert svc.resolve_approval(ev["step_id"], "approve") is True

    info, approvals, _calls = drive(svc, [
        tool_call("write_file", path="a.txt", content="hello\n"),
        ANSWER,
    ], approve)
    assert len(approvals) == 1
    (step,) = step_nodes(svc)
    assert step.metadata["status"] == "ok"
    assert step.metadata["ws_commit"]
    with open(os.path.join(svc.workspace().root, "a.txt")) as fh:
        assert fh.read() == "hello\n"
    assert svc.store.get(info["assistant_id"]).content == "all done"


def test_deny_tells_the_model_and_continues(svc):
    def deny(svc, ev):
        assert svc.resolve_approval(ev["step_id"], "deny") is True

    info, approvals, calls = drive(svc, [
        tool_call("write_file", path="a.txt", content="hello\n"),
        ANSWER,
    ], deny)
    assert len(approvals) == 1
    (step,) = step_nodes(svc)
    assert step.metadata["status"] == "denied"
    assert not os.path.exists(os.path.join(svc.workspace().root, "a.txt"))
    # The denial went back to the model as its tool result; the loop moved on.
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert any("denied by the user" in m["content"] for m in tool_msgs)
    assert svc.store.get(info["assistant_id"]).content == "all done"


def test_approval_timeout_counts_as_denied(svc, monkeypatch):
    monkeypatch.setattr(service_mod, "AGENT_APPROVAL_TIMEOUT_S", 0.3)
    started = time.monotonic()
    fake, _calls = make_fake([
        tool_call("write_file", path="a.txt", content="hello\n"),
        ANSWER,
    ])
    with patch.object(service_mod, "stream_backend", fake):
        info = svc.start_turn("do it", None, agent=True)
        events = list(svc.subscribe(info["assistant_id"], 0))
    assert time.monotonic() - started < 5  # the 10-minute default did NOT apply
    (step,) = step_nodes(svc)
    assert step.metadata["status"] == "denied"
    resolved = [e for e in events if e["type"] == "approval_resolved"]
    assert resolved and resolved[0]["decision"] == "deny"
    assert not os.path.exists(os.path.join(svc.workspace().root, "a.txt"))


def test_always_approve_skips_later_prompts(svc):
    seen = []

    def approve_always(svc, ev):
        seen.append(ev)
        svc.resolve_approval(ev["step_id"], "approve", always=True)

    drive(svc, [
        tool_call("write_file", path="a.txt", content="one\n"),
        tool_call("write_file", path="b.txt", content="two\n"),  # no prompt now
        ANSWER,
    ], approve_always)
    assert len(seen) == 1  # only the first write_file asked
    assert {s.metadata["status"] for s in step_nodes(svc)} == {"ok"}


def test_always_scope_writes_covers_both_write_tools(svc):
    prompts = []

    def approve_writes(svc, ev):
        prompts.append(ev)
        svc.resolve_approval(ev["step_id"], "approve", always=True, scope="writes")

    drive(svc, [
        tool_call("write_file", path="a.txt", content="one\n"),
        # edit_file is covered by the "writes" scope — no second prompt.
        tool_call("edit_file", path="a.txt", old_string="one", new_string="1"),
        ANSWER,
    ], approve_writes)
    assert len(prompts) == 1
    assert {s.metadata["status"] for s in step_nodes(svc)} == {"ok"}
    with open(os.path.join(svc.workspace().root, "a.txt")) as fh:
        assert fh.read() == "1\n"


def test_stop_denies_a_pending_approval(svc):
    fake, _calls = make_fake([
        tool_call("write_file", path="a.txt", content="x\n"),
        ANSWER,
    ])
    with patch.object(service_mod, "stream_backend", fake):
        info = svc.start_turn("do it", None, agent=True)
        for ev in svc.subscribe(info["assistant_id"], 0):
            if ev["type"] == "approval_required":
                assert svc.request_stop(info["assistant_id"]) is True
    (step,) = step_nodes(svc)
    assert step.metadata["status"] == "denied"
    assert not os.path.exists(os.path.join(svc.workspace().root, "a.txt"))


def test_session_approvals_query_and_reset(svc):
    """The composer pill reads/clears the conversation's always-approvals."""

    def approve_writes(svc, ev):
        svc.resolve_approval(ev["step_id"], "approve", always=True, scope="writes")

    info, _ap, _calls = drive(svc, [
        tool_call("write_file", path="a.txt", content="one\n"),
        ANSWER,
    ], approve_writes)
    assert svc.session_approvals(info["user_id"])["always"] == ["edit_file", "write_file"]
    svc.reset_session_approvals(info["user_id"])
    assert svc.session_approvals(info["user_id"])["always"] == []
    # After the reset the same tool asks again.
    prompts = []
    drive(svc, [
        tool_call("write_file", path="b.txt", content="two\n"),
        ANSWER,
    ], lambda s, ev: (prompts.append(ev), s.resolve_approval(ev["step_id"], "approve")))
    assert len(prompts) == 1


def test_shell_always_asks_by_default(svc):
    def approve(svc, ev):
        assert ev["tool"] == "run_shell"
        assert ev["preview_diff"] == ""  # no diff preview for shell
        svc.resolve_approval(ev["step_id"], "approve")

    _info, approvals, _calls = drive(svc, [
        tool_call("run_shell", command="echo hi > shell.txt"),
        ANSWER,
    ], approve)
    assert len(approvals) == 1
    assert os.path.exists(os.path.join(svc.workspace().root, "shell.txt"))


def test_read_only_tools_never_ask(svc):
    def should_not_be_called(svc, ev):  # pragma: no cover - fails if reached
        raise AssertionError("read-only tool asked for approval")

    _info, approvals, _calls = drive(svc, [
        tool_call("list_dir", path="."),
        tool_call("read_file", path="missing.txt"),
        tool_call("grep", pattern="x"),
        ANSWER,
    ], should_not_be_called)
    assert approvals == []
    assert len(step_nodes(svc)) == 3


def test_auto_approve_writes_env(tmp_path, monkeypatch):
    """CHATTABLE_AGENT_AUTO_APPROVE=writes auto-approves write/edit only."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("CHATTABLE_AGENT_AUTO_APPROVE", "writes")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    svc = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    prompts = []

    def on_approval(svc, ev):
        prompts.append(ev)
        svc.resolve_approval(ev["step_id"], "approve")

    drive(svc, [
        tool_call("write_file", path="a.txt", content="auto\n"),   # auto-approved
        tool_call("run_shell", command="true"),                     # still asks
        ANSWER,
    ], on_approval)
    assert [p["tool"] for p in prompts] == ["run_shell"]
    assert os.path.exists(os.path.join(svc.workspace().root, "a.txt"))


def test_resolve_unknown_step_returns_false(svc):
    assert svc.resolve_approval("no-such-step", "approve") is False
