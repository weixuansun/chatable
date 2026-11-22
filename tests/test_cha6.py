"""CHA-6: nested repos in snapshots, sandbox denial hints, leaked text
tool-call markup, chat-mode context notes."""
import os
import shutil
import subprocess

import pytest

import main
import service as service_mod
from service import _recover_text_tool_calls, _scrub_answer_text, _ToolCallStripper
from shell_sandbox import sandbox_denial
from workspace import ShadowGit, Workspace

from test_agent_loop import ANSWER, git_required, run_turn, step_nodes, svc  # noqa: F401

LEAK = ('我来看看。<|open|>tools<|sep|><|open|>call tool="read_file" index="1"<|sep|>'
        '<|open|>argument key="path" type="string"<|sep|>notes.txt<|close|>argument<|sep|>'
        '<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>')


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _nested_repo(path, commit=True):
    os.makedirs(os.path.join(path, "pkg"), exist_ok=True)
    _git("init", "-q", path)
    with open(os.path.join(path, "pkg", "x.py"), "w") as fh:
        fh.write("x = 1\n")
    with open(os.path.join(path, ".gitignore"), "w") as fh:
        fh.write("*.log\n")
    with open(os.path.join(path, "debug.log"), "w") as fh:
        fh.write("noise\n")
    if commit:
        _git("add", ".", cwd=path)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i", cwd=path)


# --------------------------------------------------------------- nested repos

@git_required
@pytest.mark.parametrize("commit", [True, False])
def test_snapshot_keeps_nested_repo_files_not_a_gitlink(tmp_path, commit):
    ws = Workspace(str(tmp_path / "ws"))
    sg = ShadowGit(ws, str(tmp_path / "shadow.git"))
    with open(os.path.join(ws.root, "a.txt"), "w") as fh:
        fh.write("a\n")
    sg.snapshot("base", allow_empty=True)
    _nested_repo(os.path.join(ws.root, "chatable"), commit=commit)  # e.g. a git clone
    c1 = sg.snapshot("after clone")
    files = sg.ls_tree(c1)
    assert "chatable/pkg/x.py" in files and "chatable/.gitignore" in files
    assert "chatable/debug.log" not in files          # the nested .gitignore is honoured
    assert not any(f.startswith("chatable/.git/") for f in files)
    modes = sg._git("ls-tree", "-r", c1).stdout
    assert "160000" not in modes                      # no empty gitlink
    assert sg.snapshot("again") == c1                 # unchanged → same snapshot
    with open(os.path.join(ws.root, "chatable", "pkg", "new.py"), "w") as fh:
        fh.write("n\n")
    assert {"status": "A", "path": "chatable/pkg/new.py"} in sg.stage_changes()
    c2 = sg.snapshot("new file")
    with open(os.path.join(ws.root, "chatable", "pkg", "x.py"), "w") as fh:
        fh.write("broken\n")
    sg.restore(c2)
    assert open(os.path.join(ws.root, "chatable", "pkg", "x.py")).read() == "x = 1\n"
    assert os.path.isdir(os.path.join(ws.root, "chatable", ".git"))  # left untouched


@git_required
def test_follow_up_run_sees_repo_cloned_by_previous_run(tmp_path, monkeypatch):
    """The user's bug: round 1 cloned a repo inside the run worktree; the
    follow-up run's worktree had an empty chatable/ directory."""
    from test_chat_runs import _svc, _wait
    from test_orch import FakeExecutor

    class Cloner(FakeExecutor):
        def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
                proc_reg=lambda p: None, timeout=None):
            _nested_repo(os.path.join(workdir, "chatable"))
            yield {"type": "result", "final_text": "cloned", "exit_code": 0,
                   "session_id": None, "usage": {}}

    seen = {}

    class Reader(FakeExecutor):
        def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
                proc_reg=lambda p: None, timeout=None):
            p = os.path.join(workdir, "chatable", "pkg", "x.py")
            seen["content"] = open(p).read() if os.path.exists(p) else None
            yield {"type": "result", "final_text": "reviewed", "exit_code": 0,
                   "session_id": None, "usage": {}}

    chat, orch = _svc(tmp_path, monkeypatch, {"a": Cloner(), "b": Reader()})
    (r1,) = orch.start_chat_runs("clone the repo", None, ["a"])
    _ev, md = _wait(chat, r1["assistant_id"])
    assert md["run_status"] == "ok" and md["files_changed"] >= 2
    (r2,) = orch.start_chat_runs("review it", r1["assistant_id"], ["b"])
    _wait(chat, r2["assistant_id"])
    assert seen["content"] == "x = 1\n"
    assert "chatable/pkg/x.py" in chat.shadow.ls_tree(md["ws_commit"]) \
        if hasattr(chat, "shadow") else True


# ----------------------------------------------------------- sandbox denials

def test_sandbox_denial_classification():
    net = sandbox_denial("fatal: unable to access 'https://github.com/x/y/': "
                         "Could not resolve host: github.com", False, "seatbelt", False)
    assert net["kind"] == "network" and "CHATTABLE_AGENT_SHELL_NET=1" in net["agent_hint"]
    assert "CHATTABLE_AGENT_SHELL_NET=1" in net["user_hint"]
    fs = sandbox_denial("ls: /Users/me: Operation not permitted", False, "seatbelt",
                        False, "/w")
    assert fs["kind"] == "filesystem" and "/w" in fs["agent_hint"]
    assert sandbox_denial("Could not resolve host", False, "off", False) is None
    assert sandbox_denial("Could not resolve host", True, "bwrap", False) is None
    assert sandbox_denial("Could not resolve host", False, "bwrap", True) is None  # net on
    assert sandbox_denial("AssertionError: 1 != 2", False, "bwrap", False) is None


def test_sandbox_block_reaches_step_and_user_notice(svc, monkeypatch):  # noqa: F811
    from web_tools import WebToolResult
    import agent_tools

    def fake_shell(ws, command, *a, **kw):
        hint = sandbox_denial("Could not resolve host: github.com", False, "seatbelt", False)
        return WebToolResult(False, "$ git clone …\n[exit 128]\n" + hint["agent_hint"],
                             {"sandbox_block": hint})

    monkeypatch.setattr(agent_tools, "do_run_shell", fake_shell)
    from test_agent_loop import tool_call
    info, events, calls = run_turn(svc, [
        tool_call("run_shell", command="git clone https://github.com/x/y"),
        tool_call("run_shell", command="git clone https://github.com/x/y --depth 1"),
        ANSWER,
    ])
    notices = [e for e in events if e["type"] == "notice" and e.get("kind") == "sandbox"]
    assert len(notices) == 1 and "CHATTABLE_AGENT_SHELL_NET=1" in notices[0]["text"]
    steps = step_nodes(svc)
    assert steps[0].metadata.get("sandbox_block") == "network"
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert "[sandbox]" in tool_msgs[-1]["content"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_real_bwrap_network_denial_is_explained(tmp_path, monkeypatch):
    import agent_tools
    monkeypatch.delenv("CHATTABLE_AGENT_SHELL_NET", raising=False)
    monkeypatch.setenv("CHATTABLE_AGENT_SANDBOX", "bwrap")
    from shell_sandbox import sandbox_mode
    if sandbox_mode() != "bwrap":
        pytest.skip("bwrap unusable here")
    ws = Workspace(str(tmp_path / "ws"))
    res = agent_tools.do_run_shell(
        ws, "python3 -c \"import socket; socket.getaddrinfo('github.com', 443)\"")
    assert not res.ok
    assert "[sandbox]" in res.output and res.metadata["sandbox_block"]["kind"] == "network"


# --------------------------------------------------- leaked tool-call markup

def test_pipe_marker_markup_is_normalised_scrubbed_and_recoverable():
    norm = main.normalize_model_markup(LEAK)
    assert '<invoke name="read_file">' in norm and "<|" not in norm
    assert _scrub_answer_text(LEAK) == "我来看看。"
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    assert _recover_text_tool_calls(LEAK, tools) == [
        {"id": None, "name": "read_file", "arguments": {"path": "notes.txt"}}]
    assert _recover_text_tool_calls(LEAK, [{"function": {"name": "web_fetch"}}]) == []
    fullwidth = LEAK.replace("|", "｜")
    assert _scrub_answer_text(fullwidth) == "我来看看。"


@pytest.mark.parametrize("step", [1, 2, 3, 7, 1000])
def test_streaming_stripper_hides_pipe_markup(step):
    text = LEAK + "然后 a<b|c 结束"
    st = _ToolCallStripper()
    out = "".join(st.feed(text[i:i + step]) for i in range(0, len(text), step)) + st.flush()
    assert out == "我来看看。然后 a<b|c 结束"


@git_required
def test_agent_loop_executes_a_call_leaked_as_text(svc):  # noqa: F811
    ws_root = svc.workspace().root
    with open(os.path.join(ws_root, "notes.txt"), "w") as fh:
        fh.write("secret-notes-content\n")
    info, events, calls = run_turn(svc, [
        [("delta", LEAK), ("done", {"input": 1, "output": 1})],
        ANSWER,
    ])
    node = svc.store.get(info["assistant_id"])
    assert "<|" not in node.content and "invoke" not in node.content
    tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and "secret-notes-content" in tool_msgs[-1]["content"]
    assert [s.metadata["tool"] for s in step_nodes(svc)] == ["read_file"]


# ------------------------------------------------------ chat mode / context

def test_chat_mode_turn_is_told_it_has_no_local_file_access(svc):  # noqa: F811
    info, events, calls = run_turn(svc, [ANSWER], agent=False)
    systems = [m["content"] for m in calls[0]["messages"] if m.get("role") == "system"]
    assert any("交给 agent" in s and "NO access" in s for s in systems)


def test_chat_loop_shrinks_old_tool_results_past_budget(monkeypatch):
    msgs = [{"role": "system", "content": "s"}]
    for i in range(5):
        msgs.append({"role": "tool", "content": f"<{i}>" + "x" * 200_000})
    service_mod._shrink_old_tool_messages(msgs, 100_000, 2)
    sizes = [len(m["content"]) for m in msgs[1:]]
    assert all(s < 3000 for s in sizes[:3]) and all(s > 200_000 for s in sizes[3:])


def test_chat_turn_on_an_agent_branch_gets_a_late_reminder(svc):  # noqa: F811
    from test_agent_loop import tool_call
    info, _ev, _calls = run_turn(svc, [tool_call("list_dir", path="."), ANSWER])  # agent turn
    _info2, _ev2, calls = run_turn(svc, [ANSWER], content="review the code",
                                   agent=False, parent_id=info["assistant_id"])
    msgs = calls[0]["messages"]
    last_user = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
    assert msgs[last_user - 1]["role"] == "system"
    assert "交给 agent" in msgs[last_user - 1]["content"] and "GitHub" in msgs[last_user - 1]["content"]
    # a plain chat branch does not get it
    _i3, _e3, calls3 = run_turn(svc, [ANSWER], content="hi", agent=False)
    assert not any("Reminder for THIS message" in str(m.get("content")) for m in calls3[0]["messages"])
