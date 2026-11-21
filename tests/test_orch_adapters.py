"""Executor adapters: fixture parsing (kimi real sample, claude/codex mocks)
and the full subprocess runner against fake binaries (never the real CLIs)."""
import json
import os
import stat

import pytest

from orch.executors import (
    ClaudeExecutor,
    CodexExecutor,
    KimiExecutor,
    parse_claude_line,
    parse_codex_line,
    parse_kimi_line,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _collect(parse, objs):
    state = {"final_text": "", "session_id": None, "usage": {}}
    events = []
    for obj in objs:
        events.extend(parse(obj, state))
    return events, state


def test_kimi_fixture_parses():
    events, state = _collect(parse_kimi_line, _fixture("kimi_stream.jsonl"))
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["name"] for c in calls] == ["Read", "Edit", "Bash"]
    assert calls[0]["arguments"] == {"path": "a.py"}  # JSON string args parsed
    assert [r["id"] for r in results] == [c["id"] for c in calls]
    assert state["session_id"] == "session_c13db92b-0b92-467c-b2e0-7bca25e22bf1"
    assert state["final_text"].startswith("I changed x to 2")


def test_claude_fixture_parses():
    events, state = _collect(parse_claude_line, _fixture("claude_stream.jsonl"))
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["name"] for c in calls] == ["Edit", "Bash"]
    assert calls[0]["arguments"]["file_path"] == "a.py"
    assert [r["id"] for r in results] == ["toolu_01", "toolu_02"]
    assert results[1]["output"] == "2"  # content blocks flattened
    assert state["session_id"] == "sess-claude-1111"
    assert state["usage"]["cost_usd"] == 0.0042
    assert "Conclusion" in state["final_text"]


def test_codex_fixture_parses():
    events, state = _collect(parse_codex_line, _fixture("codex_stream.jsonl"))
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert [c["name"] for c in calls] == ["shell", "edit_file"]
    assert results[0]["output"] == "a.py"
    assert state["session_id"] == "thread-codex-2222"
    assert state["usage"]["input_tokens"] == 6100
    assert state["final_text"].startswith("Updated a.py")


def _cat_script(tmp_path, fixture_name):
    """A 'fake binary' that just prints the recorded fixture lines."""
    script = tmp_path / f"fake_{fixture_name.split('_')[0]}.sh"
    fixture = os.path.join(FIXTURES, fixture_name)
    script.write_text(f"#!/bin/sh\ncat {fixture}\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_kimi_full_run_via_fake_binary(tmp_path, monkeypatch):
    script = _cat_script(tmp_path, "kimi_stream.jsonl")
    monkeypatch.setenv("CHATTABLE_KIMI_BIN", script)
    ex = KimiExecutor()
    ok, _reason = ex.available()
    assert ok
    events = list(ex.run(str(tmp_path), "do something"))
    result = [e for e in events if e["type"] == "result"][0]
    assert result["exit_code"] == 0
    assert result["session_id"] == "session_c13db92b-0b92-467c-b2e0-7bca25e22bf1"
    assert result["final_text"].startswith("I changed x to 2")
    assert any(e["type"] == "tool_call" and e["name"] == "Edit" for e in events)


def test_claude_full_run_via_fake_binary(tmp_path, monkeypatch):
    script = _cat_script(tmp_path, "claude_stream.jsonl")
    monkeypatch.setenv("CHATTABLE_CLAUDE_BIN", script)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    ex = ClaudeExecutor()
    ok, _reason = ex.available()
    assert ok
    events = list(ex.run(str(tmp_path), "do something"))
    result = [e for e in events if e["type"] == "result"][0]
    assert result["exit_code"] == 0
    assert result["session_id"] == "sess-claude-1111"
    assert result["usage"]["cost_usd"] == 0.0042
    # Provider env of OTHER agents is scrubbed; own key is kept.
    argv_env = ex._env()
    assert "DEEPSEEK_API_KEY" not in argv_env
    assert argv_env.get("ANTHROPIC_API_KEY") == "test-key"


def test_codex_full_run_via_fake_binary(tmp_path, monkeypatch):
    script = _cat_script(tmp_path, "codex_stream.jsonl")
    monkeypatch.setenv("CHATTABLE_CODEX_BIN", script)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ex = CodexExecutor()
    ok, _reason = ex.available()
    assert ok
    events = list(ex.run(str(tmp_path), "do something"))
    result = [e for e in events if e["type"] == "result"][0]
    assert result["exit_code"] == 0
    assert result["session_id"] == "thread-codex-2222"
    assert result["usage"]["input_tokens"] == 6100


def test_codex_resume_argv():
    ex = CodexExecutor()
    argv = ex._argv("go on", "thread-1")
    assert argv[:3] == ["codex", "exec", "resume"]


def test_availability_reports_missing_cleanly(monkeypatch):
    monkeypatch.delenv("CHATTABLE_CLAUDE_BIN", raising=False)
    monkeypatch.delenv("CHATTABLE_CODEX_BIN", raising=False)
    monkeypatch.delenv("CHATTABLE_KIMI_BIN", raising=False)
    # On this box kimi exists, claude/codex do not.
    ok, reason = ClaudeExecutor().available()
    assert not ok and "claude" in reason
    ok, reason = CodexExecutor().available()
    assert not ok and "codex" in reason


def test_kimi_run_missing_binary_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTABLE_KIMI_BIN", str(tmp_path / "nope"))
    # Force default bins away too.
    monkeypatch.setattr(KimiExecutor, "default_bins", ())
    result = list(KimiExecutor().run(str(tmp_path), "x"))[0]
    assert result["type"] == "result" and result["exit_code"] == 127
    assert "not found" in result["error"]
