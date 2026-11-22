"""External agent executors: CLI adapters mapping JSON-lines streams to
normalized events, plus the in-process `builtin` (DeepSeek) executor.

Interface:
    Executor.id: str
    Executor.available() -> (bool, reason)
    Executor.run(workdir, prompt, session_id=None, should_stop, proc_reg,
                 timeout=None) -> Iterator[dict]

Normalized events: tool_call / tool_result / meta / error, and one final
{"type": "result", final_text, exit_code, session_id, usage, killed}.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import Iterator, Optional

RUN_TIMEOUT_S = int(os.environ.get("CHATTABLE_ORCH_RUN_TIMEOUT_S", "1800"))

# Provider keys that must not leak into another agent's environment.
_PROVIDER_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
                  "GEMINI_API_KEY", "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN")


def _exec_env(keep: tuple = ()) -> dict:
    return {k: v for k, v in os.environ.items()
            if k not in _PROVIDER_KEYS or k in keep}


def _popen_lines(argv: list[str], cwd: str, env: dict, should_stop,
                 proc_reg, timeout: int) -> Iterator[tuple]:
    """Yield ("line", str) per stdout/stderr line, then ("end", info).

    Own process group; should_stop() or timeout kills the whole group.
    """
    proc = subprocess.Popen(argv, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True, bufsize=1)
    proc_reg(proc)
    buf: list[str] = []
    done = threading.Event()

    def reader() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                buf.append(line)
        finally:
            done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    idx = 0
    deadline = time.monotonic() + timeout
    killed = None
    try:
        while True:
            while idx < len(buf):
                line = buf[idx]
                idx += 1
                yield ("line", line)
            if done.is_set() and idx >= len(buf):
                break
            if should_stop():
                killed = "cancelled"
            elif time.monotonic() > deadline:
                killed = f"timed out after {timeout}s"
            if killed:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                break
            time.sleep(0.05)
        t.join(timeout=2)
        while idx < len(buf):
            line = buf[idx]
            idx += 1
            yield ("line", line)
        yield ("end", {"exit_code": proc.wait(), "killed": killed})
    finally:
        proc_reg(None)


# --------------------------------------------------------------------------- #
# Adapters: one parse function per CLI, shared by the runner and the tests
# --------------------------------------------------------------------------- #

def parse_kimi_line(obj: dict, state: dict) -> list[dict]:
    """kimi --output-format stream-json → normalized events."""
    role = obj.get("role")
    if role == "assistant":
        out = []
        for tc in obj.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            out.append({"type": "tool_call", "id": tc.get("id"),
                        "name": fn.get("name"), "arguments": args or {}})
        content = obj.get("content")
        if isinstance(content, str) and content.strip():
            state["final_text"] = content
            out.append({"type": "meta", "text": content})
        return out
    if role == "tool":
        return [{"type": "tool_result", "id": obj.get("tool_call_id"),
                 "output": str(obj.get("content") or "")[:2000]}]
    if role == "meta":
        if obj.get("type") == "session.resume_hint" and obj.get("session_id"):
            state["session_id"] = obj["session_id"]
        return [{"type": "meta", "kind": obj.get("type")}]
    return []


def parse_claude_line(obj: dict, state: dict) -> list[dict]:
    """claude -p --output-format stream-json → normalized events."""
    typ = obj.get("type")
    payload = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    if typ == "system" and obj.get("subtype") == "init":
        if obj.get("session_id"):
            state["session_id"] = obj["session_id"]
        return [{"type": "meta", "kind": "init"}]
    if typ == "assistant":
        out = []
        for block in payload.get("content") or []:
            btype = block.get("type")
            if btype == "text" and block.get("text", "").strip():
                state["final_text"] = block["text"]
                out.append({"type": "meta", "text": block["text"]})
            elif btype == "tool_use":
                out.append({"type": "tool_call", "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": block.get("input") or {}})
        return out
    if typ == "user":
        out = []
        for block in payload.get("content") or []:
            if block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    content = "\n".join(str(b.get("text") or "") for b in content
                                        if isinstance(b, dict))
                out.append({"type": "tool_result", "id": block.get("tool_use_id"),
                            "output": str(content or "")[:2000]})
        return out
    if typ == "result":
        state["session_id"] = obj.get("session_id") or state.get("session_id")
        state["usage"] = dict(obj.get("usage") or {})
        if obj.get("total_cost_usd") is not None:
            state["usage"]["cost_usd"] = obj["total_cost_usd"]
        if obj.get("result"):
            state["final_text"] = obj["result"]
        return [{"type": "meta", "kind": "result"}]
    return []


def parse_codex_line(obj: dict, state: dict) -> list[dict]:
    """codex exec --json → normalized events."""
    typ = obj.get("type")
    if typ == "thread.started":
        if obj.get("thread_id"):
            state["session_id"] = obj["thread_id"]
        return [{"type": "meta", "kind": "thread.started"}]
    if typ in ("item.started", "item.completed"):
        item = obj.get("item") or {}
        itype = item.get("type")
        if itype == "agent_message":
            if typ == "item.completed" and item.get("text", "").strip():
                state["final_text"] = item["text"]
                return [{"type": "meta", "text": item["text"]}]
            return []
        if itype == "command_execution":
            if typ == "item.started":
                return [{"type": "tool_call", "id": item.get("id"),
                         "name": "shell",
                         "arguments": {"command": item.get("command")}}]
            return [{"type": "tool_result", "id": item.get("id"),
                     "output": str(item.get("output") or "")[:2000],
                     "ok": item.get("exit_code") in (0, None)}]
        if itype == "file_change":
            if typ == "item.started":
                return [{"type": "tool_call", "id": item.get("id"),
                         "name": "edit_file",
                         "arguments": {k: item.get(k) for k in ("path", "kind") if k in item}}]
            return [{"type": "tool_result", "id": item.get("id"),
                     "output": str(item.get("status") or "changed")}]
        if itype == "reasoning":
            return [{"type": "meta", "kind": "reasoning"}]
        return []
    if typ == "turn.completed":
        state["usage"] = dict(obj.get("usage") or {})
        return [{"type": "meta", "kind": "turn.completed"}]
    return []


class _CliExecutor:
    """Shared runner for CLI adapters."""

    id = ""
    bin_env = ""
    default_bins: tuple = ()

    def _binary(self) -> Optional[str]:
        env = os.environ.get(self.bin_env)
        if env and os.path.isfile(env):
            return env
        for cand in self.default_bins:
            if os.path.isfile(cand):
                return cand
        return None

    def available(self) -> tuple[bool, str]:
        return (True, "") if self._binary() else (False, f"{self.id} binary not found ({self.bin_env})")

    def _argv(self, prompt: str, session_id: Optional[str]) -> list[str]:
        raise NotImplementedError

    def _parse(self, obj: dict, state: dict) -> list[dict]:
        raise NotImplementedError

    def _env(self) -> dict:
        return _exec_env()

    def run(self, workdir: str, prompt: str, session_id: Optional[str] = None,
            should_stop=lambda: False, proc_reg=lambda p: None,
            timeout: Optional[int] = None) -> Iterator[dict]:
        binary = self._binary()
        if not binary:
            yield {"type": "result", "final_text": "", "exit_code": 127,
                   "session_id": None, "usage": {}, "killed": None,
                   "error": self.available()[1]}
            return
        state = {"final_text": "", "session_id": session_id, "usage": {}}
        argv = self._argv(prompt, session_id)
        argv[0] = binary
        for kind, payload in _popen_lines(argv, workdir, self._env(), should_stop,
                                          proc_reg, timeout or RUN_TIMEOUT_S):
            if kind == "end":
                yield {"type": "result", "final_text": state["final_text"],
                       "exit_code": payload["exit_code"], "killed": payload["killed"],
                       "session_id": state.get("session_id"), "usage": state["usage"]}
                continue
            line = payload.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                yield {"type": "error", "text": line[:500]}
                continue
            for ev in self._parse(obj, state):
                yield ev


class KimiExecutor(_CliExecutor):
    id = "kimi"
    bin_env = "CHATTABLE_KIMI_BIN"
    default_bins = tuple(b for b in (shutil.which("kimi"),
                                     os.path.expanduser("~/.kimi-code/bin/kimi")) if b)

    def _argv(self, prompt: str, session_id: Optional[str]) -> list[str]:
        argv = ["kimi", "--output-format", "stream-json", "-p", prompt]
        if session_id:
            argv += ["-S", session_id]
        return argv

    def _parse(self, obj: dict, state: dict) -> list[dict]:
        return parse_kimi_line(obj, state)


class ClaudeExecutor(_CliExecutor):
    id = "claude"
    bin_env = "CHATTABLE_CLAUDE_BIN"
    default_bins = tuple(b for b in (shutil.which("claude"),) if b)

    def available(self) -> tuple[bool, str]:
        if not self._binary():
            return (False, "claude CLI not found (CHATTABLE_CLAUDE_BIN)")
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.path.isdir(os.path.expanduser("~/.claude"))):
            return (False, "claude CLI found but no auth (ANTHROPIC_API_KEY or ~/.claude)")
        return (True, "")

    def _argv(self, prompt: str, session_id: Optional[str]) -> list[str]:
        argv = ["claude", "-p", prompt, "--output-format", "stream-json",
                "--verbose", "--permission-mode", "acceptEdits"]
        if session_id:
            argv += ["--resume", session_id]
        return argv

    def _env(self) -> dict:
        return _exec_env(keep=("ANTHROPIC_API_KEY",))

    def _parse(self, obj: dict, state: dict) -> list[dict]:
        return parse_claude_line(obj, state)


class CodexExecutor(_CliExecutor):
    id = "codex"
    bin_env = "CHATTABLE_CODEX_BIN"
    default_bins = tuple(b for b in (shutil.which("codex"),) if b)

    def available(self) -> tuple[bool, str]:
        if not self._binary():
            return (False, "codex CLI not found (CHATTABLE_CODEX_BIN)")
        if not (os.environ.get("OPENAI_API_KEY")
                or os.path.isdir(os.path.expanduser("~/.codex"))):
            return (False, "codex CLI found but no auth (OPENAI_API_KEY or ~/.codex)")
        return (True, "")

    def _argv(self, prompt: str, session_id: Optional[str]) -> list[str]:
        if session_id:
            return ["codex", "exec", "resume", session_id, "--json", prompt]
        return ["codex", "exec", "--json", "--full-auto", prompt]

    def _env(self) -> dict:
        return _exec_env(keep=("OPENAI_API_KEY",))

    def _parse(self, obj: dict, state: dict) -> list[dict]:
        return parse_codex_line(obj, state)


class BuiltinExecutor:
    """The in-process DeepSeek agent loop, confined to the run's worktree
    with auto-approve (no prompts), via ChatService's override hooks."""

    id = "builtin"

    def __init__(self, chat_service=None) -> None:
        self._svc = chat_service

    def available(self) -> tuple[bool, str]:
        import main
        ok = bool(getattr(main, "OPENAI_API_KEY", ""))
        return (True, "") if ok else (False, "builtin executor: no model API key configured")

    def run(self, workdir: str, prompt: str, session_id: Optional[str] = None,
            should_stop=lambda: False, proc_reg=lambda p: None,
            timeout: Optional[int] = None) -> Iterator[dict]:
        import main  # noqa: F401 - ensures backend config is loaded
        import service as service_mod
        from store import TreeStore
        from service import ChatService
        from workspace import ShadowGit, Workspace

        run_dir = os.path.dirname(os.path.abspath(workdir))
        ws = Workspace(workdir)
        shadow = ShadowGit(ws, os.path.join(run_dir, "shadow.git"))
        svc = ChatService(TreeStore(os.path.join(run_dir, "chat.jsonl")))
        info = svc.start_turn(prompt, None, agent=True, agent_overrides={
            "workspace": ws, "shadow": shadow, "auto_approve": True,
        })
        final_text = ""
        usage: dict = {}
        deadline = time.monotonic() + (timeout or RUN_TIMEOUT_S)
        for ev in svc.subscribe(info["assistant_id"], 0):
            if should_stop() or time.monotonic() > deadline:
                svc.request_stop(info["assistant_id"])
            typ = ev.get("type")
            if typ == "step_start":
                yield {"type": "tool_call", "id": ev.get("id"), "name": ev.get("tool"),
                       "arguments": ev.get("args") or {}}
            elif typ == "tool_result":
                yield {"type": "tool_result", "id": ev.get("name"),
                       "output": str(ev.get("output") or "")[:2000], "ok": ev.get("ok")}
            elif typ == "plan":
                yield {"type": "meta", "kind": "plan", "items": ev.get("items")}
            elif typ == "error":
                yield {"type": "error", "text": ev.get("message") or "error"}
            elif typ == "notice":
                yield {"type": "meta", "kind": "notice", "text": ev.get("text") or ""}
            elif typ == "done":
                final_text = ev.get("content") or ""
                usage = ev.get("usage") or {}
        yield {"type": "result", "final_text": final_text,
               "exit_code": 0 if final_text else 1, "killed": None,
               "session_id": None, "usage": usage}


def default_executors(chat_service=None) -> dict:
    return {
        "kimi": KimiExecutor(),
        "claude": ClaudeExecutor(),
        "codex": CodexExecutor(),
        "builtin": BuiltinExecutor(chat_service),
    }
