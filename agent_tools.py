"""Agent-mode tools: workspace-confined file and shell tools.

These tools are only registered (and only reachable by the model) in agent
mode.  Every path argument goes through ``Workspace.resolve``, so nothing
here can read or write outside the workspace.  ``write_file``/``edit_file``
return a unified diff of the change; ``preview_change`` computes that diff
without applying it, for the approval UI.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
from typing import Any, Optional

from tools import ToolRegistry, ToolSpec
from web_tools import WebToolResult
from workspace import Workspace, WorkspaceError

READ_CAP_BYTES = 200 * 1024          # reads truncate past this, with a note
WRITE_CAP_BYTES = 1024 * 1024        # refuse larger writes
GREP_MAX_MATCHES = 100
LIST_MAX_ENTRIES = 500
SHELL_OUTPUT_CAP = 30_000            # chars kept (head + tail)
SHELL_MAX_TIMEOUT_S = 60

WRITE_TOOLS = ("write_file", "edit_file")
MUTATING_TOOLS = ("write_file", "edit_file", "run_shell")

# Never handed to shell subprocesses: the agent must not exfiltrate the
# server's credentials or the auth token that gates agent mode itself.
_SCRUB_ENV_RE = re.compile(r"(^OPENAI_API_KEY$|^DEEPSEEK_API_KEY$|"
                           r"^CHATTABLE_AUTH_TOKEN$|^CHATABLE_AUTH_TOKEN$|"
                           r"_API_KEY$|_SECRET$|_TOKEN$)")


def _err(message: str) -> WebToolResult:
    return WebToolResult.error(message)


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return False


def _read_text(abspath: str) -> tuple[Optional[str], Optional[str]]:
    """(content, error) — UTF-8 text only, binary files refused."""
    try:
        with open(abspath, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        return None, f"cannot read file: {e}"
    if b"\x00" in raw[:8192]:
        return None, "refusing to read a binary file"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "file is not valid UTF-8 text"


def _unified_diff(old: str, new: str, rel: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
    ))


# --------------------------------------------------------------------------- #
# Tool implementations (each takes the already-confined workspace)
# --------------------------------------------------------------------------- #

def do_list_dir(ws: Workspace, path: str = ".", depth: int = 2) -> WebToolResult:
    root = ws.resolve(path)
    if not os.path.isdir(root):
        return _err(f"not a directory: {path}")
    depth = max(1, min(int(depth or 1), 5))
    lines: list[str] = []
    truncated = False

    def walk(d: str, level: int, prefix: str) -> None:
        nonlocal truncated
        if len(lines) >= LIST_MAX_ENTRIES:
            truncated = True
            return
        try:
            names = sorted(os.listdir(d), key=str.casefold)
        except OSError:
            return
        for name in names:
            if name == ".git":
                continue
            if len(lines) >= LIST_MAX_ENTRIES:
                truncated = True
                return
            full = os.path.join(d, name)
            is_dir = os.path.isdir(full)
            lines.append(f"{prefix}{'[dir] ' if is_dir else ''}{name}")
            if is_dir and level < depth:
                walk(full, level + 1, prefix + "  ")

    walk(root, 1, "")
    out = "\n".join(lines) or "(empty directory)"
    if truncated:
        out += f"\n[listing truncated at {LIST_MAX_ENTRIES} entries]"
    return WebToolResult.success(out)


def do_read_file(ws: Workspace, path: str, offset: int = 1, limit: int = 0) -> WebToolResult:
    abspath = ws.resolve(path)
    if os.path.isdir(abspath):
        return _err(f"is a directory: {path}")
    content, error = _read_text(abspath)
    if error:
        return _err(error)
    assert content is not None
    truncated_note = ""
    if len(content.encode("utf-8")) > READ_CAP_BYTES:
        raw = content.encode("utf-8")[:READ_CAP_BYTES]
        content = raw.decode("utf-8", errors="ignore")
        truncated_note = (f"\n[file truncated at {READ_CAP_BYTES} bytes; "
                          f"use offset/limit to page through it]")
    lines = content.splitlines()
    total = len(lines)
    start = max(1, int(offset or 1))
    end = total if not limit else min(total, start - 1 + max(1, int(limit)))
    if start > total and total > 0:
        return _err(f"offset {start} past end of file ({total} lines)")
    numbered = [f"{i}\t{lines[i - 1]}" for i in range(start, end + 1)]
    header = f"{ws.rel(abspath)} — lines {start}-{end} of {total}"
    return WebToolResult.success(header + "\n" + "\n".join(numbered) + truncated_note)


def do_grep(ws: Workspace, pattern: str, path: str = ".", glob: str = "") -> WebToolResult:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return _err(f"invalid regex: {e}")
    root = ws.resolve(path or ".")
    targets: list[str] = []
    if os.path.isfile(root):
        targets.append(root)
    elif os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in sorted(filenames):
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                targets.append(os.path.join(dirpath, name))
    else:
        return _err(f"no such file or directory: {path}")
    matches: list[str] = []
    files_searched = 0
    for full in sorted(targets):
        if len(matches) >= GREP_MAX_MATCHES:
            break
        if _is_binary(full):
            continue
        content, error = _read_text(full)
        if error:
            continue
        files_searched += 1
        for lineno, line in enumerate(content.splitlines(), start=1):
            if rx.search(line):
                matches.append(f"{ws.rel(full)}:{lineno}: {line.strip()[:300]}")
                if len(matches) >= GREP_MAX_MATCHES:
                    break
    out = "\n".join(matches)
    if len(matches) >= GREP_MAX_MATCHES:
        out += f"\n[truncated at {GREP_MAX_MATCHES} matches]"
    if not out:
        out = f"(no matches in {files_searched} file(s))"
    return WebToolResult.success(out)


def _plan_write(ws: Workspace, path: str, content: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(abspath, old_text, diff, error) for a write_file call."""
    if len(content.encode("utf-8")) > WRITE_CAP_BYTES:
        return None, None, None, f"content too large (max {WRITE_CAP_BYTES} bytes)"
    abspath = ws.resolve(path)
    if os.path.isdir(abspath):
        return None, None, None, f"is a directory: {path}"
    old = ""
    if os.path.exists(abspath):
        old, error = _read_text(abspath)
        if error:
            return None, None, None, error
        assert old is not None
    if old == content:
        return None, None, None, "no change: content is identical"
    return abspath, old, _unified_diff(old, content, ws.rel(abspath)), None


def _plan_edit(ws: Workspace, path: str, old_string: str, new_string: str,
               replace_all: bool = False
               ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(abspath, new_text, diff, error) for an edit_file call."""
    abspath = ws.resolve(path)
    if not os.path.exists(abspath):
        return None, None, None, f"file not found: {path}"
    old, error = _read_text(abspath)
    if error:
        return None, None, None, error
    assert old is not None
    count = old.count(old_string)
    if count == 0:
        return None, None, None, "old_string not found in file"
    if count > 1 and not replace_all:
        return None, None, None, (
            f"old_string is not unique ({count} occurrences); "
            "pass replace_all=true or a longer, unique old_string"
        )
    new = old.replace(old_string, new_string) if replace_all \
        else old.replace(old_string, new_string, 1)
    if new == old:
        return None, None, None, "no change: old_string equals new_string"
    if len(new.encode("utf-8")) > WRITE_CAP_BYTES:
        return None, None, None, f"result too large (max {WRITE_CAP_BYTES} bytes)"
    return abspath, new, _unified_diff(old, new, ws.rel(abspath)), None


def do_write_file(ws: Workspace, path: str, content: str) -> WebToolResult:
    try:
        abspath, _old, diff, error = _plan_write(ws, path, content)
    except WorkspaceError as e:
        return _err(str(e))
    if error:
        return _err(error)
    assert abspath is not None and diff is not None
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, "w", encoding="utf-8") as fh:
        fh.write(content)
    return WebToolResult.success(
        f"wrote {ws.rel(abspath)} ({len(content)} chars)\n{diff}", diff=diff)


def do_edit_file(ws: Workspace, path: str, old_string: str, new_string: str,
                 replace_all: bool = False) -> WebToolResult:
    try:
        abspath, new, diff, error = _plan_edit(ws, path, old_string, new_string, replace_all)
    except WorkspaceError as e:
        return _err(str(e))
    if error:
        return _err(error)
    assert abspath is not None and new is not None and diff is not None
    with open(abspath, "w", encoding="utf-8") as fh:
        fh.write(new)
    return WebToolResult.success(
        f"edited {ws.rel(abspath)}\n{diff}", diff=diff)


def preview_change(ws: Workspace, name: str, args: dict[str, Any]) -> str:
    """Unified diff a write_file/edit_file call WOULD produce ("" if none).

    Used for the approval prompt: the user sees the exact change before the
    tool runs.  Errors come back as a short text note instead of a diff.
    """
    try:
        if name == "write_file":
            _a, _o, diff, error = _plan_write(
                ws, str(args.get("path") or ""), str(args.get("content") or ""))
        elif name == "edit_file":
            _a, _n, diff, error = _plan_edit(
                ws, str(args.get("path") or ""), str(args.get("old_string") or ""),
                str(args.get("new_string") or ""), bool(args.get("replace_all")))
        else:
            return ""
    except (WorkspaceError, OSError) as e:
        return f"[preview unavailable: {e}]"
    if error:
        return f"[change would fail: {error}]"
    return diff or ""


def do_run_shell(ws: Workspace, command: str, timeout: int = 30) -> WebToolResult:
    timeout = max(1, min(int(timeout or 30), SHELL_MAX_TIMEOUT_S))
    env = {k: v for k, v in os.environ.items() if not _SCRUB_ENV_RE.search(k)}
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=ws.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return _err(f"command timed out after {timeout}s (killed)")
    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + proc.stderr
    if len(out) > SHELL_OUTPUT_CAP:
        head = SHELL_OUTPUT_CAP * 2 // 3
        tail = SHELL_OUTPUT_CAP - head
        out = (out[:head] + f"\n[... output truncated, {len(out) - SHELL_OUTPUT_CAP} "
               f"chars omitted ...]\n" + out[-tail:])
    text = f"$ {command}\n[exit {proc.returncode}]\n{out}".strip()
    return WebToolResult(proc.returncode == 0, text)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def _str(args: dict, key: str, default: str = "") -> str:
    value = args.get(key)
    return default if value is None else str(value)


def build_agent_registry(ws: Workspace) -> ToolRegistry:
    """The agent-mode tool set, all confined to *ws*."""
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="list_dir",
        description="List files and directories in the workspace as a tree.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory, relative to the workspace root (default '.')."},
                "depth": {"type": "integer", "description": "Tree depth 1-5 (default 2)."},
            },
        },
        handler=lambda a: do_list_dir(ws, _str(a, "path", "."), int(a.get("depth") or 2)),
        read_only=True,
        result_max_chars=20_000,
    ))
    registry.register(ToolSpec(
        name="read_file",
        description="Read a workspace text file with line numbers. Use offset/limit to page.",
        input_schema={
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the workspace root."},
                "offset": {"type": "integer", "description": "First line to show, 1-based (default 1)."},
                "limit": {"type": "integer", "description": "Max lines to show (default: whole file, capped at 200KB)."},
            },
        },
        handler=lambda a: do_read_file(ws, _str(a, "path"), int(a.get("offset") or 1),
                                       int(a.get("limit") or 0)),
        read_only=True,
        result_max_chars=220_000,
    ))
    registry.register(ToolSpec(
        name="grep",
        description="Search workspace files for a regex pattern. Returns path:line: text matches.",
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "File or directory to search (default: whole workspace)."},
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
            },
        },
        handler=lambda a: do_grep(ws, _str(a, "pattern"), _str(a, "path", "."),
                                  _str(a, "glob")),
        read_only=True,
        result_max_chars=20_000,
    ))
    registry.register(ToolSpec(
        name="write_file",
        description="Create or overwrite a workspace file (parent directories are created). Returns a unified diff.",
        input_schema={
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the workspace root."},
                "content": {"type": "string", "description": "Full new file content."},
            },
        },
        handler=lambda a: do_write_file(ws, _str(a, "path"), _str(a, "content")),
        read_only=False,
        result_max_chars=30_000,
    ))
    registry.register(ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact string in a workspace file. Fails when old_string "
            "is not found or occurs more than once (unless replace_all=true). "
            "Returns a unified diff."
        ),
        input_schema={
            "type": "object",
            "required": ["path", "old_string", "new_string"],
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the workspace root."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
            },
        },
        handler=lambda a: do_edit_file(ws, _str(a, "path"), _str(a, "old_string"),
                                       _str(a, "new_string"), bool(a.get("replace_all"))),
        read_only=False,
        result_max_chars=30_000,
    ))
    registry.register(ToolSpec(
        name="run_shell",
        description=(
            "Run a shell command in the workspace (always requires approval). "
            f"Timeout max {SHELL_MAX_TIMEOUT_S}s; output is truncated."
        ),
        input_schema={
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string", "description": "Shell command, run with cwd=workspace."},
                "timeout": {"type": "integer", "description": f"Seconds, 1-{SHELL_MAX_TIMEOUT_S} (default 30)."},
            },
        },
        handler=lambda a: do_run_shell(ws, _str(a, "command"), int(a.get("timeout") or 30)),
        read_only=False,
        result_max_chars=40_000,
    ))
    return registry
