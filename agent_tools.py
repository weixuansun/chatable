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
import glob as globmod
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from tools import ToolRegistry, ToolSpec
from web_tools import WebToolResult
from workspace import Workspace, WorkspaceError

READ_CAP_BYTES = 200 * 1024          # reads truncate past this, with a note
WRITE_CAP_BYTES = 1024 * 1024        # refuse larger writes
GREP_MAX_MATCHES = 100
GLOB_MAX_RESULTS = 200
LIST_MAX_ENTRIES = 500
SHELL_OUTPUT_CAP = 30_000            # chars kept (head + tail) for small outputs
SHELL_STREAM_KEEP = 8192             # head and tail bytes kept from streamed output
SHELL_DEFAULT_TIMEOUT_S = 120
SHELL_MAX_TIMEOUT_S = 600

WRITE_TOOLS = ("write_file", "edit_file")
MUTATING_TOOLS = ("write_file", "edit_file", "run_shell")

# Directories that are noise for exploration (never listed/grepped/globbed).
SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "dist", "build", ".idea", ".vscode",
})

PLAN_STATUSES = ("pending", "in_progress", "completed")

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


class _Gitignore:
    """Cheap subset of a workspace-root .gitignore: plain names, globs,
    dir-only patterns (trailing /) and root-anchored patterns (leading /).
    Negations and nested .gitignore files are intentionally unsupported."""

    def __init__(self, ws_root: str) -> None:
        self.patterns: list[str] = []
        try:
            with open(os.path.join(ws_root, ".gitignore"), encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            self.patterns.append(line)

    def match(self, rel: str, is_dir: bool) -> bool:
        rel = rel.replace(os.sep, "/")
        base = rel.rsplit("/", 1)[-1]
        for pat in self.patterns:
            p = pat
            dir_only = p.endswith("/")
            p = p.rstrip("/")
            if not p or (dir_only and not is_dir):
                continue
            if p.startswith("/"):
                if fnmatch.fnmatch(rel, p[1:]):
                    return True
            elif "/" in p:
                if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p + "/*"):
                    return True
            elif fnmatch.fnmatch(base, p):
                return True
        return False


def _skipped(rel: str, is_dir: bool, gi: _Gitignore) -> bool:
    parts = rel.replace(os.sep, "/").split("/")
    if any(part in SKIP_DIRS for part in parts):
        return True
    return gi.match(rel, is_dir)


# --------------------------------------------------------------------------- #
# Tool implementations (each takes the already-confined workspace)
# --------------------------------------------------------------------------- #

def do_list_dir(ws: Workspace, path: str = ".", depth: int = 2) -> WebToolResult:
    root = ws.resolve(path)
    if not os.path.isdir(root):
        return _err(f"not a directory: {path}")
    depth = max(1, min(int(depth or 1), 5))
    gi = _Gitignore(ws.root)
    lines: list[str] = []
    truncated = False

    def walk(d: str, level: int) -> None:
        nonlocal truncated
        if truncated:
            return
        try:
            names = sorted(os.listdir(d), key=str.casefold)
        except OSError:
            return
        for name in names:
            full = os.path.join(d, name)
            is_dir = os.path.isdir(full)
            rel = ws.rel(full)
            if _skipped(rel, is_dir, gi):
                continue
            lines.append(rel + ("/" if is_dir else ""))
            if len(lines) >= LIST_MAX_ENTRIES:
                truncated = True
                return
            if is_dir and level < depth:
                walk(full, level + 1)
                if truncated:
                    return

    walk(root, 1)
    out = "\n".join(lines) or "(empty directory)"
    if truncated:
        out += f"\n[... listing truncated at {LIST_MAX_ENTRIES} entries]"
    return WebToolResult.success(out)


def do_glob(ws: Workspace, pattern: str, path: str = ".") -> WebToolResult:
    pattern = (pattern or "").strip()
    if not pattern:
        return _err("pattern is required")
    if pattern.startswith("/") or ".." in pattern.split("/"):
        return _err("pattern must stay inside the workspace")
    root = ws.resolve(path or ".")
    if not os.path.isdir(root):
        return _err(f"not a directory: {path}")
    gi = _Gitignore(ws.root)
    matches: list[str] = []
    for rel in globmod.glob(pattern, root_dir=root, recursive=True):
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        if _skipped(ws.rel(full), False, gi):
            continue
        matches.append(full)
    # Newest first — usually what the agent is looking for.
    matches.sort(key=lambda f: (-os.path.getmtime(f), f))
    total = len(matches)
    lines = [ws.rel(f) for f in matches[:GLOB_MAX_RESULTS]]
    out = "\n".join(lines) or "(no matches)"
    if total > GLOB_MAX_RESULTS:
        out += f"\n[... {total - GLOB_MAX_RESULTS} more matches]"
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


def do_grep(ws: Workspace, pattern: str, path: str = ".", glob: str = "",
            ignore_case: bool = False, context: int = 0) -> WebToolResult:
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return _err(f"invalid regex: {e}")
    context = max(0, min(int(context or 0), 5))
    root = ws.resolve(path or ".")
    gi = _Gitignore(ws.root)
    targets: list[str] = []
    if os.path.isfile(root):
        targets.append(root)
    elif os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not _skipped(ws.rel(os.path.join(dirpath, d)), True, gi)]
            for name in sorted(filenames):
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                full = os.path.join(dirpath, name)
                if _skipped(ws.rel(full), False, gi):
                    continue
                targets.append(full)
    else:
        return _err(f"no such file or directory: {path}")
    out_lines: list[str] = []
    total = 0
    for full in sorted(targets):
        if _is_binary(full):
            continue
        content, error = _read_text(full)
        if error:
            continue
        lines = content.splitlines()
        rel = ws.rel(full)
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            total += 1
            if total > GREP_MAX_MATCHES:
                continue  # keep counting for the "N more" note
            if context:
                lo = max(0, i - context)
                hi = min(len(lines), i + context + 1)
                for j in range(lo, hi):
                    mark = ":" if j == i else "-"
                    out_lines.append(f"{rel}{mark}{j + 1}{mark} {lines[j][:300]}")
            else:
                out_lines.append(f"{rel}:{i + 1}: {line[:300]}")
    out = "\n".join(out_lines)
    if total > GREP_MAX_MATCHES:
        note = f"[... {total - GREP_MAX_MATCHES} more matches]"
        out = (out + "\n" + note) if out else note
    if not out:
        out = "(no matches)"
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


def _occurrence_lines(content: str, needle: str) -> list[int]:
    """1-based line numbers of each occurrence of *needle* in *content*."""
    lines: list[int] = []
    pos = 0
    while True:
        idx = content.find(needle, pos)
        if idx == -1:
            return lines
        lines.append(content.count("\n", 0, idx) + 1)
        pos = idx + 1


def _most_similar_region(content: str, old_string: str) -> str:
    """The file region most similar to old_string (line-numbered), for the
    not-found error. Quick upper-bound ratios keep this cheap on big files."""
    old_lines = old_string.splitlines()
    content_lines = content.splitlines()
    n = len(old_lines)
    if not n or len(content_lines) < n:
        return ""
    best_ratio, best_start = 0.0, 0
    last = min(len(content_lines) - n, 5000)
    for i in range(last + 1):
        window = "\n".join(content_lines[i:i + n])
        r = difflib.SequenceMatcher(None, old_string, window).quick_ratio()
        if r > best_ratio:
            best_ratio, best_start = r, i
    if best_ratio < 0.5:
        return ""
    region = "\n".join(
        f"{best_start + j + 1}\t{content_lines[best_start + j]}" for j in range(n)
    )
    return f"Most similar region (similarity ~{best_ratio:.0%}):\n{region}"


def _plan_edit(ws: Workspace, path: str, old_string: str, new_string: str,
               replace_all: bool = False,
               expected_replacements: Optional[int] = None,
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
        msg = ("old_string not found in file. Whitespace and indentation "
               "must match exactly what is in the file — re-read the file "
               "with read_file and copy the exact text.")
        similar = _most_similar_region(old, old_string)
        if similar:
            msg += "\n" + similar
        return None, None, None, msg
    if expected_replacements is not None and count != expected_replacements:
        return None, None, None, (
            f"expected {expected_replacements} occurrence(s) of old_string, "
            f"found {count}"
        )
    if count > 1 and not replace_all and expected_replacements is None:
        lines = ", ".join(str(n) for n in _occurrence_lines(old, old_string))
        return None, None, None, (
            f"old_string is not unique ({count} occurrences at lines {lines}); "
            "pass replace_all=true, set expected_replacements, or use a "
            "longer, unique old_string"
        )
    if replace_all or expected_replacements is not None:
        new = old.replace(old_string, new_string)
    else:
        new = old.replace(old_string, new_string, 1)
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
                 replace_all: bool = False,
                 expected_replacements: Optional[int] = None) -> WebToolResult:
    try:
        abspath, new, diff, error = _plan_edit(
            ws, path, old_string, new_string, replace_all, expected_replacements)
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
                str(args.get("new_string") or ""), bool(args.get("replace_all")),
                args.get("expected_replacements"))
        else:
            return ""
    except (WorkspaceError, OSError) as e:
        return f"[preview unavailable: {e}]"
    if error:
        return f"[change would fail: {error}]"
    return diff or ""


def do_run_shell(ws: Workspace, command: str, timeout: int = SHELL_DEFAULT_TIMEOUT_S,
                 on_output: Optional[Callable[[str], None]] = None,
                 on_proc: Optional[Callable] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> WebToolResult:
    """Run a shell command, streaming output chunks to ``on_output``.

    The command runs in its own process group (start_new_session); a stop
    request (``should_stop``) or the timeout kills the WHOLE group with
    SIGKILL, so children of the command die with it.  Stored output keeps the
    first and last SHELL_STREAM_KEEP bytes with an omission note.
    """
    timeout = max(1, min(int(timeout or SHELL_DEFAULT_TIMEOUT_S), SHELL_MAX_TIMEOUT_S))
    env = {k: v for k, v in os.environ.items() if not _SCRUB_ENV_RE.search(k)}
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=ws.root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
    except OSError as e:
        return _err(f"failed to start command: {e}")
    if on_proc:
        on_proc(proc)

    head: list[str] = []
    head_len = 0
    tail: deque = deque(maxlen=SHELL_STREAM_KEEP)
    total = 0

    def reader() -> None:
        nonlocal head_len, total
        try:
            while True:
                chunk = proc.stdout.read(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                total += len(chunk)
                if on_output:
                    try:
                        on_output(chunk)
                    except Exception:  # noqa: BLE001 - streaming must not kill the reader
                        pass
                rest = chunk
                if head_len < SHELL_STREAM_KEEP:
                    take = min(len(chunk), SHELL_STREAM_KEEP - head_len)
                    head.append(chunk[:take])
                    head_len += take
                    rest = chunk[take:]
                if rest:
                    tail.extend(rest)
        except (OSError, ValueError):
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    deadline = time.monotonic() + timeout
    killed = ""
    while proc.poll() is None:
        if should_stop and should_stop():
            killed = "killed: stopped by user"
        elif time.monotonic() >= deadline:
            killed = f"killed: timed out after {timeout}s"
        if killed:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            break
        time.sleep(0.05)
    proc.wait()
    t.join(timeout=2)
    if on_proc:
        on_proc(None)

    out = "".join(head)
    omitted = total - head_len - len(tail)
    if omitted > 0:
        out += f"\n[... {omitted} bytes omitted ...]\n"
    out += "".join(tail)
    rc = proc.returncode
    status = f"[{killed}]" if killed else f"[exit {rc}]"
    text = f"$ {command}\n{status}\n{out}".strip()
    return WebToolResult(rc == 0 and not killed, text)


def do_update_plan(items: Any) -> WebToolResult:
    """Validate and render the agent's working plan (TodoWrite-style)."""
    if not isinstance(items, list) or not items:
        return _err("items must be a non-empty list of {content, status}")
    normalized: list[dict] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return _err(f"item {i} must be an object with content and status")
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "").strip()
        if not content:
            return _err(f"item {i}: content is required")
        if status not in PLAN_STATUSES:
            return _err(f"item {i}: status must be one of {', '.join(PLAN_STATUSES)}")
        normalized.append({"content": content, "status": status})
    marks = {"pending": "○", "in_progress": "◐", "completed": "●"}
    rendered = "Plan:\n" + "\n".join(
        f"{marks[i['status']]} [{i['status']}] {i['content']}" for i in normalized
    )
    return WebToolResult.success(rendered, items=normalized)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def _str(args: dict, key: str, default: str = "") -> str:
    value = args.get(key)
    return default if value is None else str(value)


def build_agent_registry(ws: Workspace, ctx: Optional[dict] = None) -> ToolRegistry:
    """The agent-mode tool set, all confined to *ws*.

    ``ctx`` is the per-turn execution context (all keys optional):
    ``on_output(chunk)`` for streamed shell output, ``on_proc(proc|None)``
    for the live shell process (stop kills its group), ``should_stop()``.
    """
    ctx = ctx or {}
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="list_dir",
        description=(
            "List files in the workspace as workspace-relative paths "
            "(directories end with '/'). Noise dirs (.git, __pycache__, "
            "node_modules, …) are skipped automatically."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory, relative to the workspace root (default '.')."},
                "depth": {"type": "integer", "description": "How deep to recurse, 1-5 (default 2)."},
            },
        },
        handler=lambda a: do_list_dir(ws, _str(a, "path", "."), int(a.get("depth") or 2)),
        read_only=True,
        result_max_chars=20_000,
    ))
    registry.register(ToolSpec(
        name="glob",
        description=(
            "Find workspace files by glob pattern (e.g. '**/*.py', 'src/*.ts'). "
            "Returns workspace-relative paths, newest first."
        ),
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern; '**' matches any depth."},
                "path": {"type": "string", "description": "Directory to search from (default: workspace root)."},
            },
        },
        handler=lambda a: do_glob(ws, _str(a, "pattern"), _str(a, "path", ".")),
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
        description=(
            "Search workspace files for a regex pattern. Output is "
            "'path:line: text' (context lines use '-' separators)."
        ),
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "File or directory to search (default: whole workspace)."},
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default false)."},
                "context": {"type": "integer", "description": "Show this many lines before/after each match, 0-5 (default 0)."},
            },
        },
        handler=lambda a: do_grep(ws, _str(a, "pattern"), _str(a, "path", "."),
                                  _str(a, "glob"), bool(a.get("ignore_case")),
                                  int(a.get("context") or 0)),
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
            "Replace an exact string in a workspace file. Whitespace and "
            "indentation must match the file exactly. Fails when old_string is "
            "not found (returns the most similar region) or occurs more than "
            "once (unless replace_all or expected_replacements is set). "
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
                "expected_replacements": {"type": "integer", "description": "Replace all, but fail unless exactly this many occurrences exist."},
            },
        },
        handler=lambda a: do_edit_file(ws, _str(a, "path"), _str(a, "old_string"),
                                       _str(a, "new_string"), bool(a.get("replace_all")),
                                       a.get("expected_replacements")),
        read_only=False,
        result_max_chars=30_000,
    ))
    registry.register(ToolSpec(
        name="run_shell",
        description=(
            "Run a shell command in the workspace (always requires approval). "
            f"Timeout default {SHELL_DEFAULT_TIMEOUT_S}s, max {SHELL_MAX_TIMEOUT_S}s; "
            "long output is truncated keeping head and tail."
        ),
        input_schema={
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "string", "description": "Shell command, run with cwd=workspace."},
                "timeout": {"type": "integer", "description": f"Seconds, 1-{SHELL_MAX_TIMEOUT_S} (default {SHELL_DEFAULT_TIMEOUT_S})."},
            },
        },
        handler=lambda a: do_run_shell(
            ws, _str(a, "command"), int(a.get("timeout") or SHELL_DEFAULT_TIMEOUT_S),
            on_output=ctx.get("on_output"), on_proc=ctx.get("on_proc"),
            should_stop=ctx.get("should_stop")),
        read_only=False,
        result_max_chars=40_000,
    ))
    registry.register(ToolSpec(
        name="update_plan",
        description=(
            "Track the plan for a multi-step task. Call with the full updated "
            "list whenever you start or finish a step."
        ),
        input_schema={
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array", "description": "[{content: string, status: 'pending'|'in_progress'|'completed'}]"},
            },
        },
        handler=lambda a: do_update_plan(a.get("items")),
        read_only=True,
        result_max_chars=8_000,
    ))
    return registry
