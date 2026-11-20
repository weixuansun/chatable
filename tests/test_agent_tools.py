"""Agent workspace tools: list_dir / glob / read_file / grep / write_file / edit_file / run_shell / update_plan."""
import os
import time

import pytest

from agent_tools import (
    GREP_MAX_MATCHES,
    READ_CAP_BYTES,
    SHELL_STREAM_KEEP,
    build_agent_registry,
    do_edit_file,
    do_glob,
    do_grep,
    do_list_dir,
    do_read_file,
    do_run_shell,
    do_update_plan,
    do_write_file,
    preview_change,
)
from workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "ws"))


def _read(ws, rel):
    with open(os.path.join(ws.root, rel), encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------ write_file

def test_write_file_creates_parents_and_returns_diff(ws):
    r = do_write_file(ws, "sub/dir/a.txt", "hello\n")
    assert r.ok
    assert _read(ws, "sub/dir/a.txt") == "hello\n"
    diff = r.metadata["diff"]
    assert "--- a/sub/dir/a.txt" in diff
    assert "+++ b/sub/dir/a.txt" in diff
    assert "+hello" in diff


def test_write_file_overwrite_diff_shows_old_and_new(ws):
    do_write_file(ws, "a.txt", "v1\n")
    r = do_write_file(ws, "a.txt", "v2\n")
    assert r.ok
    assert "-v1" in r.metadata["diff"] and "+v2" in r.metadata["diff"]


def test_write_file_rejects_escape(ws):
    r = do_write_file(ws, "../evil.txt", "x")
    assert not r.ok
    assert "escapes" in r.output
    assert not os.path.exists(os.path.join(os.path.dirname(ws.root), "evil.txt"))


def test_write_file_no_change_is_an_error(ws):
    do_write_file(ws, "a.txt", "same\n")
    r = do_write_file(ws, "a.txt", "same\n")
    assert not r.ok and "no change" in r.output


# ------------------------------------------------------------------- edit_file

def test_edit_file_replaces_unique_string(ws):
    do_write_file(ws, "a.txt", "foo bar\n")
    r = do_edit_file(ws, "a.txt", "bar", "baz")
    assert r.ok
    assert _read(ws, "a.txt") == "foo baz\n"
    assert "-foo bar" in r.metadata["diff"] and "+foo baz" in r.metadata["diff"]


def test_edit_file_requires_unique_old_string(ws):
    do_write_file(ws, "a.txt", "foo x\nfoo y\nfoo z\n")
    r = do_edit_file(ws, "a.txt", "foo", "baz")
    assert not r.ok
    assert "not unique" in r.output
    assert "lines 1, 2, 3" in r.output  # occurrences reported with line numbers
    assert _read(ws, "a.txt") == "foo x\nfoo y\nfoo z\n"  # untouched


def test_edit_file_replace_all(ws):
    do_write_file(ws, "a.txt", "foo x foo\n")
    r = do_edit_file(ws, "a.txt", "foo", "baz", replace_all=True)
    assert r.ok
    assert _read(ws, "a.txt") == "baz x baz\n"


def test_edit_file_expected_replacements(ws):
    do_write_file(ws, "a.txt", "foo x foo\n")
    r = do_edit_file(ws, "a.txt", "foo", "baz", expected_replacements=2)
    assert r.ok
    assert _read(ws, "a.txt") == "baz x baz\n"
    r = do_edit_file(ws, "a.txt", "baz", "qux", expected_replacements=1)
    assert not r.ok and "expected 1 occurrence" in r.output and "found 2" in r.output


def test_edit_file_not_found_shows_similar_region(ws):
    do_write_file(ws, "a.txt", "def fib(n):\n    a, b = 0, 1\n    return a\n")
    # Wrong indentation: the real line starts with 4 spaces, we ask for 2.
    r = do_edit_file(ws, "a.txt", "\n  a, b = 0, 1", "\n  a, b = 1, 1")
    assert not r.ok
    assert "must match exactly" in r.output
    assert "Most similar region" in r.output
    assert "2\t    a, b = 0, 1" in r.output  # line-numbered hint


def test_edit_file_missing_file(ws):
    r = do_edit_file(ws, "nope.txt", "a", "b")
    assert not r.ok and "not found" in r.output


# ------------------------------------------------------------------- read_file

def test_read_file_line_numbers_offset_limit(ws):
    do_write_file(ws, "a.txt", "".join(f"line{i}\n" for i in range(1, 11)))
    r = do_read_file(ws, "a.txt", offset=3, limit=2)
    assert r.ok
    assert "3\tline3" in r.output
    assert "4\tline4" in r.output
    assert "5\tline5" not in r.output
    assert "lines 3-4 of 10" in r.output


def test_read_file_truncates_huge_files(ws):
    do_write_file(ws, "big.txt", "x" * (READ_CAP_BYTES + 1000))
    r = do_read_file(ws, "big.txt")
    assert r.ok
    assert "truncated" in r.output


def test_read_file_refuses_binary_and_dirs(ws):
    with open(os.path.join(ws.root, "b.bin"), "wb") as fh:
        fh.write(b"\x00\x01\x02")
    assert not do_read_file(ws, "b.bin").ok
    assert not do_read_file(ws, ".").ok


def test_read_file_offset_past_end(ws):
    do_write_file(ws, "a.txt", "one\n")
    r = do_read_file(ws, "a.txt", offset=99)
    assert not r.ok


# --------------------------------------------------------------------- list_dir

def test_list_dir_relative_paths_and_depth(ws):
    do_write_file(ws, "pkg/sub/mod.py", "x\n")
    do_write_file(ws, "top.txt", "y\n")
    r = do_list_dir(ws, ".")
    assert r.ok
    # Workspace-relative paths, dirs with trailing slash — no indentation.
    assert "pkg/" in r.output and "pkg/sub/" in r.output
    assert "top.txt" in r.output
    assert "pkg/sub/mod.py" not in r.output  # level 3, beyond the default depth 2
    r3 = do_list_dir(ws, ".", depth=3)
    assert "pkg/sub/mod.py" in r3.output
    r1 = do_list_dir(ws, ".", depth=1)
    assert "pkg/" in r1.output and "pkg/sub/" not in r1.output


def test_list_dir_skips_noise_dirs_and_gitignore(ws):
    for d in (".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"):
        os.makedirs(os.path.join(ws.root, d), exist_ok=True)
        do_write_file(ws, f"{d}/x.txt", "noise\n")
    do_write_file(ws, "keep.txt", "k\n")
    do_write_file(ws, ".gitignore", "secret.txt\n")
    do_write_file(ws, "secret.txt", "s\n")
    r = do_list_dir(ws, ".")
    lines = r.output.splitlines()
    assert "keep.txt" in lines
    for line in lines:
        for noise in (".git/", "__pycache__/", ".pytest_cache/", "node_modules/",
                      ".venv/", "secret.txt", "x.txt"):
            assert noise != line and not line.startswith(noise), line


def test_list_dir_not_a_directory(ws):
    do_write_file(ws, "a.txt", "x\n")
    assert not do_list_dir(ws, "a.txt").ok


# ------------------------------------------------------------------------ glob

def test_glob_finds_files_newest_first(ws):
    do_write_file(ws, "old.py", "x = 1\n")
    time.sleep(0.02)
    do_write_file(ws, "pkg/new.py", "y = 2\n")
    r = do_glob(ws, "**/*.py")
    assert r.ok
    lines = r.output.splitlines()
    assert lines[0] == "pkg/new.py" and lines[1] == "old.py"


def test_glob_rejects_escapes_and_skips_noise(ws):
    assert not do_glob(ws, "../*.py").ok
    assert not do_glob(ws, "/etc/*.conf").ok
    do_write_file(ws, "__pycache__/m.cpython-313.pyc", "junk")
    do_write_file(ws, "m.py", "x\n")
    r = do_glob(ws, "**/*")
    assert "m.py" in r.output and "__pycache__" not in r.output


# ------------------------------------------------------------------------ grep

def test_grep_finds_matches_with_glob(ws):
    do_write_file(ws, "a.py", "def hello():\n    pass\n")
    do_write_file(ws, "b.txt", "hello world\n")
    r = do_grep(ws, "hello")
    assert "a.py:1:" in r.output and "b.txt:1:" in r.output
    r = do_grep(ws, "hello", glob="*.py")
    assert "a.py:1:" in r.output and "b.txt" not in r.output


def test_grep_ignore_case_and_context(ws):
    do_write_file(ws, "a.py", "class Foo:\n    def Hello(self):\n        return 1\n")
    assert "no matches" in do_grep(ws, "hello").output
    r = do_grep(ws, "hello", ignore_case=True, context=1)
    assert "a.py:2:     def Hello(self):" in r.output
    assert "a.py-1- class Foo:" in r.output          # context lines use '-'
    assert "a.py-3-         return 1" in r.output


def test_grep_counts_beyond_cap(ws):
    body = "".join(f"match line {i}\n" for i in range(GREP_MAX_MATCHES + 7))
    do_write_file(ws, "big.txt", body)
    r = do_grep(ws, "match line")
    assert r.ok
    assert f"{7} more matches" in r.output


def test_grep_no_matches_bad_regex_and_noise_skip(ws):
    do_write_file(ws, "a.txt", "nothing here\n")
    assert "no matches" in do_grep(ws, "zzzz").output
    assert not do_grep(ws, "(unclosed").ok
    do_write_file(ws, "__pycache__/cached.txt", "needle\n")
    assert "no matches" in do_grep(ws, "needle").output


# -------------------------------------------------------------------- run_shell

def test_run_shell_executes_in_workspace(ws):
    r = do_run_shell(ws, "pwd")
    assert r.ok
    assert os.path.realpath(ws.root) in r.output.replace("\n", "")
    r = do_run_shell(ws, "echo hi > out.txt")
    assert r.ok and _read(ws, "out.txt") == "hi\n"


def test_run_shell_nonzero_exit_and_stderr(ws):
    r = do_run_shell(ws, "echo oops >&2; exit 3")
    assert not r.ok
    assert "[exit 3]" in r.output and "oops" in r.output


def test_run_shell_timeout(ws):
    r = do_run_shell(ws, "sleep 5", timeout=1)
    assert not r.ok and "timed out" in r.output


def test_run_shell_timeout_bounds(ws):
    # Default 120s, capped at 600s — a bogus huge value must not hang the test.
    r = do_run_shell(ws, "echo ok", timeout=999999)
    assert r.ok and "ok" in r.output


def test_run_shell_streams_output_chunks(ws):
    chunks = []
    r = do_run_shell(ws, "echo line1; echo line2", on_output=chunks.append)
    assert r.ok
    assert "line1" in "".join(chunks) and "line2" in "".join(chunks)


def test_run_shell_output_head_tail_cap(ws):
    chunks = []
    # ~30KB of numbered output; stored result keeps head + tail only.
    r = do_run_shell(ws, "seq 1 4000", on_output=chunks.append)
    assert r.ok
    assert "bytes omitted" in r.output
    assert "1\n" in r.output          # head kept
    assert "4000" in r.output         # tail kept
    assert len(r.output) < 2 * SHELL_STREAM_KEEP + 1000
    assert "2500" in "".join(chunks)  # streaming saw the full output


def test_run_shell_stop_kills_the_process_group(ws):
    started = time.monotonic()

    def should_stop():
        return time.monotonic() - started > 0.3

    r = do_run_shell(ws, "sleep 30", timeout=60, should_stop=should_stop)
    assert time.monotonic() - started < 10  # killed, not the full timeout
    assert not r.ok and "stopped by user" in r.output


def test_run_shell_scrubs_secret_env(ws, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "tok-should-not-leak")
    r = do_run_shell(ws, "echo K=$OPENAI_API_KEY T=$CHATTABLE_AUTH_TOKEN")
    assert r.ok
    assert "sk-should-not-leak" not in r.output
    assert "tok-should-not-leak" not in r.output


# ------------------------------------------------------------------ update_plan

def test_update_plan_validates_and_renders():
    r = do_update_plan([
        {"content": "explore the repo", "status": "completed"},
        {"content": "fix the bug", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ])
    assert r.ok
    assert "[completed] explore the repo" in r.output
    assert "[in_progress] fix the bug" in r.output
    assert r.metadata["items"][1]["status"] == "in_progress"


def test_update_plan_rejects_bad_input():
    assert not do_update_plan([]).ok
    assert not do_update_plan([{"content": "x", "status": "done"}]).ok
    assert not do_update_plan([{"status": "pending"}]).ok


# ---------------------------------------------------------------------- preview

def test_preview_change_shows_diff_without_writing(ws):
    diff = preview_change(ws, "write_file", {"path": "p.txt", "content": "new\n"})
    assert "+new" in diff
    assert not os.path.exists(os.path.join(ws.root, "p.txt"))

    do_write_file(ws, "p.txt", "one two\n")
    diff = preview_change(ws, "edit_file",
                          {"path": "p.txt", "old_string": "two", "new_string": "2"})
    assert "-one two" in diff and "+one 2" in diff
    assert _read(ws, "p.txt") == "one two\n"  # untouched

    assert preview_change(ws, "run_shell", {"command": "rm -rf ."}) == ""


# --------------------------------------------------------------------- registry

def test_registry_marks_read_only(ws):
    registry = build_agent_registry(ws)
    read_only = {s.name: s.read_only for s in registry.list()}
    assert read_only == {
        "list_dir": True,
        "glob": True,
        "read_file": True,
        "grep": True,
        "write_file": False,
        "edit_file": False,
        "run_shell": False,
        "update_plan": True,
    }
    result = registry.run("write_file", {"path": "r.txt", "content": "via registry\n"})
    assert result.ok
    assert _read(ws, "r.txt") == "via registry\n"
