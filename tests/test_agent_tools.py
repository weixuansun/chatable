"""Agent workspace tools: list_dir / read_file / grep / write_file / edit_file / run_shell."""
import os

import pytest

from agent_tools import (
    READ_CAP_BYTES,
    build_agent_registry,
    do_edit_file,
    do_grep,
    do_list_dir,
    do_read_file,
    do_run_shell,
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
    do_write_file(ws, "a.txt", "foo x foo\n")
    r = do_edit_file(ws, "a.txt", "foo", "baz")
    assert not r.ok and "not unique" in r.output
    assert _read(ws, "a.txt") == "foo x foo\n"  # untouched


def test_edit_file_replace_all(ws):
    do_write_file(ws, "a.txt", "foo x foo\n")
    r = do_edit_file(ws, "a.txt", "foo", "baz", replace_all=True)
    assert r.ok
    assert _read(ws, "a.txt") == "baz x baz\n"


def test_edit_file_old_string_not_found(ws):
    do_write_file(ws, "a.txt", "content\n")
    r = do_edit_file(ws, "a.txt", "missing", "x")
    assert not r.ok and "not found" in r.output


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


# ------------------------------------------------------------------------ grep

def test_grep_finds_matches_with_glob(ws):
    do_write_file(ws, "a.py", "def hello():\n    pass\n")
    do_write_file(ws, "b.txt", "hello world\n")
    r = do_grep(ws, "hello")
    assert "a.py:1:" in r.output and "b.txt:1:" in r.output
    r = do_grep(ws, "hello", glob="*.py")
    assert "a.py:1:" in r.output and "b.txt" not in r.output


def test_grep_no_matches_and_bad_regex(ws):
    do_write_file(ws, "a.txt", "nothing here\n")
    assert "no matches" in do_grep(ws, "zzzz").output
    assert not do_grep(ws, "(unclosed").ok


def test_grep_skips_git_dirs(ws):
    os.makedirs(os.path.join(ws.root, ".git"))
    with open(os.path.join(ws.root, ".git", "config"), "w") as fh:
        fh.write("hello\n")
    assert "no matches" in do_grep(ws, "hello").output


# --------------------------------------------------------------------- list_dir

def test_list_dir_tree_and_git_skip(ws):
    do_write_file(ws, "pkg/mod.py", "x\n")
    os.makedirs(os.path.join(ws.root, ".git"))
    r = do_list_dir(ws, ".")
    assert r.ok
    assert "[dir] pkg" in r.output and "mod.py" in r.output
    assert ".git" not in r.output


def test_list_dir_not_a_directory(ws):
    do_write_file(ws, "a.txt", "x\n")
    assert not do_list_dir(ws, "a.txt").ok


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


def test_run_shell_scrubs_secret_env(ws, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "tok-should-not-leak")
    r = do_run_shell(ws, "echo K=$OPENAI_API_KEY T=$CHATTABLE_AUTH_TOKEN")
    assert r.ok
    assert "sk-should-not-leak" not in r.output
    assert "tok-should-not-leak" not in r.output


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
        "read_file": True,
        "grep": True,
        "write_file": False,
        "edit_file": False,
        "run_shell": False,
    }
    result = registry.run("write_file", {"path": "r.txt", "content": "via registry\n"})
    assert result.ok
    assert _read(ws, "r.txt") == "via registry\n"
