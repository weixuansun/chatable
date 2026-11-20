"""Workspace path confinement and shadow-git snapshots."""
import os
import shutil

import pytest

from workspace import ShadowGit, Workspace, WorkspaceError

git_required = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "ws"))


# ---------------------------------------------------------------- confinement

def test_resolve_rejects_traversal(ws):
    for bad in ("../outside", "a/../../..", "..", "/etc/passwd", "/"):
        with pytest.raises(WorkspaceError):
            ws.resolve(bad)


def test_resolve_allows_inside_paths(ws):
    assert ws.resolve(".") == ws.root
    assert ws.resolve("") == ws.root
    inside = ws.resolve("sub/dir/file.txt")
    assert inside.startswith(ws.root + os.sep)
    # An absolute path that stays inside the workspace is fine.
    assert ws.resolve(inside) == inside


def test_resolve_rejects_symlink_escape(ws, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    os.symlink(str(secret), os.path.join(ws.root, "link.txt"))
    with pytest.raises(WorkspaceError):
        ws.resolve("link.txt")
    # A symlinked directory escaping the root is rejected even for new files.
    os.symlink(str(tmp_path), os.path.join(ws.root, "dirlink"))
    with pytest.raises(WorkspaceError):
        ws.resolve("dirlink/secret.txt")
    with pytest.raises(WorkspaceError):
        ws.resolve("dirlink/new-file.txt")


def test_resolve_allows_symlink_staying_inside(ws):
    real = os.path.join(ws.root, "real")
    os.mkdir(real)
    os.symlink(real, os.path.join(ws.root, "alias"))
    resolved = ws.resolve("alias/f.txt")
    assert resolved.startswith(ws.root + os.sep)
    assert resolved == os.path.join(real, "f.txt")


# ---------------------------------------------------------------- shadow git

@git_required
def test_snapshot_and_restore(ws):
    shadow = ShadowGit(ws, os.path.join(os.path.dirname(ws.root), "shadow.git"))
    open(os.path.join(ws.root, "a.txt"), "w").write("v1")
    c1 = shadow.snapshot("first")
    assert c1

    open(os.path.join(ws.root, "a.txt"), "w").write("v2")
    open(os.path.join(ws.root, "b.txt"), "w").write("new")
    c2 = shadow.snapshot("second")
    assert c2 and c2 != c1

    # A file that was never part of any snapshot must also go on restore.
    open(os.path.join(ws.root, "manual.txt"), "w").write("m")
    shadow.restore(c1)

    assert open(os.path.join(ws.root, "a.txt")).read() == "v1"
    assert not os.path.exists(os.path.join(ws.root, "b.txt"))
    assert not os.path.exists(os.path.join(ws.root, "manual.txt"))
    assert shadow.head() == c1


@git_required
def test_restore_deletes_files_added_after_snapshot_in_subdirs(ws):
    shadow = ShadowGit(ws, os.path.join(os.path.dirname(ws.root), "shadow.git"))
    c1 = shadow.snapshot("empty base", allow_empty=True)
    os.makedirs(os.path.join(ws.root, "pkg"))
    open(os.path.join(ws.root, "pkg", "m.py"), "w").write("x = 1\n")
    shadow.snapshot("add pkg")
    shadow.restore(c1)
    assert not os.path.exists(os.path.join(ws.root, "pkg", "m.py"))


@git_required
def test_workspace_git_dir_never_tracked_nor_deleted(ws):
    gitdir = os.path.join(ws.root, ".git")
    os.makedirs(gitdir)
    open(os.path.join(gitdir, "HEAD"), "w").write("ref: refs/heads/x\n")
    open(os.path.join(ws.root, "a.txt"), "w").write("v1")

    shadow = ShadowGit(ws, os.path.join(os.path.dirname(ws.root), "shadow.git"))
    c1 = shadow.snapshot("s1")
    tracked = shadow._git("ls-files").stdout.split()
    assert tracked == ["a.txt"]  # the workspace's own .git is excluded

    open(os.path.join(ws.root, "a.txt"), "w").write("v2")
    shadow.restore(c1)
    assert open(os.path.join(gitdir, "HEAD")).read() == "ref: refs/heads/x\n"
    assert open(os.path.join(ws.root, "a.txt")).read() == "v1"


@git_required
def test_diff_between_commits(ws):
    shadow = ShadowGit(ws, os.path.join(os.path.dirname(ws.root), "shadow.git"))
    open(os.path.join(ws.root, "a.txt"), "w").write("one\n")
    c1 = shadow.snapshot("one")
    open(os.path.join(ws.root, "a.txt"), "w").write("one\ntwo\n")
    c2 = shadow.snapshot("two")
    diff = shadow.diff(c2, c1)
    assert "+two" in diff
    root_diff = shadow.diff(c1)
    assert "+one" in root_diff


def test_no_git_degrades_gracefully(ws, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    shadow = ShadowGit(ws, os.path.join(os.path.dirname(ws.root), "shadow.git"))
    assert not shadow.available
    assert shadow.snapshot("s") is None
    assert shadow.head() is None
    with pytest.raises(WorkspaceError):
        shadow.restore("abc123")
    with pytest.raises(WorkspaceError):
        shadow.diff("abc123")
