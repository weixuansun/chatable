"""Agent workspace: path confinement and shadow-git snapshots.

The workspace is the only directory the agent-mode file tools may touch.
``Workspace.resolve`` confines every path to the workspace root: ``..``
traversal, absolute paths outside the root and symlink escapes are all
rejected (the fully resolved real path must stay under the root realpath).

``ShadowGit`` keeps a git repository OUTSIDE the workspace
(``git --git-dir=<shadow> --work-tree=<workspace>``) so each mutating agent
step can snapshot the workspace without ever touching a ``.git`` directory
inside it.  ``.git`` entries are excluded from tracking, so restores never
delete or overwrite a workspace's own repository.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from typing import Optional


class WorkspaceError(ValueError):
    """Raised when a path escapes the workspace or git is unavailable."""


def default_workspace_path(data_dir: str) -> str:
    return os.path.join(data_dir, "workspace")


def default_shadow_git_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "agent_shadow.git")


class Workspace:
    """A directory the agent tools are confined to."""

    def __init__(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.root = os.path.realpath(path)

    def resolve(self, path: str = "") -> str:
        """Resolve *path* to an absolute real path inside the workspace.

        Relative paths are taken from the workspace root; absolute paths must
        already point inside it.  Symlinks are fully resolved, so a link
        (anywhere in the chain) pointing outside the root is rejected.
        """
        raw = (path or ".").strip() or "."
        if os.path.isabs(raw):
            cand = os.path.realpath(raw)
        else:
            cand = os.path.realpath(os.path.join(self.root, raw))
        if cand != self.root and not cand.startswith(self.root + os.sep):
            raise WorkspaceError(f"path escapes the workspace: {path!r}")
        return cand

    def rel(self, abspath: str) -> str:
        """Workspace-relative display path for an absolute confined path."""
        return os.path.relpath(abspath, self.root)


class ShadowGit:
    """Snapshots of a workspace in a git dir kept outside of it.

    All git invocations use ``--git-dir``/``--work-tree`` and never
    ``shell=True``.  When git is not installed every operation degrades to a
    no-op (snapshots return None) except ``restore``/``diff``, which raise
    ``WorkspaceError`` so callers can surface a clear message.
    """

    def __init__(self, workspace: Workspace, git_dir: str) -> None:
        self.workspace = workspace
        self.git_dir = os.path.abspath(git_dir)
        self.available = shutil.which("git") is not None
        self._lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------ #

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            [
                "git",
                f"--git-dir={self.git_dir}",
                f"--work-tree={self.workspace.root}",
                "-c", "user.name=chatable-agent",
                "-c", "user.email=agent@chatable.local",
                "-c", "core.autocrlf=false",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check and proc.returncode != 0:
            raise WorkspaceError(
                f"git {args[0]} failed: {(proc.stderr or proc.stdout).strip()[:400]}"
            )
        return proc

    def _require_git(self) -> None:
        if not self.available:
            raise WorkspaceError("git is not installed; workspace snapshots are unavailable")

    def _ensure_init(self) -> None:
        self._require_git()
        if self._initialized and os.path.isdir(self.git_dir):
            return
        if not os.path.isdir(self.git_dir):
            self._git("init")
            # Never track (or delete on clean) any .git inside the workspace.
            info = os.path.join(self.git_dir, "info")
            os.makedirs(info, exist_ok=True)
            with open(os.path.join(info, "exclude"), "a", encoding="utf-8") as fh:
                fh.write(".git\n")
        self._initialized = True

    # ------------------------------------------------------------------ #

    def _head(self) -> Optional[str]:
        """HEAD commit id, or None. Caller must hold ``_lock``."""
        proc = self._git("rev-parse", "HEAD", check=False)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip() or None

    def head(self) -> Optional[str]:
        if not self.available or not os.path.isdir(self.git_dir):
            return None
        with self._lock:
            return self._head()

    def snapshot(self, message: str, allow_empty: bool = False) -> Optional[str]:
        """Commit the current workspace state; return the commit id (or None)."""
        if not self.available:
            return None
        with self._lock:
            self._ensure_init()
            self._git("add", "-A")
            dirty = self._git("status", "--porcelain").stdout.strip()
            if not dirty and not allow_empty:
                return self._head()
            args = ["commit", "-m", message, "--no-verify"]
            if allow_empty and not dirty:
                args.append("--allow-empty")
            self._git(*args)
            return self._head()

    def restore(self, commit: str) -> None:
        """Reset the workspace to *commit*.

        Files absent from the snapshot are deleted; anything named ``.git``
        inside the workspace is left untouched (it is excluded from the
        shadow repo, so ``git clean`` ignores it).
        """
        self._require_git()
        with self._lock:
            self._ensure_init()
            proc = self._git("cat-file", "-e", f"{commit}^{{commit}}", check=False)
            if proc.returncode != 0:
                raise WorkspaceError(f"unknown snapshot commit: {commit}")
            # Index+work-tree := commit (deletes tracked files not in it),
            # move HEAD, then drop untracked leftovers (ignored files kept).
            self._git("read-tree", "--reset", "-u", commit)
            self._git("update-ref", "HEAD", commit)
            self._git("clean", "-fd")

    def diff(self, commit: str, parent: Optional[str] = None) -> str:
        """Unified diff for *commit* against *parent* (or its first parent)."""
        self._require_git()
        with self._lock:
            self._ensure_init()
            if parent:
                return self._git("diff", parent, commit).stdout
            proc = self._git("diff-tree", "--root", "-p", commit)
            return proc.stdout
