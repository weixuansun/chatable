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


# Directories never searched for nested repositories (also excluded from
# snapshots by the shadow repo's info/exclude).
_NESTED_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
                     ".pytest_cache", ".mypy_cache", ".chatable"}


def find_nested_repos(root: str, limit: int = 200) -> list[str]:
    """Relative paths of git repositories nested inside *root* (a directory
    holding its own ``.git`` dir or file), outermost only."""
    out: list[str] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, _files in os.walk(root):
        if dirpath != root and os.path.lexists(os.path.join(dirpath, ".git")):
            out.append(os.path.relpath(dirpath, root).replace(os.sep, "/"))
            dirnames[:] = []
            if len(out) >= limit:
                break
            continue
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _NESTED_SKIP_DIRS
                             and not os.path.islink(os.path.join(dirpath, d)))
    return out


def _nested_tree(git_dir: str, repo_dir: str, depth: int = 0, keep_index: bool = True) -> str:
    """Tree object (in *git_dir*) holding the work-tree files of the nested
    repository at *repo_dir* — its own .gitignore is honoured, its ``.git``
    is not stored. Nested-in-nested repos are expanded recursively."""
    import hashlib
    idx_dir = os.path.join(git_dir, "chatable-nested-index")
    os.makedirs(idx_dir, exist_ok=True)
    key = hashlib.sha1(os.path.abspath(repo_dir).encode()).hexdigest()[:16]
    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(idx_dir, key))
    base = ["git", f"--git-dir={git_dir}", f"--work-tree={repo_dir}",
            "-c", "core.autocrlf=false"]

    def run(*args: str) -> subprocess.CompletedProcess:
        proc = subprocess.run([*base, *args], capture_output=True, text=True,
                              timeout=300, env=env)
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git {args[0]} (nested {repo_dir}) failed: "
                f"{(proc.stderr or proc.stdout).strip()[:300]}")
        return proc

    inner = find_nested_repos(repo_dir) if depth < 4 else []
    run("add", "-A", "--", ".", *[f":(exclude){r}" for r in inner])
    for sub in inner:
        tree = _nested_tree(git_dir, os.path.join(repo_dir, sub), depth + 1, keep_index)
        run("rm", "-r", "--cached", "-q", "--ignore-unmatch", "--", sub)
        run("read-tree", f"--prefix={sub}/", tree)
    tree = run("write-tree").stdout.strip()
    if not keep_index:
        try:
            os.remove(env["GIT_INDEX_FILE"])
        except OSError:
            pass
    return tree


def add_all_with_nested(git_dir: str, work_tree: str, git_argv: list[str],
                        keep_index: bool = True) -> list[str]:
    """``git add -A`` for a snapshot, with nested git repositories (e.g. a
    ``git clone`` / ``git init`` the agent ran inside the workspace) stored as
    their actual files instead of empty gitlinks — so snapshots, and the run
    worktrees / forks checked out from them, contain the code. A plain
    ``git add -A`` records a gitlink (or fails outright for a repo without a
    commit). *git_argv* runs git against the index being built (workspace or
    run worktree). Returns the nested paths that were expanded."""
    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run([*git_argv, *args], capture_output=True, text=True,
                              timeout=300)
        if check and proc.returncode != 0:
            raise WorkspaceError(
                f"git {args[0]} failed: {(proc.stderr or proc.stdout).strip()[:400]}")
        return proc

    nested = find_nested_repos(work_tree)
    run("add", "-A", "--", ".", *[f":(exclude){rel}" for rel in nested])
    expanded = []
    for rel in nested:
        if run("check-ignore", "-q", "--", rel, check=False).returncode == 0:
            continue  # ignored on purpose by the user's .gitignore
        tree = _nested_tree(git_dir, os.path.join(work_tree, rel), keep_index=keep_index)
        run("rm", "-r", "--cached", "-q", "--ignore-unmatch", "--", rel)
        run("read-tree", f"--prefix={rel}/", tree)
        expanded.append(rel)
    return expanded


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
            # Never track (or delete on clean) any .git inside the workspace,
            # build/cache noise, or orchestrator handoff files (.chatable/).
            info = os.path.join(self.git_dir, "info")
            os.makedirs(info, exist_ok=True)
            exclude = os.path.join(info, "exclude")
            wanted = (".git", "__pycache__/", "*.pyc", ".pytest_cache/",
                      ".mypy_cache/", "node_modules/", ".venv/", "venv/", ".chatable/")
            existing = ""
            if os.path.isfile(exclude):
                with open(exclude, encoding="utf-8") as fh:
                    existing = fh.read()
            with open(exclude, "a", encoding="utf-8") as fh:
                for line in wanted:
                    if line not in existing.splitlines():
                        fh.write(line + "\n")
        self._initialized = True

    # ------------------------------------------------------------------ #

    def _add_all(self) -> None:
        """``git add -A`` plus nested repositories as plain files (no empty
        gitlinks). Caller must hold ``_lock``."""
        add_all_with_nested(self.git_dir, self.workspace.root, [
            "git", f"--git-dir={self.git_dir}", f"--work-tree={self.workspace.root}",
            "-c", "core.autocrlf=false"])

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
            self._add_all()
            dirty = self._git("status", "--porcelain").stdout.strip()
            if not dirty and not allow_empty:
                return self._head()
            args = ["commit", "-m", message, "--no-verify"]
            if allow_empty and not dirty:
                args.append("--allow-empty")
            self._git(*args)
            return self._head()

    def stage_changes(self, limit: int = 50) -> list[dict]:
        """Files changed in the work tree since the last stage/snapshot, then
        stage them (no commit). Called around a shell step so the step can
        list the files it touched without a per-step snapshot commit."""
        if not self.available:
            return []
        with self._lock:
            self._ensure_init()
            if self._head() is None:
                # No baseline yet: everything would look "touched".
                self._add_all()
                return []
            proc = self._git("status", "--porcelain", "--untracked-files=all",
                             check=False)
            out: list[dict] = []
            for line in proc.stdout.splitlines():
                if len(line) < 4:
                    continue
                y, path = line[1], line[3:]
                if y == " ":
                    continue  # already staged → not changed by this step
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                out.append({"status": {"?": "A"}.get(y, y), "path": path.strip('"')})
            self._add_all()
            return out[:limit]

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

    def diff_numstat(self, a: str, b: str) -> list[dict]:
        """Per-file change summary between two commits (a → b)."""
        self._require_git()
        with self._lock:
            self._ensure_init()
            stats = self._git("diff", "--numstat", a, b).stdout
            status = self._git("diff", "--name-status", a, b).stdout
        status_map: dict[str, str] = {}
        for line in status.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status_map[parts[-1]] = parts[0]
        files = []
        for line in stats.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            add, delete, path = parts[0], parts[1], parts[2]
            binary = not (add.isdigit() and delete.isdigit())
            files.append({
                "path": path,
                "status": status_map.get(path, "M"),
                "additions": int(add) if add.isdigit() else 0,
                "deletions": int(delete) if delete.isdigit() else 0,
                # numstat prints "-\t-" for binary files; there is no text diff.
                "binary": binary,
            })
        return files

    def ls_tree(self, commit: str) -> list[str]:
        """All tracked file paths at *commit*."""
        self._require_git()
        with self._lock:
            self._ensure_init()
            out = self._git("ls-tree", "-r", "--name-only", commit).stdout
        return [line for line in out.splitlines() if line.strip()]

    def show_file(self, commit: str, path: str, cap: int = 200 * 1024) -> tuple[str, bool]:
        """(content, truncated) of one file at *commit*, capped at *cap* bytes."""
        self._require_git()
        if path.startswith("/") or ".." in path.split("/") or ":" in path:
            raise WorkspaceError(f"bad snapshot path: {path!r}")
        with self._lock:
            self._ensure_init()
            proc = subprocess.run(
                ["git", f"--git-dir={self.git_dir}", f"--work-tree={self.workspace.root}",
                 "show", f"{commit}:{path}"],
                capture_output=True, timeout=60,
            )
        if proc.returncode != 0:
            raise WorkspaceError(f"not in snapshot {commit[:8]}: {path}")
        data = proc.stdout
        truncated = len(data) > cap
        if truncated:
            data = data[:cap]
        return data.decode("utf-8", errors="replace"), truncated
