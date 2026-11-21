"""Run worktrees: isolated directories on the shadow repo.

Each run checks out its parent node's snapshot into a linked git worktree
(``git --git-dir=<shadow> worktree add``), so the run never touches the real
workspace. At the end the worktree state is committed back into the shadow
repo (the run's own index/HEAD; the main workspace HEAD is unaffected) and
the worktree is removed. The shadow repo's info/exclude (`.git`,
`.chatable/`, caches) applies everywhere, so the handoff context file is
never snapshotted.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from workspace import ShadowGit, WorkspaceError


def _run(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _check(proc: subprocess.CompletedProcess, what: str) -> None:
    if proc.returncode != 0:
        raise WorkspaceError(f"{what} failed: {(proc.stderr or proc.stdout).strip()[:400]}")


def worktree_create(shadow: ShadowGit, commit: str, dest: str) -> None:
    """Check out *commit* into a fresh linked worktree at *dest*."""
    shadow._require_git()
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    proc = _run(["git", f"--git-dir={shadow.git_dir}", "worktree", "add",
                 "--detach", dest, commit])
    _check(proc, "git worktree add")


def worktree_snapshot(shadow: ShadowGit, wt: str, message: str) -> Optional[str]:
    """Commit the worktree state back into the shadow repo (or None if
    unchanged). Uses the worktree's own git link, so the workspace index
    and HEAD are untouched."""
    base = ["git", "-C", wt, "-c", "user.name=chatable-orchestrator",
            "-c", "user.email=orchestrator@chatable.local"]
    _check(_run(base + ["add", "-A"]), "git add")
    dirty = _run(base + ["status", "--porcelain"]).stdout.strip()
    if not dirty:
        return None
    _check(_run(base + ["commit", "-m", message, "--no-verify"]), "git commit")
    head = _run(base + ["rev-parse", "HEAD"])
    _check(head, "git rev-parse")
    return head.stdout.strip() or None


def facts_files(shadow: ShadowGit, start_commit: str, end_commit: str) -> list[dict]:
    """Changed files between two snapshots, spec-shaped: path/added/removed."""
    if start_commit == end_commit:
        return []
    return [{"path": f["path"], "added": f["additions"], "removed": f["deletions"]}
            for f in shadow.diff_numstat(start_commit, end_commit)]


def worktree_remove(shadow: ShadowGit, wt: str) -> None:
    """Remove a run worktree (best-effort; leftovers are deleted directly)."""
    _run(["git", f"--git-dir={shadow.git_dir}", "worktree", "remove", "--force", wt])
    shutil.rmtree(wt, ignore_errors=True)
