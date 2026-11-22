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

from workspace import ShadowGit, WorkspaceError, add_all_with_nested


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
    # A repo the agent cloned inside the worktree: store its files, not an
    # empty gitlink (follow-ups / forks would otherwise see an empty dir).
    add_all_with_nested(shadow.git_dir, wt, ["git", "-C", wt], keep_index=False)
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


# --------------------------------------------------------------------------- #
# Merge (3-way, sequential, in a throwaway worktree on the shadow repo)
# --------------------------------------------------------------------------- #

def merge_base(shadow: ShadowGit, commits: list[str]) -> Optional[str]:
    """Best common ancestor of the commits (None when unrelated)."""
    if not commits:
        return None
    base = commits[0]
    for c in commits[1:]:
        proc = _run(["git", f"--git-dir={shadow.git_dir}", "merge-base", base, c])
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        base = proc.stdout.strip()
    return base


def merge_into_worktree(shadow: ShadowGit, commits: list[str], wt: str) -> dict:
    """Apply commits[1:] on top of commits[0] in the worktree *wt*.

    Each input is merged with `git merge --no-ff`; conflicts are left in the
    files as conflict markers and committed as intermediate merge commits, so
    later inputs still apply cleanly on top. Returns {base, conflict_files}.
    """
    worktree_create(shadow, commits[0], wt)
    base = merge_base(shadow, commits)
    conflict_files: list[str] = []
    idcfg = ["-c", "user.name=chatable-orchestrator",
             "-c", "user.email=orchestrator@chatable.local"]
    for commit in commits[1:]:
        proc = _run(["git", "-C", wt] + idcfg +
                    ["merge", "--no-commit", "--no-ff", commit])
        if proc.returncode != 0:
            names = _run(["git", "-C", wt, "diff", "--name-only",
                          "--diff-filter=U"]).stdout.splitlines()
            conflict_files.extend(n for n in names if n.strip())
        _check(_run(["git", "-C", wt, "add", "-A"]), "git add")
        _check(_run(["git", "-C", wt] + idcfg +
                    ["commit", "--no-verify", "-m", f"merge {commit[:8]}"]),
               "git merge commit")
    seen: set[str] = set()
    unique = [f for f in conflict_files if not (f in seen or seen.add(f))]
    return {"base": base, "conflict_files": unique}


def commit_worktree_tree(shadow: ShadowGit, wt: str, parents: list[str],
                         message: str) -> str:
    """Commit the worktree's current index as a commit with explicit parents."""
    tree = _run(["git", "-C", wt, "write-tree"])
    _check(tree, "git write-tree")
    argv = ["git", f"--git-dir={shadow.git_dir}",
            "-c", "user.name=chatable-orchestrator",
            "-c", "user.email=orchestrator@chatable.local",
            "commit-tree", tree.stdout.strip(), "-m", message]
    for p in parents:
        argv += ["-p", p]
    proc = _run(argv)
    _check(proc, "git commit-tree")
    return proc.stdout.strip()
