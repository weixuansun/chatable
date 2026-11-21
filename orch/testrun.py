"""Run + parse the project test command for orchestrator runs.

Results are baseline-relative: a run/merge is only "red" when it introduces
NEW failures vs its parent's (or merge base's) test result — pre-existing
failures on a sibling branch don't turn a docs-only change red.
"""
from __future__ import annotations

import re

from agent_tools import do_run_shell
from workspace import Workspace

_FAILED_LINE_RE = re.compile(r"^FAILED\s+(\S+)")
_SUBFAILED_AFTER_RE = re.compile(r"^SUBFAILED\([^)]*\)\s+(\S+::\S+)")
_SUBFAILED_INSIDE_RE = re.compile(r"^SUBFAILED\((\S+::\S+)\)")
_SUMMARY_LINES = 15
_FAILED_IDS_CAP = 20


def parse_test_output(output: str) -> dict:
    """Parse a test runner's output into counts + failing ids.

    pytest style: "2 failed, 770 passed, 3 skipped in 4.1s" with
    "FAILED path::test - reason" and "SUBFAILED(...) path::test" lines.
    Other runners: counts stay 0; ok/not-ok comes from the exit code.
    """
    failed_ids: list[str] = []
    for line in output.splitlines():
        m = _FAILED_LINE_RE.match(line)
        if m:
            failed_ids.append(m.group(1))
            continue
        m = _SUBFAILED_AFTER_RE.match(line) or _SUBFAILED_INSIDE_RE.match(line)
        if m:
            failed_ids.append(m.group(1))

    def last_count(pattern: str) -> int:
        matches = re.findall(pattern, output)
        return int(matches[-1]) if matches else 0

    passed = last_count(r"(\d+)\s+passed")
    failed = last_count(r"(\d+)\s+failed")
    errors = last_count(r"(\d+)\s+errors?")
    return {
        "passed": passed,
        "failed": failed + errors,
        "failed_ids": failed_ids[:_FAILED_IDS_CAP],
    }


def summarize(output: str, lines: int = _SUMMARY_LINES) -> str:
    tail = output.strip().splitlines()[-lines:]
    return "\n".join(tail)


def run_test_cmd(wt: str, cmd: str, timeout: int) -> dict:
    """Run the test command in a worktree and parse the result."""
    res = do_run_shell(Workspace(wt), cmd, timeout=timeout)
    parsed = parse_test_output(res.output or "")
    return {
        "cmd": cmd,
        "exit_ok": res.ok,
        **parsed,
        "summary": summarize(res.output or ""),
    }


def compare_to_baseline(result: dict, baseline: dict) -> dict:
    """Diff a test result against the baseline's: new failures and fixed.

    When both sides have parseable failing ids, compare them; otherwise fall
    back to the failure counts.
    """
    base_ids = set(baseline.get("failed_ids") or [])
    new_ids = set(result.get("failed_ids") or [])
    if base_ids or new_ids:
        return {
            "new_failures": sorted(i for i in new_ids - base_ids)[:_FAILED_IDS_CAP],
            "fixed": sorted(i for i in base_ids - new_ids)[:_FAILED_IDS_CAP],
        }
    extra = result.get("failed", 0) - baseline.get("failed", 0)
    return {
        "new_failures": ([f"{extra} test(s)"] if extra > 0 else []),
        "fixed": ([f"{-extra} test(s)"] if extra < 0 else []),
    }
