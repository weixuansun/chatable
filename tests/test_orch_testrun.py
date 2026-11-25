"""Baseline-relative test results: output parsing (incl. SUBFAILED), baseline
caching, status rules, merge baseline."""
import os
import time

import pytest

from orch.testrun import compare_to_baseline, parse_test_output
from orch.service import OrchService
from orch.store import new_id

from test_orch import _orch, ChatOrchWait, git_required
from test_orch_merge import _make_child, ResolvingFake, _with_venv_on_path


# ---------------------------------------------------------------- parsing

def test_parse_pytest_summary_counts():
    out = ("...\n===== 2 failed, 770 passed, 4 skipped in 4.13s =====\n")
    parsed = parse_test_output(out)
    assert parsed["passed"] == 770
    assert parsed["failed"] == 2
    assert parsed["failed_ids"] == []


def test_parse_failed_and_subfailed_ids():
    out = (
        "FAILED tests/test_more.py::test_split_at_max - assert [1] == [2]\n"
        "FAILED tests/test_more.py::test_other - ValueError\n"
        "SUBFAILED(1 != 2) tests/test_more.py::test_split_at\n"
        "SUBFAILED(lengths differ) tests/test_more.py::test_chunked_even\n"
        "===== 4 failed, 100 passed in 1.0s =====\n"
    )
    parsed = parse_test_output(out)
    assert parsed["failed"] == 4
    assert parsed["failed_ids"] == [
        "tests/test_more.py::test_split_at_max",
        "tests/test_more.py::test_other",
        "tests/test_more.py::test_split_at",
        "tests/test_more.py::test_chunked_even",
    ]


def test_parse_errors_count_into_failed():
    out = "==== 1 passed, 2 errors in 0.5s ===="
    parsed = parse_test_output(out)
    assert parsed["failed"] == 2


def test_parse_fallback_unknown_runner():
    parsed = parse_test_output("$ make test\n[exit 2]\nmake: *** [test] Error 2\n")
    assert parsed["passed"] == 0 and parsed["failed"] == 0


def test_compare_to_baseline_counts_fallback():
    delta = compare_to_baseline({"failed": 3, "failed_ids": []},
                                {"failed": 1, "failed_ids": []})
    assert delta["new_failures"] == ["2 test(s)"]
    assert delta["fixed"] == []
    delta2 = compare_to_baseline({"failed": 1, "failed_ids": []},
                                 {"failed": 3, "failed_ids": []})
    assert delta2["new_failures"] == [] and delta2["fixed"] == ["2 test(s)"]


def test_compare_to_baseline_by_ids():
    base = {"failed_ids": ["t::a", "t::b"]}
    delta = compare_to_baseline({"failed_ids": ["t::a", "t::c"]}, base)
    assert delta["new_failures"] == ["t::c"]
    assert delta["fixed"] == ["t::b"]


# ------------------------------------------------------- baseline caching

def test_baseline_computed_once_per_snapshot(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    calls = {"n": 0}
    import orch.service as svc_mod
    orig = svc_mod.run_test_cmd

    def counting(wt, cmd, timeout):
        calls["n"] += 1
        return orig(wt, cmd, timeout)

    monkeypatch.setattr(svc_mod, "run_test_cmd", counting)
    r1 = svc.baseline_test_result(root["snapshot"], "true")
    r2 = svc.baseline_test_result(root["snapshot"], "true")
    assert r1 == r2
    assert calls["n"] == 1  # second call served from the cache


# --------------------------------------------------------- status rules

def _pytest_project():
    return {
        "conftest.py": "",
        "test_pre.py": ("def test_preexisting():\n"
                        "    assert False, 'pre-existing failure'\n"),
    }


@git_required
def test_preexisting_failures_do_not_fail_a_docs_run(tmp_path, monkeypatch):
    """A run that changes nothing relevant is ok even when its parent's
    tests are red — the failures are pre-existing."""
    _with_venv_on_path(monkeypatch)
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    parent = _make_child(svc, root, _pytest_project(), "add failing test")
    svc.store.set_meta(test_cmd="python -m pytest -q")

    docs = ResolvingFake({"NOTES.md": "docs only\n"})
    svc.executors["fake"] = docs
    rid = svc.submit_runs(parent["id"], [{"instruction": "docs only",
                                          "agents": ["fake"]}])[0]
    run = ChatOrchWait(svc).wait(rid)
    node = svc.store.nodes[run["node_id"]]
    assert node["status"] == "ok"
    fact = node["facts"]["test"]
    assert fact["ok"] is True
    assert fact["failed"] == 1
    assert fact["baseline"]["failed"] == 1
    assert fact["new_failures"] == []
    assert "NOTES.md" in {f["path"] for f in node["facts"]["files"]}


@git_required
def test_new_failure_fails_the_run(tmp_path, monkeypatch):
    _with_venv_on_path(monkeypatch)
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    parent = _make_child(svc, root, _pytest_project(), "add failing test")
    svc.store.set_meta(test_cmd="python -m pytest -q")

    breaker = ResolvingFake({
        "test_new.py": "def test_new():\n    assert False, 'new failure'\n",
    })
    svc.executors["fake"] = breaker
    rid = svc.submit_runs(parent["id"], [{"instruction": "add a new failing test",
                                          "agents": ["fake"]}])[0]
    run = ChatOrchWait(svc).wait(rid)
    node = svc.store.nodes[run["node_id"]]
    assert node["status"] == "failed"
    fact = node["facts"]["test"]
    assert fact["ok"] is False
    assert fact["new_failures"] == ["test_new.py::test_new"]
    assert fact["fixed"] == []


@git_required
def test_fixed_failures_are_reported(tmp_path, monkeypatch):
    _with_venv_on_path(monkeypatch)
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    parent = _make_child(svc, root, _pytest_project(), "add failing test")
    svc.store.set_meta(test_cmd="python -m pytest -q")

    fixer = ResolvingFake({
        "test_pre.py": "def test_preexisting():\n    assert True\n",
    })
    svc.executors["fake"] = fixer
    rid = svc.submit_runs(parent["id"], [{"instruction": "fix the bug",
                                          "agents": ["fake"]}])[0]
    run = ChatOrchWait(svc).wait(rid)
    node = svc.store.nodes[run["node_id"]]
    assert node["status"] == "ok"
    assert node["facts"]["test"]["fixed"] == ["test_pre.py::test_preexisting"]
    assert node["facts"]["test"]["new_failures"] == []


# -------------------------------------------------------------- merge baseline

@git_required
def test_merge_baseline_preexisting_failures_ok(tmp_path, monkeypatch):
    """Merge keeps a pre-existing failure from the base → still ok."""
    _with_venv_on_path(monkeypatch)
    # The bug lives at the ROOT snapshot (the demo scenario): siblings branch
    # off it, the merge base carries the failure → pre-existing, not new.
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "conftest.py").write_text("")
    (ws / "test_pre.py").write_text("def test_preexisting():\n    assert False\n")
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {"b.txt": "B\n"}, "write B")
    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=True)
    m = svc.store.nodes[res["node_id"]]
    assert m["status"] == "ok", m["facts"].get("test")
    fact = m["facts"]["test"]
    assert fact["ok"] is True and fact["failed"] == 1
    assert fact["baseline"]["failed"] == 1
    assert fact["new_failures"] == []
    assert m["conclusion"]["done"].startswith("Merged 2 branches cleanly")
    assert "write A" in m["conclusion"]["done"] and "write B" in m["conclusion"]["done"]
    assert m["facts"]["duration_s"] >= 0


@git_required
def test_merge_baseline_new_failure_is_tests_failed(tmp_path, monkeypatch):
    _with_venv_on_path(monkeypatch)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "conftest.py").write_text("")
    resolver = ResolvingFake({
        "test_new.py": "def test_new():\n    assert True\n",
    })
    svc, _tmp = _orch(tmp_path, monkeypatch, fake_exec=resolver)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"conftest.py": "", "a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {
        "test_new.py": "def test_new():\n    assert False, 'new failure'\n",
    }, "add new failing test")
    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=True)
    m = svc.store.nodes[res["node_id"]]
    assert m["status"] == "tests_failed"
    assert m["facts"]["test"]["new_failures"] == ["test_new.py::test_new"]
    assert res["run_id"]  # resolver dispatched
    run = ChatOrchWait(svc).wait(res["run_id"])
    m2 = svc.store.nodes[run["node_id"]]
    assert m2["status"] == "ok"


@git_required
def test_test_command_that_cannot_run_is_not_a_pass(tmp_path, monkeypatch):
    """A missing runner (exit != 0, nothing parsed) is reported as "not run",
    never as a vacuous pass, and doesn't fail the run either."""
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    parent = _make_child(svc, root, {"README.md": "x\n"}, "base")
    svc.store.set_meta(test_cmd="definitely-not-a-test-runner-xyz")
    svc.executors["fake"] = ResolvingFake({"NOTES.md": "docs\n"})
    rid = svc.submit_runs(parent["id"], [{"instruction": "docs",
                                          "agents": ["fake"]}])[0]
    run = ChatOrchWait(svc).wait(rid)
    node = svc.store.nodes[run["node_id"]]
    assert node["status"] == "ok"
    fact = node["facts"]["test"]
    assert fact["ran"] is False and fact["ok"] is None
    assert fact["new_failures"] == []
