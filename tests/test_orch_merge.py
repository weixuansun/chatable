"""Orchestrator merge (DAG): clean/conflict/red-test merges, resolver dispatch,
merge-aware context, project notes, localized conclusion headings."""
import os
import time

import pytest

from orch.context import parse_conclusion
from orch.service import OrchService
from orch.store import new_id
from orch.worktree import worktree_create, worktree_remove, worktree_snapshot

from service import ChatService
from store import TreeStore

from test_orch import FakeExecutor, _orch, ChatOrchWait, git_required  # noqa: F401

_VENV_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".venv", "bin")


def _with_venv_on_path(monkeypatch):
    """pytest must be on PATH for the post-merge test runs (as in production)."""
    monkeypatch.setenv("PATH", _VENV_BIN + os.pathsep + os.environ.get("PATH", ""))


def _make_child(svc, parent, files, instruction, conclusion=None):
    """A node with real content, created directly via a throwaway worktree."""
    wt = os.path.join(svc.runs_dir, f"m-{new_id()}", "wt")
    worktree_create(svc.shadow, parent["snapshot"], wt)
    for path, content in files.items():
        full = os.path.join(wt, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(content)
    commit = worktree_snapshot(svc.shadow, wt, instruction) or parent["snapshot"]
    worktree_remove(svc.shadow, wt)
    node = {"id": new_id(), "snapshot": commit, "instruction": instruction,
            "parents": [parent["id"]], "agent": "fake", "model": None,
            "status": "ok", "adopted": False,
            "conclusion": conclusion or {"done": instruction, "decisions": "",
                                         "not_done": "", "next": ""},
            "facts": {"files": []}, "session_id": None, "events": [],
            "created_at": time.time()}
    svc.store.add_node(node)
    return node


class ResolvingFake:
    """Resolver that reads CONTEXT.md and writes the resolution."""

    def __init__(self, files):
        self.files = files
        self.contexts = []

    def available(self):
        return (True, "")

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        ctx = os.path.join(workdir, ".chatable", "CONTEXT.md")
        self.contexts.append(open(ctx).read() if os.path.exists(ctx) else "")
        for path, content in self.files.items():
            full = os.path.join(workdir, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as fh:
                fh.write(content)
        yield {"type": "result",
               "final_text": "resolved\n\n## Conclusion\n### Done\nResolved the merge.",
               "exit_code": 0, "session_id": None, "usage": {}}


def _read_at(svc, commit, path):
    return svc.shadow.show_file(commit, path)[0]


# ------------------------------------------------------------------ clean merge

@git_required
def test_clean_merge_of_three_siblings(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {"b.txt": "B\n"}, "write B")
    c = _make_child(svc, root, {"c.txt": "C\n"}, "write C")

    res = svc.merge([a["id"], b["id"], c["id"]], agent="fake", run_tests=False)
    assert res["status"] == "ok" and res["run_id"] is None
    m = svc.store.nodes[res["node_id"]]
    assert m["parents"] == [a["id"], b["id"], c["id"]]
    assert m["agent"] == "merge"
    assert _read_at(svc, m["snapshot"], "a.txt") == "A\n"
    assert _read_at(svc, m["snapshot"], "b.txt") == "B\n"
    assert _read_at(svc, m["snapshot"], "c.txt") == "C\n"
    assert {f["path"] for f in m["facts"]["files"]} == {"a.txt", "b.txt", "c.txt"}


@git_required
def test_clean_merge_runs_project_tests(tmp_path, monkeypatch):
    _with_venv_on_path(monkeypatch)
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    a = _make_child(svc, root, {
        "conftest.py": "",
        "test_ok.py": "def test_ok():\n    assert True\n",
    }, "add green tests")
    b = _make_child(svc, root, {"b.txt": "B\n"}, "write B")

    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=True)
    assert res["status"] == "ok"
    m = svc.store.nodes[res["node_id"]]
    assert m["facts"]["test"]["ok"] is True
    assert "pytest" in m["facts"]["test"]["cmd"]


# -------------------------------------------------------------- conflict merge

@git_required
def test_conflict_merge_dispatches_resolver(tmp_path, monkeypatch):
    resolver = ResolvingFake({"a.txt": "resolved\n"})
    svc, _tmp = _orch(tmp_path, monkeypatch, fake_exec=resolver)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {"a.txt": "B\n"}, "write B")

    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=False)
    assert res["status"] == "conflict"
    m = svc.store.nodes[res["node_id"]]
    assert m["facts"]["conflict_files"] == ["a.txt"]
    assert m["parents"] == [a["id"], b["id"]]

    # The resolver run was dispatched with the merge context.
    assert res["run_id"]
    run = ChatOrchWait(svc).wait(res["run_id"])
    assert run["status"] == "ok"
    ctx = resolver.contexts[0]
    assert "## Merge task" in ctx
    assert "write A" in ctx and "write B" in ctx
    assert "a.txt" in ctx
    m2 = svc.store.nodes[run["node_id"]]
    assert m2["status"] == "ok"
    assert m2["parents"] == [m["id"]]
    assert _read_at(svc, m2["snapshot"], "a.txt") == "resolved\n"
    assert m2["conclusion"]["done"] == "Resolved the merge."


@git_required
def test_red_tests_after_merge_dispatch_resolver(tmp_path, monkeypatch):
    _with_venv_on_path(monkeypatch)
    resolver = ResolvingFake({"test_fail.py": "def test_ok():\n    assert True\n"})
    svc, _tmp = _orch(tmp_path, monkeypatch, fake_exec=resolver)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"b.txt": "B\n"}, "write B")
    b = _make_child(svc, root, {
        "conftest.py": "",
        "test_fail.py": "def test_fail():\n    assert False\n",
    }, "add failing test")

    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=True)
    assert res["status"] == "tests_failed"
    assert res["run_id"]
    run = ChatOrchWait(svc).wait(res["run_id"])
    m2 = svc.store.nodes[run["node_id"]]
    assert m2["status"] == "ok"
    ctx = resolver.contexts[0]
    assert "Failing test output" in ctx
    assert "write B" in ctx and "add failing test" in ctx


# --------------------------------------------------------- merge-aware context

@git_required
def test_context_below_merge_includes_all_parents(tmp_path, monkeypatch):
    resolver = ResolvingFake({"x.txt": "x\n"})
    svc, _tmp = _orch(tmp_path, monkeypatch, fake_exec=resolver)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {"b.txt": "B\n"}, "write B")
    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=False)
    rid = svc.submit_runs(res["node_id"], [{"instruction": "next step",
                                            "agents": ["fake"]}])[0]
    run = ChatOrchWait(svc).wait(rid)
    assert run["status"] == "ok"
    ctx = resolver.contexts[0]
    # Path layer below the merge walks BOTH parents.
    assert "write A" in ctx and "write B" in ctx
    assert "Merge #" in ctx
    assert "next step" in ctx


def test_ancestors_of_dedupes_and_orders(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    root = svc.store.root_node()
    a = _make_child(svc, root, {"a.txt": "A\n"}, "write A")
    b = _make_child(svc, root, {"b.txt": "B\n"}, "write B")
    res = svc.merge([a["id"], b["id"]], agent="fake", run_tests=False)
    m_id = res["node_id"]
    anc = svc.store.ancestors_of(m_id)
    ids = [n["id"] for n in anc]
    assert set(ids) == {root["id"], a["id"], b["id"], m_id}
    assert len(ids) == len(set(ids))  # no duplicates
    assert ids == sorted(ids, key=lambda i: svc.store.nodes[i].get("created_at", 0))


# ------------------------------------------------------- conclusion robustness

def test_parse_conclusion_localized_and_colon_headings():
    zh = parse_conclusion("做了些事\n\n## 结论\n### Done\n改了 a.py\n### Next steps\n跑测试")
    assert zh["done"] == "改了 a.py"
    assert zh["next"] == "跑测试"
    colon = parse_conclusion("work\n\nConclusion:\nDone: fixed it")
    assert colon["done"] == "fixed it"


# -------------------------------------------------------------- project notes

def test_project_notes_override(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    info = svc.update_project("Always keep the public API stable.",
                              "pytest -q tests/unit")
    assert info["notes"].startswith("Always keep")
    assert info["test_cmd"] == "pytest -q tests/unit"
    # Reloaded from disk.
    svc2 = OrchService.__new__(OrchService)
    from orch.store import OrchStore
    store = OrchStore(svc.store.path)
    assert store.meta["notes"].startswith("Always keep")
    assert store.meta["test_cmd"] == "pytest -q tests/unit"


def test_project_notes_used_in_context(tmp_path, monkeypatch):
    svc, _tmp = _orch(tmp_path, monkeypatch)
    svc.update_project("Respect the freeze.", None)
    root = svc.store.root_node()
    from orch.context import build_context
    content = build_context(svc.store, root, "do x", None, str(tmp_path))
    assert "Project notes" in content and "Respect the freeze." in content
