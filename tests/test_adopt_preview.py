"""Adopt-preview endpoints: will_delete / local_edit flags and counts.

Covers both /api/agent/node/{id}/adopt-preview (chat run nodes) and
/api/orch/node/{id}/adopt-preview (orchestrator DAG nodes), plus the 404 /
agent-guard behaviour, and that previewing does not change adopt/undo-adopt.

Imports server with an isolated CHATTABLE_DATA_DIR (same pattern as
tests/test_agent_api.py — this module sorts before it alphabetically, so it
may be the one to import server first) so no real user data is touched.
"""
import os
import tempfile
import time

_tmp = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k)
              for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION",
                        "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
                        "CHATTABLE_WORKSPACE")}
os.environ["CHATTABLE_DATA_DIR"] = _tmp
os.environ["CHATTABLE_VISION"] = "0"
for _k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN", "CHATTABLE_WORKSPACE"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from orch.service import OrchService  # noqa: E402
from service import ChatService  # noqa: E402
from store import TreeStore  # noqa: E402

for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

from test_agent_loop import git_required  # noqa: E402
from test_orch import FakeExecutor  # noqa: E402

BASE = "http://localhost:8000"
ORIGIN = {"Origin": BASE}
AUTH = {"Authorization": "Bearer preview-test-token"}


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


class EditExecutor(FakeExecutor):
    """Modifies a.txt, deletes b.txt and adds d.txt inside the run worktree."""

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        _write(workdir, "a.txt", "node a\n")
        os.remove(os.path.join(workdir, "b.txt"))
        _write(workdir, "d.txt", "node d\n")
        yield {"type": "tool_call", "id": "c1", "name": "edit_files", "arguments": {}}
        yield {"type": "tool_result", "id": "c1", "output": "edited"}
        yield {"type": "result",
               "final_text": "Edited.\n\n## Conclusion\n### Done\nEdited files.",
               "exit_code": 0, "session_id": "edit-session", "usage": {}}


def _web(tmp_path, monkeypatch):
    """server.service/server.orch pointed at a tmp workspace holding a/b/c."""
    monkeypatch.setenv("CHATTABLE_AUTH_TOKEN", "preview-test-token")
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("CHATTABLE_ORCH_LIMITS", "fake=5")
    monkeypatch.setenv("CHATTABLE_RETAIN_FINISHED_TURN_S", "60")
    chat = ChatService(TreeStore(str(tmp_path / "chat.jsonl")))
    ws = chat.workspace().root
    os.makedirs(ws, exist_ok=True)
    for name in ("a.txt", "b.txt", "c.txt"):
        _write(ws, name, f"base {name}\n")
    orch = OrchService(chat, executors={"fake": EditExecutor()})
    chat.run_stop_hook = orch.stop_chat_run
    monkeypatch.setattr(server, "service", chat)
    monkeypatch.setattr(server, "orch", orch)
    return TestClient(server.app, base_url=BASE), chat, orch, ws


def _wait_chat_run(chat, aid, timeout=20):
    for _ in chat.subscribe(aid, 0):
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        md = chat.store.get(aid).metadata or {}
        if md.get("run_status") not in (None, "queued", "running"):
            return md
        time.sleep(0.05)
    raise AssertionError("chat run did not finish")


def _wait_orch_run(orch, run_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = orch.store.runs[run_id]
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} stuck: {run['status']}")


def _expect_preview(body, target_commit, base_commit):
    """The shared shape: a M (locally edited), b D (node deleted it),
    d A (node added it), e D (locally added) — and the matching counts."""
    assert body["ok"] is True
    assert body["target_commit"] == target_commit
    assert body["base_commit"] == base_commit
    assert body["workspace_commit"]
    files = {f["path"]: f for f in body["files"]}
    assert set(files) == {"a.txt", "b.txt", "d.txt", "e.txt"}
    for f in files.values():
        assert {"path", "status", "additions", "deletions", "binary",
                "will_delete", "local_edit"} <= set(f)
        assert f["binary"] is False
    a, b, d, e = files["a.txt"], files["b.txt"], files["d.txt"], files["e.txt"]
    # Node and the user both changed a.txt: local edit, kept (not deleted).
    assert a["status"] == "M" and a["local_edit"] and not a["will_delete"]
    # b.txt: deleted by the node, locally untouched → delete, no local edit.
    assert b["status"] == "D" and b["will_delete"] and not b["local_edit"]
    # d.txt: added by the node → plain add on adopt.
    assert d["status"] == "A" and not d["will_delete"] and not d["local_edit"]
    # e.txt: added locally, not in the snapshot → deleted AND a local edit.
    assert e["status"] == "D" and e["will_delete"] and e["local_edit"]
    assert body["counts"] == {"added": 1, "modified": 1, "deleted": 2,
                              "local_edits": 2}
    return files


@git_required
def test_chat_node_adopt_preview(tmp_path, monkeypatch):
    client, chat, orch, ws = _web(tmp_path, monkeypatch)
    (run,) = orch.start_chat_runs("edit files", None, ["fake"])
    aid = run["assistant_id"]
    md = _wait_chat_run(chat, aid)
    # Local edits in the real workspace after the run finished.
    _write(ws, "a.txt", "manual a\n")
    _write(ws, "e.txt", "manual e\n")

    r = client.get(f"/api/agent/node/{aid}/adopt-preview", headers=AUTH)
    assert r.status_code == 200, r.text
    _expect_preview(r.json(), md["ws_commit"], md["ws_parent_commit"])

    # Previewing changes nothing about adopt/undo-adopt itself.
    r = client.post("/api/agent/adopt", json={"node_id": aid},
                    headers={**ORIGIN, **AUTH})
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) == {"ok", "adopted_commit", "previous_commit", "files"}
    assert data["ok"] and data["adopted_commit"] == md["ws_commit"]
    assert _read(ws, "a.txt") == "node a\n"
    assert not os.path.exists(os.path.join(ws, "b.txt"))
    assert _read(ws, "c.txt") == "base c.txt\n"
    assert _read(ws, "d.txt") == "node d\n"
    assert not os.path.exists(os.path.join(ws, "e.txt"))

    r = client.post("/api/agent/undo-adopt",
                    json={"previous_commit": data["previous_commit"]},
                    headers={**ORIGIN, **AUTH})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "restored": data["previous_commit"]}
    assert _read(ws, "a.txt") == "manual a\n"
    assert _read(ws, "b.txt") == "base b.txt\n"
    assert not os.path.exists(os.path.join(ws, "d.txt"))
    assert _read(ws, "e.txt") == "manual e\n"


@git_required
def test_orch_node_adopt_preview(tmp_path, monkeypatch):
    client, chat, orch, ws = _web(tmp_path, monkeypatch)
    state = client.get("/api/orch/state", headers=AUTH).json()
    root_id = state["project"]["root"]
    root_snap = next(n["snapshot"] for n in state["nodes"] if n.get("root"))

    r = client.post("/api/orch/runs",
                    json={"from_node": root_id,
                          "items": [{"instruction": "edit files",
                                     "agents": ["fake"]}]},
                    headers={**ORIGIN, **AUTH})
    assert r.status_code == 200, r.text
    run = _wait_orch_run(orch, r.json()["runs"][0])
    assert run["status"] == "ok"
    node_id = run["node_id"]

    _write(ws, "a.txt", "manual a\n")
    _write(ws, "e.txt", "manual e\n")

    r = client.get(f"/api/orch/node/{node_id}/adopt-preview", headers=AUTH)
    assert r.status_code == 200, r.text
    _expect_preview(r.json(), orch.store.nodes[node_id]["snapshot"], root_snap)

    # The root node has no starting snapshot: base_commit is null and no
    # file can be flagged local_edit.
    r = client.get(f"/api/orch/node/{root_id}/adopt-preview", headers=AUTH)
    assert r.status_code == 200, r.text
    root_body = r.json()
    assert root_body["ok"] and root_body["base_commit"] is None
    assert root_body["files"] and all(not f["local_edit"]
                                      for f in root_body["files"])

    # Previewing changes nothing about adopt/undo-adopt itself.
    r = client.post("/api/orch/adopt", json={"node_id": node_id},
                    headers={**ORIGIN, **AUTH})
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) == {"ok", "adopted_commit", "previous_commit", "files"}
    assert data["ok"]
    assert _read(ws, "a.txt") == "node a\n"
    assert not os.path.exists(os.path.join(ws, "b.txt"))
    assert _read(ws, "d.txt") == "node d\n"
    assert not os.path.exists(os.path.join(ws, "e.txt"))

    r = client.post("/api/orch/undo-adopt",
                    json={"previous_commit": data["previous_commit"]},
                    headers={**ORIGIN, **AUTH})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "restored": data["previous_commit"]}
    assert _read(ws, "a.txt") == "manual a\n"
    assert _read(ws, "b.txt") == "base b.txt\n"
    assert _read(ws, "e.txt") == "manual e\n"


@git_required
def test_preview_unknown_node_404(tmp_path, monkeypatch):
    client, _chat, _orch, _ws = _web(tmp_path, monkeypatch)
    r = client.get("/api/agent/node/nope/adopt-preview", headers=AUTH)
    assert r.status_code == 404
    r = client.get("/api/orch/node/nope/adopt-preview", headers=AUTH)
    assert r.status_code == 404


def test_preview_guarded_when_agent_unavailable(monkeypatch):
    """Exposed beyond loopback without CHATTABLE_AUTH_TOKEN, the adopt
    previews refuse like the other agent endpoints."""
    monkeypatch.delenv("CHATTABLE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CHATABLE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CHATTABLE_BIND_HOST", "0.0.0.0")
    client = TestClient(server.app, base_url=BASE)
    for url in ("/api/agent/node/x/adopt-preview",
                "/api/orch/node/x/adopt-preview"):
        r = client.get(url, headers=ORIGIN)
        assert r.status_code == 403
        assert "CHATTABLE_AUTH_TOKEN" in r.json()["detail"]


class RenameExecutor(FakeExecutor):
    """Renames c.txt (content unchanged) — git rename detection must not hide
    that the old path will be deleted."""

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        os.rename(os.path.join(workdir, "c.txt"), os.path.join(workdir, "renamed.txt"))
        yield {"type": "result", "final_text": "Renamed.", "exit_code": 0,
               "session_id": None, "usage": {}}


def test_preview_flags_renamed_away_file_as_deleted(tmp_path, monkeypatch):
    client, chat, orch, ws = _web(tmp_path, monkeypatch)
    orch.executors["fake"] = RenameExecutor()
    root_id = client.get("/api/orch/state", headers=AUTH).json()["project"]["root"]
    r = client.post("/api/orch/runs",
                    json={"from_node": root_id,
                          "items": [{"instruction": "rename", "agents": ["fake"]}]},
                    headers={**ORIGIN, **AUTH})
    run = _wait_orch_run(orch, r.json()["runs"][0])
    assert run["status"] == "ok"
    body = client.get(f"/api/orch/node/{run['node_id']}/adopt-preview",
                      headers=AUTH).json()
    by_path = {f["path"]: f for f in body["files"]}
    assert set(by_path) == {"c.txt", "renamed.txt"}, body["files"]
    assert by_path["c.txt"]["will_delete"] is True
    assert by_path["renamed.txt"]["will_delete"] is False
    assert body["counts"]["deleted"] == 1 and body["counts"]["added"] == 1
