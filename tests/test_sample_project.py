"""The onboarding sample-project endpoint (POST /api/sample-project).

Covers copying the bundled examples/sample-project into the data dir,
idempotency (an existing copy is never overwritten), and the project
directory being switched to the copy.
"""
import json
import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k) for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION")}
os.environ["CHATTABLE_DATA_DIR"] = _tmp
os.environ["CHATTABLE_VISION"] = "0"  # kill switch: never probe the backend

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import server  # noqa: E402
from orch.service import OrchService  # noqa: E402
from service import ChatService  # noqa: E402
from store import TreeStore  # noqa: E402
from test_orch import FakeExecutor  # noqa: E402

for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

BASE = "http://localhost:8000"
ORIGIN = {"Origin": BASE}


@pytest.fixture
def web(tmp_path, monkeypatch):
    """The sample-project endpoint on a ChatService/OrchService in tmp_path."""
    monkeypatch.delenv("CHATTABLE_WORKSPACE", raising=False)
    monkeypatch.setattr(main, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    chat = ChatService(TreeStore(str(tmp_path / "data" / "chatable.jsonl")))
    orch = OrchService(chat, executors={"a": FakeExecutor()})
    monkeypatch.setattr(server, "service", chat)
    monkeypatch.setattr(server, "orch", orch)
    return TestClient(server.app, base_url=BASE), chat, orch


def test_sample_project_copies_and_switches_workspace(web):
    client, chat, orch = web
    r = client.post("/api/sample-project", headers=ORIGIN)
    assert r.status_code == 200, r.json()
    data = r.json()
    dest = os.path.join(chat.data_dir(), "sample-project")
    assert data["ok"] and data["created"] is True
    assert data["path"] == os.path.realpath(dest)
    # The bundled files landed, tests included.
    assert os.path.isfile(os.path.join(dest, "sample_project", "calc.py"))
    assert os.path.isfile(os.path.join(dest, "tests", "test_calc.py"))
    # The project directory was switched and persisted.
    cfg = data["config"]
    assert cfg["agent"]["workspace"] == os.path.realpath(dest)
    assert cfg["agent"]["workspace_source"] == "settings"
    assert chat.workspace().root == os.path.realpath(dest)
    with open(main.SETTINGS_PATH, encoding="utf-8") as fh:
        assert json.load(fh)["workspace_dir"] == os.path.realpath(dest)


def test_sample_project_idempotent_no_overwrite(web):
    client, chat, _orch = web
    r = client.post("/api/sample-project", headers=ORIGIN)
    assert r.status_code == 200 and r.json()["created"] is True
    # User edits inside the copy must survive a second call.
    marker = os.path.join(chat.data_dir(), "sample-project", "sample_project", "calc.py")
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write("\n# user edit\n")
    r = client.post("/api/sample-project", headers=ORIGIN)
    assert r.status_code == 200 and r.json()["created"] is False
    with open(marker, encoding="utf-8") as fh:
        assert fh.read().endswith("# user edit\n")


def test_sample_project_pinned_by_env(web, tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "envws"))
    client, chat, _orch = web
    r = client.post("/api/sample-project", headers=ORIGIN)
    assert r.status_code == 400 and "CHATTABLE_WORKSPACE" in r.json()["detail"]
    assert not os.path.exists(os.path.join(chat.data_dir(), "sample-project"))


def test_sample_project_busy_409(web, monkeypatch):
    client, _chat, orch = web
    monkeypatch.setattr(orch, "busy", lambda: True)
    r = client.post("/api/sample-project", headers=ORIGIN)
    assert r.status_code == 409


def test_core_routes_registered():
    """Guard: every API path the page fetches on load is still routed."""
    paths = {getattr(r, "path", None) for r in server.app.routes}
    for p in ("/api/tree", "/api/config", "/api/settings", "/api/sample-project"):
        assert p in paths, p
