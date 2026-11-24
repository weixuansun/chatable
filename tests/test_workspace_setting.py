"""Settings-driven agent project directory (workspace_dir).

Covers the CHATTABLE_WORKSPACE > settings > default priority, the settings
endpoint's validation, the busy 409, and the orchestrator rebinding onto the
new project after a switch.
"""
import hashlib
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
    """The settings endpoint running on a ChatService/OrchService in tmp_path."""
    monkeypatch.delenv("CHATTABLE_WORKSPACE", raising=False)
    monkeypatch.setattr(main, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    chat = ChatService(TreeStore(str(tmp_path / "data" / "chatable.jsonl")))
    orch = OrchService(chat, executors={"a": FakeExecutor()})
    monkeypatch.setattr(server, "service", chat)
    monkeypatch.setattr(server, "orch", orch)
    return TestClient(server.app, base_url=BASE), chat, orch


def _chat(tmp_path, monkeypatch):
    monkeypatch.delenv("CHATTABLE_WORKSPACE", raising=False)
    monkeypatch.setattr(main, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return ChatService(TreeStore(str(tmp_path / "data" / "chatable.jsonl")))


def test_workspace_priority_default_settings_env(tmp_path, monkeypatch):
    chat = _chat(tmp_path, monkeypatch)
    default_root = os.path.realpath(os.path.join(chat.data_dir(), "workspace"))
    assert chat.workspace().root == default_root
    assert chat.workspace_source()[1] == "default"

    proj = tmp_path / "proj"
    proj.mkdir()
    main.persist_config(workspace_dir=str(proj))
    assert chat.workspace_source()[1] == "settings"
    assert chat.workspace().root == os.path.realpath(str(proj))

    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "envws"))
    assert chat.workspace_source()[1] == "env"
    assert chat.workspace().root == os.path.realpath(str(tmp_path / "envws"))


def test_validate_workspace_dir(tmp_path, monkeypatch):
    chat = _chat(tmp_path, monkeypatch)
    assert chat.validate_workspace_dir("") == ""
    assert chat.validate_workspace_dir("   ") == ""
    proj = tmp_path / "proj"
    proj.mkdir()
    assert chat.validate_workspace_dir(str(proj)) == os.path.realpath(str(proj))
    # ~ expands to the home directory.
    assert chat.validate_workspace_dir("~") == os.path.realpath(os.path.expanduser("~"))
    with pytest.raises(ValueError):
        chat.validate_workspace_dir(str(tmp_path / "missing"))
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        chat.validate_workspace_dir(str(f))
    # The data folder itself and its orch/shadow dirs are off-limits.
    data = chat.data_dir()
    os.makedirs(os.path.join(data, "orch"), exist_ok=True)
    os.makedirs(os.path.join(data, "agent_shadow.git"), exist_ok=True)
    for bad in (data, os.path.join(data, "orch"), os.path.join(data, "agent_shadow.git")):
        with pytest.raises(ValueError):
            chat.validate_workspace_dir(bad)


def test_settings_workspace_rejects_bad_paths(web, tmp_path):
    client, chat, _orch = web
    r = client.post("/api/settings",
                    json={"workspace_dir": str(tmp_path / "missing")}, headers=ORIGIN)
    assert r.status_code == 400 and "does not exist" in r.json()["detail"]
    r = client.post("/api/settings",
                    json={"workspace_dir": chat.data_dir()}, headers=ORIGIN)
    assert r.status_code == 400 and "data folder" in r.json()["detail"]


def test_settings_workspace_pinned_by_env(web, tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTABLE_WORKSPACE", str(tmp_path / "envws"))
    client, _chat_svc, _orch = web
    r = client.post("/api/settings", json={"workspace_dir": str(tmp_path)}, headers=ORIGIN)
    assert r.status_code == 400 and "CHATTABLE_WORKSPACE" in r.json()["detail"]


def test_settings_workspace_busy_409(web, tmp_path, monkeypatch):
    client, _chat_svc, orch = web
    monkeypatch.setattr(orch, "busy", lambda: True)
    proj = tmp_path / "proj"
    proj.mkdir()
    r = client.post("/api/settings", json={"workspace_dir": str(proj)}, headers=ORIGIN)
    assert r.status_code == 409


def test_settings_workspace_switch_rebinds_orch(web, tmp_path):
    client, chat, orch = web
    proj = tmp_path / "proj"
    proj.mkdir()
    old_id, old_store = orch.project_id, orch.store.path
    r = client.post("/api/settings", json={"workspace_dir": str(proj)}, headers=ORIGIN)
    assert r.status_code == 200, r.json()
    root = os.path.realpath(str(proj))
    cfg = r.json()
    assert cfg["agent"]["workspace"] == root
    assert cfg["agent"]["workspace_source"] == "settings"
    assert cfg["agent"]["workspace_configured"] == root
    # The orchestrator follows: new project id, new store under the data dir.
    assert orch.project_id == hashlib.sha1(root.encode()).hexdigest()[:12]
    assert orch.project_id != old_id
    assert orch.store.path == os.path.join(chat.data_dir(), "orch",
                                           orch.project_id + ".jsonl")
    assert orch.store.path != old_store
    assert chat.workspace().root == root
    with open(main.SETTINGS_PATH, encoding="utf-8") as fh:
        assert json.load(fh)["workspace_dir"] == root


def test_settings_workspace_reset_to_default(web, tmp_path):
    client, chat, orch = web
    proj = tmp_path / "proj"
    proj.mkdir()
    r = client.post("/api/settings", json={"workspace_dir": str(proj)}, headers=ORIGIN)
    assert r.status_code == 200
    r = client.post("/api/settings", json={"workspace_dir": ""}, headers=ORIGIN)
    assert r.status_code == 200, r.json()
    cfg = r.json()
    default_root = os.path.realpath(os.path.join(chat.data_dir(), "workspace"))
    assert cfg["agent"]["workspace"] == default_root
    assert cfg["agent"]["workspace_source"] == "default"
    assert orch.project_id == hashlib.sha1(default_root.encode()).hexdigest()[:12]
    with open(main.SETTINGS_PATH, encoding="utf-8") as fh:
        assert "workspace_dir" not in json.load(fh)
