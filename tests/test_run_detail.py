"""GET /api/run/{run_id}: single-run detail — status, timing, step trail,
changed files (snapshot diff) and failure reason; 404 for unknown ids.

The API tests import server with an isolated CHATTABLE_DATA_DIR (same
pattern as tests/test_agent_api.py) so no real user data is touched.
"""
import os
import tempfile
import time

import pytest

_tmp = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k)
              for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION",
                        "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN")}
os.environ["CHATTABLE_DATA_DIR"] = _tmp
os.environ["CHATTABLE_VISION"] = "0"
for _k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN"):
    os.environ.pop(_k, None)

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

from orch.store import OrchStore  # noqa: E402

from test_agent_loop import git_required  # noqa: E402
from test_chat_runs import WriterExecutor, _svc, _wait  # noqa: E402
from test_orch import FakeExecutor  # noqa: E402


class BoomExecutor(FakeExecutor):
    """Emits one failed tool call, then dies mid-run."""

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        yield {"type": "tool_call", "id": "c1", "name": "run_shell",
               "arguments": {"command": "make it fail"}}
        yield {"type": "tool_result", "id": "c1", "ok": False, "output": "boom output"}
        raise RuntimeError("executor exploded")


@git_required
def test_run_detail_unknown_run(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    with pytest.raises(KeyError):
        orch.run_detail("nope")


@git_required
def test_run_detail_running_then_ok(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt", sleep=1)})
    (r,) = orch.start_chat_runs("make a.txt", None, ["a"])
    deadline = time.monotonic() + 5
    while True:
        md = chat.store.get(r["assistant_id"]).metadata or {}
        if md.get("run_status") == "running":
            break
        assert time.monotonic() < deadline, "run did not start"
        time.sleep(0.02)
    # The node links back to its run from the start (queued already).
    assert md["run_id"] == r["run_id"]

    d = orch.run_detail(r["run_id"])
    assert d["id"] == r["run_id"] and d["agent"] == "a"
    assert d["status"] == "running"
    assert d["user_id"] == r["user_id"] and d["assistant_id"] == r["assistant_id"]
    assert d["started_at"] and d["finished_at"] is None
    assert d["duration_s"] is not None and d["duration_s"] >= 0
    assert d["steps"] and d["steps"][0]["name"] == "write_file"
    assert d["steps"][0]["ok"] is None  # still open
    assert d["current_step"]["name"] == "write_file"
    assert d["steps_done"] == 0
    assert isinstance(d["files"], list)  # empty until the file is written
    assert d["error"] == ""

    _wait(chat, r["assistant_id"])
    d = orch.run_detail(r["run_id"])
    assert d["status"] == "ok"
    assert d["finished_at"] and d["duration_s"] is not None
    assert d["steps_done"] == 1
    assert d["current_step"]["ok"] is True
    assert d["files"] == [{"path": "a.txt", "status": "A", "added": 1, "removed": 0}]
    assert d["error"] == "" and "output_tail" not in d

    # The step trail is persisted: a fresh store on the same file has it.
    reloaded = OrchStore(orch.store.path)
    assert reloaded.runs[r["run_id"]]["steps"]
    assert reloaded.runs[r["run_id"]]["steps_done"] == 1


@git_required
def test_run_detail_failed(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": BoomExecutor()})
    (r,) = orch.start_chat_runs("fail", None, ["a"])
    _ev, md = _wait(chat, r["assistant_id"])
    assert md["run_status"] == "failed"
    d = orch.run_detail(r["run_id"])
    assert d["status"] == "failed"
    assert "executor exploded" in d["error"]
    assert d["steps_done"] == 1 and d["steps"][0]["ok"] is False
    assert d["output_tail"] and "boom output" in d["output_tail"][-1]
    assert d["current_step"]["name"] == "run_shell"


@git_required
def test_api_run_detail_404_and_ok(tmp_path, monkeypatch):
    chat, orch = _svc(tmp_path, monkeypatch, {"a": WriterExecutor("a.txt")})
    monkeypatch.setattr(server, "orch", orch)
    client = TestClient(server.app, base_url="http://localhost:8000")
    auth = {"Authorization": "Bearer t"}  # _svc sets CHATTABLE_AUTH_TOKEN=t

    r = client.get("/api/run/nope", headers=auth)
    assert r.status_code == 404

    (run,) = orch.start_chat_runs("make a.txt", None, ["a"])
    _wait(chat, run["assistant_id"])
    r = client.get(f"/api/run/{run['run_id']}", headers=auth)
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["agent"] == "a"
    assert d["assistant_id"] == run["assistant_id"]
    assert d["files"] == [{"path": "a.txt", "status": "A", "added": 1, "removed": 0}]
    assert d["steps_done"] == 1


@git_required
def test_run_detail_mid_run_files_without_commits(tmp_path, monkeypatch):
    """While running, changed files come from the live worktree, read-only:
    polling must not add commits to the shadow repo or lock the index."""
    import subprocess

    class SlowAfterWrite(WriterExecutor):
        def run(self, workdir, prompt, **kw):
            with open(os.path.join(workdir, "base.txt"), "a") as fh:
                fh.write("more\n")
            with open(os.path.join(workdir, "new.txt"), "w") as fh:
                fh.write("a\nb\n")
            yield {"type": "tool_call", "id": "c1", "name": "write_file",
                   "arguments": {"path": "new.txt"}}
            time.sleep(1.0)
            yield {"type": "tool_result", "id": "c1", "output": "ok"}
            yield {"type": "result", "final_text": "done", "exit_code": 0,
                   "session_id": None, "usage": {}}

    chat, orch = _svc(tmp_path, monkeypatch, {"a": SlowAfterWrite("x")})
    (r,) = orch.start_chat_runs("go", None, ["a"])
    deadline = time.monotonic() + 5
    while not (orch.run_detail(r["run_id"])["steps"]):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    count = lambda: subprocess.run(
        ["git", "--git-dir", orch.shadow.git_dir, "rev-list", "--all", "--count"],
        capture_output=True, text=True).stdout.strip()
    before = count()
    d = orch.run_detail(r["run_id"])
    assert d["status"] == "running"
    files = {f["path"]: f for f in d["files"]}
    assert files["new.txt"] == {"path": "new.txt", "status": "A", "added": 2, "removed": 0}
    assert files["base.txt"]["status"] == "M" and files["base.txt"]["added"] == 1
    assert not any(p.startswith(".chatable/") for p in files)
    orch.run_detail(r["run_id"])
    assert count() == before  # no progress commits
    _events, md = _wait(chat, r["assistant_id"])
    assert md["run_status"] == "ok"
