"""Orchestrator HTTP endpoints with a fake executor (never the real CLIs)."""
import os
import tempfile
import time
import unittest

_tmp_data = tempfile.mkdtemp()
_tmp_ws = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k)
              for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION",
                        "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
                        "CHATTABLE_WORKSPACE", "CHATTABLE_ORCH_LIMITS")}
os.environ["CHATTABLE_DATA_DIR"] = _tmp_data
os.environ["CHATTABLE_WORKSPACE"] = _tmp_ws
os.environ["CHATTABLE_VISION"] = "0"
os.environ["CHATTABLE_ORCH_LIMITS"] = "fake=5"
os.environ["CHATTABLE_AUTH_TOKEN"] = "orch-test-token"
for _k in ("CHATABLE_AUTH_TOKEN",):
    os.environ.pop(_k, None)

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

BASE = "http://localhost:8000"
AUTH = {"Authorization": "Bearer orch-test-token"}
ORIGIN = {"origin": BASE}


class FakeExecutor:
    def available(self):
        return (True, "")

    def run(self, workdir, prompt, session_id=None, should_stop=lambda: False,
            proc_reg=lambda p: None, timeout=None):
        with open(os.path.join(workdir, "fake.txt"), "w") as fh:
            fh.write("api run\n")
        yield {"type": "tool_call", "name": "write_file", "arguments": {"path": "fake.txt"}}
        yield {"type": "result",
               "final_text": "done\n\n## Conclusion\n### Done\nWrote fake.txt.",
               "exit_code": 0, "session_id": "s-1", "usage": {}}


def _client() -> TestClient:
    return TestClient(server.app, base_url=BASE)


def _wait(run_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = server.orch.store.runs[run_id]
        if run["status"] not in ("queued", "running"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} stuck: {run['status']}")


class OrchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = {k: os.environ.get(k)
                      for k in ("CHATTABLE_AUTH_TOKEN", "CHATTABLE_WORKSPACE")}
        os.environ["CHATTABLE_AUTH_TOKEN"] = "orch-test-token"
        os.environ["CHATTABLE_WORKSPACE"] = _tmp_ws
        from service import ChatService
        from store import TreeStore
        from orch.service import OrchService
        chat = ChatService(TreeStore(os.path.join(_tmp_data, "chat.jsonl")))
        cls.orch = OrchService(chat, executors={"fake": FakeExecutor()})
        server.orch = cls.orch

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_state_reports_executors_and_root(self):
        r = _client().get("/api/orch/state", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("root", body["project"])
        self.assertTrue(any(n.get("root") for n in body["nodes"]))
        self.assertTrue(body["executors"]["fake"]["available"])
        # claude/codex report unavailable cleanly on this box.
        self.assertIn("fake", body["executors"])

    def test_state_requires_token(self):
        r = _client().get("/api/orch/state")
        self.assertEqual(r.status_code, 401)

    def test_submit_run_and_node_detail_and_compare_and_adopt(self):
        client = _client()
        state = client.get("/api/orch/state", headers=AUTH).json()
        root_id = state["project"]["root"]

        r = client.post("/api/orch/runs",
                        json={"from_node": root_id,
                              "items": [{"instruction": "make fake.txt",
                                         "agents": ["fake"]}]},
                        headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 200, r.text)
        run_id = r.json()["runs"][0]
        run = _wait(run_id)
        self.assertEqual(run["status"], "ok")

        node_id = run["node_id"]
        r = client.get(f"/api/orch/node/{node_id}", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        node = r.json()
        self.assertEqual(node["conclusion"]["done"], "Wrote fake.txt.")
        self.assertTrue(node["events"])

        r = client.get(f"/api/orch/compare?a={root_id}&b={node_id}", headers=AUTH)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(r.json()["files"][0]["path"], "fake.txt")

        r = client.post("/api/orch/adopt", json={"node_id": node_id},
                        headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(os.path.exists(os.path.join(_tmp_ws, "fake.txt")))
        prev = r.json()["previous_commit"]
        r = client.post("/api/orch/undo-adopt", json={"previous_commit": prev},
                        headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(os.path.join(_tmp_ws, "fake.txt")))

    def test_submit_unknown_node_and_executor(self):
        client = _client()
        r = client.post("/api/orch/runs",
                        json={"from_node": "nope",
                              "items": [{"instruction": "x", "agents": ["fake"]}]},
                        headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 404)
        state = client.get("/api/orch/state", headers=AUTH).json()
        r = client.post("/api/orch/runs",
                        json={"from_node": state["project"]["root"],
                              "items": [{"instruction": "x", "agents": ["nope"]}]},
                        headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 400)

    def test_cancel_unknown_run_404(self):
        r = _client().post("/api/orch/runs/nope/cancel",
                           headers={**ORIGIN, **AUTH})
        self.assertEqual(r.status_code, 404)

    def test_stream_replays_events(self):
        client = _client()
        state = client.get("/api/orch/state", headers=AUTH).json()
        r = client.post("/api/orch/runs",
                        json={"from_node": state["project"]["root"],
                              "items": [{"instruction": "stream me", "agents": ["fake"]}]},
                        headers={**ORIGIN, **AUTH})
        run_id = r.json()["runs"][0]
        _wait(run_id)
        with client.stream("GET", "/api/orch/stream", headers=AUTH) as resp:
            types = set()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    import json as _json
                    evt = _json.loads(line[6:])
                    if evt.get("type") == "end":
                        break
                    types.add(evt.get("type"))
                if {"run_started", "run_finished", "node_created"} <= types:
                    break
        self.assertIn("run_started", types)
        self.assertIn("run_finished", types)
        self.assertIn("node_created", types)


if __name__ == "__main__":
    unittest.main()
