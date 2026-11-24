"""Agent HTTP endpoints: token gating, status, node diff / adopt validation.

Only the run-result endpoints remain (status, node diff, compare, files,
file, adopt, undo-adopt) — the in-conversation agent-mode endpoints (mode
toggle, approve, restore, step diff) are gone with the retired in-place
agent mode.

Imports server with an isolated CHATTABLE_DATA_DIR (same pattern as
tests/test_server_security.py) so no real user data is touched.
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k)
              for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION",
                        "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
                        "CHATTABLE_WORKSPACE")}
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
import service as _service_mod  # noqa: E402

_service_mod.VISION_ENABLED = os.environ.get("CHATTABLE_VISION", "1").lower() not in ("0", "false", "no")

BASE = "http://localhost:8000"
ORIGIN = {"origin": BASE}


def _client() -> TestClient:
    return TestClient(server.app, base_url=BASE)


class AgentTokenGateTests(unittest.TestCase):
    """Exposed beyond loopback without CHATTABLE_AUTH_TOKEN, the agent
    endpoints must refuse."""

    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
                                 "CHATTABLE_BIND_HOST")}
        for k in self._saved:
            os.environ.pop(k, None)
        os.environ["CHATTABLE_BIND_HOST"] = "0.0.0.0"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_status_reports_unavailable(self):
        r = _client().get("/api/agent/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["available"])
        self.assertIn("CHATTABLE_AUTH_TOKEN", body["reason"])
        self.assertTrue(body["workspace"])

    def test_mutating_endpoints_refused_without_token(self):
        refused = [
            _client().post("/api/agent/adopt", json={"node_id": "x"}, headers=ORIGIN),
            _client().post("/api/agent/undo-adopt", json={"previous_commit": "x"},
                           headers=ORIGIN),
            _client().get("/api/agent/compare?a=x&b=y", headers=ORIGIN),
            _client().get("/api/agent/files?node=x", headers=ORIGIN),
            _client().get("/api/agent/file?node=x&path=a.txt", headers=ORIGIN),
        ]
        for r in refused:
            self.assertEqual(r.status_code, 403)
            self.assertIn("CHATTABLE_AUTH_TOKEN", r.json()["detail"])


class AgentEndpointValidationTests(unittest.TestCase):
    """With a token, agent endpoints work (and require the bearer token)."""

    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN")}
        os.environ["CHATTABLE_AUTH_TOKEN"] = "agent-test-token"
        self.auth = {"Authorization": "Bearer agent-test-token"}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_status_available_with_token(self):
        r = _client().get("/api/agent/status", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["available"])
        self.assertEqual(body["reason"], "")

    def test_endpoints_require_the_token_when_set(self):
        # Middleware-level: no credentials at all -> 401.
        self.assertEqual(_client().get("/api/agent/status").status_code, 401)
        r = _client().post("/api/agent/adopt", json={"node_id": "x"}, headers=ORIGIN)
        self.assertEqual(r.status_code, 401)

    def test_adopt_unknown_node_404(self):
        r = _client().post("/api/agent/adopt", json={"node_id": "nope"},
                           headers={**ORIGIN, **self.auth})
        self.assertEqual(r.status_code, 404)

    def test_node_diff_unknown_404(self):
        r = _client().get("/api/agent/node/nope/diff", headers=self.auth)
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
