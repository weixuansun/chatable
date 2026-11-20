"""Host/Origin/token hardening of the FastAPI server (server.SecurityMiddleware).

The app is imported with an isolated CHATTABLE_DATA_DIR so no real user data
is touched, and with the vision probe disabled (no network).
"""
import os
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
_saved_env = {k: os.environ.get(k)
              for k in ("CHATTABLE_DATA_DIR", "CHATTABLE_VISION",
                        "CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
                        "CHATTABLE_ALLOWED_HOSTS", "CHATABLE_ALLOWED_HOSTS")}
os.environ["CHATTABLE_DATA_DIR"] = _tmp
os.environ["CHATTABLE_VISION"] = "0"  # kill switch: never probe the backend
for _k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN",
           "CHATTABLE_ALLOWED_HOSTS", "CHATABLE_ALLOWED_HOSTS"):
    os.environ.pop(_k, None)

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

# Restore the environment exactly: pytest collects (imports) every test module
# before running any test, and service.py reads CHATTABLE_VISION into its
# module-level VISION_ENABLED at import time — leaving the kill switch set
# would disable image injection for the whole session.
for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v
import service as _service_mod  # noqa: E402

_service_mod.VISION_ENABLED = os.environ.get("CHATTABLE_VISION", "1").lower() not in ("0", "false", "no")

BASE = "http://localhost:8000"


def _client() -> TestClient:
    return TestClient(server.app, base_url=BASE)


class HostAllowlistTests(unittest.TestCase):
    def test_localhost_host_allowed(self):
        r = _client().get("/api/config")
        self.assertEqual(r.status_code, 200)

    def test_bad_host_rejected(self):
        for host in ("evil.com", "localhost.evil.com", "192.168.1.10", ""):
            r = _client().get("/api/config", headers={"host": host})
            self.assertEqual(r.status_code, 400, host)

    def test_bind_host_and_ipv6_allowed(self):
        for host in ("127.0.0.1:8000", "[::1]:8000", "localhost"):
            r = _client().get("/api/config", headers={"host": host})
            self.assertEqual(r.status_code, 200, host)

    def test_extra_allowed_hosts_env(self):
        old = os.environ.get("CHATTABLE_ALLOWED_HOSTS")
        os.environ["CHATTABLE_ALLOWED_HOSTS"] = "chat.local, example.com:9000"
        try:
            for host in ("chat.local", "chat.local:8000", "example.com:9000"):
                r = _client().get("/api/config", headers={"host": host})
                self.assertEqual(r.status_code, 200, host)
            r = _client().get("/api/config", headers={"host": "other.com"})
            self.assertEqual(r.status_code, 400)
        finally:
            if old is None:
                os.environ.pop("CHATTABLE_ALLOWED_HOSTS", None)
            else:
                os.environ["CHATTABLE_ALLOWED_HOSTS"] = old

    def test_single_t_alias_allowed_hosts(self):
        old = os.environ.get("CHATABLE_ALLOWED_HOSTS")
        os.environ["CHATABLE_ALLOWED_HOSTS"] = "legacy.host"
        try:
            r = _client().get("/api/config", headers={"host": "legacy.host"})
            self.assertEqual(r.status_code, 200)
        finally:
            if old is None:
                os.environ.pop("CHATABLE_ALLOWED_HOSTS", None)
            else:
                os.environ["CHATABLE_ALLOWED_HOSTS"] = old

    def test_bind_host_variable_allowed(self):
        # main() records --host in server._BIND_HOST so the server never
        # rejects its own Host header.
        old = server._BIND_HOST
        server._BIND_HOST = "192.168.1.5"
        try:
            r = _client().get("/api/config", headers={"host": "192.168.1.5"})
            self.assertEqual(r.status_code, 200)
            r = _client().get("/api/config", headers={"host": "192.168.1.6"})
            self.assertEqual(r.status_code, 400)
        finally:
            server._BIND_HOST = old


class OriginCheckTests(unittest.TestCase):
    def test_cross_origin_post_rejected(self):
        r = _client().post(
            "/api/node/nope/rename",
            json={"title": "x"},
            headers={"origin": "http://evil.com"},
        )
        self.assertEqual(r.status_code, 403)

    def test_cross_origin_referer_rejected(self):
        r = _client().post(
            "/api/node/nope/rename",
            json={"title": "x"},
            headers={"referer": "http://evil.com/page"},
        )
        self.assertEqual(r.status_code, 403)

    def test_same_origin_post_allowed(self):
        # 404 (node missing) proves the middleware let the request through.
        r = _client().post(
            "/api/node/nope/rename",
            json={"title": "x"},
            headers={"origin": BASE},
        )
        self.assertEqual(r.status_code, 404)

    def test_headerless_post_from_nonloopback_client_rejected(self):
        # TestClient's peer is "testclient" (not loopback); a browser always
        # sends Origin/Referer, curl from the same machine is loopback.
        r = _client().post("/api/node/nope/rename", json={"title": "x"})
        self.assertEqual(r.status_code, 403)

    def test_fs_list_cross_origin_rejected(self):
        r = _client().get("/api/fs/list", headers={"origin": "http://evil.com"})
        self.assertEqual(r.status_code, 403)

    def test_fs_list_headerless_rejected(self):
        r = _client().get("/api/fs/list")
        self.assertEqual(r.status_code, 403)

    def test_fs_list_same_origin_allowed(self):
        r = _client().get("/api/fs/list", headers={"referer": BASE + "/"})
        self.assertEqual(r.status_code, 200)

    def test_plain_get_not_origin_checked(self):
        r = _client().get("/api/config", headers={"origin": "http://evil.com"})
        self.assertEqual(r.status_code, 200)  # Host allowlist still applies


class AuthTokenTests(unittest.TestCase):
    TOKEN = "s3cret-token"

    def setUp(self):
        self._old = {k: os.environ.get(k)
                     for k in ("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN")}
        os.environ["CHATTABLE_AUTH_TOKEN"] = self.TOKEN
        os.environ.pop("CHATABLE_AUTH_TOKEN", None)

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_token_required(self):
        r = _client().get("/api/config")
        self.assertEqual(r.status_code, 401)

    def test_wrong_token_rejected(self):
        r = _client().get("/api/config",
                          headers={"authorization": "Bearer wrong"})
        self.assertEqual(r.status_code, 401)

    def test_bearer_token_accepted(self):
        r = _client().get(
            "/api/config",
            headers={"authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(r.status_code, 200)

    def test_login_sets_cookie_and_redirects(self):
        client = _client()
        r = client.get(f"/?token={self.TOKEN}", follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertIn("chatable_token=", r.headers["set-cookie"])
        # The cookie authenticates follow-up API calls (UI, SSE, uploads).
        r = client.get("/api/config")
        self.assertEqual(r.status_code, 200)

    def test_bad_login_token_rejected(self):
        r = _client().get("/?token=nope", follow_redirects=False)
        self.assertEqual(r.status_code, 403)

    def test_single_t_alias_token(self):
        os.environ.pop("CHATTABLE_AUTH_TOKEN", None)
        os.environ["CHATABLE_AUTH_TOKEN"] = self.TOKEN
        r = _client().get(
            "/api/config",
            headers={"authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(r.status_code, 200)

    def test_cookie_parsing_robust(self):
        # Odd spacing / extra cookies / a malformed segment must not break
        # token lookup.
        cookie = f"foo=bar; chatable_token={self.TOKEN};  ;baz=qux"
        r = _client().get("/api/config", headers={"cookie": cookie})
        self.assertEqual(r.status_code, 200)

    def test_token_allows_headerless_post(self):
        # curl with a Bearer token needs no Origin header.
        r = _client().post(
            "/api/node/nope/rename",
            json={"title": "x"},
            headers={"authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(r.status_code, 404)  # passed middleware, node missing

    def test_token_still_rejects_cross_origin(self):
        r = _client().post(
            "/api/node/nope/rename",
            json={"title": "x"},
            headers={"authorization": f"Bearer {self.TOKEN}",
                     "origin": "http://evil.com"},
        )
        self.assertEqual(r.status_code, 403)


class FileServingTests(unittest.TestCase):
    def test_traversal_blocked(self):
        headers = {"referer": BASE + "/"}
        for name in ("..%2F..%2Fetc%2Fpasswd", "..\\..\\etc\\passwd",
                     "%2Fetc%2Fpasswd"):
            r = _client().get(f"/api/files/{name}", headers=headers)
            self.assertIn(r.status_code, (404, 405), name)

    def test_uploaded_file_served(self):
        headers = {"referer": BASE + "/"}
        up = _client().post(
            "/api/upload",
            files={"file": ("note.txt", b"hello upload", "text/plain")},
            headers={"origin": BASE},
        )
        self.assertEqual(up.status_code, 200)
        r = _client().get(f"/api/files/{up.json()['filename']}", headers=headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b"hello upload")


if __name__ == "__main__":
    unittest.main()
