"""SSRF / local-file-read defences in web_tools.web_fetch.

No real network: DNS is stubbed via socket.getaddrinfo and the HTTP transport
via httpx.MockTransport.
"""
import asyncio
import os
import socket
import unittest

import httpx

import store as store_mod
import web_tools
from web_tools import (
    _FetchRefused,
    _fetch_redirected,
    _stream_get,
    _validate_fetch_url,
    web_fetch,
)


# A fake DNS table: host → resolved IP. Unknown hosts raise gaierror.
_DNS = {
    "good.example": "93.184.216.34",
    "localhost": "127.0.0.1",
    "evil.example": "192.168.1.1",
    "2130706433": "127.0.0.1",  # inet_aton form of 127.0.0.1
    "fake.example": "198.18.0.1",   # TUN/fake-IP proxy range (Clash/Mihomo)
    "intranet.example": "10.1.2.3",
}


def _fake_getaddrinfo(host, port, *args, **kwargs):
    ip = _DNS.get(host)
    if ip is None:
        raise socket.gaierror(f"Name or service not known: {host}")
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]


def _mock_client(handler):
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=web_tools._FETCH_TIMEOUT,
        follow_redirects=False,
        headers=web_tools._FETCH_HEADERS,
    )


class UrlValidationTests(unittest.TestCase):
    def setUp(self):
        self._orig = socket.getaddrinfo
        socket.getaddrinfo = _fake_getaddrinfo

    def tearDown(self):
        socket.getaddrinfo = self._orig

    def test_public_host_allowed(self):
        self.assertIsNone(_validate_fetch_url("https://good.example/page"))

    def test_public_literal_allowed(self):
        self.assertIsNone(_validate_fetch_url("http://93.184.216.34/"))

    def test_loopback_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://127.0.0.1/"))

    def test_ipv6_loopback_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://[::1]/"))

    def test_ipv4_mapped_ipv6_loopback_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://[::ffff:127.0.0.1]/"))

    def test_cloud_metadata_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://169.254.169.254/latest/meta-data"))

    def test_cgnat_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://100.64.0.1/"))

    def test_localhost_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://localhost:8000/api/tree"))

    def test_decimal_ip_refused(self):
        self.assertIsNotNone(_validate_fetch_url("http://2130706433/"))

    def test_private_resolution_refused(self):
        self.assertIsNotNone(_validate_fetch_url("https://evil.example/"))

    def test_non_http_schemes_refused(self):
        for url in ("file:///etc/passwd", "ftp://good.example/x",
                    "data:text/html,hi", "gopher://good.example/"):
            self.assertIsNotNone(_validate_fetch_url(url), url)

    def test_unresolvable_host(self):
        self.assertIsNotNone(_validate_fetch_url("https://no-such-host.invalid/"))


class FakeIpAndProxyTests(unittest.TestCase):
    """TUN/fake-IP proxies (Clash/Mihomo) resolve every hostname into
    198.18.0.0/15; CHATTABLE_FETCH_ALLOW_NETS controls the exception."""

    ENV = "CHATTABLE_FETCH_ALLOW_NETS"
    PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                  "ALL_PROXY", "all_proxy")

    def setUp(self):
        self._orig_dns = socket.getaddrinfo
        socket.getaddrinfo = _fake_getaddrinfo
        self._old_env = {k: os.environ.get(k) for k in (self.ENV,) + self.PROXY_VARS}
        for k in self.PROXY_VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        socket.getaddrinfo = self._orig_dns
        for k, v in self._old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_fake_ip_allowed_by_default(self):
        os.environ.pop(self.ENV, None)
        self.assertIsNone(_validate_fetch_url("https://fake.example/"))
        # Literal fake-IP addresses are allowed by default too (documented:
        # 198.18/15 is a benchmarking range, never a real LAN).
        self.assertIsNone(_validate_fetch_url("https://198.18.0.1/"))

    def test_fake_ip_disabled_when_env_empty(self):
        os.environ[self.ENV] = ""
        self.assertIsNotNone(_validate_fetch_url("https://fake.example/"))

    def test_custom_cidr_honored(self):
        os.environ[self.ENV] = "10.0.0.0/8"
        self.assertIsNone(_validate_fetch_url("https://intranet.example/"))
        # The custom list REPLACES the default: fake-IP is refused again.
        self.assertIsNotNone(_validate_fetch_url("https://fake.example/"))

    def test_loopback_and_metadata_stay_refused(self):
        os.environ.pop(self.ENV, None)  # default 198.18/15 allowed
        for url in ("http://127.0.0.1/", "http://169.254.169.254/",
                    "http://10.0.0.5/", "http://[::1]/"):
            self.assertIsNotNone(_validate_fetch_url(url), url)

    def test_proxy_fallback_when_resolution_fails(self):
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        # Unresolvable locally, but the proxy resolves server-side.
        self.assertIsNone(_validate_fetch_url("https://no-such-host.invalid/"))

    def test_no_proxy_still_refuses_unresolvable(self):
        self.assertIsNotNone(_validate_fetch_url("https://no-such-host.invalid/"))

    def test_proxy_fallback_does_not_skip_literal_ip_checks(self):
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        self.assertIsNotNone(_validate_fetch_url("http://127.0.0.1/"))


class SchemeRefusalTests(unittest.TestCase):
    """Explicit non-http(s) schemes get a clear refusal; bare hosts still
    get https:// prepended."""

    def test_explicit_schemes_refused(self):
        for url in ("gopher://example.com/", "ftp://example.com/x",
                    "data:text/html,hi", "javascript:alert(1)"):
            r = web_fetch(url)
            self.assertFalse(r.ok, url)
            self.assertIn("only http", r.output, url)

    def test_bare_host_still_works(self):
        # "example.com/x" and "example.com:8080/x" are bare hosts, not
        # schemes — they reach DNS resolution rather than a scheme refusal.
        orig = socket.getaddrinfo
        socket.getaddrinfo = _fake_getaddrinfo
        saved = {k: os.environ.pop(k, None) for k in
                 ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                  "ALL_PROXY", "all_proxy")}
        try:
            r = web_fetch("no-such-host.invalid:8080/x")
        finally:
            socket.getaddrinfo = orig
            os.environ.update({k: v for k, v in saved.items() if v is not None})
        self.assertFalse(r.ok)
        self.assertIn("Cannot resolve host", r.output)


class WebFetchRefusalTests(unittest.TestCase):
    """End-to-end refusals through web_fetch (validation fails before any
    connection, so no transport is needed)."""

    def test_file_scheme_refused(self):
        r = web_fetch("file:///etc/passwd")
        self.assertFalse(r.ok)
        self.assertIn("file://", r.output)

    def test_loopback_refused(self):
        r = web_fetch("http://127.0.0.1:8000/api/tree")
        self.assertFalse(r.ok)
        self.assertIn("Refused", r.output)

    def test_metadata_ip_refused(self):
        r = web_fetch("http://169.254.169.254/latest/meta-data")
        self.assertFalse(r.ok)
        self.assertIn("Refused", r.output)

    def test_arbitrary_local_path_refused(self):
        r = web_fetch("/etc/passwd")
        self.assertFalse(r.ok)
        self.assertIn("Refused", r.output)

    def test_home_path_refused(self):
        r = web_fetch("~/.chatable/settings.json")
        self.assertFalse(r.ok)
        self.assertIn("Refused", r.output)


class LocalPathAllowlistTests(unittest.TestCase):
    """Uploaded attachments stay readable: paths inside a registered
    uploads/images root (ChatService registers its active data dir)."""

    def test_uploads_path_allowed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            web_tools.register_local_read_root(os.path.join(d, "uploads"))
            os.makedirs(os.path.join(d, "uploads"))
            p = os.path.join(d, "uploads", "note.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello from an uploaded file")
            r = web_fetch(p)
            self.assertTrue(r.ok, r.output)
            self.assertIn("hello from an uploaded file", r.output)

    def test_symlink_escape_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            web_tools.register_local_read_root(os.path.join(d, "uploads"))
            uploads = os.path.join(d, "uploads")
            os.makedirs(uploads)
            secret = os.path.join(d, "secret.txt")
            with open(secret, "w", encoding="utf-8") as f:
                f.write("top secret")
            link = os.path.join(uploads, "link.txt")
            os.symlink(secret, link)
            r = web_fetch(link)
            self.assertFalse(r.ok)
            self.assertIn("Refused", r.output)

    def test_default_dir_fallback_when_nothing_registered(self):
        """Without a ChatService (CLI use), the DEFAULT data dir's
        uploads/images dirs are the allowlist."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            orig_resolve = store_mod.resolve_db_path
            saved_roots = set(web_tools._LOCAL_READ_ROOTS)
            web_tools._LOCAL_READ_ROOTS.clear()
            store_mod.resolve_db_path = lambda data_dir=None: os.path.join(d, "chatable.jsonl")
            try:
                self.assertTrue(web_tools.local_path_allowed(
                    os.path.join(d, "uploads", "a.txt")))
                self.assertFalse(web_tools.local_path_allowed(
                    os.path.join(d, "elsewhere", "a.txt")))
            finally:
                store_mod.resolve_db_path = orig_resolve
                web_tools._LOCAL_READ_ROOTS.update(saved_roots)


class RedirectAndCapTests(unittest.TestCase):
    def setUp(self):
        self._orig = socket.getaddrinfo
        socket.getaddrinfo = _fake_getaddrinfo

    def tearDown(self):
        socket.getaddrinfo = self._orig

    def test_redirect_to_private_ip_refused(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "good.example":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
            return httpx.Response(200, text="should never be reached")

        async def go():
            async with _mock_client(handler) as client:
                await _fetch_redirected(client, "https://good.example/start")

        with self.assertRaises(_FetchRefused):
            asyncio.run(go())

    def test_too_many_redirects_refused(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://good.example/loop"})

        async def go():
            async with _mock_client(handler) as client:
                await _fetch_redirected(client, "https://good.example/loop")

        with self.assertRaises(_FetchRefused):
            asyncio.run(go())

    def test_public_redirect_allowed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(301, headers={"location": "/final"})
            return httpx.Response(200, text="hello world",
                                  headers={"content-type": "text/plain"})

        async def go():
            async with _mock_client(handler) as client:
                return await _fetch_redirected(client, "https://good.example/start")

        status, headers, body, final_url = asyncio.run(go())
        self.assertEqual(status, 200)
        self.assertEqual(body, b"hello world")
        self.assertEqual(final_url, "https://good.example/final")

    def test_response_size_cap(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"x" * 4096,
                                  headers={"content-type": "text/plain"})

        async def go():
            async with _mock_client(handler) as client:
                await _stream_get(client, "https://good.example/big", max_bytes=1024)

        with self.assertRaises(_FetchRefused):
            asyncio.run(go())

    def test_web_fetch_allowed_public_url(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="<html><body><p>Readable public content here.</p></body></html>",
                headers={"content-type": "text/html; charset=utf-8"},
            )

        orig_client = httpx.AsyncClient

        def factory(**kwargs):
            return orig_client(
                transport=httpx.MockTransport(handler),
                timeout=web_tools._FETCH_TIMEOUT,
                follow_redirects=False,
                headers=web_tools._FETCH_HEADERS,
            )

        httpx.AsyncClient = factory
        try:
            r = web_fetch("https://good.example/article", no_cache=True)
        finally:
            httpx.AsyncClient = orig_client
        self.assertTrue(r.ok, r.output)
        self.assertIn("Readable public content", r.output)


if __name__ == "__main__":
    unittest.main()
