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
    """Uploaded attachments stay readable: paths inside <data_dir>/uploads."""

    def test_uploads_path_allowed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            orig = store_mod.resolve_db_path
            store_mod.resolve_db_path = lambda data_dir=None: os.path.join(d, "chatable.jsonl")
            try:
                os.makedirs(os.path.join(d, "uploads"))
                p = os.path.join(d, "uploads", "note.txt")
                with open(p, "w", encoding="utf-8") as f:
                    f.write("hello from an uploaded file")
                r = web_fetch(p)
                self.assertTrue(r.ok, r.output)
                self.assertIn("hello from an uploaded file", r.output)
            finally:
                store_mod.resolve_db_path = orig

    def test_symlink_escape_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            orig = store_mod.resolve_db_path
            store_mod.resolve_db_path = lambda data_dir=None: os.path.join(d, "chatable.jsonl")
            try:
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
            finally:
                store_mod.resolve_db_path = orig


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
