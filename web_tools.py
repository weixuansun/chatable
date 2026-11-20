"""Read-only web tools for chatable.

`web_search()` and `web_fetch()` return `WebToolResult` objects that can be
rendered directly and stored as `system_note` nodes in the conversation tree.
"""
from __future__ import annotations

import asyncio
import html as html_mod
import io
import ipaddress
import os
import re
import socket
import threading
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import httpx
except ImportError:  # pragma: no cover - depends on local environment
    httpx = None

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import pymupdf as fitz  # PyMuPDF — PDF embedded-image extraction / page rendering
except ImportError:  # pragma: no cover
    try:
        import fitz  # legacy module name
    except ImportError:
        fitz = None

import base64 as _base64


def _images_wanted() -> bool:
    """True when the active backend accepts image inputs (probed at startup /
    on config change). Deferred import avoids a module cycle."""
    try:
        import main  # noqa: PLC0415
        return bool(getattr(main, "VISION_SUPPORTED", False))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Image capture (only used when the backend is vision-capable)
# ---------------------------------------------------------------------------

_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r'\bsrc\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_IMG_SRCSET_RE = re.compile(r'\bsrcset\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
# URL fragments that mark decorative/tracking images rather than content.
_IMG_NOISE_RE = re.compile(
    r"icon|logo|sprite|avatar|badge|tracking|pixel|spinner|blank|emoji|favicon",
    re.IGNORECASE,
)


def _extract_image_urls(html: str, base_url: str, limit: int = 8) -> list[str]:
    """Candidate content-image URLs from raw HTML, noise-filtered, in order."""
    urls: list[str] = []
    seen: set[str] = set()
    for tag in _IMG_TAG_RE.findall(html):
        m = _IMG_SRC_RE.search(tag)
        src = m.group(1).strip() if m else ""
        if not src:
            sm = _IMG_SRCSET_RE.search(tag)
            if sm:
                src = sm.group(1).split(",")[0].strip().split(" ")[0]
        if not src or src.startswith("data:"):
            continue
        full = urllib.parse.urljoin(base_url, src)
        if not full.startswith(("http://", "https://")):
            continue
        if _IMG_NOISE_RE.search(full):
            continue
        if full not in seen:
            seen.add(full)
            urls.append(full)
            if len(urls) >= limit:
                break
    return urls


async def _download_images(urls: list[str], limit: int = 4,
                           max_bytes: int = 5 * 1024 * 1024) -> list[dict]:
    """Download images and inline them as base64 data URLs. Failures skipped.

    Each candidate URL goes through the same SSRF validation as a top-level
    fetch (page-controlled image URLs could otherwise point at loopback or
    cloud-metadata addresses), and redirects are NOT followed — a redirect
    target would be an unvalidated hop.
    """
    if httpx is None:
        return []
    out: list[dict] = []
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=False, headers=_FETCH_HEADERS,
    ) as client:
        for u in urls:
            if len(out) >= limit:
                break
            try:
                status, headers, body = await _stream_get(client, u, max_bytes=max_bytes)
                ct = headers.get("content-type", "").split(";")[0].strip()
                if status == 200 and ct.startswith("image/") and 0 < len(body) <= max_bytes:
                    out.append({
                        "url": u,
                        "data_url": f"data:{ct};base64,{_base64.b64encode(body).decode()}",
                    })
            except Exception:  # noqa: BLE001 - a broken image never fails the fetch
                continue
    return out


def _extract_pdf_images(raw_bytes: bytes, limit: int = 6) -> list[dict]:
    """Embedded images from a PDF; for text-less (scanned) PDFs, render the
    first pages instead. Empty list when PyMuPDF is unavailable."""
    if fitz is None:
        return []
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    try:
        for page in doc:
            for info in page.get_images(full=True):
                if len(out) >= limit:
                    break
                try:
                    pix = fitz.Pixmap(doc, info[0])
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    data = pix.tobytes("jpeg", jpg_quality=80)
                except Exception:  # noqa: BLE001
                    continue
                if len(data) < 10 * 1024:  # skip icons/fragments
                    continue
                out.append({
                    "url": f"embedded#{len(out) + 1}",
                    "data_url": "data:image/jpeg;base64," + _base64.b64encode(data).decode(),
                })
        if not out:
            # Scanned PDF: render the first pages so the model can read them.
            for pno in range(min(3, len(doc))):
                pix = doc[pno].get_pixmap(dpi=110)
                data = pix.tobytes("jpeg", jpg_quality=80)
                out.append({
                    "url": f"page#{pno + 1}",
                    "data_url": "data:image/jpeg;base64," + _base64.b64encode(data).decode(),
                })
    except Exception:  # noqa: BLE001
        return []
    finally:
        doc.close()
    return out


@dataclass
class WebToolResult:
    ok: bool
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def success(output: str, **metadata: Any) -> "WebToolResult":
        return WebToolResult(True, output, metadata)

    @staticmethod
    def error(message: str, **metadata: Any) -> "WebToolResult":
        return WebToolResult(False, message, metadata)


# ---------------------------------------------------------------------------
# Web search: Tavily → Exa → DuckDuckGo fallback chain
# ---------------------------------------------------------------------------

_TAVILY_URL = "https://api.tavily.com/search"
_EXA_URL = "https://api.exa.ai/search"
_DDG_URL = "https://html.duckduckgo.com/html/"

_DDG_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_attr(tag_html: str, attr: str) -> str:
    m = re.search(rf'\b{attr}="([^"]*)"', tag_html)
    return m.group(1) if m else ""


def _decode_ddg_url(url: str) -> str:
    if url.startswith("//duckduckgo.com/l/?"):
        qs = urllib.parse.urlparse("https:" + url).query
        params = urllib.parse.parse_qs(qs)
        return params.get("uddg", [""])[0]
    url_for_parse = "https://" + url if "://" not in url else url
    parsed = urllib.parse.urlparse(url_for_parse)
    if parsed.netloc.endswith("duckduckgo.com"):
        return ""
    return url


def _parse_ddg_results(html: str, max_results: int = 10) -> list[dict[str, str]]:
    results = []
    blocks = re.findall(
        r'<div[^>]*class="[^"]*\bresult\b[^"]*"[^>]*>(.*?)</div>\s*</div>',
        html,
        re.DOTALL,
    )
    for block in blocks[:max_results * 3]:
        title_match = re.search(
            r'(<a[^>]*class="[^"]*result__a[^"]*"[^>]*>)(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not title_match:
            continue
        title = re.sub(r"<[^>]+>", "", title_match.group(2)).strip()
        url = _decode_ddg_url(_extract_attr(title_match.group(1), "href"))
        snippet_match = re.search(
            r'<(?:a|span)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
            block,
            re.DOTALL,
        )
        snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip() if snippet_match else ""
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
    return results


async def _tavily_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    response = await client.post(_TAVILY_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = item.get("content", "").strip()
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _exa_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    # contents.text comes free with the first 10 results per search (Exa's
    # March 2026 pricing), so asking for a short text snippet costs nothing extra.
    payload = {
        "query": query,
        "numResults": max_results,
        "contents": {"text": {"maxCharacters": 300}},
    }
    response = await client.post(
        _EXA_URL,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _ddg_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int,
    language: str = "",
) -> list[dict[str, str]]:
    # Auto-detect Chinese: use cn-zh region for better Chinese results
    kl = language or ""
    if not kl:
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in query)
        if has_cjk:
            kl = "cn-zh"
    response = await client.post(
        _DDG_URL,
        data={"q": query, "b": "", "kl": kl},
        headers=_DDG_HEADERS,
    )
    response.raise_for_status()
    return _parse_ddg_results(response.text, max_results)


def _format_search_results(results: list[dict[str, str]], query: str, source: str) -> str:
    lines = [f"Search results for: {query} [{source}]\n"]
    for i, result in enumerate(results, 1):
        lines.append(f"{i}. **{result['title']}**")
        lines.append(f"   URL: {result['url']}")
        if result["snippet"]:
            lines.append(f"   {result['snippet']}")
        lines.append("")
    return "\n".join(lines)


async def web_search_async(query: str, max_results: int = 8, language: str = "") -> WebToolResult:
    query = query.strip()
    max_results = max(1, min(int(max_results), 20))
    if not query:
        return WebToolResult.error("No search query provided")
    if httpx is None:
        return WebToolResult.error("httpx is required for web_search; install requirements.txt")

    # Keyed providers in fallback order: recurring free quotas first (Tavily
    # 1k/month, then Exa $10/month). DuckDuckGo (free, keyless) is always the
    # final fallback.
    keyed: list[tuple[str, Any, str]] = []
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    exa_key = os.environ.get("EXA_API_KEY", "").strip()
    if tavily_key:
        keyed.append(("Tavily", _tavily_search, tavily_key))
    if exa_key:
        keyed.append(("Exa", _exa_search, exa_key))

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        fail_parts: list[str] = []
        for name, fn, key in keyed:
            try:
                results = await fn(client, key, query, max_results)
            except Exception as exc:  # noqa: BLE001
                fail_parts.append(f"{name}: {exc}")
                continue
            if results:
                source = name
                if fail_parts:
                    source += f" (fallback from: {'; '.join(fail_parts)})"
                return WebToolResult.success(
                    _format_search_results(results, query, source),
                    query=query,
                    result_count=len(results),
                    source=name.lower(),
                )
            fail_parts.append(f"{name}: no results")

        ddg_error: Optional[str] = None
        try:
            results = await _ddg_search(client, query, max_results, language=language)
        except Exception as exc:  # noqa: BLE001
            ddg_error = str(exc)
            results = []

        if results:
            source = "DuckDuckGo"
            if fail_parts:
                source += f" (fallback from: {'; '.join(fail_parts)})"
            return WebToolResult.success(
                _format_search_results(results, query, source),
                query=query,
                result_count=len(results),
                source="duckduckgo",
            )

        search_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"
        parts = list(fail_parts)
        if ddg_error:
            parts.append(f"DuckDuckGo: {ddg_error}")
        else:
            parts.append("DuckDuckGo: no results found")
        reason = "; ".join(parts)
        return WebToolResult.error(
            f"Search failed ({reason})\nManual search URL: {search_url}",
            query=query,
            source="none",
        )


def web_search(query: str, max_results: int = 8, language: str = "") -> WebToolResult:
    return asyncio.run(web_search_async(query, max_results, language=language))


# ---------------------------------------------------------------------------
# SSRF guard: web_fetch must only ever talk to public http(s) addresses
# ---------------------------------------------------------------------------

_MAX_REDIRECTS = 5
# Hard cap on a fetched response body (HTML or PDF). Prevents a hostile or
# broken endpoint from exhausting memory; the tool truncates for display far
# below this anyway.
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0) if httpx is not None else None


class _FetchRefused(Exception):
    """A fetch was refused (SSRF guard, size cap) — surfaced as a tool error."""


def _ip_is_public(ip_str: str) -> bool:
    """True only for globally routable addresses.

    ``ipaddress.is_global`` already refuses loopback (127/8, ::1), private
    ranges (10/8, 172.16/12, 192.168/16, fc00::/7), link-local (169.254/16
    including cloud metadata 169.254.169.254, fe80::/10), CGNAT (100.64/10),
    multicast, reserved and unspecified (0.0.0.0, ::) addresses. IPv4-mapped
    IPv6 forms (e.g. ::ffff:127.0.0.1) are unwrapped and judged as IPv4.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def _validate_fetch_url(url: str) -> Optional[str]:
    """SSRF check for one hop. Returns a refusal message, or None if allowed.

    Only http/https schemes pass. The host is resolved and EVERY returned
    address must be public — this also catches numeric oddities
    (http://2130706433/, http://0x7f000001/) because getaddrinfo applies
    inet_aton semantics to them. Callers must re-validate every redirect hop.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return f"Refused: invalid URL {url!r}"
    if scheme not in ("http", "https"):
        return (f"Refused: only http:// and https:// URLs can be fetched "
                f"(got scheme {scheme!r} or none)")
    if not host:
        return f"Refused: URL has no host: {url!r}"
    # Literal IP host (hostname strips the brackets of [::1] forms).
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not _ip_is_public(host):
            return f"Refused: {host} is not a public IP address"
        return None
    # Hostname: resolve and require every candidate address to be public.
    try:
        infos = socket.getaddrinfo(
            host, port or (443 if scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        return f"Cannot resolve host {host!r}: {exc}"
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return f"Cannot resolve host {host!r}: no addresses"
    for addr in addrs:
        if not _ip_is_public(addr):
            return (f"Refused: host {host!r} resolves to non-public address "
                    f"{addr} (SSRF protection)")
    return None


async def _stream_get(
    client: "httpx.AsyncClient", url: str, max_bytes: int = _MAX_RESPONSE_BYTES,
) -> tuple[int, "httpx.Headers", bytes]:
    """One validated, size-capped GET. Returns (status, headers, body).

    Raises _FetchRefused on SSRF refusal or when the body exceeds the size
    cap. The URL is re-validated immediately before connecting: validation
    and connection both resolve DNS, so a rebind between the two stays a
    race — validating at the last moment shrinks that window as far as
    httpx allows.
    """
    refusal = _validate_fetch_url(url)
    if refusal is not None:
        raise _FetchRefused(refusal)
    async with client.stream("GET", url) as resp:
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes(65536):
            total += len(chunk)
            if total > max_bytes:
                raise _FetchRefused(
                    f"Refused: response from {url} exceeds the "
                    f"{max_bytes // (1024 * 1024)} MB size cap")
            chunks.append(chunk)
        return resp.status_code, resp.headers, b"".join(chunks)


_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


async def _fetch_redirected(
    client: "httpx.AsyncClient", url: str,
) -> tuple[int, "httpx.Headers", bytes, str]:
    """GET ``url`` following redirects MANUALLY (cap ``_MAX_REDIRECTS``), so
    every hop is re-validated by the SSRF guard — httpx's follow_redirects
    would happily land on 169.254.169.254 after one 302.

    Returns (status, headers, body, final_url). Raises _FetchRefused on
    refusal, httpx.TimeoutException / httpx.RequestError on transport failure.
    """
    for _ in range(_MAX_REDIRECTS + 1):
        status, headers, body = await _stream_get(client, url)
        location = headers.get("location")
        if status in _REDIRECT_STATUSES and location:
            url = urllib.parse.urljoin(url, location)
            continue
        return status, headers, body, url
    raise _FetchRefused(f"Refused: too many redirects (>{_MAX_REDIRECTS})")


def _decode_body(headers: "httpx.Headers", body: bytes) -> str:
    ct = headers.get("content-type", "")
    m = re.search(r"charset=([\w.\-]+)", ct, re.IGNORECASE)
    enc = m.group(1) if m else "utf-8"
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


# Local-read allowlist: directories web_fetch may read local files from
# (realpath'd). The ACTIVE data dir is not knowable here — it can be set by
# --data-dir, the settings UI, or ChatService.rebind_store() — so ChatService
# registers its uploads/images dirs on init and on every rebind. Additive by
# design: files uploaded under a previous data dir stay readable after a
# switch (nodes may still reference them).
_LOCAL_READ_ROOTS: set[str] = set()
_LOCAL_READ_ROOTS_LOCK = threading.Lock()


def register_local_read_root(directory: str) -> None:
    """Allow web_fetch / attachment injection to read files under ``directory``.

    Called by ChatService with its uploads/ and images/ dirs. Thread-safe;
    duplicates collapse.
    """
    if not directory:
        return
    with _LOCAL_READ_ROOTS_LOCK:
        _LOCAL_READ_ROOTS.add(os.path.realpath(directory))


def _local_read_roots() -> list[str]:
    """Snapshot of the allowlist. When nothing was registered (CLI, direct
    web_fetch use without a ChatService), fall back to the uploads/images
    dirs of the DEFAULT data dir — the zero-config case."""
    with _LOCAL_READ_ROOTS_LOCK:
        roots = set(_LOCAL_READ_ROOTS)
    if roots:
        return sorted(roots)
    try:
        from store import resolve_db_path  # noqa: PLC0415 - avoid import cycle
        data_dir = os.path.dirname(os.path.abspath(resolve_db_path()))
    except Exception:  # noqa: BLE001
        return []
    return [os.path.realpath(os.path.join(data_dir, sub))
            for sub in ("uploads", "images")]


def local_path_allowed(file_path: str) -> bool:
    """True when a local path may be read: only inside a registered
    uploads/images folder (user attachments). Anything else on the server's
    filesystem (settings.json with the API key, chat history, /etc/…) is
    refused. Symlinks are resolved before the check."""
    real = os.path.realpath(file_path)
    return any(real.startswith(root + os.sep) for root in _local_read_roots())


# ---------------------------------------------------------------------------
# Web fetch
# ---------------------------------------------------------------------------

_CACHE_TTL = 15 * 60
_CACHE_MAX = 50


class _FetchCache:
    """Thread-safe: web_fetch/grep run on concurrent background-turn threads."""

    def __init__(self, maxsize: int = _CACHE_MAX, ttl: float = _CACHE_TTL) -> None:
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._cache:
                return None
            ts, value = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.monotonic(), value)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


_fetch_cache = _FetchCache()

# Store raw text of fetches so grep can search across them.
# {url: full_text}  — guarded by _content_store_lock (concurrent turn threads).
_content_store: dict[str, str] = {}
_content_store_lock = threading.Lock()
_CONTENT_STORE_MAX = 20

# Upper bound on the `max_length` a caller may request for the RETURNED
# (model-visible) text. Raised from 100k to comfortably hold a full paper.
_MAX_RETURN_CHARS = 500_000
# Absolute cap on how much text we extract from a PDF into the grep content
# store, regardless of the caller's display `max_length`. Generous enough for a
# long paper/report; bounds memory for pathological inputs.
_PDF_FULL_TEXT_CAP = 2_000_000


def _register_content(url: str, text: str) -> None:
    """Record a fetched document's text so ``grep_content`` can search it.

    Every successful fetch path (HTML, PDF, local file, arXiv fallback, cache
    hit) must call this — otherwise ``grep`` reports "No content has been
    fetched yet" even though ``web_fetch`` succeeded. Keeps at most
    ``_CONTENT_STORE_MAX`` documents, evicting the oldest.
    """
    if not text:
        return
    with _content_store_lock:
        _content_store[url] = text
        while len(_content_store) > _CONTENT_STORE_MAX:
            _content_store.pop(next(iter(_content_store)))

_FETCH_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _html_to_text(html: str) -> str:
    # Remove non-content elements entirely
    for tag in ("script", "style", "noscript", "meta", "link", "svg",
                "nav", "footer", "header", "aside", "iframe"):
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html,
                      flags=re.DOTALL | re.IGNORECASE)
    # Structural elements → newlines
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|article|section|main|pre|blockquote)[^>]*>",
                  "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", "", html)
    html = html_mod.unescape(html)
    # Collapse whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    lines = [line.strip() for line in html.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


async def _get_http_client() -> httpx.AsyncClient:
    """Deprecated: kept for API compatibility.

    A persistent module-global client cannot be reused across ``asyncio.run()``
    calls (each closes its event loop), which caused "Event loop is closed".
    Callers now create a fresh client per call via ``async with``. This helper
    returns a fresh, unbound client if anything still imports it.
    """
    return httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers=_FETCH_HEADERS,
    )


def _truncate_for_display(full_text: str, max_length: int, url: str) -> str:
    """Truncate the model-visible text, pointing at grep for the rest.

    The COMPLETE document is always registered in ``_content_store`` (see the
    callers), so when we truncate what the model sees, we tell it the remainder
    is retrievable with ``grep`` rather than leaving it to re-fetch in a loop —
    which is exactly what produced the "PDF truncated, let me fetch again"
    behaviour. ``url`` is included so the model can pass it as grep's ``url=``.
    """
    if len(full_text) <= max_length:
        return full_text
    shown = full_text[:max_length]
    return (
        shown
        + f"\n\n[Content truncated for display at {max_length} of {len(full_text)} chars. "
        + "The FULL document is indexed and searchable: call grep with a pattern "
        + f"(optionally url=\"{url}\") to retrieve any later section — e.g. specific "
        + "methods, results, tables, or terms — instead of fetching this URL again.]"
    )


def _read_pdf_bytes(content: bytes, max_chars: int = _PDF_FULL_TEXT_CAP) -> str:
    """Extract text from PDF bytes. Returns empty string on failure.

    Extracts the WHOLE document (every page) up to ``max_chars``, which defaults
    to a generous paper-sized cap. Previously this stopped as soon as the running
    total reached the caller's display ``max_length`` (often 8k–100k), so later
    pages of a long PDF were never read — which both truncated the answer and
    left those pages out of the grep content store. The display-length limit is
    now applied by the caller to the RETURNED text only; the full text is what
    gets registered for grep.
    """
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(pages)
    except Exception:
        return ""


async def web_fetch_async(url: str, max_length: int = 8000, no_cache: bool = False) -> WebToolResult:
    url = url.strip()
    max_length = max(500, min(int(max_length), _MAX_RETURN_CHARS))
    if not url:
        return WebToolResult.error("No URL provided")
    if httpx is None:
        return WebToolResult.error("httpx is required for web_fetch; install requirements.txt")

    # file:// is refused outright — web_fetch must never become a local file
    # read primitive for a prompt-injected page or a misbehaving model.
    if url.lower().startswith("file://"):
        return WebToolResult.error(
            "Refused: file:// URLs are not supported. Only http:// and "
            "https:// URLs (or files uploaded to this chat) can be fetched.",
            url=url,
        )

    # Local file path — allowed ONLY inside the uploads/images folders of the
    # data directory (user attachments the model is told to read). Any other
    # path (/etc/passwd, settings.json with the API key, …) is refused.
    if url.startswith("/") or url.startswith("~/"):
        file_path = os.path.expanduser(url)
        if not local_path_allowed(file_path):
            return WebToolResult.error(
                f"Refused: local file access is restricted to files uploaded "
                f"to this chat (the uploads/images folders of the chatable "
                f"data directory): {file_path}",
                url=url,
            )
        if not os.path.isfile(file_path):
            return WebToolResult.error(f"File not found: {file_path}", url=url)
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
        except OSError as exc:
            return WebToolResult.error(f"Cannot read file: {exc}", url=url)
        # Try PDF extraction — extract the FULL document, register it all for
        # grep, and only truncate the model-visible text.
        pdf_text = _read_pdf_bytes(raw_bytes)
        if pdf_text.strip():
            _register_content(file_path, pdf_text)
            text = _truncate_for_display(pdf_text, max_length, file_path)
            return WebToolResult.success(
                f"Content from {file_path}:\n\n{text}",
                url=file_path, local_file=True,
            )
        # Not a PDF or empty — try as plain text. Register the full decoded
        # text for grep; only truncate what the model sees.
        try:
            full = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            return WebToolResult.error(f"Cannot decode file as text or PDF: {file_path}", url=file_path)
        if not full.strip():
            return WebToolResult.error(f"File is empty: {file_path}", url=file_path)
        _register_content(file_path, full)
        text = _truncate_for_display(full, max_length, file_path)
        return WebToolResult.success(
            f"Content from {file_path}:\n\n{text}",
            url=file_path, local_file=True,
        )

    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    # arXiv PDFs are binary — redirect to the abstract page (or HTML if available)
    # so the fetcher returns readable text instead of raw PDF bytes.
    arxiv_match = re.match(
        r"https?://arxiv\.org/pdf/(\d+\.\d+)(?:v\d+)?(?:\.pdf)?$",
        url,
    )
    if arxiv_match:
        paper_id = arxiv_match.group(1)
        # Try the experimental HTML version first; fall back to abstract page
        url = f"https://arxiv.org/html/{paper_id}"
        print(f"  [web_fetch] arXiv PDF → redirecting to HTML: {url}")

    cache_key = f"{url}\x00{max_length}"
    if not no_cache:
        cached = _fetch_cache.get(cache_key)
        if cached is not None:
            # Re-register so grep works even when the fetch is served from cache.
            # All cached outputs start with a "Content from …:\n\n" header; the
            # searchable document is everything after the first blank line.
            body = cached.split("\n\n", 1)[1] if "\n\n" in cached else cached
            _register_content(url, body)
            return WebToolResult.success(cached, url=url, from_cache=True)

    # Fresh client per call (bound to this asyncio.run loop). Redirects are
    # followed manually so every hop is re-validated by the SSRF guard, and
    # the body is streamed with a hard size cap instead of buffered whole.
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT, follow_redirects=False, headers=_FETCH_HEADERS,
    ) as client:
        try:
            status, resp_headers, raw_bytes, final_url = await _fetch_redirected(client, url)
        except _FetchRefused as exc:
            return WebToolResult.error(str(exc), url=url)
        except httpx.TimeoutException:
            return WebToolResult.error(f"Request timed out fetching {url}", url=url)
        except httpx.RequestError as exc:
            return WebToolResult.error(f"Request error fetching {url}: {exc}", url=url)

        # arXiv HTML version may not exist — fall back to the abstract page.
        if status == 404 and arxiv_match:
            fallback_url = f"https://arxiv.org/abs/{arxiv_match.group(1)}"
            try:
                st2, hd2, body2, fin2 = await _fetch_redirected(client, fallback_url)
            except Exception:  # noqa: BLE001 - fall through to the 404 error below
                pass
            else:
                raw_text2 = _decode_body(hd2, body2)
                text = _html_to_text(raw_text2) if "html" in hd2.get("content-type", "") else raw_text2
                if st2 < 400 and text.strip():
                    # Register full text for grep; truncate only what's shown.
                    _register_content(fin2, text)
                    display = _truncate_for_display(text, max_length, fin2)
                    return WebToolResult.success(
                        f"Content from {fin2}:\n\n{display}",
                        url=fin2, status_code=st2,
                    )
        if status >= 400:
            return WebToolResult.error(
                f"HTTP {status} error fetching {url}",
                url=url, status_code=status,
            )
        if final_url != url:
            print(f"  [web_fetch] followed redirect → {final_url}")
        url = final_url

    content_type = resp_headers.get("content-type", "")
    raw_text = _decode_body(resp_headers, raw_bytes)
    raw_len = len(raw_text)

    # PDF — extract text from binary content
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        # Extract the FULL document, register it all for grep, and only truncate
        # the model-visible text.
        pdf_text = _read_pdf_bytes(raw_bytes)
        images = _extract_pdf_images(raw_bytes) if _images_wanted() else []
        if pdf_text.strip():
            _register_content(url, pdf_text)
            text = _truncate_for_display(pdf_text, max_length, url)
            note = f"\n\n（另抓取 {len(images)} 张图片，已随结果提供给模型）" if images else ""
            output = f"Content from {url} (PDF, {len(raw_bytes)} bytes):\n\n{text}{note}"
            _fetch_cache.set(cache_key, output)
            return WebToolResult.success(
                output, url=url, status_code=status,
                content_type=content_type, is_pdf=True, images=images,
            )
        if images:
            # No extractable text (scanned PDF) but page renders are available.
            output = (
                f"Content from {url} (PDF, {len(raw_bytes)} bytes): no extractable "
                f"text — {len(images)} page image(s) captured and provided to the model."
            )
            return WebToolResult.success(
                output, url=url, status_code=status,
                content_type=content_type, is_pdf=True, images=images,
            )
        return WebToolResult.error(
            f"PDF text extraction failed for {url} ({len(raw_bytes)} bytes). "
            f"The PDF may be image-based or encrypted.",
            url=url, status_code=status, is_pdf=True,
        )

    if "html" in content_type:
        text = _html_to_text(raw_text)
        # If HTML extraction produced very little relative to raw size, the
        # page is likely JS-rendered — include a note and fallback.
        if len(text) < 200 and raw_len > 1000:
            raw_snippet = _html_to_text(re.sub(
                r"<script[^>]*>.*?</script>", " ",
                re.sub(r"<style[^>]*>.*?</style>", " ", raw_text, flags=re.DOTALL),
                flags=re.DOTALL | re.IGNORECASE,
            )).strip()
            if len(raw_snippet) < 80:
                # Truly empty page — likely a SPA or block page
                title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "no title"
                return WebToolResult.error(
                    f"Page contains almost no readable text (likely JS-rendered or blocked). "
                    f"Page title: \"{title}\". Raw HTML size: {raw_len} chars. "
                    f"Try a different URL or use a service that renders JavaScript.",
                    url=url, status_code=status, content_type=content_type,
                    raw_size=raw_len,
                )
            text = raw_snippet + (
                f"\n\n[Note: Full HTML-to-text extraction produced very little output "
                f"({len(text)} chars from {raw_len} raw chars). "
                f"The page may be heavily JS-dependent. Showing best-effort extraction above.]"
            )
    else:
        text = raw_text

    if not text.strip():
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_text, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "unknown"
        return WebToolResult.error(
            f"No readable content found at {url} (page title: \"{title}\", raw size: {raw_len} chars)",
            url=url, status_code=status, content_type=content_type,
            raw_size=raw_len,
        )

    # Register the FULL extracted text for grep, then truncate only what the
    # model sees (with a grep hint pointing at the rest).
    _register_content(url, text)
    display = _truncate_for_display(text, max_length, url)
    images = []
    if _images_wanted() and "html" in content_type:
        images = await _download_images(_extract_image_urls(raw_text, url))
    note = f"\n\n（另抓取 {len(images)} 张图片，已随结果提供给模型）" if images else ""
    output = f"Content from {url}:\n\n{display}{note}"
    _fetch_cache.set(cache_key, output)
    return WebToolResult.success(
        output, url=url, status_code=status,
        content_type=content_type, raw_size=raw_len, images=images,
    )


def web_fetch(url: str, max_length: int = 8000, no_cache: bool = False) -> WebToolResult:
    return asyncio.run(web_fetch_async(url, max_length, no_cache))


def grep_content(pattern: str, url: str = "", max_lines: int = 30, ignore_case: bool = True) -> WebToolResult:
    """Search for *pattern* in previously fetched content.

    - If *url* is given, search only that document.
    - Otherwise, search across all cached fetches.
    Returns matching lines with line numbers and surrounding context.
    """
    if not pattern.strip():
        return WebToolResult.error("No search pattern provided")
    with _content_store_lock:
        # Snapshot under the lock: a concurrent web_fetch on another turn
        # thread may mutate the store while we iterate it below.
        snapshot = dict(_content_store)
    if not snapshot:
        return WebToolResult.error("No content has been fetched yet. Use web_fetch first.")

    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return WebToolResult.error(f"Invalid regex pattern: {exc}")

    targets = {url: snapshot[url]} if url and url in snapshot else snapshot
    if url and url not in snapshot:
        return WebToolResult.error(
            f"URL not in cache: {url}. Available: {', '.join(list(snapshot.keys())[:5])}",
        )

    lines_out: list[str] = []
    total_matches = 0
    for src_url, content in targets.items():
        src_lines = content.splitlines()
        matches: list[int] = []
        for i, line in enumerate(src_lines):
            if regex.search(line):
                matches.append(i)
        if not matches:
            continue

        total_matches += len(matches)
        lines_out.append(f"\n--- {src_url} ({len(matches)} matches) ---")
        # Show matches with context (1 line before, 1 line after)
        shown: set[int] = set()
        for m in matches[:max_lines]:
            for ctx in range(max(0, m - 1), min(len(src_lines), m + 2)):
                if ctx not in shown:
                    prefix = ">" if ctx == m else " "
                    lines_out.append(f"  {prefix} {ctx+1}: {src_lines[ctx]}")
                    shown.add(ctx)
        if len(matches) > max_lines:
            lines_out.append(f"  ... ({len(matches) - max_lines} more matches)")

    if total_matches == 0:
        return WebToolResult.error(f"No matches found for pattern: {pattern}")

    return WebToolResult.success(
        "\n".join(lines_out),
        pattern=pattern, total_matches=total_matches, urls_searched=list(targets.keys()),
    )
