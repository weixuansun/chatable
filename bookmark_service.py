"""Bookmark business logic: fetch, summarize, and prepare open prompts."""
from __future__ import annotations

import re
import threading
from typing import Callable, Optional

from bookmark_store import BookmarkStore, BookmarkNotFoundError
from models import Bookmark
from web_tools import web_fetch
import main


# Fallback title extraction if web_fetch doesn't yield a usable title.
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# Open Graph / Twitter card titles carry the real article title more often
# than <title> (which sites pad with their brand name).
_META_TAG_RE = re.compile(r"<meta[^>]*>", re.IGNORECASE)
_OG_TITLE_TAG_RE = re.compile(
    r"(?:property|name)=[\"'](?:og:title|twitter:title)[\"']", re.IGNORECASE
)
_CONTENT_ATTR_RE = re.compile(r"content=[\"'](.*?)[\"']", re.IGNORECASE | re.DOTALL)
# Prefix that web_fetch prepends to returned text, e.g.
#   "Content from https://example.com:\n\n"
#   "Content from /path/to/file:\n\n"
#   "Content from https://example.com (PDF, 1234567 bytes):\n\n"
_FETCH_PREFIX_RE = re.compile(
    r"^Content from .+?(?: \(PDF, \d+ bytes\))?:\n\n",
    re.IGNORECASE | re.DOTALL,
)


def _strip_fetch_prefix(text: str) -> str:
    """Remove the 'Content from ...' prefix that web_fetch adds to output."""
    m = _FETCH_PREFIX_RE.match(text)
    return text[m.end():] if m else text


def _extract_title_from_html(html: str) -> str:
    # Prefer og:title / twitter:title (the real blog/paper title) over <title>.
    for tag in _META_TAG_RE.findall(html):
        if _OG_TITLE_TAG_RE.search(tag):
            c = _CONTENT_ATTR_RE.search(tag)
            if c and c.group(1).strip():
                return c.group(1).strip()
    m = _TITLE_RE.search(html)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return ""


def _is_noise_line(s: str) -> bool:
    """True for lines that should not be used as a title."""
    if not s:
        return True
    # Pure numbers (page counts, dates, IDs).
    if re.match(r"^\d+([\.,]\d+)*$", s):
        return True
    # URLs or simple file paths.
    if re.match(r"^https?://", s):
        return True
    if s.startswith("/") and len(s.split()) <= 2:
        return True
    # Section markers / navigation noise.
    if s.lower() in {"home", "menu", "search", "login", "sign up", "skip to content"}:
        return True
    return False


def _extract_title(text: str) -> str:
    """Best-effort title from fetched text or raw HTML."""
    text = _strip_fetch_prefix(text)
    # First try HTML <title> if the text looks like HTML.
    stripped = text.strip()
    if stripped.startswith(("<", "<!DOCTYPE", "<html")):
        title = _extract_title_from_html(stripped)
        if title:
            return title
    # Otherwise use the first meaningful line, cleaned.
    for line in text.splitlines():
        s = line.strip()
        if _is_noise_line(s):
            continue
        s = re.sub(r"^#+\s*", "", s)  # drop markdown heading markers
        return s[:120]
    return ""


class BookmarkService:
    """Coordinates bookmark storage, web fetching, and summarization."""

    def __init__(
        self,
        store: Optional[BookmarkStore] = None,
        fetcher: Optional[Callable[[str], object]] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.store = store or BookmarkStore()
        self.fetcher = fetcher or web_fetch
        self.summarizer = summarizer or main.summarize_for_bookmark

    def list_bookmarks(self) -> list[Bookmark]:
        return self.store.list()

    def get_bookmark(self, bid: str) -> Bookmark:
        return self.store.get(bid)

    def delete_bookmark(self, bid: str) -> bool:
        return self.store.delete(bid)

    def rename_bookmark(self, bid: str, title: str) -> Bookmark:
        """Manually name a bookmark; a non-empty title locks it against the
        auto-extracted title on refresh. Empty title clears the lock."""
        bm = self.store.get(bid)  # raises BookmarkNotFoundError
        title = (title or "").strip()
        metadata = dict(bm.metadata or {})
        metadata["title_locked"] = bool(title)
        return self.store.update(bid, title=title, metadata=metadata)

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            raise ValueError("url is required")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    def _default_title(self, url: str) -> str:
        """Best-effort fallback title when the page hasn't been fetched yet."""
        # Try to use the last path segment or domain.
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            last = path.split("/")[-1].replace("-", " ").replace("_", " ")
            if last:
                return last[:80]
        return parsed.netloc or url

    def add_bookmark_quick(self, url: str) -> Bookmark:
        """Persist a bookmark immediately with only the URL and a placeholder title."""
        url = self._normalize_url(url)
        bm = Bookmark.new(url)
        bm.title = self._default_title(url)
        bm.metadata = {"fetch_ok": False, "enriching": True}
        return self.store.add(bm)

    def enrich_bookmark(self, bid: str) -> Bookmark:
        """Fetch content, extract title, and generate summary for an existing bookmark."""
        bm = self.store.get(bid)
        result = self.fetcher(bm.url)
        content = result.output if hasattr(result, "output") else str(result)
        ok = getattr(result, "ok", True)
        metadata = dict(bm.metadata or {})
        metadata["fetch_ok"] = ok
        metadata["enriching"] = False
        metadata.pop("fetch_error", None)
        metadata.pop("summary_error", None)
        error = getattr(result, "error", "")
        if isinstance(error, str) and error:
            metadata["fetch_error"] = error

        title = bm.title if metadata.get("title_locked") else (_extract_title(content) if ok else bm.title)
        summary = bm.summary
        if ok and content:
            try:
                summary = self.summarizer(content)
            except Exception as exc:  # noqa: BLE001 - summary is best-effort
                metadata["summary_error"] = str(exc)

        return self.store.update(
            bid,
            title=title or bm.title or bm.url,
            summary=summary,
            content=content,
            metadata=metadata,
        )

    def add_bookmark(self, url: str) -> Bookmark:
        """Fetch ``url``, extract a title, summarize, and persist (synchronous)."""
        bm = self.add_bookmark_quick(url)
        return self.enrich_bookmark(bm.id)

    def refresh_bookmark(self, bid: str) -> Bookmark:
        """Re-fetch and re-summarize an existing bookmark."""
        bm = self.store.get(bid)
        result = self.fetcher(bm.url)
        content = result.output if hasattr(result, "output") else str(result)
        ok = getattr(result, "ok", True)
        metadata = dict(bm.metadata or {})
        metadata["fetch_ok"] = ok
        metadata.pop("fetch_error", None)
        metadata.pop("summary_error", None)
        error = getattr(result, "error", "")
        if isinstance(error, str) and error:
            metadata["fetch_error"] = error

        title = bm.title if metadata.get("title_locked") else (_extract_title(content) if ok else bm.title)
        summary = bm.summary
        if ok and content:
            try:
                summary = self.summarizer(content)
            except Exception as exc:  # noqa: BLE001
                metadata["summary_error"] = str(exc)

        return self.store.update(
            bid,
            title=title or bm.title or bm.url,
            summary=summary,
            content=content,
            metadata=metadata,
        )

    def build_open_prompt(self, bm: Bookmark) -> str:
        """Build the user prompt that seeds a chat turn from a bookmark."""
        title = bm.title or bm.url
        summary = bm.summary or "（暂无摘要）"
        lines = [
            f"请详细阅读并分析这篇文章：{bm.url}",
            "",
            f"标题：{title}",
            f"摘要：{summary}",
            "",
            "请对文章进行详细解读。如果需要重新获取完整正文，可以使用 web_fetch；"
            "如果需要定位特定段落，可以使用 grep。",
        ]
        return "\n".join(lines)
