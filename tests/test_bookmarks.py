import os
import tempfile
import unittest

from bookmark_service import BookmarkService, _extract_title, _strip_fetch_prefix
from bookmark_store import BookmarkStore
from models import Bookmark


class FakeFetchResult:
    def __init__(self, output: str, ok: bool = True, error: str = ""):
        self.output = output
        self.ok = ok
        self.error = error


def fake_fetcher(output: str, ok: bool = True):
    def _fetch(url):
        return FakeFetchResult(output, ok=ok)
    return _fetch


def fake_summarizer(text: str) -> str:
    return "Summary of: " + text[:40]


class ExtractTitleTests(unittest.TestCase):
    def test_extract_title_from_html(self):
        html = "<html><head><title>My Article Title</title></head><body>text</body></html>"
        self.assertEqual(_extract_title(html), "My Article Title")

    def test_extract_title_prefers_og_title(self):
        html = ('<html><head><title>Site Name | Padded Title</title>'
                '<meta property="og:title" content="Real Article Title" />'
                '</head><body>text</body></html>')
        self.assertEqual(_extract_title(html), "Real Article Title")

    def test_extract_title_from_first_line(self):
        text = "# Markdown Heading\n\nSome body text."
        self.assertEqual(_extract_title(text), "Markdown Heading")

    def test_extract_title_strips_web_fetch_prefix(self):
        text = "Content from https://example.com:\n\nReal Article Title\nSome body."
        self.assertEqual(_extract_title(text), "Real Article Title")

    def test_extract_title_strips_pdf_fetch_prefix(self):
        text = "Content from https://example.com/paper.pdf (PDF, 1234567 bytes):\n\nPaper Title\nAbstract here."
        self.assertEqual(_extract_title(text), "Paper Title")

    def test_extract_title_skips_noise_lines(self):
        text = "Content from https://example.com:\n\n12345\nhttps://other.com\n/section\nActual Title"
        self.assertEqual(_extract_title(text), "Actual Title")

    def test_strip_fetch_prefix(self):
        self.assertEqual(_strip_fetch_prefix("Content from /tmp/file.txt:\n\nbody"), "body")
        self.assertEqual(_strip_fetch_prefix("No prefix here"), "No prefix here")


class BookmarkStoreTests(unittest.TestCase):
    def test_add_list_delete(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            bm = Bookmark.new("https://example.com/article")
            bm.title = "Example"
            store.add(bm)

            self.assertEqual(len(store.list()), 1)
            self.assertEqual(store.get(bm.id).title, "Example")

            self.assertTrue(store.delete(bm.id))
            self.assertEqual(len(store.list()), 0)
            self.assertFalse(store.delete(bm.id))

    def test_search(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            a = Bookmark.new("https://example.com/attention")
            a.title = "Attention Is All You Need"
            b = Bookmark.new("https://example.com/rope")
            b.summary = "Rotary position embeddings explained"
            store.add(a)
            store.add(b)

            self.assertEqual(len(store.search("attention")), 1)
            self.assertEqual(len(store.search("embeddings")), 1)
            self.assertEqual(len(store.search("rope")), 1)


class BookmarkServiceTests(unittest.TestCase):
    def test_add_bookmark_fetches_and_summarizes(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            svc = BookmarkService(
                store=store,
                fetcher=fake_fetcher("<title>Great Article</title><p>Body here.</p>"),
                summarizer=fake_summarizer,
            )

            bm = svc.add_bookmark("example.com/article")

            self.assertEqual(bm.url, "https://example.com/article")
            self.assertEqual(bm.title, "Great Article")
            self.assertTrue(bm.summary.startswith("Summary of:"))
            self.assertIn("Body here", bm.content)

    def test_add_bookmark_rejects_empty_url(self):
        svc = BookmarkService(store=BookmarkStore(db_path="/tmp/dummy.jsonl"))
        with self.assertRaises(ValueError):
            svc.add_bookmark("  ")

    def test_refresh_bookmark_updates_content(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            svc = BookmarkService(
                store=store,
                fetcher=fake_fetcher("old"),
                summarizer=fake_summarizer,
            )
            bm = svc.add_bookmark("example.com/x")
            self.assertIn("old", bm.content)

            svc.fetcher = fake_fetcher("new")
            refreshed = svc.refresh_bookmark(bm.id)
            self.assertIn("new", refreshed.content)

    def test_build_open_prompt_includes_url_title_summary(self):
        bm = Bookmark.new("https://example.com/x")
        bm.title = "Title"
        bm.summary = "Summary"
        svc = BookmarkService(store=BookmarkStore(db_path="/tmp/dummy.jsonl"))
        prompt = svc.build_open_prompt(bm)
        self.assertIn("https://example.com/x", prompt)
        self.assertIn("Title", prompt)
        self.assertIn("Summary", prompt)


class BookmarkRenameTests(unittest.TestCase):
    def test_rename_locks_title_against_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            svc = BookmarkService(
                store=store,
                fetcher=fake_fetcher("<title>Auto Title</title><p>body</p>"),
                summarizer=fake_summarizer,
            )
            bm = svc.add_bookmark("example.com/x")
            self.assertEqual(bm.title, "Auto Title")

            renamed = svc.rename_bookmark(bm.id, "我的收藏")
            self.assertEqual(renamed.title, "我的收藏")
            self.assertTrue(renamed.metadata["title_locked"])

            # Refresh re-extracts content/summary but keeps the manual title.
            refreshed = svc.refresh_bookmark(bm.id)
            self.assertEqual(refreshed.title, "我的收藏")
            self.assertTrue(refreshed.summary.startswith("Summary of:"))

            # Empty rename unlocks; the next refresh re-extracts the title.
            svc.rename_bookmark(bm.id, "  ")
            self.assertFalse(store.get(bm.id).metadata["title_locked"])
            self.assertEqual(svc.refresh_bookmark(bm.id).title, "Auto Title")


if __name__ == "__main__":
    unittest.main()
