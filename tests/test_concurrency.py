"""Stress the thread-safety of the in-memory stores.

The web server runs generation on background threads while request threads
read the same dicts — these tests hammer exactly those read/write pairs and
fail on 'dictionary changed size during iteration' / lost updates.
"""
import concurrent.futures
import os
import tempfile
import unittest

import web_tools
from bookmark_store import BookmarkStore
from models import Bookmark, Node
from store import ROOT_ID, TreeStore


class TreeStoreConcurrencyTests(unittest.TestCase):
    def test_concurrent_reads_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            store = TreeStore(os.path.join(d, "chatable.jsonl"))
            seed = Node.new("user", "seed", ROOT_ID)
            store.add(seed)
            errors: list[Exception] = []

            def writer(w: int) -> None:
                try:
                    for i in range(40):
                        n = Node.new("user", f"w{w}-{i}", seed.id)
                        store.add(n)
                        store.set_title(n.id, f"title {i}")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            def reader() -> None:
                try:
                    for _ in range(80):
                        store.all_nodes()
                        store.children_map()
                        store.pinned_nodes()
                        store.path_to_root(seed.id)
                        store.schema_info()
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(writer, w) for w in range(4)]
                futs += [pool.submit(reader) for _ in range(4)]
                for f in futs:
                    f.result()
            self.assertEqual(errors, [])
            self.assertEqual(len(store.all_nodes()), 4 * 40 + 1)


class BookmarkStoreConcurrencyTests(unittest.TestCase):
    def test_concurrent_reads_and_writes(self):
        with tempfile.TemporaryDirectory() as d:
            store = BookmarkStore(db_path=os.path.join(d, "bookmarks.jsonl"))
            errors: list[Exception] = []

            def writer(w: int) -> None:
                try:
                    for i in range(30):
                        bm = Bookmark.new(f"https://example.com/{w}/{i}")
                        store.add(bm)
                        store.update(bm.id, title=f"t{i}")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            def reader() -> None:
                try:
                    for _ in range(60):
                        store.list()
                        store.search("example")
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(writer, w) for w in range(4)]
                futs += [pool.submit(reader) for _ in range(4)]
                for f in futs:
                    f.result()
            self.assertEqual(errors, [])
            self.assertEqual(len(store.list()), 4 * 30)


class WebToolsContentStoreConcurrencyTests(unittest.TestCase):
    def test_register_and_grep_concurrently(self):
        errors: list[Exception] = []

        def writer(w: int) -> None:
            try:
                for i in range(50):
                    web_tools._register_content(f"https://x.example/{w}/{i}",
                                                f"body {w} {i} needle")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(50):
                    web_tools.grep_content("needle")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(writer, w) for w in range(4)]
            futs += [pool.submit(reader) for _ in range(4)]
            for f in futs:
                f.result()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
