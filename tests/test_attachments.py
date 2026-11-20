"""The local-read allowlist follows the ACTIVE ChatService data dir (not the
default ~/.chatable), and client-supplied attachment paths are confined to it.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import web_tools
from service import ChatService
from store import TreeStore


def _svc(d: str) -> ChatService:
    return ChatService(TreeStore(os.path.join(d, "chatable.jsonl")))


class LocalReadRootsFollowStoreTests(unittest.TestCase):
    def test_non_default_data_dir(self):
        # Regression: with --data-dir X, uploads under X must be readable
        # (the default-dir fallback would point at ~/.chatable instead).
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            self.assertTrue(web_tools.local_path_allowed(
                os.path.join(d, "uploads", "a.txt")))
            self.assertTrue(web_tools.local_path_allowed(
                os.path.join(d, "images", "i.png")))
            # The data dir itself (history file, settings) stays off-limits.
            self.assertFalse(web_tools.local_path_allowed(
                os.path.join(d, "chatable.jsonl")))
            self.assertFalse(web_tools.local_path_allowed("/etc/passwd"))

    def test_rebind_registers_new_dir(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            svc = _svc(d1)
            svc.rebind_store(d2)
            self.assertTrue(web_tools.local_path_allowed(
                os.path.join(d2, "uploads", "a.txt")))
            # The previous dir stays allowed: nodes may still reference
            # files uploaded before the switch.
            self.assertTrue(web_tools.local_path_allowed(
                os.path.join(d1, "uploads", "a.txt")))

    def test_web_fetch_reads_upload_in_non_default_dir(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            desc = svc.save_upload("note.txt", b"uploaded content")
            r = web_tools.web_fetch(desc["path"])
            self.assertTrue(r.ok, r.output)
            self.assertIn("uploaded content", r.output)


class AttachmentPathTests(unittest.TestCase):
    def _start(self, svc: ChatService, attachments) -> dict:
        with patch.object(svc, "_run_turn", lambda *a, **k: None):
            return svc.start_turn("hi", None, attachments=attachments)

    def test_arbitrary_path_silently_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            info = self._start(svc, [
                {"filename": "passwd", "path": "/etc/passwd", "mime": "text/plain"},
            ])
            meta = svc.store.get(info["user_id"]).metadata
            self.assertFalse((meta or {}).get("attachments"))

    def test_traversal_path_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            evil = os.path.join(d, "uploads", "..", "chatable.jsonl")
            info = self._start(svc, [
                {"filename": "x", "path": evil, "mime": "text/plain"},
            ])
            meta = svc.store.get(info["user_id"]).metadata
            self.assertFalse((meta or {}).get("attachments"))

    def test_uploaded_path_kept(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            desc = svc.save_upload("note.txt", b"hello")
            info = self._start(svc, [
                {"filename": "note.txt", "path": desc["path"], "mime": "text/plain"},
            ])
            meta = svc.store.get(info["user_id"]).metadata
            self.assertEqual(meta["attachments"][0]["path"], desc["path"])

    def test_image_data_url_refuses_outside_path(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(d)
            self.assertEqual(svc._image_data_url("/etc/passwd", "text/plain"), "")
            desc = svc.save_upload("pic.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
            self.assertTrue(
                svc._image_data_url(desc["path"], "image/png")
                .startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
