"""settings.json writes: centralized, atomic, lock-serialized (main.py)."""
import json
import os
import tempfile
import threading
import unittest

import main


class SettingsWriteTests(unittest.TestCase):
    def setUp(self):
        self._old_path = main.SETTINGS_PATH
        self._dir = tempfile.mkdtemp()
        main.SETTINGS_PATH = os.path.join(self._dir, "settings.json")

    def tearDown(self):
        main.SETTINGS_PATH = self._old_path

    def test_write_is_atomic_and_0600(self):
        main.persist_config()
        # No temp files left behind, final file parses, mode is 0600.
        self.assertEqual(os.listdir(self._dir), ["settings.json"])
        with open(main.SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["model"], main.OPENAI_MODEL)
        mode = os.stat(main.SETTINGS_PATH).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_read_modify_write_merges(self):
        main.set_reasoning_effort("low")
        main.persist_config(data_dir="/tmp/somewhere")
        with open(main.SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["reasoning_effort"], "low")
        self.assertEqual(data["data_dir"], "/tmp/somewhere")
        self.assertIn("api_key", data)

    def test_concurrent_writers_never_tear_or_lose_fields(self):
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                for n in range(30):
                    main._update_settings(lambda d, i=i, n=n: d.__setitem__(f"w{i}", n))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(60):
                    if os.path.exists(main.SETTINGS_PATH):
                        with open(main.SETTINGS_PATH, encoding="utf-8") as f:
                            json.load(f)  # a torn write would raise here
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        with open(main.SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for i in range(4):
            self.assertEqual(data[f"w{i}"], 29)  # every writer's last value survived


if __name__ == "__main__":
    unittest.main()
