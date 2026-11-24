import pytest

import main


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Never read or write the developer's real settings.json: a persisted
    project directory there would point agent tests at a real repo."""
    monkeypatch.setattr(main, "SETTINGS_PATH", str(tmp_path / "settings.json"))
