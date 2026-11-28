"""DeepSeek Search API provider: result mapping, query/count limits,
key resolution, and fallback through the web_search chain."""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import web_tools


def _item(**overrides):
    base = {
        "rerank_score": 2.4,
        "title": "DeepSeek V4 model released",
        "url": "https://example.com/deepseek-v4",
        "snippet": "short snippet about V4",
        "summary": "long summary meant for models about the V4 release",
        "content": "",
        "site_name": "Example News",
        "site_icon": "https://example.com/favicon.ico",
        "date": "2026-06-01T00:00:00+00:00",
        "authority": 3,
    }
    base.update(overrides)
    return base


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, headers=None, json=None, **_kwargs):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


def _run(coro):
    return asyncio.run(coro)


class DeepSeekMappingTests(unittest.TestCase):
    def test_maps_summary_date_and_site_name(self):
        client = FakeClient(FakeResponse(200, {"results": [_item()]}))
        results = _run(web_tools._deepseek_search(client, "test-key", "DeepSeek V4", 5))
        self.assertEqual(len(results), 1)
        rec = results[0]
        self.assertEqual(rec["title"], "DeepSeek V4 model released")
        self.assertEqual(rec["url"], "https://example.com/deepseek-v4")
        self.assertEqual(rec["snippet"], "long summary meant for models about the V4 release")
        self.assertEqual(rec["site_name"], "Example News")
        self.assertEqual(rec["date"], "2026-06-01T00:00:00+00:00")

        text = web_tools._format_search_results(results, "DeepSeek V4", "DeepSeek")
        self.assertIn("long summary meant for models", text)
        self.assertIn("Example News", text)
        self.assertIn("2026-06-01", text)
        self.assertNotIn("short snippet about V4", text)

        self.assertEqual(client.calls[0]["url"], web_tools._DEEPSEEK_SEARCH_URL)
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer test-key")

    def test_falls_back_to_snippet_when_summary_empty(self):
        item = _item(summary="", snippet="only a snippet")
        client = FakeClient(FakeResponse(200, {"results": [item]}))
        results = _run(web_tools._deepseek_search(client, "k", "q", 5))
        self.assertEqual(results[0]["snippet"], "only a snippet")

    def test_skips_results_without_title_or_url(self):
        client = FakeClient(FakeResponse(200, {"results": [
            _item(title=""),
            _item(url=""),
            _item(title="Keep", url="https://ok.example/x", summary="ok"),
        ]}))
        results = _run(web_tools._deepseek_search(client, "k", "q", 5))
        self.assertEqual([r["title"] for r in results], ["Keep"])

    def test_truncates_query_and_clamps_count(self):
        client = FakeClient(FakeResponse(200, {"results": []}))
        long_q = "α" * 200
        _run(web_tools._deepseek_search(client, "k", long_q, 99))
        payload = client.calls[0]["json"]
        self.assertEqual(payload["count"], 50)
        self.assertEqual(len(payload["query"]), 128)
        self.assertTrue(payload["query"])

        client2 = FakeClient(FakeResponse(200, {"results": []}))
        _run(web_tools._deepseek_search(client2, "k", "q", 0))
        self.assertEqual(client2.calls[0]["json"]["count"], 1)

    def test_passes_ori_question_when_given(self):
        client = FakeClient(FakeResponse(200, {"results": []}))
        _run(web_tools._deepseek_search(
            client, "k", "V4 release", 5, ori_question="  When was DeepSeek V4 released?  ",
        ))
        self.assertEqual(
            client.calls[0]["json"]["ori_question"],
            "When was DeepSeek V4 released?",
        )

    def test_skips_ori_question_when_empty(self):
        client = FakeClient(FakeResponse(200, {"results": []}))
        _run(web_tools._deepseek_search(client, "k", "q", 5, ori_question="  "))
        self.assertNotIn("ori_question", client.calls[0]["json"])

    def test_maps_language_to_lang_operator(self):
        self.assertEqual(web_tools._deepseek_lang_operator("cn-zh"), "lang:zh")
        self.assertEqual(web_tools._deepseek_lang_operator("zh-CN"), "lang:zh")
        self.assertEqual(web_tools._deepseek_lang_operator("en"), "lang:en")
        self.assertEqual(web_tools._deepseek_lang_operator("zh,en"), "lang:zh,en")
        self.assertEqual(web_tools._deepseek_lang_operator("us-en"), "lang:en")
        self.assertEqual(web_tools._deepseek_lang_operator("chinese"), "")
        self.assertEqual(web_tools._deepseek_lang_operator("wt-wt"), "")

        client = FakeClient(FakeResponse(200, {"results": []}))
        _run(web_tools._deepseek_search(client, "k", "DeepSeek V4", 5, language="cn-zh"))
        self.assertEqual(client.calls[0]["json"]["query"], "DeepSeek V4 lang:zh")

    def test_does_not_duplicate_existing_lang_operator(self):
        client = FakeClient(FakeResponse(200, {"results": []}))
        _run(web_tools._deepseek_search(
            client, "k", "DeepSeek lang:en", 5, language="zh",
        ))
        self.assertEqual(client.calls[0]["json"]["query"], "DeepSeek lang:en")

    def test_error_messages_for_401_402_429(self):
        cases = [
            (401, "invalid api key", "unauthorized"),
            (402, "account balance is not enough", "insufficient balance"),
            (429, "rate limit exceeded", "rate limited"),
        ]
        for status, message, hint in cases:
            client = FakeClient(FakeResponse(status, {
                "error": {"message": message, "type": "api_error", "param": None, "code": "api_error"},
            }))
            with self.assertRaises(RuntimeError) as ctx:
                _run(web_tools._deepseek_search(client, "k", "q", 5))
            text = str(ctx.exception)
            self.assertIn(f"HTTP {status}", text)
            self.assertIn(hint, text)
            self.assertIn(message, text)


class DeepSeekKeyTests(unittest.TestCase):
    def test_dedicated_env_wins_over_model_key(self):
        env = {"DEEPSEEK_SEARCH_API_KEY": "search-key", "DEEPSEEK_API_KEY": "env-key"}
        import main
        with patch.dict(os.environ, env, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "model-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.deepseek.com"):
            self.assertEqual(web_tools._deepseek_search_api_key(), "search-key")

    def test_reuses_live_deepseek_model_key(self):
        import main
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "model-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.deepseek.com"):
            self.assertEqual(web_tools._deepseek_search_api_key(), "model-key")

    def test_non_deepseek_backend_uses_env_deepseek_key(self):
        import main
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "openai-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.openai.com/v1"), \
             patch.object(web_tools, "_persisted_settings", lambda: {}):
            self.assertEqual(web_tools._deepseek_search_api_key(), "env-key")

    def test_saved_deepseek_key_used_when_backend_is_not_deepseek(self):
        import main
        saved = {
            "base_url": "https://api.openai.com",
            "api_key": "openai-key",
            "deepseek_api_key": "saved-ds-key",
        }
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "openai-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.openai.com"), \
             patch.object(web_tools, "_persisted_settings", lambda: saved):
            self.assertEqual(web_tools._deepseek_search_api_key(), "saved-ds-key")

    def test_settings_api_key_used_when_saved_backend_is_deepseek(self):
        import main
        saved = {"base_url": "https://api.deepseek.com", "api_key": "settings-ds-key"}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "kimi-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.moonshot.cn/v1"), \
             patch.object(web_tools, "_persisted_settings", lambda: saved):
            # Live backend is Kimi, but settings still hold the DeepSeek key.
            self.assertEqual(web_tools._deepseek_search_api_key(), "settings-ds-key")

    def test_non_deepseek_live_key_is_never_sent_to_deepseek(self):
        import main
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "openai-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.openai.com"), \
             patch.object(web_tools, "_persisted_settings", lambda: {
                 "base_url": "https://api.openai.com", "api_key": "openai-key",
             }):
            self.assertEqual(web_tools._deepseek_search_api_key(), "")

    def test_no_key_when_backend_is_not_deepseek_and_env_empty(self):
        import main
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(main, "OPENAI_API_KEY", "openai-key"), \
             patch.object(main, "OPENAI_BASE_URL", "https://api.openai.com"), \
             patch.object(web_tools, "_persisted_settings", lambda: {}):
            self.assertEqual(web_tools._deepseek_search_api_key(), "")


class DeepSeekChainTests(unittest.TestCase):
    def test_deepseek_used_first_when_key_available(self):
        calls = []

        async def ok_ds(client, key, query, max_results, language="", ori_question=""):
            calls.append("deepseek")
            return [{"title": "T", "url": "https://x.example/", "snippet": "s"}]

        async def fail(*_a, **_k):
            calls.append("other")
            raise RuntimeError("should not be called")

        with patch.dict(os.environ, {"TAVILY_API_KEY": "t", "EXA_API_KEY": "e"}, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: "ds-key"), \
             patch.object(web_tools, "_deepseek_search", ok_ds), \
             patch.object(web_tools, "_tavily_search", fail), \
             patch.object(web_tools, "_exa_search", fail):
            res = web_tools.web_search("q")

        self.assertTrue(res.ok)
        self.assertEqual(calls, ["deepseek"])
        self.assertEqual(res.metadata["source"], "deepseek")
        self.assertIn("[DeepSeek]", res.output)

    def test_falls_back_on_402_to_tavily(self):
        calls = []

        async def fail_ds(client, key, query, max_results, language="", ori_question=""):
            calls.append("deepseek")
            raise RuntimeError("HTTP 402: insufficient balance: account balance is not enough")

        async def ok_tavily(client, key, query, max_results):
            calls.append("tavily")
            return [{"title": "T", "url": "https://x.example/", "snippet": "s"}]

        env = {"TAVILY_API_KEY": "t"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: "ds-key"), \
             patch.object(web_tools, "_deepseek_search", fail_ds), \
             patch.object(web_tools, "_tavily_search", ok_tavily):
            res = web_tools.web_search("q")

        self.assertTrue(res.ok)
        self.assertEqual(calls, ["deepseek", "tavily"])
        self.assertEqual(res.metadata["source"], "tavily")
        self.assertIn("fallback from", res.output)
        self.assertIn("insufficient balance", res.output)

    def test_falls_back_to_ddg_after_keyed_failures(self):
        async def fail_ds(*_a, **_k):
            raise RuntimeError("HTTP 429: rate limited")

        async def fail(*_a, **_k):
            raise RuntimeError("boom")

        async def ok_ddg(client, query, max_results, language=""):
            return [{"title": "T", "url": "https://x.example/", "snippet": "s"}]

        env = {"TAVILY_API_KEY": "t", "EXA_API_KEY": "e"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: "ds-key"), \
             patch.object(web_tools, "_deepseek_search", fail_ds), \
             patch.object(web_tools, "_tavily_search", fail), \
             patch.object(web_tools, "_exa_search", fail), \
             patch.object(web_tools, "_ddg_search", ok_ddg):
            res = web_tools.web_search("q")

        self.assertTrue(res.ok)
        self.assertEqual(res.metadata["source"], "duckduckgo")
        self.assertIn("DeepSeek", res.output)
        self.assertIn("429", res.output)

    def test_provider_env_forces_duckduckgo(self):
        calls = []

        async def fail_ds(*_a, **_k):
            calls.append("deepseek")
            raise RuntimeError("should not be called")

        async def ok_ddg(client, query, max_results, language=""):
            calls.append("ddg")
            return [{"title": "T", "url": "https://x.example/", "snippet": "s"}]

        env = {"CHATTABLE_WEB_SEARCH_PROVIDER": "ddg"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: "ds-key"), \
             patch.object(web_tools, "_deepseek_search", fail_ds), \
             patch.object(web_tools, "_ddg_search", ok_ddg):
            res = web_tools.web_search("q")

        self.assertTrue(res.ok)
        self.assertEqual(calls, ["ddg"])
        self.assertEqual(res.metadata["source"], "duckduckgo")

    def test_provider_env_reports_missing_key(self):
        env = {"CHATTABLE_WEB_SEARCH_PROVIDER": "deepseek"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: ""):
            res = web_tools.web_search("q")
        self.assertFalse(res.ok)
        self.assertIn("DeepSeek: no API key", res.output)

    def test_missing_deepseek_key_logs_once_then_falls_back(self):
        import io
        from contextlib import redirect_stdout

        async def ok_ddg(client, query, max_results, language=""):
            return [{"title": "T", "url": "https://x.example/", "snippet": "s"}]

        web_tools._missing_deepseek_key_warned = False
        buf = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(web_tools, "_deepseek_search_api_key", lambda: ""), \
             patch.object(web_tools, "_ddg_search", ok_ddg), \
             redirect_stdout(buf):
            res1 = web_tools.web_search("q")
            res2 = web_tools.web_search("q")
        self.assertTrue(res1.ok)
        self.assertTrue(res2.ok)
        self.assertEqual(res1.metadata["source"], "duckduckgo")
        log = buf.getvalue()
        self.assertEqual(log.count("[web_search] no DeepSeek API key found"), 1)
        self.assertIn("DEEPSEEK_API_KEY", log)
