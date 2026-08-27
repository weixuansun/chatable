"""Core LLM and context-building code for chatable.

Uses the OpenAI Python SDK against an OpenAI-compatible endpoint. Defaults to
DeepSeek but can be pointed at any compatible provider via env vars:

  DEEPSEEK_API_KEY  — API key (OPENAI_API_KEY fallback)
  DEEPSEEK_BASE_URL — base URL (OPENAI_BASE_URL fallback)
  DEEPSEEK_MODEL    — model name (OPENAI_MODEL fallback)
  DEEPSEEK_REASONING_EFFORT — default "high"
  DEEPSEEK_THINKING_TYPE    — default "enabled"

The CLI uses streaming chat-completions when possible. Reasoning chunks
(DeepSeek's `reasoning_content`) are surfaced separately from final answer
tokens so the web UI can render them differently.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
from copy import deepcopy
from typing import Any

from models import Node


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FROZEN. Do not interpolate. Do not add dates, UUIDs, user context, or any
# per-request data. Keeps the prefix byte-identical across branches so any
# automatic server-side caching has a chance to land.
SYSTEM_PROMPT = (
    "You are a helpful assistant in a tree-structured chat interface for learning "
    "new topics and reading papers. "
    "The user may fork from earlier points to explore papers, math, proofs, "
    "and algorithm ideas without carrying unrelated later context. Use only "
    "the supplied branch path and pinned context. Respond concisely and clearly. "
    "When reading papers, fetch content then grep for specific sections (methods, "
    "results, metrics). "
    "IMPORTANT: When the question involves a technical term, jargon, a newly coined "
    "name, or a model/method/dataset abbreviation or acronym (e.g. 'RoPE', 'GRPO', "
    "'Mamba', 'KV cache', a paper's nickname) that you are not fully certain about, "
    "do NOT answer from memory or guess. First use web_search (and web_fetch on a "
    "primary source) to look it up, then answer grounded in what you found, citing "
    "the source. Prefer verifying over hallucinating a plausible-sounding definition. "
    "CRITICAL — about tool calls: tools are issued through the system's tool-call "
    "mechanism BEFORE you write your reply, never from inside the reply. Your reply "
    "is the final answer shown to the user; it cannot trigger any tool. Therefore "
    "you must NEVER describe, announce, plan, or print a tool call in your reply — "
    "in ANY form. This includes XML/function-call syntax (e.g. <function_calls>), "
    "and ALSO plain-language narration such as '行动 1：再次获取…' or '我先再抓取一次' "
    "followed by a JSON object/array of arguments like [{\"url\": …}] or "
    "{\"query\": …}. None of that runs anything; it only leaks raw machinery to the "
    "user. If, while drafting your answer, you realise you still need more data "
    "(e.g. the fetched content was truncated), do NOT write out the call you wish "
    "you could make. Instead either: (a) answer fully from the content you already "
    "have, or (b) state briefly and in prose what is missing and why (e.g. 'the PDF "
    "was truncated at page 30, so the later sections are not covered'), without "
    "emitting any call syntax or argument JSON. The system runs a separate tool "
    "phase that decides and executes calls for you; trust it and keep your reply "
    "free of tool-call mechanics."
)


# DeepSeek emits its private tool-call markup using fullwidth pipes, e.g.
# `<｜｜DSML｜｜tool_calls> … <｜｜DSML｜｜invoke name="…"> … </｜｜DSML｜｜tool_calls>`
# (the bars are U+FF5C FULLWIDTH VERTICAL LINE, not ASCII '|'). When this leaks
# as plain text, neither the tool-call parser nor the answer scrubber recognise
# it. ``normalize_model_markup`` rewrites the markers to the standard tag forms
# so the existing parse/scrub paths in service.py handle them.
#
# The trailing ``(?=[A-Za-z/])`` lookahead requires the marker to be followed by
# a tag-name character. This matters for streaming: a marker split across deltas
# (``<｜｜DSML｜`` then ``｜tool_calls>``) must NOT be rewritten on the partial
# buffer — without the lookahead the regex would greedily eat the separator pipe
# and corrupt the still-incomplete marker.
_DSML_PREFIX_RE = re.compile(r"<\s*[｜|]+\s*DSML\s*[｜|]+\s*(?=[A-Za-z/])", re.IGNORECASE)
_DSML_CLOSE_PREFIX_RE = re.compile(r"</\s*[｜|]+\s*DSML\s*[｜|]+\s*(?=[A-Za-z])", re.IGNORECASE)


def normalize_model_markup(text: str) -> str:
    """Rewrite DeepSeek DSML tool-call markers to the standard tag forms.

    ``<｜｜DSML｜｜invoke name="x">`` → ``<invoke name="x">``; the closing
    ``</｜｜DSML｜｜tool_calls>`` → ``</tool_calls>``. Leaves all other text
    untouched and is idempotent on text containing no DSML markers.
    """
    if not text or "DSML" not in text:
        return text
    text = _DSML_CLOSE_PREFIX_RE.sub("</", text)
    text = _DSML_PREFIX_RE.sub("<", text)
    return text


# Wrapper tags a model may emit when it "describes" a tool call as plain text
# instead of issuing a real one. These blocks must never reach the answer.
TOOL_CALL_TAG_PAIRS = (
    ("<function_calls>", "</function_calls>"),
    ("<tool_calls>", "</tool_calls>"),
    ("<tool_call>", "</tool_call>"),
    ("<function_call>", "</function_call>"),
)
_TOOL_CALL_BLOCK_RE = re.compile(
    "|".join(
        rf"{re.escape(open_)}.*?{re.escape(close)}"
        for open_, close in TOOL_CALL_TAG_PAIRS
    ),
    flags=re.DOTALL | re.IGNORECASE,
)


def scrub_answer_text(text: str) -> str:
    """Normalise DSML markers then strip any leaked tool-call block.

    Single source of truth for the final-pass cleanup used by service.py.
    """
    return _TOOL_CALL_BLOCK_RE.sub("", normalize_model_markup(text))


# ---- openai mode defaults --------------------------------------------------
OPENAI_API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
OPENAI_BASE_URL = (
    os.environ.get("DEEPSEEK_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api.deepseek.com"
)
OPENAI_MODEL = (
    os.environ.get("DEEPSEEK_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "deepseek-v4-pro"
)
OPENAI_REASONING_EFFORT = (
    os.environ.get("DEEPSEEK_REASONING_EFFORT")
    or os.environ.get("OPENAI_REASONING_EFFORT")
    or "high"
)
OPENAI_THINKING_TYPE = (
    os.environ.get("DEEPSEEK_THINKING_TYPE")
    or os.environ.get("OPENAI_THINKING_TYPE")
    or "enabled"
)

# ---- persisted UI settings -------------------------------------------------
# The web settings panel saves the model config here so a server restart does
# not require re-entering it. Env vars provide the defaults; the persisted
# file, when present, wins (it is the user's most recent explicit choice).
SETTINGS_PATH = os.environ.get(
    "CHATTABLE_SETTINGS",
    os.path.join(
        os.environ.get("CHATTABLE_HOME", os.path.join(os.path.expanduser("~"), ".chatable")),
        "settings.json",
    ),
)


def _load_persisted_config() -> dict:
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def persist_config(data_dir: str = "") -> None:
    """Write the live config to SETTINGS_PATH.

    The file holds the API key, so it is created with 0600 permissions.
    Merges with the existing file: a model-only save keeps the persisted
    storage folder, and a folder-only save keeps the model fields.
    Best-effort: a write failure never breaks the running process.
    """
    data = _load_persisted_config()
    data.update({
        "base_url": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
    })
    if data_dir:
        data["data_dir"] = data_dir
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        fd = os.open(SETTINGS_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def load_persisted_data_dir() -> str:
    """The storage folder persisted by the web settings UI, or "" if none."""
    return str(_load_persisted_config().get("data_dir") or "")


# ---- vision capability probe ------------------------------------------------
# Whether the current backend accepts image inputs. None = not probed yet
# (treated as unsupported: we never fetch images on a guess).
VISION_SUPPORTED: Optional[bool] = None

# 1x1 transparent PNG — enough to make the API parse an image part.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _vision_cache_key() -> str:
    return f"{OPENAI_BASE_URL}|{OPENAI_MODEL}"


def _persist_vision_flag() -> None:
    """Record the probe outcome in settings.json (merged, best-effort)."""
    try:
        data = _load_persisted_config()
        vs = data.get("vision_support")
        if not isinstance(vs, dict):
            vs = {}
        vs[_vision_cache_key()] = VISION_SUPPORTED
        data["vision_support"] = vs
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        fd = os.open(SETTINGS_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def probe_vision_support() -> Optional[bool]:
    """Probe the backend with a 1px image: success ⇒ images accepted.

    A 4xx marks the backend as not vision-capable; network/5xx leaves the flag
    untouched (unknown, retried on the next config change).
    """
    global VISION_SUPPORTED
    client = _get_openai_client()
    try:
        client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Reply with: ok"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_B64}"}},
            ]}],
            stream=False,
            max_tokens=16,
            extra_body={"thinking": {"type": "disabled"}},
        )
        VISION_SUPPORTED = True
    except Exception as e:  # noqa: BLE001 - classify by status when available
        status = getattr(e, "status_code", None)
        if status and 400 <= status < 500:
            VISION_SUPPORTED = False
        # else: unknown — keep the previous value
    if VISION_SUPPORTED is not None:
        _persist_vision_flag()
    return VISION_SUPPORTED


def init_vision_flag() -> None:
    """Resolve VISION_SUPPORTED from the settings cache, else probe in the
    background. Called at startup and after every manual config change."""
    global VISION_SUPPORTED
    if os.environ.get("CHATTABLE_VISION", "1").lower() in ("0", "false", "no"):
        VISION_SUPPORTED = False  # hard kill switch: never probe, never fetch
        return
    cached = _load_persisted_config().get("vision_support")
    if isinstance(cached, dict):
        value = cached.get(_vision_cache_key())
        if isinstance(value, bool):
            VISION_SUPPORTED = value
            return
    VISION_SUPPORTED = None
    threading.Thread(target=probe_vision_support, daemon=True).start()


_saved = _load_persisted_config()
if _saved.get("model"):
    OPENAI_API_KEY = str(_saved.get("api_key") or OPENAI_API_KEY)
    OPENAI_BASE_URL = str(_saved.get("base_url") or OPENAI_BASE_URL).rstrip("/")
    OPENAI_MODEL = str(_saved["model"])

# ---- shared ----------------------------------------------------------------
MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "16384"))
CONTEXT_MAX_PATH_MESSAGES = int(os.environ.get("CHATTABLE_CONTEXT_MAX_PATH_MESSAGES", "80"))
CONTEXT_KEEP_RECENT_MESSAGES = int(os.environ.get("CHATTABLE_CONTEXT_KEEP_RECENT_MESSAGES", "24"))
CACHE_POLICY = os.environ.get("CHATTABLE_CACHE_POLICY", "auto").lower()
CACHE_CONTROL_MAX_BREAKPOINTS = int(os.environ.get("CHATTABLE_CACHE_CONTROL_MAX_BREAKPOINTS", "4"))

# Auto-titling: after a turn completes, a short title can be generated using the
# OpenAI-compatible backend with a dedicated (usually cheaper/faster) model.
# Empty string disables the feature.
SUMMARY_MODEL = os.environ.get("CHATTABLE_SUMMARY_MODEL", "")
SUMMARY_TITLE_MAX_CHARS = int(os.environ.get("CHATTABLE_SUMMARY_TITLE_MAX_CHARS", "48"))


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when a manual runtime config (from the web UI) is invalid."""


def apply_manual_config(
    *,
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> dict:
    """Rebind the LLM backend from explicit fields (not env vars).

    Used by the web UI's settings panel so the model/credentials can be changed
    after launch without env vars. Keeps the existing api_key when a blank one
    is supplied. Clears the cached OpenAI client so a new base_url/key takes
    effect. Persists the result to SETTINGS_PATH so restarts keep it.
    Returns the masked live config.
    """
    global OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, _openai_client

    model = (model or "").strip()
    if not model:
        raise ConfigError("model is required")

    eff_api_key = str(api_key) if api_key else OPENAI_API_KEY
    if not eff_api_key:
        raise ConfigError("api_key is required")

    OPENAI_API_KEY = eff_api_key
    OPENAI_BASE_URL = (base_url or OPENAI_BASE_URL).strip().rstrip("/")
    OPENAI_MODEL = model
    _openai_client = None  # force re-init with the new key/base_url
    persist_config()       # survive server restarts
    init_vision_flag()     # re-probe (or load cached) vision support for the new backend
    return current_config()


def current_config() -> dict:
    """Return the live backend config for the web UI, with secrets masked."""
    def mask(s: str) -> str:
        return "" if not s else "•" * min(8, len(s))
    return {
        "backend": "openai",
        "auth_mode": "api_key",
        "base_url": OPENAI_BASE_URL,
        "model": OPENAI_MODEL,
        "has_api_key": bool(OPENAI_API_KEY),
        "api_key_masked": mask(OPENAI_API_KEY),
    }


def list_models(
    *,
    base_url: str = "",
    api_key: str = "",
) -> list[str]:
    """Return available model ids from the backend's ``/v1/models`` endpoint.

    Powers the settings panel's model dropdown. Uses the given credentials/URL
    when provided (so the UI can preview a list for values being typed), else
    falls back to the live config. Returns a sorted list; empty on any failure
    (the UI keeps free-text entry as a fallback).
    """
    import requests

    bu = (base_url or OPENAI_BASE_URL).strip().rstrip("/")
    key = api_key or OPENAI_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if not bu:
        return []
    try:
        r = requests.get(f"{bu}/v1/models", headers=headers, timeout=20)
        if r.status_code >= 400:
            return []
        data = r.json()
    except Exception:  # noqa: BLE001 - dropdown is best-effort
        return []
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    ids = []
    for it in items:
        mid = it.get("id") if isinstance(it, dict) else (it if isinstance(it, str) else None)
        if mid:
            ids.append(str(mid))
    return sorted(set(ids))


# Lazy-init: only import openai when needed to avoid hard dep
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import httpx  # noqa: PLC0415
        from openai import OpenAI  # noqa: PLC0415
        # Finite read timeout: long enough for slow reasoning streams, but a
        # stalled upstream connection fails loudly instead of hanging forever.
        _openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
    return _openai_client


class StreamTruncatedError(RuntimeError):
    """The upstream closed the SSE stream without finish_reason or usage.

    A relay/proxy dropping the connection mid-answer looks like a clean end of
    stream to the SDK; without this check a truncated reply would be persisted
    as if it were complete.
    """


class SummaryUnavailable(RuntimeError):
    """Raised when auto-title generation cannot run (misconfig, no backend)."""


def _clean_title(text: str, max_chars: int = SUMMARY_TITLE_MAX_CHARS) -> str:
    """Normalise a model reply into a one-line title."""
    text = normalize_model_markup(text or "")
    # First non-empty line, stripped of markdown/quote decoration and quotes.
    line = ""
    for raw in text.splitlines():
        s = raw.strip().lstrip("#>-*• ").strip().strip('"“”\'')
        if s:
            line = s
            break
    line = " ".join(line.split())
    if len(line) > max_chars:
        line = line[: max_chars - 1].rstrip() + "…"
    return line


def summarize_turn(user_text: str, assistant_text: str) -> dict[str, str]:
    """Generate a short title AND a one-sentence summary for one Q/A turn.

    Enabled by default: uses ``CHATTABLE_SUMMARY_MODEL`` when set, otherwise
    the main conversation model (thinking disabled, inputs truncated). Set
    ``CHATTABLE_AUTO_TITLE=0`` to opt out at the service layer.
    Returns ``{"title", "summary"}``; raises ``SummaryUnavailable`` on failure.
    """
    model = SUMMARY_MODEL or OPENAI_MODEL
    client = _get_openai_client()

    # Keep inputs bounded — a gist only needs the essence, not the whole turn.
    u = " ".join((user_text or "").split())[:1500]
    a = " ".join((assistant_text or "").split())[:3000]
    sys_prompt = (
        "Given a Q&A exchange, output exactly two lines:\n"
        "Title: <a single concise title, max ~12 words, capturing the topic>\n"
        "Summary: <one sentence, max ~30 words, capturing the key point of the "
        "answer>\n"
        "Use the same language as the exchange. No quotes, no punctuation at "
        "the end of the title, no bullet points, no extra lines."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Question:\n{u}\n\nAnswer:\n{a}"},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 192,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    try:
        resp = client.chat.completions.create(**kwargs)
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    except Exception as e:  # noqa: BLE001
        raise SummaryUnavailable(f"summary request failed: {e}") from e

    title = ""
    summary = ""
    for raw in extract_text(data).splitlines():
        s = raw.strip()
        if not title and s.lower().startswith("title:"):
            title = _clean_title(s[len("title:"):])
        elif not summary and s.lower().startswith("summary:"):
            summary = s[len("summary:"):].strip()
    if not title or not summary:
        raise SummaryUnavailable("summary model returned an incomplete gist")
    summary = " ".join(summary.split())
    if len(summary) > 120:
        summary = summary[:119].rstrip() + "…"
    return {"title": title, "summary": summary}


def summarize_for_title(user_text: str, assistant_text: str) -> str:
    """Generate a short node title from one Q/A turn using the OpenAI backend.

    Returns a trimmed one-line title, or raises ``SummaryUnavailable`` if the
    backend fails.
    """
    if not SUMMARY_MODEL:
        raise SummaryUnavailable("auto-title disabled (no CHATTABLE_SUMMARY_MODEL)")
    return summarize_turn(user_text, assistant_text)["title"]


def summarize_for_bookmark(text: str, max_chars: int = 160) -> str:
    """Generate a minimal one-sentence summary of an article/blog/paper.

    Uses the currently configured OpenAI backend. Returns a single-sentence
    summary (capped at ``max_chars``) or an empty string if the backend is
    unavailable.
    """
    client = _get_openai_client()
    truncated = " ".join(text.split())[:6000]
    sys_prompt = (
        "Summarize the following article in ONE short sentence. Capture the "
        "core claim or topic and nothing else. Use the same language as the "
        "article. No preamble, no bullet points, no quotes around the output."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": truncated},
    ]
    kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 128,
    }
    try:
        resp = client.chat.completions.create(**kwargs)
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    except Exception:  # noqa: BLE001 - summary is best-effort
        return ""
    summary = extract_text(data).strip()
    summary = " ".join(summary.split())
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def backend_description() -> str:
    extra = []
    if OPENAI_REASONING_EFFORT:
        extra.append(f"effort={OPENAI_REASONING_EFFORT}")
    if OPENAI_THINKING_TYPE:
        extra.append(f"thinking={OPENAI_THINKING_TYPE}")
    suffix = f" [{', '.join(extra)}]" if extra else ""
    return f"backend=openai base_url={OPENAI_BASE_URL} model={OPENAI_MODEL}{suffix}"


# ---------------------------------------------------------------------------
# Message builder (shared)
# ---------------------------------------------------------------------------

def _node_as_message(node: Node) -> dict:
    if node.role == "raw_message":
        # Ephemeral replay node: content is the canonical (sorted-keys) JSON of
        # an exact message dict persisted at turn end. Parse it back so the
        # request carries the native shape (assistant tool_calls / tool role).
        return json.loads(node.content)
    if node.role in ("summary", "system_note"):
        return {"role": "system", "content": node.content}
    return {"role": node.role, "content": node.content}


def _node_excerpt(node: Node, limit: int = 220) -> str:
    if node.role == "raw_message":
        # Replayed tool exchange: summarize instead of dumping the JSON.
        try:
            m = json.loads(node.content)
        except ValueError:
            m = {}
        if m.get("role") == "assistant":
            names = [
                str((c.get("function") or {}).get("name") or "?")
                for c in (m.get("tool_calls") or [])
            ]
            text = "tool call: " + ", ".join(names)
        else:
            text = "tool result: " + str(m.get("content") or "")
    else:
        text = " ".join((node.summary or node.content).split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"- {node.role} {node.id}: {text}"


def _pinned_context_message(pinned_nodes: list[Node]) -> dict | None:
    if not pinned_nodes:
        return None
    lines = [
        "Pinned context. Treat these as stable definitions, claims, excerpts, "
        "or constraints for this branch when relevant:"
    ]
    lines.extend(_node_excerpt(n, limit=600) for n in pinned_nodes)
    return {"role": "system", "content": "\n".join(lines)}


def canonical_messages(messages: list[dict]) -> str:
    """Stable serialization used for prefix-cache diagnostics."""
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def messages_hash(messages: list[dict]) -> str:
    return hashlib.sha256(canonical_messages(messages).encode("utf-8")).hexdigest()


def _last_user_index(messages: list[dict]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _cache_info(messages: list[dict]) -> dict:
    """Describe stable request bytes without assuming provider cache support."""
    latest_user_index = _last_user_index(messages)
    before_latest_user = messages[:latest_user_index] if latest_user_index is not None else messages
    return {
        "policy": CACHE_POLICY,
        "request_hash": messages_hash(messages),
        "request_message_count": len(messages),
        "request_bytes": len(canonical_messages(messages).encode("utf-8")),
        "next_turn_prefix_hash": messages_hash(messages),
        "next_turn_prefix_message_count": len(messages),
        "prefix_before_latest_user_hash": messages_hash(before_latest_user),
        "prefix_before_latest_user_message_count": len(before_latest_user),
        "latest_user_message_index": latest_user_index,
        "explicit_cache_control_indexes": explicit_cache_control_indexes(messages),
    }


def _add_cache_control_to_content(content: Any) -> Any:
    marker = {"type": "ephemeral"}
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": marker}]
    if isinstance(content, list):
        out = deepcopy(content)
        for part in reversed(out):
            if isinstance(part, dict) and part.get("type") == "text":
                part["cache_control"] = marker
                return out
        if out:
            last = out[-1]
            if isinstance(last, dict):
                last["cache_control"] = marker
        return out
    return content


def explicit_cache_control_indexes(messages: list[dict]) -> list[int]:
    """Indexes that would receive Anthropic-style cache_control in explicit mode."""
    if CACHE_POLICY not in ("explicit", "anthropic"):
        return []
    latest_user_index = _last_user_index(messages)
    end = latest_user_index if latest_user_index is not None else len(messages)
    candidates = [
        i for i, msg in enumerate(messages[:end])
        if msg.get("role") == "system" and msg.get("content")
    ]
    max_breakpoints = max(0, min(CACHE_CONTROL_MAX_BREAKPOINTS, 4))
    return candidates[-max_breakpoints:]


def apply_explicit_cache_control(messages: list[dict]) -> list[dict]:
    """Apply Anthropic-style ephemeral cache breakpoints when explicitly enabled.

    OpenAI-compatible proxies vary in whether they pass through extra content
    block fields. The default policy does not call this; users must opt in with
    CHATTABLE_CACHE_POLICY=explicit (or anthropic).
    """
    indexes = set(explicit_cache_control_indexes(messages))
    if not indexes:
        return messages
    out = deepcopy(messages)
    for index in indexes:
        out[index]["content"] = _add_cache_control_to_content(out[index].get("content"))
    return out


def build_context(path: list[Node], pinned_nodes: list[Node] | None = None) -> dict:
    """Build the model context and the UI preview from a branch path."""
    pinned_nodes = pinned_nodes or []
    path_ids = {n.id for n in path}
    pinned_extra = [n for n in pinned_nodes if n.id not in path_ids]
    chat_path = [n for n in path if n.role in ("user", "assistant", "summary", "system_note", "raw_message")]

    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    pinned_msg = _pinned_context_message(pinned_extra)
    if pinned_msg:
        msgs.append(pinned_msg)

    omitted: list[Node] = []
    kept = chat_path
    if len(chat_path) > CONTEXT_MAX_PATH_MESSAGES:
        keep_count = max(1, CONTEXT_KEEP_RECENT_MESSAGES)
        omitted = chat_path[:-keep_count]
        kept = chat_path[-keep_count:]
        # Never let the kept window start mid tool-exchange: a leading replayed
        # role="tool" message without its assistant tool_calls parent makes the
        # API reject the request ("tool_call_id is not found"). Push such
        # orphans back into the omitted excerpt block.
        while kept and kept[0].role == "raw_message":
            try:
                payload = json.loads(kept[0].content)
            except ValueError:
                break
            if payload.get("role") != "tool":
                break
            omitted.append(kept.pop(0))
        msgs.append({
            "role": "system",
            "content": (
                "Earlier branch messages were shortened for context budget. "
                "Use these excerpts only as a rough map; prefer pinned context "
                "and the recent exact messages below.\n" +
                "\n".join(_node_excerpt(n) for n in omitted)
            ),
        })

    msgs.extend(_node_as_message(n) for n in kept)
    estimated_tokens = sum(max(1, len(m["content"]) // 4) for m in msgs)
    return {
        "messages": msgs,
        "pinned_nodes": pinned_extra,
        "omitted_nodes": omitted,
        "kept_nodes": kept,
        "estimated_tokens": estimated_tokens,
        "cache": _cache_info(msgs),
    }


# ---------------------------------------------------------------------------
# Backend: openai SDK
# ---------------------------------------------------------------------------

def _openai_call(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """Call via the OpenAI Python SDK and return an OpenAI-shaped dict."""
    client = _get_openai_client()

    kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": apply_explicit_cache_control(messages),
        "stream": False,
        "max_tokens": MAX_TOKENS,
    }
    if disable_thinking:
        # Non-streaming decision calls skip reasoning to cut latency.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        if OPENAI_REASONING_EFFORT:
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
        if OPENAI_THINKING_TYPE:
            kwargs["extra_body"] = {"thinking": {"type": OPENAI_THINKING_TYPE}}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    resp = client.chat.completions.create(**kwargs)
    # Convert SDK object → plain dict using model_dump (Pydantic v2) / dict()
    try:
        return resp.model_dump()
    except AttributeError:
        return dict(resp)


# ---------------------------------------------------------------------------
# Unified call + extract helpers
# ---------------------------------------------------------------------------

def call_backend(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    return _openai_call(messages, tools=tools, tool_choice=tool_choice, disable_thinking=disable_thinking)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
#
# Protocol: the backend speaks OpenAI-compatible chat-completions SSE. We
# yield `("delta", text_chunk)` for each content chunk and finally
# `("done", usage_dict)` once the stream closes.
#
# Reasoning chunks (DeepSeek's `reasoning_content`) are yielded as
# `("reasoning", text_chunk)` so the UI can render them differently.

def _openai_stream(messages: list[dict], tools: list[dict] | None = None):
    """Stream via the OpenAI SDK.

    Yields ("reasoning", str) / ("delta", str) for text chunks, then — when the
    model requests tools — ("tool_calls", [{id, name, arguments}]) assembled
    from the index-based streaming deltas, and finally ("done", usage_dict).
    """
    client = _get_openai_client()

    kwargs: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": apply_explicit_cache_control(messages),
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "stream_options": {"include_usage": True},
    }
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
    if OPENAI_THINKING_TYPE:
        kwargs["extra_body"] = {"thinking": {"type": OPENAI_THINKING_TYPE}}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_usage_wrapper: dict[str, Any] = {}
    saw_finish = False
    # Streaming tool_calls arrive as index-keyed deltas: the first chunk for an
    # index carries id/name, later chunks carry arguments string fragments.
    tool_parts: dict[int, dict[str, str]] = {}
    stream = client.chat.completions.create(**kwargs)
    try:
        for chunk in stream:
            # chunk is a Pydantic model — convert lazily
            c = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            if c.get("usage"):
                last_usage_wrapper = {"usage": c["usage"]}
                saw_finish = True
            choices = c.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            if choice.get("finish_reason"):
                saw_finish = True
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                part = tool_parts.setdefault(
                    tc.get("index") or 0, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    part["id"] += tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    part["name"] += fn["name"]
                if fn.get("arguments"):
                    part["arguments"] += fn["arguments"]
            rc = delta.get("reasoning_content")
            if rc:
                yield ("reasoning", rc)
            content = delta.get("content")
            if content:
                yield ("delta", content)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    if not saw_finish:
        raise StreamTruncatedError("upstream stream ended without finish_reason (truncated reply)")
    if tool_parts:
        calls = []
        for idx in sorted(tool_parts):
            part = tool_parts[idx]
            call = _normalize_tool_call({
                "id": part["id"],
                "function": {"name": part["name"], "arguments": part["arguments"]},
            })
            if call is not None:
                calls.append(call)
        if calls:
            yield ("tool_calls", calls)
    yield ("done", extract_usage(last_usage_wrapper))


def stream_backend(messages: list[dict], tools: list[dict] | None = None):
    """Yield streaming events; see _openai_stream for the protocol."""
    yield from _openai_stream(messages, tools=tools)


def extract_text(resp: dict) -> str:
    """First assistant choice's message content.

    OpenAI sometimes returns `content` as a list of parts (for multimodal
    replies); handle both string and list.
    """
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return ""


def _normalize_tool_call(raw: dict) -> dict[str, Any] | None:
    """Normalise one OpenAI-shaped tool_call into ``{id, name, arguments}``.

    Shared by the non-streaming response parser and the streaming delta
    accumulator. ``arguments`` is parsed into a dict; unparseable JSON falls
    back to ``{}`` (same historical behaviour).
    """
    if not isinstance(raw, dict):
        return None
    function = raw.get("function") or {}
    name = function.get("name")
    raw_args = function.get("arguments") or "{}"
    if not isinstance(name, str) or not name:
        return None
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(raw_args)
        except (TypeError, json.JSONDecodeError):
            args = {}
    if not isinstance(args, dict):
        return None
    return {"id": raw.get("id") or "", "name": name, "arguments": args}


def extract_tool_calls(resp: dict) -> list[dict[str, Any]]:
    """Extract OpenAI-compatible assistant tool calls as simple dicts."""
    choices = resp.get("choices") or []
    if not choices:
        return []
    msg = (choices[0] or {}).get("message") or {}
    raw_calls = msg.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for raw in raw_calls:
        call = _normalize_tool_call(raw)
        if call is not None:
            out.append(call)
    return out


def extract_usage(resp: dict) -> dict:
    """Map provider-specific usage fields into our schema.

    Cache-hit reporting varies across providers:
      - DeepSeek      : usage.prompt_cache_hit_tokens / prompt_cache_miss_tokens
      - OpenAI/std    : usage.prompt_tokens_details.cached_tokens
      - Anthropic     : usage.cache_read_input_tokens / cache_creation_input_tokens
                        (passed through some Bedrock-compatible proxies)

    We read all of them and surface a single `cache_read` / `cache_creation`
    pair so downstream code doesn't care which provider it is.
    """
    u = resp.get("usage") or {}
    if hasattr(u, "model_dump"):
        u = u.model_dump()
    elif hasattr(u, "__dict__"):
        u = dict(u.__dict__)

    prompt = int(u.get("prompt_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    total = int(u.get("total_tokens") or (prompt + completion))

    # cache read — try each provider's field in turn
    cache_read = int(u.get("prompt_cache_hit_tokens") or 0)  # DeepSeek
    if not cache_read:
        details = u.get("prompt_tokens_details") or {}
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        elif hasattr(details, "__dict__"):
            details = dict(details.__dict__)
        cache_read = int(details.get("cached_tokens") or 0)   # OpenAI
    if not cache_read:
        cache_read = int(u.get("cache_read_input_tokens") or 0)   # Anthropic

    # cache creation — currently Anthropic-only on the wire
    cache_creation = int(u.get("cache_creation_input_tokens") or 0)

    # DeepSeek also reports misses explicitly — useful for sanity checks
    cache_miss = int(u.get("prompt_cache_miss_tokens") or 0)

    return {
        "input": prompt,
        "output": completion,
        "total": total,
        "cache_read": cache_read,
        "cache_creation": cache_creation,
        "cache_miss": cache_miss,
    }
