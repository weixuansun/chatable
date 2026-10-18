"""Print-free orchestration layer for the chatable web UI.

This module lifts the business logic into a terminal-agnostic generator,
``ChatService.send_stream``, that yields structured events instead of writing
to stdout. The web server turns those events into Server-Sent Events.

It reuses the existing modules unchanged:
  - ``store.TreeStore``  — SQLite parent-pointer tree
  - ``main.build_context`` / ``main.stream_backend`` / ``main.call_backend`` …
  - ``tools.TOOL_REGISTRY`` — read-only tool registry

Event shapes yielded by ``send_stream`` (each is a plain dict):
  {"type": "user_node",      "id", "parent_id"}
  {"type": "assistant_node", "id", "parent_id"}
  {"type": "tool",           "name", "arguments", "node_id"}
  {"type": "tool_skipped",   "error"}
  {"type": "assistant_start","id", "context": {...}}
  {"type": "reasoning",      "text"}
  {"type": "delta",          "text"}
  {"type": "error",          "message"}
  {"type": "done",           "id", "usage", "stats", "elapsed"}
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Iterator, Optional

import httpx
from openai import APIConnectionError

import main
import report_store
from main import build_context, call_backend, extract_text, extract_tool_calls, extract_usage, stream_backend
from models import Node
from store import ROOT_ID, TreeStore, resolve_db_path
from tools import TOOL_REGISTRY, ToolResult

from bookmark_service import BookmarkService
from bookmark_store import resolve_bookmarks_path


# Mirror the CLI configuration so behaviour matches across front-ends.
AUTO_TOOLS_ENABLED = os.environ.get("CHATTABLE_AUTO_TOOLS", "1").lower() not in ("0", "false", "no")
AUTO_TOOL_MAX_ROUNDS = int(os.environ.get("CHATTABLE_AUTO_TOOL_MAX_ROUNDS", "10"))
TOOL_DECISION_MODE = os.environ.get("CHATTABLE_TOOL_DECISION_MODE", "native").lower()
# How long a finished turn's event buffer is retained for late reconnects.
RETAIN_FINISHED_TURN_S = float(os.environ.get("CHATTABLE_RETAIN_FINISHED_TURN_S", "20"))
# Auto title+summary per turn is on by default (main model, thinking off);
# CHATTABLE_AUTO_TITLE=0 opts out, CHATTABLE_SUMMARY_MODEL picks a cheaper model.
AUTO_TITLE_ENABLED = os.environ.get("CHATTABLE_AUTO_TITLE", "1").lower() not in ("0", "false", "no")
# Follow-up suggestions (candidate next questions under each reply) are on by
# default, on the same model profile; CHATTABLE_AUTO_SUGGEST=0 opts out.
AUTO_SUGGEST_ENABLED = os.environ.get("CHATTABLE_AUTO_SUGGEST", "1").lower() not in ("0", "false", "no")
# Vision: attach images as inline data URLs to the current turn's user message.
# Disable if the backend rejects image inputs.
VISION_ENABLED = os.environ.get("CHATTABLE_VISION", "1").lower() not in ("0", "false", "no")

# Transient mid-stream failures worth one automatic retry: a cleanly-ended but
# truncated stream, or the upstream/middlebox dropping the connection
# mid-response. Partial text already emitted is discarded via `reset`, so
# retrying is safe.
# Looked up dynamically because tests importlib.reload(main), which re-creates
# StreamTruncatedError — a module-level tuple would go stale.
def _is_transient_stream_error(e: Exception) -> bool:
    return isinstance(e, (main.StreamTruncatedError, httpx.RemoteProtocolError, APIConnectionError))
# Replay past turns' tool exchanges as native OpenAI tool messages (instead of
# folded system text) so the next turn's request shares a byte-identical prefix
# with the previous turn's — the whole investigation stays cache-hittable.
NATIVE_TOOL_HISTORY = os.environ.get("CHATTABLE_NATIVE_TOOL_HISTORY", "1").lower() not in ("0", "false", "no")


def _canon_assistant_tool_msg(content: str, raw_calls: list[dict], reasoning: str = "") -> dict:
    """Assistant message carrying tool_calls, keys in sorted order.

    ``reasoning_content`` must be passed back on later requests in DeepSeek's
    thinking mode (a 400 otherwise), so it is part of the canonical shape and
    of the persisted replay.
    """
    return {
        "content": content,
        "reasoning_content": reasoning,
        "role": "assistant",
        "tool_calls": raw_calls,
    }


def _canon_raw_call(call_id: str, name: str, arguments: dict) -> dict:
    """One tool_calls entry, keys in sorted order throughout."""
    return {
        "function": {
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
            "name": name,
        },
        "id": call_id,
        "type": "function",
    }


def _canon_tool_result_msg(call_id: str, output: str) -> dict:
    """role=tool result message, keys in sorted order."""
    return {"content": output, "role": "tool", "tool_call_id": call_id}


# Stand-in result for a call that was issued in an assistant tool_calls message
# but never executed (stop pressed mid-round). Every id must be answered or the
# exchange is invalid — both for the follow-up request and for the replay
# persisted on the node.
_UNEXECUTED_TOOL_OUTPUT = "[tool call not executed: generation was stopped]"


class _LiveTurn:
    """Server-side buffer for one in-progress assistant turn.

    Generation runs in a background thread that appends events here; HTTP
    subscribers (the initial /api/send and any later /api/stream reconnect after
    a browser refresh) replay the buffered events from an offset and then block
    for new ones until the turn is ``done``. This decouples generation from any
    single client connection, so a refresh no longer aborts the reply.
    """

    def __init__(self, assistant_id: str) -> None:
        self.assistant_id = assistant_id
        self.events: list[dict] = []
        self.done = False
        self.stop_requested = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def append(self, event: dict) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def request_stop(self) -> None:
        with self._cond:
            self.stop_requested = True
            self._cond.notify_all()

    def read_from(self, index: int, timeout: float = 15.0) -> Iterator[dict]:
        """Yield events starting at ``index``, blocking for new ones until done.

        Returns when the turn is finished and all events have been delivered.
        After ``timeout`` idle seconds a ``_ping`` keepalive is yielded — the
        HTTP layer serializes it as an SSE comment, so it never reaches client
        event handling or replay offsets. This keeps the connection warm across
        proxies during long tool/planning phases.
        """
        i = max(0, index)
        while True:
            with self._cond:
                while i >= len(self.events) and not self.done:
                    if not self._cond.wait(timeout=timeout):
                        break  # idle: fall through to emit a keepalive
                idle = i >= len(self.events) and not self.done
                batch = [] if idle else self.events[i:]
                finished = self.done
            if idle:
                yield {"type": "_ping"}
                continue
            for ev in batch:
                yield ev
            i += len(batch)
            if finished and i >= len(self.events):
                return


class _LiveTurns:
    """Registry of in-progress turns, keyed by assistant node id."""

    def __init__(self) -> None:
        self._turns: dict[str, _LiveTurn] = {}
        self._lock = threading.Lock()

    def create(self, assistant_id: str) -> _LiveTurn:
        turn = _LiveTurn(assistant_id)
        with self._lock:
            self._turns[assistant_id] = turn
        return turn

    def get(self, assistant_id: str) -> Optional[_LiveTurn]:
        with self._lock:
            return self._turns.get(assistant_id)

    def remove(self, assistant_id: str) -> None:
        with self._lock:
            self._turns.pop(assistant_id, None)

    def discard(self, key: str, turn: _LiveTurn) -> None:
        """Remove ``key`` only if it still maps to ``turn``.

        For registries whose keys are reused (one report job per conversation):
        a finished job's delayed cleanup must not drop a newer job.
        """
        with self._lock:
            if self._turns.get(key) is turn:
                del self._turns[key]

    def active_ids(self) -> list[str]:
        with self._lock:
            return [tid for tid, t in self._turns.items() if not t.done]


def _heading_text(node: Node) -> str:
    """A node's one-line heading: its title, else the start of its content."""
    text = (node.title or "").strip() or " ".join((node.content or "").split())[:60]
    text = text.lstrip("#").strip()  # never let content inject heading level
    return text or "(untitled)"


def _title_slug(title: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "-", title).strip("-")[:40]


# ---- survey report (综述) helpers -------------------------------------------

# Per-message character caps tried in order when a tree's text is over the
# input budget (None = untouched); the first that fits wins.
_REPORT_NODE_CAPS = (None, 8000, 4000, 2000, 1000, 500)
_REPORT_MAX_SOURCES = 80
_REPORT_FOCUS_MAX_CHARS = 500

# CJK and full-width characters cost about a token each, everything else about
# a token per four characters. (models.estimate_tokens is len//4, which
# under-counts Chinese text several times over.)
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def _estimate_report_tokens(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def _clip_middle(text: str, cap: Optional[int]) -> str:
    """Keep the head and tail of ``text`` within ``cap`` characters."""
    if cap is None or len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return (f"{text[:head].rstrip()}\n\n[……中间省略约 {len(text) - cap} 字……]\n\n"
            f"{text[-tail:].lstrip()}")


def _strip_attachment_preamble(content: str) -> str:
    """Drop the composer's attachment note from a user message.

    The note tells the model to web_fetch server-side paths (see send() in
    static/index.html); mirrors ``_stripAttachmentPreamble`` there.
    """
    text = content or ""
    if text.startswith("Attached image(s)"):
        _, _, text = text.partition("\n")
        text = text.lstrip()
    if text.startswith("Attached file(s)"):
        lines = text.split("\n")
        i = 1
        while i < len(lines) and lines[i].startswith("- "):
            i += 1
        text = "\n".join(lines[i:])
    return text.lstrip()


def _report_fingerprint(nodes: list[Node]) -> str:
    """Identity of a conversation's chat content, to tell a saved report is
    out of date. Covers in-progress replies too (their length changes when
    they finish), so a report written mid-turn goes stale afterwards."""
    h = hashlib.sha1()
    for n in sorted(nodes, key=lambda n: n.id):
        if n.role in ("user", "assistant"):
            h.update(f"{n.id}:{len(n.content or '')}\n".encode())
    return h.hexdigest()


def _fmt_stats(u: dict, elapsed: float) -> str:
    """Format usage stats including cache hit info."""
    inp = u.get("input", 0)
    out = u.get("output", 0)
    total = u.get("total", inp + out)
    cr = u.get("cache_read", 0)
    cw = u.get("cache_creation", 0)
    rate = (cr / inp * 100) if inp > 0 else 0.0
    tok_s = (out / elapsed) if elapsed > 0 else 0.0
    parts = [f"in={inp}", f"out={out}", f"total={total}"]
    if cr or cw:
        parts.append(f"cache_read={cr} ({rate:.0f}%)")
        if cw:
            parts.append(f"cache_write={cw}")
    parts.append(f"{elapsed:.1f}s")
    if tok_s > 0:
        parts.append(f"{tok_s:.0f}tok/s")
    return "[" + " · ".join(parts) + "]"


# Wrapper tags that some models hallucinate into their final answer text when
# they "describe" a tool call instead of issuing a real one. We never want any
# of these blocks to reach the user-visible answer. (open, close) pairs:
_TOOL_CALL_TAG_PAIRS = (
    ("<function_calls>", "</function_calls>"),
    ("<tool_calls>", "</tool_calls>"),
    ("<tool_call>", "</tool_call>"),
    ("<function_call>", "</function_call>"),
)

# DeepSeek emits its private tool-call markup using fullwidth pipes, e.g.
# `<｜｜DSML｜｜tool_calls> … <｜｜DSML｜｜invoke name="…"> … </｜｜DSML｜｜tool_calls>`
# (the bars are U+FF5C FULLWIDTH VERTICAL LINE, not ASCII '|'). When this leaks
# as plain text neither the parser nor the scrubber recognised it. We normalise
# it to the standard tag forms (via main.normalize_model_markup) so the existing
# parse/scrub paths below handle it.
normalize_model_markup = main.normalize_model_markup

_TOOL_CALL_OPEN_TAGS = tuple(open_ for open_, _ in _TOOL_CALL_TAG_PAIRS)
_TOOL_CALL_CLOSE_FOR = {open_: close for open_, close in _TOOL_CALL_TAG_PAIRS}
_TOOL_CALL_MAX_OPEN_LEN = max(len(t) for t in _TOOL_CALL_OPEN_TAGS)

# Final-pass scrub for whatever survives streaming (and for non-streamed text).
_TOOL_CALL_BLOCK_RE = re.compile(
    "|".join(
        rf"{re.escape(open_)}.*?{re.escape(close)}"
        for open_, close in _TOOL_CALL_TAG_PAIRS
    ),
    flags=re.DOTALL | re.IGNORECASE,
)

# An opening tag that never got closed (stream ended mid-markup): drop the
# rest of the text, otherwise the empty-reply check below can't see it.
_TOOL_CALL_OPEN_TAIL_RE = re.compile(
    "|".join(rf"{re.escape(open_)}.*$" for open_, _ in _TOOL_CALL_TAG_PAIRS),
    flags=re.DOTALL | re.IGNORECASE,
)

# Some models dump the raw tool-call JSON object into the answer text instead
# of (or after failing to) issue a real tool call. Fenced ```json variant.
_TOOL_CALL_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_TOOL_CALL_JSON_MARKER_RE = re.compile(r'\{\s*"tool_calls"')


def _parse_tool_call_dump(s: str):
    """Return the parsed object if ``s`` is a {"tool_calls": [...]} dump."""
    s = s.strip()
    if not s.startswith("{") or '"tool_calls"' not in s:
        return None
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        return obj
    return None


def _strip_raw_tool_call_json(text: str) -> str:
    """Remove raw {"tool_calls": …} JSON dumps from answer text.

    The tag-pair scrubber above only handles XML-ish markup; models also leak
    the OpenAI tool_calls JSON shape as plain text (whole message, fenced code
    block, or embedded in prose). Uses the JSON decoder for span detection so
    nested braces don't fool a regex. Best-effort: an answer that legitimately
    quotes such an object (e.g. discussing the API format) loses the quote.
    """
    text = _TOOL_CALL_JSON_FENCE_RE.sub(
        lambda m: "" if _parse_tool_call_dump(m.group(1)) else m.group(0), text)
    if _parse_tool_call_dump(text) is not None:
        return ""
    # Bare dump embedded in prose: decode from each marker occurrence.
    decoder = json.JSONDecoder()
    out: list[str] = []
    i = 0
    for m in _TOOL_CALL_JSON_MARKER_RE.finditer(text):
        j = m.start()
        if j < i:
            continue  # inside an already-removed span
        try:
            obj, end = decoder.raw_decode(text[j:])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            out.append(text[i:j])
            i = j + end
    out.append(text[i:])
    return "".join(out)


def _scrub_answer_text(text: str) -> str:
    """Remove every flavour of hallucinated tool-call markup from answer text:
    DSML markers, XML-ish tag pairs (terminated or not), and raw JSON dumps."""
    text = normalize_model_markup(text)
    text = _TOOL_CALL_BLOCK_RE.sub("", text)
    text = _TOOL_CALL_OPEN_TAIL_RE.sub("", text)
    return _strip_raw_tool_call_json(text)


# Appended once when a turn's visible answer scrubbed to nothing: tell the
# model to stop trying to call tools and just write the answer.
_NO_TOOL_NUDGE = (
    "系统提示：工具调用阶段已结束。不要再输出任何工具调用标记或 "
    "tool_calls JSON；请直接基于已获得的工具结果给出完整的最终回答。"
)


class _ToolCallStripper:
    """Streaming filter that suppresses tool-call XML blocks split across deltas.

    Handles multiple wrapper tag pairs. ``feed`` returns the text safe to emit
    now; while an opening tag has been seen but its matching close has not, the
    remainder is buffered. ``flush`` returns any trailing text that turned out
    not to be a block.
    """

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        # Rewrite any DeepSeek DSML markers to standard tags first, so the
        # block-suppression scan below catches them. Done on the whole buffer
        # because a marker may straddle two deltas.
        self.buf = normalize_model_markup(self.buf)
        out: list[str] = []
        while True:
            # Find the earliest opening tag currently in the buffer.
            idx = -1
            which = None
            for tag in _TOOL_CALL_OPEN_TAGS:
                p = self.buf.find(tag)
                if p != -1 and (idx == -1 or p < idx):
                    idx, which = p, tag
            if idx == -1:
                break
            close = _TOOL_CALL_CLOSE_FOR[which]
            end = self.buf.find(close, idx + len(which))
            if end == -1:
                # Opening tag present, close not yet — emit text before it, hold the rest.
                out.append(self.buf[:idx])
                self.buf = self.buf[idx:]
                return "".join(out)
            # Complete block — drop it and keep scanning.
            out.append(self.buf[:idx])
            self.buf = self.buf[end + len(close):]
        # No opening tag in buffer — emit all but a tail that might begin one.
        tail = self._partial_tail_len()
        if tail:
            out.append(self.buf[:len(self.buf) - tail])
            self.buf = self.buf[-tail:]
        else:
            out.append(self.buf)
            self.buf = ""
        return "".join(out)

    def _partial_tail_len(self) -> int:
        # Hold back a tail that could be the start of a standard open tag…
        for length in range(min(len(self.buf), _TOOL_CALL_MAX_OPEN_LEN - 1), 0, -1):
            suffix = self.buf[-length:]
            if any(tag.startswith(suffix) for tag in _TOOL_CALL_OPEN_TAGS):
                return length
        # …or the start of a DeepSeek DSML marker split across deltas
        # (e.g. "<｜｜DS" arriving before "ML｜｜tool_calls>"). normalize_model_markup
        # only fires once the literal "DSML" is present, so buffer any trailing
        # "<" whose stripped remainder is still a prefix of "DSML".
        lt = self.buf.rfind("<")
        if lt != -1:
            core = re.sub(r"[｜|\s]", "", self.buf[lt + 1:])
            if "dsml".startswith(core.lower()):  # "" counts as a prefix → hold a lone "<"
                return len(self.buf) - lt
        return 0

    def flush(self) -> str:
        if self.buf and not any(tag in self.buf for tag in _TOOL_CALL_OPEN_TAGS):
            out, self.buf = self.buf, ""
            return out
        return ""


class ChatService:
    """Stateless-ish wrapper around a single TreeStore.

    The only mutable session state the web UI needs is "which node is the
    current fork point", but that is best kept in the client and passed in
    explicitly per request, so this service does not hold a selected_id.
    """

    def __init__(
        self,
        store: Optional[TreeStore] = None,
        bookmark_service: Optional[BookmarkService] = None,
    ) -> None:
        self.store = store or TreeStore()
        if bookmark_service is None:
            # Keep bookmarks next to the chat history file.
            from bookmark_store import BookmarkStore
            bm_path = resolve_bookmarks_path(
                data_dir=os.path.dirname(os.path.abspath(self.store.db_path))
            )
            bookmark_service = BookmarkService(store=BookmarkStore(db_path=bm_path))
        self.bookmark_service = bookmark_service
        # Memoized base64 data URLs for image attachments, keyed by file path.
        self._img_data_urls: dict[str, str] = {}
        # In-progress turns, so generation survives a client disconnect/refresh.
        self.live = _LiveTurns()
        # In-progress survey-report jobs, keyed by trunk id. Kept apart from
        # `live`: /api/active lists `live` ids and the UI reconnects to each as
        # a chat turn.
        self.reports = _LiveTurns()
        # Serializes starting a job, saving its report, and deleting the
        # conversation it belongs to.
        self._report_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Uploads (browser drag-and-drop → server-side file the tools can read)
    # ------------------------------------------------------------------ #

    def uploads_dir(self) -> str:
        """Directory holding uploaded files, alongside the SQLite store."""
        base = os.path.dirname(os.path.abspath(self.store.db_path))
        path = os.path.join(base, "uploads")
        os.makedirs(path, exist_ok=True)
        return path

    def save_upload(self, filename: str, data: bytes) -> dict:
        """Persist an uploaded file and return a descriptor the UI can reference.

        The returned ``path`` is a server-side filesystem path that the existing
        ``web_fetch`` tool already knows how to read (PDF text extraction, plain
        text, …) — so uploads need no new parsing logic. Names are sanitised and
        de-duplicated to avoid traversal and collisions.
        """
        safe = os.path.basename(filename or "upload").strip() or "upload"
        # Keep it filesystem-safe: drop path separators / control chars.
        safe = re.sub(r"[^\w.\- ]+", "_", safe).strip(". ") or "upload"
        dest_dir = self.uploads_dir()
        dest = os.path.join(dest_dir, safe)
        stem, ext = os.path.splitext(safe)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{stem}-{n}{ext}")
            n += 1
        with open(dest, "wb") as f:
            f.write(data)
        return {"ok": True, "path": dest, "filename": os.path.basename(dest), "size": len(data)}

    # ------------------------------------------------------------------ #
    # Read helpers (for tree / node / path views)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_tags(node: Node) -> list[str]:
        if isinstance(node.metadata, dict) and isinstance(node.metadata.get("tags"), list):
            return [str(t) for t in node.metadata["tags"]]
        return []

    def _node_summary_dict(self, node: Node, preview_len: int = 160) -> dict:
        text = " ".join((node.title or node.content).split())
        if len(text) > preview_len:
            text = text[: preview_len - 1] + "…"
        summary = " ".join((node.summary or "").split())
        if len(summary) > preview_len:
            summary = summary[: preview_len - 1] + "…"
        return {
            "id": node.id,
            "role": node.role,
            "parent_id": node.parent_id,
            "title": node.title,
            "preview": text,
            "summary": summary,
            "pinned": node.pinned,
            "tags": self._node_tags(node),
            "token_count": node.token_count,
            "created_at": node.created_at,
            "tool": (node.metadata or {}).get("tool") if isinstance(node.metadata, dict) else None,
        }

    def tree(self) -> dict:
        """Return the whole tree as nested JSON rooted at the synthetic ROOT.

        Top-level conversations also carry ``report``: "running" while a survey
        report is being generated, "ready" once one is saved, else None.
        """
        children = self.store.children_map()
        nodes = {n.id: n for n in self.store.all_nodes()}
        reported = report_store.list_ids(self._reports_dir())
        generating = set(self.reports.active_ids())

        def build(nid: str) -> dict | None:
            # children_map() and all_nodes() are separate snapshots; a node can
            # be deleted (e.g. an interrupted turn) between them. Skip any child
            # id no longer present rather than raising KeyError.
            kids = [c for c in (build(cid) for cid in children.get(nid, [])) if c is not None]
            if nid == ROOT_ID:
                return {"id": ROOT_ID, "role": "root", "children": kids}
            node = nodes.get(nid)
            if node is None:
                return None
            data = self._node_summary_dict(node)
            if node.parent_id == ROOT_ID:
                data["report"] = ("running" if nid in generating
                                  else "ready" if nid in reported else None)
            data["children"] = kids
            return data

        return build(ROOT_ID)

    def node_detail(self, nid: str) -> dict:
        node = self.store.get(nid)
        path = self.store.path_to_root(nid)
        return {
            "id": node.id,
            "role": node.role,
            "content": node.content,
            "parent_id": node.parent_id,
            "title": node.title,
            "summary": node.summary,
            "pinned": node.pinned,
            "tags": self._node_tags(node),
            "usage": node.usage,
            "metadata": node.metadata,
            "token_count": node.token_count,
            "created_at": node.created_at,
            "path": [self._node_summary_dict(n) for n in path],
        }

    def path_messages(self, nid: str) -> list[dict]:
        """root→nid full nodes, for rendering a conversation view.

        Assistant nodes expose their folded-in ``tool_results`` so the UI can
        render tool calls inline instead of as separate, navigable nodes.
        """
        if nid == ROOT_ID:
            return []
        out = []
        for n in self.store.path_to_root(nid):
            meta = n.metadata if isinstance(n.metadata, dict) else {}
            tool_results = [
                {k: r.get(k) for k in ("name", "arguments", "ok", "error", "duration_ms", "output")}
                for r in (meta.get("tool_results") or [])
            ]
            out.append({
                "id": n.id,
                "role": n.role,
                "content": n.content,
                "title": n.title,
                "pinned": n.pinned,
                "tags": self._node_tags(n),
                "tool": meta.get("tool"),
                "tool_results": tool_results,
                "reasoning": meta.get("reasoning", ""),
                "attachments": meta.get("attachments") or [],
                "suggestions": meta.get("suggestions") or [],
                "usage": n.usage,
            })
        return out

    def config(self) -> dict:
        return {
            "backend": main.backend_description(),
            "db_path": self.store.db_path,
            "data_dir": os.path.dirname(os.path.abspath(self.store.db_path)),
            "bookmarks_path": self.bookmark_service.store.db_path,
            "auto_tools": AUTO_TOOLS_ENABLED,
            "auto_tool_max_rounds": AUTO_TOOL_MAX_ROUNDS,
            "tool_decision_mode": TOOL_DECISION_MODE,
            "tools": sorted(TOOL_REGISTRY.names()),
            "auto_title": AUTO_TITLE_ENABLED,
            "auto_suggest": AUTO_SUGGEST_ENABLED,
            "vision_supported": main.VISION_SUPPORTED,
            "summary_model": main.SUMMARY_MODEL,
            "llm": main.current_config(),
            "active_streams": len(self.live.active_ids()),
            "bookmarks_count": len(self.bookmark_service.list_bookmarks()),
        }

    def has_active_streams(self) -> bool:
        """True if any reply or survey report is still generating (blocks
        model/store switches)."""
        return bool(self.live.active_ids() or self.reports.active_ids())

    # ------------------------------------------------------------------ #
    # Bookmarks
    # ------------------------------------------------------------------ #

    def list_bookmarks(self) -> list[dict]:
        return [
            {
                "id": bm.id,
                "url": bm.url,
                "title": bm.title,
                "summary": bm.summary,
                "created_at": bm.created_at,
                "updated_at": bm.updated_at,
                "metadata": bm.metadata,
            }
            for bm in self.bookmark_service.list_bookmarks()
        ]

    def add_bookmark(self, url: str) -> dict:
        """Add bookmark immediately, then enrich title/summary in background."""
        bm = self.bookmark_service.add_bookmark_quick(url)
        info = {
            "id": bm.id,
            "url": bm.url,
            "title": bm.title,
            "summary": bm.summary,
            "created_at": bm.created_at,
            "updated_at": bm.updated_at,
            "metadata": bm.metadata,
        }

        def _enrich():
            try:
                self.bookmark_service.enrich_bookmark(bm.id)
            except Exception:  # noqa: BLE001 - background enrichment is best-effort
                pass

        threading.Thread(target=_enrich, daemon=True).start()
        return info

    def refresh_bookmark(self, bid: str) -> dict:
        bm = self.bookmark_service.refresh_bookmark(bid)
        return {
            "id": bm.id,
            "url": bm.url,
            "title": bm.title,
            "summary": bm.summary,
            "created_at": bm.created_at,
            "updated_at": bm.updated_at,
            "metadata": bm.metadata,
        }

    def delete_bookmark(self, bid: str) -> dict:
        return {"ok": self.bookmark_service.delete_bookmark(bid)}

    def rename_bookmark(self, bid: str, title: str) -> dict:
        bm = self.bookmark_service.rename_bookmark(bid, title)
        return {
            "id": bm.id,
            "url": bm.url,
            "title": bm.title,
            "summary": bm.summary,
            "created_at": bm.created_at,
            "updated_at": bm.updated_at,
            "metadata": bm.metadata,
        }

    def open_bookmark(self, bid: str, parent_id: Optional[str] = None) -> dict:
        """Seed a new chat turn from a bookmark and return the new node ids."""
        bm = self.bookmark_service.get_bookmark(bid)
        prompt = self.bookmark_service.build_open_prompt(bm)
        return self.start_turn(prompt, parent_id=parent_id)

    def rebind_store(self, data_dir: str) -> dict:
        """Point the service at a different data folder.

        Refuses while a reply is generating (the background thread holds the old
        store). Returns the new config on success. The path is resolved the same
        way as the --data-dir launch flag.
        """
        if self.has_active_streams():
            raise RuntimeError("cannot switch storage while a reply is generating")
        new_path = resolve_db_path(data_dir)
        # No-op if it's already the active store.
        if os.path.abspath(new_path) != os.path.abspath(self.store.db_path):
            self.store = TreeStore(db_path=new_path)
            from bookmark_store import BookmarkStore
            bookmarks_path = resolve_bookmarks_path(
                data_dir=os.path.dirname(os.path.abspath(new_path))
            )
            self.bookmark_service = BookmarkService(
                store=BookmarkStore(db_path=bookmarks_path),
                fetcher=self.bookmark_service.fetcher,
                summarizer=self.bookmark_service.summarizer,
            )
        return {"ok": True, "db_path": self.store.db_path}

    def generate_title(self, node_id: str, force: bool = False) -> dict:
        """Fill a node's short title AND one-sentence summary from its Q/A turn.

        One ``main.summarize_turn`` call produces both. Title is skipped when
        already set or locked (manual rename) unless ``force=True``; an empty
        summary is always filled. Returns ``{"ok", "id", ...}``.
        """
        node = self.store.get(node_id)  # raises NodeNotFoundError if missing
        if node.role != "assistant":
            return {"ok": False, "id": node_id, "skipped": "not an assistant node"}
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        title_locked = bool(meta.get("title_locked"))
        need_title = force or (not node.title and not title_locked)
        need_summary = force or not node.summary
        if not need_title and not need_summary:
            return {"ok": True, "id": node_id, "title": node.title, "summary": node.summary,
                    "skipped": "already titled"}

        # The user turn that prompted this reply is the assistant node's parent.
        user_text = ""
        if node.parent_id and self.store.exists(node.parent_id):
            parent = self.store.get(node.parent_id)
            if parent.role == "user":
                user_text = parent.content

        try:
            gist = main.summarize_turn(user_text, node.content)
        except main.SummaryUnavailable as e:
            return {"ok": False, "id": node_id, "error": str(e)}
        except Exception as e:  # noqa: BLE001 - never break the turn over a title
            return {"ok": False, "id": node_id, "error": f"{type(e).__name__}: {e}"}

        if need_title:
            self.store.set_title(node_id, gist["title"])
        if need_summary:
            self._set_node_summary(node_id, gist["summary"])
        fresh = self.store.get(node_id)
        return {"ok": True, "id": node_id, "title": fresh.title, "summary": fresh.summary}

    def _set_node_summary(self, node_id: str, summary: str) -> None:
        """Write node.summary and flush — in one locked store write, so it
        can't undo a metadata write (follow-up suggestions) landing meanwhile."""
        self.store.set_summary(node_id, summary)

    # Context budget for follow-up suggestions: the latest turn nearly whole,
    # earlier turns as short excerpts.
    _SUGGEST_EARLIER_NODES = 8
    _SUGGEST_EARLIER_CHARS = {"user": 300, "assistant": 500}
    _SUGGEST_LAST_CHARS = {"user": 2000, "assistant": 6000}

    def _suggest_context(self, node_id: str) -> str:
        """The conversation up to an assistant reply, rendered for the
        follow-up-suggestion call."""
        path = [n for n in self.store.path_to_root(node_id) if n.role in ("user", "assistant")]
        reply = path.pop()
        question = path.pop() if path and path[-1].role == "user" else None

        def said(n: Node) -> str:
            text = _strip_attachment_preamble(n.content) if n.role == "user" else n.content
            return (text or "").strip()

        def who(n: Node) -> str:
            return "用户" if n.role == "user" else "助手"

        lines: list[str] = []
        earlier = path[-self._SUGGEST_EARLIER_NODES:]
        if earlier:
            lines.append("【较早的对话】（只保留每条的开头）")
            for n in earlier:
                text = " ".join(said(n).split())
                cap = self._SUGGEST_EARLIER_CHARS[n.role]
                lines.append(f"{who(n)}：{text[:cap]}{'…' if len(text) > cap else ''}")
            lines.append("")
        lines.append("【最后一轮】")
        if question is not None:
            lines.append(f"用户：{_clip_middle(said(question), self._SUGGEST_LAST_CHARS['user'])}")
        lines.append(f"助手：{_clip_middle(said(reply), self._SUGGEST_LAST_CHARS['assistant'])}")
        return "\n".join(lines)

    def generate_suggestions(self, node_id: str, force: bool = False) -> dict:
        """Candidate next questions for an assistant reply, saved on the node
        (``metadata["suggestions"]``) and shown as buttons under it.

        Called by the UI after a turn completes, like auto-title. Keeps
        existing suggestions unless ``force=True``. Never raises over an LLM
        failure: returns ``{"ok": False, "error"}`` instead.
        """
        node = self.store.get(node_id)  # raises NodeNotFoundError if missing
        if node.role != "assistant":
            return {"ok": False, "id": node_id, "skipped": "not an assistant node"}
        text = (node.content or "").strip()
        if not text or text.startswith(("[error:", "[empty reply:")) \
                or node_id in self.live.active_ids():
            return {"ok": False, "id": node_id, "skipped": "no finished answer to follow up on"}
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        if meta.get("suggestions") and not force:
            return {"ok": True, "id": node_id, "suggestions": meta["suggestions"],
                    "skipped": "already suggested"}
        try:
            questions = main.suggest_followups(self._suggest_context(node_id))
        except main.SummaryUnavailable as e:
            return {"ok": False, "id": node_id, "error": str(e)}
        except Exception as e:  # noqa: BLE001 - suggestions are best-effort
            return {"ok": False, "id": node_id, "error": f"{type(e).__name__}: {e}"}
        if not self.store.exists(node_id):  # deleted while the model was thinking
            return {"ok": False, "id": node_id, "skipped": "node deleted"}
        self.store.update_metadata(node_id, {"suggestions": questions})
        return {"ok": True, "id": node_id, "suggestions": questions}

    def rename_node(self, node_id: str, title: str) -> dict:
        """Manually name a user/assistant node; locks it against auto-titling.

        An empty title clears the manual name and the lock, restoring the
        automatic title flow.
        """
        node = self.store.get(node_id)  # raises NodeNotFoundError
        if node.role not in ("user", "assistant"):
            raise ValueError(f"cannot rename node of role: {node.role}")
        title = (title or "").strip()
        self.store.set_title(node_id, title)
        self.store.update_metadata(node_id, {"title_locked": bool(title)})
        return {"ok": True, "id": node_id, "title": title}

    def delete_subtree(self, nid: str) -> int:
        """Delete a node and its descendants (used to discard an interrupted turn).

        Deleting a whole conversation also drops its survey report and stops a
        report job still running for it.
        """
        if not self.store.exists(nid):
            return 0
        if self.store.get(nid).parent_id != ROOT_ID:
            return self.store.delete_subtree(nid)
        with self._report_lock:
            job = self.reports.get(nid)
            if job is not None:
                job.request_stop()
            removed = self.store.delete_subtree(nid)
            report_store.delete(self._reports_dir(), nid)
        return removed

    def export_tree_markdown(self, trunk_id: str) -> tuple[str, str]:
        """Export one whole conversation tree (a trunk and all its branches)
        as a readable Markdown document. Returns (filename, content).

        Depth-first, hierarchical sections: each user node opens a section
        whose heading level grows with depth, so forks become sibling
        sub-sections; assistant replies follow as raw Markdown content, so
        code fences, LaTeX math and blockquotes pass through untouched.
        Tool records and reasoning are intentionally omitted.

        Raises ValueError if ``trunk_id`` is not a top-level trunk node.
        """
        trunk = self.store.get(trunk_id)  # raises NodeNotFoundError if missing
        if trunk.parent_id != ROOT_ID:
            raise ValueError(f"not a trunk node: {trunk_id}")

        children = self.store.children_map()

        def kids_of(nid: str) -> list[str]:
            # children_map is newest-first (sidebar order); export chronologically.
            return sorted(children.get(nid, []),
                          key=lambda i: self.store.get(i).created_at)

        lines: list[str] = []
        count = 0

        def emit(nid: str, depth: int) -> None:
            nonlocal count
            node = self.store.get(nid)
            if node.role not in ("user", "assistant"):
                for kid in kids_of(nid):  # summary/system_note: skip, keep descending
                    emit(kid, depth)
                return
            count += 1
            if node.role == "assistant":
                lines.extend([node.content, ""])
                for kid in kids_of(nid):
                    emit(kid, depth + 1)  # user forks become sections one level down
                return
            # user node: new section at this depth
            level = min(depth + 2, 6)
            lines.extend(["#" * level + " " + _heading_text(node), "", node.content, ""])
            kids = kids_of(nid)
            forks = [k for k in kids if self.store.get(k).role != "assistant"]
            replies = [k for k in kids if self.store.get(k).role == "assistant"]
            for i, rid in enumerate(replies):
                if i > 0:
                    lines.extend(["*(regenerated answer)*", ""])
                emit(rid, depth)  # reply body stays in this section
            for fid in forks:
                emit(fid, depth + 1)

        emit(trunk_id, 0)

        title = _heading_text(trunk)
        ts = time.strftime("%Y-%m-%d %H:%M")
        doc = [f"# {title}", "", f"> Exported {ts} · {count} messages", ""] + lines
        fname = f"chatable-{_title_slug(title) or trunk_id}-{time.strftime('%Y%m%d-%H%M%S')}.md"
        return fname, "\n".join(doc).rstrip() + "\n"

    # ------------------------------------------------------------------ #
    # Survey report (综述): one LLM pass over a whole conversation tree
    # ------------------------------------------------------------------ #
    #
    # The report is a standalone document (report_store), not a tree node: it
    # never enters any chat context. Generation runs as a background job with
    # the same event-buffer machinery as a chat turn, in its own registry keyed
    # by trunk id — /api/active must keep listing chat turns only.

    def _reports_dir(self) -> str:
        return report_store.reports_dir(os.path.dirname(os.path.abspath(self.store.db_path)))

    def _require_trunk(self, trunk_id: str) -> Node:
        trunk = self.store.get(trunk_id)  # raises NodeNotFoundError if missing
        if trunk.parent_id != ROOT_ID:
            raise ValueError(f"not a trunk node: {trunk_id}")
        return trunk

    @staticmethod
    def _report_title(trunk: Node) -> str:
        if (trunk.title or "").strip():
            return _heading_text(trunk)
        text = " ".join(_strip_attachment_preamble(trunk.content).split())[:60]
        return text.lstrip("#").strip() or "(untitled)"

    def _report_turns(self, trunk_id: str, nodes: list[Node]) -> list[dict]:
        """Group one conversation's nodes into question/answer turns.

        A turn is a user message plus its assistant replies (a second reply is
        a regenerated answer). Turns come back depth-first — each branch is
        followed to its end before the next sibling — labelled T1…Tn, each
        knowing which turn it follows and which of that turn's branches it is.
        Iterative, so a very long chat can't hit the recursion limit.
        """
        by_id = {n.id: n for n in nodes}
        kids: dict[str, list[Node]] = {}
        for n in nodes:  # oldest first, so every child list is chronological
            if n.parent_id in by_id:
                kids.setdefault(n.parent_id, []).append(n)

        def chat_kids(nid: str) -> list[Node]:
            # summary/system_note nodes are transparent: their children count
            # as children of the nearest chat node above.
            out: list[Node] = []
            pending = list(reversed(kids.get(nid, [])))
            while pending:
                n = pending.pop()
                if n.role in ("user", "assistant"):
                    out.append(n)
                else:
                    pending.extend(reversed(kids.get(n.id, [])))
            return out

        generating = set(self.live.active_ids())

        def usable(reply: Node) -> bool:
            # Skip replies with nothing to report: still generating, empty, or
            # only a failure marker written by the turn loop.
            text = (reply.content or "").strip()
            return bool(text) and reply.id not in generating \
                and not text.startswith(("[error:", "[empty reply:"))

        turns: list[dict] = []
        stack: list[dict] = [{"head": by_id[trunk_id], "parent": "", "parent_replies": 0,
                              "after": 0, "branch": 0, "branches": 0, "indent": 0}]
        while stack:
            item = stack.pop()
            head: Node = item["head"]
            # A turn normally opens with a user message. An assistant head (an
            # orphan repaired onto ROOT, or a reply under a reply) is a turn
            # with no question.
            user = head if head.role == "user" else None
            candidates = [k for k in chat_kids(head.id) if k.role == "assistant"] if user else [head]
            replies: list[Node] = []
            follow_ups: list[tuple[Node, int]] = []  # (head of next turn, reply no. it follows)
            for reply in candidates:
                no = 0
                if usable(reply):
                    replies.append(reply)
                    no = len(replies)
                follow_ups.extend((k, no) for k in chat_kids(reply.id))
            if user:
                follow_ups.extend((k, 0) for k in chat_kids(head.id) if k.role == "user")
            follow_ups.sort(key=lambda f: f[0].created_at)

            label = f"T{len(turns) + 1}"
            turns.append({
                "label": label, "user": user, "replies": replies,
                "parent": item["parent"], "parent_replies": item["parent_replies"],
                "after": item["after"], "branch": item["branch"],
                "branches": item["branches"], "indent": item["indent"],
            })
            indent = item["indent"] + (1 if len(follow_ups) > 1 else 0)
            for i in range(len(follow_ups), 0, -1):  # reversed: pop() takes the oldest first
                kid, after = follow_ups[i - 1]
                stack.append({"head": kid, "parent": label, "parent_replies": len(replies),
                              "after": after, "branch": i, "branches": len(follow_ups),
                              "indent": indent})
        return turns

    @staticmethod
    def _report_sources(turns: list[dict]) -> list[dict]:
        """Web pages the assistant actually fetched, in first-seen order.

        Only http(s) URLs: web_fetch also reads uploaded files by local path.
        """
        seen: dict[str, str] = {}
        for turn in turns:
            for reply in turn["replies"]:
                meta = reply.metadata if isinstance(reply.metadata, dict) else {}
                for rec in meta.get("tool_results") or []:
                    if not isinstance(rec, dict) or rec.get("name") != "web_fetch" or not rec.get("ok"):
                        continue
                    url = str((rec.get("arguments") or {}).get("url") or "").strip()
                    if url.startswith(("http://", "https://")) and url not in seen:
                        seen[url] = turn["label"]
        return [{"url": u, "turn": t} for u, t in list(seen.items())[:_REPORT_MAX_SOURCES]]

    @staticmethod
    def _render_report_source(turns: list[dict], sources: list[dict],
                              cap: Optional[int]) -> tuple[str, int]:
        """Render the turns as the model's input. Returns (text, clipped count).

        Structure is carried by explicit turn labels rather than heading depth,
        which flattens after a few levels. Tool output, reasoning and image
        data are never included — only what the user and assistant wrote.
        """
        clipped = 0

        def body(text: str) -> str:
            nonlocal clipped
            out = _clip_middle(text.strip(), cap)
            clipped += out != text.strip()
            return out

        def relation(turn: dict) -> list[str]:
            parts: list[str] = []
            if turn["parent"]:
                after = f"承接 {turn['parent']}"
                if turn["parent_replies"] > 1 and turn["after"]:
                    after += f" 的第 {turn['after']} 版回答"
                parts.append(after)
            if turn["branches"] > 1:
                parts.append(f"分支 {turn['branch']}/{turn['branches']}")
            return parts

        def question(turn: dict) -> str:
            user: Optional[Node] = turn["user"]
            if user is None:
                return ""
            text = _strip_attachment_preamble(user.content)
            meta = user.metadata if isinstance(user.metadata, dict) else {}
            names = [str(a.get("filename") or "") for a in meta.get("attachments") or []
                     if isinstance(a, dict)]
            names = [n for n in names if n]
            if names:
                text = f"[附件: {', '.join(names)}]\n{text}"
            return text.strip()

        def title_of(turn: dict) -> str:
            user: Optional[Node] = turn["user"]
            title = (user.title if user else "") or next(
                (r.title for r in turn["replies"] if r.title), "")
            title = " ".join((title or question(turn)).split())[:60]
            return title or "(无提问)"

        forks = sum(1 for t in turns if t["branch"] == 1 and t["branches"] > 1)
        lines = [
            f"【对话树结构】共 {len(turns)} 个回合，{forks} 处分叉。"
            "每进入一处分叉缩进加深一级；回合之间的承接关系以括号里的说明为准。",
        ]
        for turn in turns:
            rel = "，".join(relation(turn))
            lines.append(f"{'  ' * turn['indent']}- {turn['label']}"
                         f"{'（' + rel + '）' if rel else ''}：{title_of(turn)}")
        lines += ["", "【对话内容】"]
        for turn in turns:
            lines += ["", "=== " + " ｜ ".join([turn["label"]] + relation(turn)) + " ==="]
            q = question(turn)
            if q:
                lines += ["【用户】", body(q)]
            if not turn["replies"]:
                lines += ["【助手】", "（这个提问没有得到有效回答）"]
            for i, reply in enumerate(turn["replies"], 1):
                lines.append("【助手】" if i == 1 else f"【助手 · 重新生成的第 {i} 版回答】")
                lines.append(body(reply.content))
        if sources:
            lines += ["", "【对话中读取过的网页】"]
            lines += [f"{i}. {s['url']}（{s['turn']}）" for i, s in enumerate(sources, 1)]
        return "\n".join(lines) + "\n", clipped

    def _report_source(self, trunk_id: str) -> dict:
        """Everything one report run needs, from a consistent snapshot of the
        tree: the rendered input (fitted to the input budget), counts, sources
        and the tree's fingerprint.

        Raises ValueError when there is nothing to report on, or the tree is
        too large even at the tightest per-message cap.
        """
        trunk = self._require_trunk(trunk_id)
        nodes = self.store.trunk_nodes(trunk_id)
        turns = self._report_turns(trunk_id, nodes)
        if not any(t["replies"] for t in turns):
            raise ValueError("这个对话还没有可以综述的回答")
        sources = self._report_sources(turns)
        budget = main.REPORT_MAX_INPUT_TOKENS
        overhead = _estimate_report_tokens(main.REPORT_SYSTEM_PROMPT) + 256
        estimated = 0
        for cap in _REPORT_NODE_CAPS:
            text, clipped = self._render_report_source(turns, sources, cap)
            estimated = _estimate_report_tokens(text) + overhead
            if estimated <= budget:
                return {
                    "text": text,
                    "title": self._report_title(trunk),
                    "turns": len(turns),
                    "messages": sum((1 if t["user"] else 0) + len(t["replies"]) for t in turns),
                    "truncated_nodes": clipped,
                    "sources": [s["url"] for s in sources],
                    "estimated_tokens": estimated,
                    "fingerprint": _report_fingerprint(nodes),
                }
        raise ValueError(
            f"对话过大：压缩后仍约 {estimated} tokens，超过综述的输入上限 {budget} tokens"
            "（可用环境变量 CHATTABLE_REPORT_MAX_INPUT_TOKENS 调整）")

    def get_report(self, trunk_id: str) -> dict:
        """The saved report for a conversation, plus whether one is being
        generated and whether the tree has changed since it was written."""
        trunk = self._require_trunk(trunk_id)
        saved = report_store.get(self._reports_dir(), trunk_id)
        job = self.reports.get(trunk_id)
        stale = bool(saved) and saved.get("fingerprint") != _report_fingerprint(
            self.store.trunk_nodes(trunk_id))
        return {
            "trunk_id": trunk_id,
            "title": self._report_title(trunk),
            "running": job is not None and not job.done,
            "stale": stale,
            "report": saved,
        }

    def start_report(self, trunk_id: str, focus: str = "") -> dict:
        """Start generating a report in the background.

        If one is already being generated for this conversation, attaches to it
        instead (``started`` False; the new ``focus`` is ignored). Raises
        NodeNotFoundError / ValueError before anything starts.
        """
        self._require_trunk(trunk_id)
        focus = (focus or "").strip()[:_REPORT_FOCUS_MAX_CHARS]
        with self._report_lock:
            running = self.reports.get(trunk_id)
            if running is not None and not running.done:
                return {"trunk_id": trunk_id, "started": False}
            source = self._report_source(trunk_id)
            job = self.reports.create(trunk_id)
        thread = threading.Thread(
            target=self._run_report, args=(job, trunk_id, source, focus), daemon=True)
        thread.start()
        return {"trunk_id": trunk_id, "started": True}

    def subscribe_report(self, trunk_id: str, from_index: int = 0) -> Optional[Iterator[dict]]:
        """Events of a report job, or None if none is known for this conversation."""
        job = self.reports.get(trunk_id)
        if job is None:
            return None
        return job.read_from(from_index)

    def stop_report(self, trunk_id: str) -> bool:
        job = self.reports.get(trunk_id)
        if job is None or job.done:
            return False
        job.request_stop()
        return True

    def report_markdown(self, trunk_id: str) -> Optional[tuple[str, str]]:
        """(filename, content) of the saved report as a Markdown download, or
        None when there is no report yet."""
        self._require_trunk(trunk_id)
        saved = report_store.get(self._reports_dir(), trunk_id)
        if not saved or not saved.get("content"):
            return None
        when = time.localtime(saved.get("created_at") or time.time())
        slug = _title_slug(str(saved.get("title") or ""))
        fname = f"chatable-综述-{slug or trunk_id}-{time.strftime('%Y%m%d-%H%M%S', when)}.md"
        footer = (f"\n\n---\n\n> 综述由 {saved.get('model') or 'LLM'} 根据对话树生成 · "
                  f"{time.strftime('%Y-%m-%d %H:%M', when)}\n")
        return fname, saved["content"].rstrip() + footer

    def _run_report(self, job: "_LiveTurn", trunk_id: str, source: dict, focus: str) -> None:
        """Background body of one report job: stream the report, save it, and
        push events into ``job`` until done."""
        emit = job.append
        # A storage switch is refused while a job runs (has_active_streams),
        # so these stay valid for the job's lifetime.
        store, directory = self.store, self._reports_dir()
        t0 = time.monotonic()
        try:
            emit({
                "type": "report_start",
                "trunk_id": trunk_id,
                "title": source["title"],
                "focus": focus,
                "model": main.OPENAI_MODEL,
                "turns": source["turns"],
                "messages": source["messages"],
                "estimated_tokens": source["estimated_tokens"],
                "truncated_nodes": source["truncated_nodes"],
                "sources": len(source["sources"]),
            })
            messages = main.build_report_messages(source["text"], source["title"], focus)
            usage: dict = {}
            finish_reason = ""
            text_parts: list[str] = []
            # One retry on a transient upstream failure, as in a chat turn:
            # clients drop the partial text on `reset`.
            for attempt in (1, 2):
                text_parts = []
                if attempt > 1:
                    emit({"type": "reset", "reason": "upstream stream truncated; regenerating"})
                try:
                    for kind, payload in stream_backend(messages, max_tokens=main.REPORT_MAX_TOKENS):
                        if job.stop_requested:
                            break
                        if kind == "done":
                            payload = payload or {}
                            usage = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
                            finish_reason = str(payload.get("finish_reason") or "")
                            break
                        if kind == "reasoning":
                            emit({"type": "reasoning", "text": payload})
                        elif kind == "delta":
                            text_parts.append(payload)
                            emit({"type": "delta", "text": payload})
                    break
                except Exception as e:  # noqa: BLE001 - surface stream failures to the client
                    if _is_transient_stream_error(e) and attempt == 1 and not job.stop_requested:
                        continue
                    emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
                    return

            if job.stop_requested:
                emit({"type": "stopped"})
                return
            content = "".join(text_parts).strip()
            if not content:
                emit({"type": "error", "message": "模型没有返回综述内容，请重试"})
                return
            elapsed = time.monotonic() - t0
            report = {
                "trunk_id": trunk_id,
                "title": source["title"],
                "content": content,
                "focus": focus,
                "created_at": time.time(),
                "model": main.OPENAI_MODEL,
                "usage": usage,
                "stats": _fmt_stats(usage, elapsed),
                "elapsed": round(elapsed, 2),
                "turns": source["turns"],
                "messages": source["messages"],
                "truncated_nodes": source["truncated_nodes"],
                "sources": source["sources"],
                "estimated_tokens": source["estimated_tokens"],
                # "length" = cut off at the output cap: the report is incomplete.
                "output_truncated": finish_reason == "length",
                "fingerprint": source["fingerprint"],
            }
            # Under the lock that delete_subtree takes, so a conversation
            # deleted mid-run can't get its report written back afterwards.
            with self._report_lock:
                if job.stop_requested or not store.exists(trunk_id):
                    emit({"type": "stopped"})
                    return
                report_store.save(directory, report)
            stale = report["fingerprint"] != _report_fingerprint(store.trunk_nodes(trunk_id))
            emit({"type": "done", "report": report, "stale": stale})
        except Exception as e:  # noqa: BLE001 - last-resort guard for the thread
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            job.finish()
            # Retained briefly for a late reconnect, like a chat turn. Removed
            # by identity: a new job for this conversation may have started.
            timer = threading.Timer(RETAIN_FINISHED_TURN_S, self.reports.discard, args=(trunk_id, job))
            timer.daemon = True
            timer.start()

    # ------------------------------------------------------------------ #
    # Tool decision + execution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_tool_calls(raw_calls: list) -> list[dict[str, Any]]:
        if not isinstance(raw_calls, list):
            return []
        out = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            args = call.get("arguments") or {}
            if name not in TOOL_REGISTRY.names() or not isinstance(args, dict):
                continue
            out.append({"name": name, "arguments": args})
        return out[:3]

    @staticmethod
    def _parse_fc_xml(text: str) -> list[dict[str, Any]]:
        """Fallback: extract tool calls from <function_calls> XML blocks."""
        out = []
        for match in re.finditer(
            r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>",
            text, flags=re.DOTALL,
        ):
            name = match.group(1)
            if name not in TOOL_REGISTRY.names():
                continue
            inner = match.group(2)
            args: dict[str, Any] = {}
            for pm in re.finditer(
                r'<parameter\s+name="([^"]+)"([^>]*)>(.*?)</parameter>',
                inner, flags=re.DOTALL,
            ):
                key = pm.group(1)
                attrs = pm.group(2)
                value: Any = pm.group(3).strip()
                int_types = {"int", "integer", "number"}
                for attr_name in int_types:
                    m = re.search(rf'\b{attr_name}="([^"]*)"', attrs)
                    if m:
                        try:
                            value = int(m.group(1))
                        except ValueError:
                            pass
                        break
                args[key] = value
            out.append({"name": name, "arguments": args})
        return out[:3]

    @classmethod
    def _parse_tool_decision(cls, text: str) -> list[dict[str, Any]]:
        text = normalize_model_markup(text.strip())
        if not text:
            return []
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
        calls = cls._normalize_tool_calls(data.get("tool_calls", []))
        if not calls and ("<function_calls>" in text or "<invoke" in text):
            calls = cls._parse_fc_xml(text)
        return calls

    def _decide_tool_calls(self, messages: list[dict]) -> tuple[list[dict[str, Any]], dict]:
        if TOOL_DECISION_MODE in ("native", "openai"):
            try:
                return self._decide_tool_calls_native(messages)
            except Exception:  # noqa: BLE001 - fall back to prompt-based decision
                pass
        return self._decide_tool_calls_prompt(messages)

    def _decide_tool_calls_prompt(self, messages: list[dict]) -> tuple[list[dict[str, Any]], dict]:
        decision_messages = list(messages) + [
            {"role": "system", "content": TOOL_REGISTRY.decision_prompt()}
        ]
        resp = call_backend(decision_messages)
        text = extract_text(resp)
        return self._parse_tool_decision(text), extract_usage(resp)

    def _decide_tool_calls_native(self, messages: list[dict]) -> tuple[list[dict[str, Any]], dict]:
        resp = call_backend(messages, tools=TOOL_REGISTRY.openai_tools(), tool_choice="auto")
        calls = []
        for call in extract_tool_calls(resp):
            name = call.get("name")
            args = call.get("arguments") or {}
            if name in TOOL_REGISTRY.names() and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
        return calls[:3], extract_usage(resp)

    # ------------------------------------------------------------------ #
    # Tool results live INSIDE the assistant node (not as tree nodes)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tool_record(call: dict, result: ToolResult) -> dict:
        """One auto-tool result, stored in the assistant node's metadata.

        ``content`` is the exact text injected into the model context on this
        and later turns; ``output`` is the (possibly truncated) raw tool output
        kept for display. Keeping ``content`` lets us reconstruct byte-stable
        context without persisting a separate tree node.
        """
        return {
            "name": result.name,
            "arguments": call.get("arguments", {}),
            "ok": result.ok,
            "error": result.error,
            "call_id": result.call_id,
            "duration_ms": result.duration_ms,
            "output": result.output,
            "content": result.node_content(automatic=True),
            # Images captured by the tool (vision-capable backends only).
            "images": (result.metadata or {}).get("images") or [],
        }

    def _images_dir(self) -> str:
        """Sidecar folder for captured images: <data_dir>/images/. Written by
        the image-capture feature; replay must resolve these references."""
        return os.path.join(os.path.dirname(os.path.abspath(self.store.db_path)), "images")

    def _restore_image_message(self, m: dict) -> Optional[dict]:
        """Reconstruct a persisted image message for replay: image_file parts
        become image_url data-URLs read back from the sidecar files (same
        bytes → same base64 → byte-identical to the live turn's message).
        Returns None when the backend is blind or a sidecar file is missing —
        the message is then dropped from the context."""
        content = m.get("content")
        if not isinstance(content, list) or not any(
                isinstance(p, dict) and p.get("type") == "image_file" for p in content):
            return m
        if not main.VISION_SUPPORTED:
            return None  # a blind backend would 400 on image parts
        img_dir = self._images_dir()
        parts: list[dict] = []
        for p in content:
            if p.get("type") != "image_file":
                parts.append(p)
                continue
            try:
                with open(os.path.join(img_dir, os.path.basename(p["path"])), "rb") as f:
                    raw = f.read()
            except OSError:
                return None
            b64 = base64.b64encode(raw).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{p.get('mime', 'image/jpeg')};base64,{b64}"}})
        return {**m, "content": parts}

    def _expand_path(self, path: list[Node]) -> list[Node]:
        """Insert ephemeral nodes for each assistant's tool exchange.

        With native replay (the default), the exact tool messages persisted at
        turn end are re-emitted verbatim — the next request's prefix then
        matches the previous turn's byte for byte, keeping it cache-hittable.
        Older nodes (or CHATTABLE_NATIVE_TOOL_HISTORY=0) fall back to folding
        results into system_note text. These nodes are never persisted; they
        are reconstructed deterministically from metadata so forks from the
        same ancestor still send byte-identical prefixes to the backend.
        """
        out: list[Node] = []
        for n in path:
            if n.role == "assistant" and isinstance(n.metadata, dict):
                replay = n.metadata.get("tool_messages") if NATIVE_TOOL_HISTORY else None
                if replay:
                    for m in replay:
                        # image_file sidecar refs (written by the image-capture
                        # feature) must be resolved back to image_url parts —
                        # upstreams reject the image_file type with a 400.
                        m = self._restore_image_message(m)
                        if m is None:
                            continue
                        ephem = Node.new(
                            role="raw_message",
                            # Sorted keys: Node serialization flushes with
                            # sort_keys=True, so this round-trips byte-stable.
                            content=json.dumps(m, ensure_ascii=False, sort_keys=True),
                            parent_id=n.parent_id,
                        )
                        out.append(ephem)
                else:
                    for rec in (n.metadata.get("tool_results") or []):
                        ephem = Node.new(
                            role="system_note",
                            content=rec.get("content", ""),
                            parent_id=n.parent_id,
                        )
                        ephem.id = rec.get("call_id") or ephem.id
                        ephem.title = rec.get("name", "")
                        out.append(ephem)
            out.append(n)
        return out

    def _live_context(self, anchor_id: str, tool_results: list[dict]) -> dict:
        """Context for the current turn: expanded path + this turn's tool results."""
        path = self._expand_path(self.store.path_to_root(anchor_id))
        for rec in tool_results:
            ephem = Node.new(role="system_note", content=rec.get("content", ""), parent_id=anchor_id)
            ephem.id = rec.get("call_id") or ephem.id
            ephem.title = rec.get("name", "")
            path.append(ephem)
        context = build_context(path, self.store.pinned_nodes())
        if VISION_ENABLED and main.VISION_SUPPORTED:
            self._inject_images(context["messages"], anchor_id)
        return context

    def _inject_images(self, messages: list[dict], anchor_id: str) -> None:
        """Attach the current turn's uploaded images to the final user message
        as inline data-URL parts (the model API can't fetch localhost URLs).
        Older turns keep only the text mention — no re-injection, which keeps
        the context light and the cache prefix stable."""
        node = self.store.get(anchor_id)
        meta = node.metadata if isinstance(node.metadata, dict) else {}
        images = [
            a for a in (meta.get("attachments") or [])
            if str(a.get("mime", "")).startswith("image/") and a.get("path")
        ]
        if not images:
            return
        for msg in reversed(messages):
            if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
                continue
            parts: list[dict] = [{"type": "text", "text": msg["content"]}]
            for a in images:
                url = self._image_data_url(a["path"], a.get("mime") or "image/jpeg")
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            if len(parts) > 1:
                msg["content"] = parts
            return

    def _image_data_url(self, path: str, mime: str) -> str:
        """base64 data URL for an uploaded image, memoized per path."""
        if path not in self._img_data_urls:
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                return ""
            self._img_data_urls[path] = (
                f"data:{mime};base64,{base64.b64encode(data).decode()}"
            )
        return self._img_data_urls[path]

    # ------------------------------------------------------------------ #
    # The core: streamed send (generation runs in the background so a client
    # disconnect/refresh never aborts the reply; clients subscribe/reconnect).
    # ------------------------------------------------------------------ #

    def start_turn(self, content: str, parent_id: Optional[str] = None,
                   attachments: Optional[list[dict]] = None) -> dict:
        """Create the user+assistant nodes and start generating in the background.

        Returns immediately with the new node ids. Generation proceeds in a
        daemon thread, pushing events into a ``_LiveTurn`` buffer that clients
        read via ``subscribe``. Because generation no longer depends on the HTTP
        connection, refreshing or closing the tab does not stop the reply.
        ``attachments`` (uploaded files, e.g. images) are persisted on the user
        node's metadata — only descriptors, the bytes live in the uploads dir.
        """
        parent_id = parent_id or ROOT_ID
        if not self.store.exists(parent_id):
            parent_id = ROOT_ID
        parent = self.store.get(parent_id)

        user_node = Node.new(role="user", content=content, parent_id=parent.id)
        if attachments:
            user_node.metadata = {
                "attachments": [
                    {k: a[k] for k in ("filename", "path", "mime") if a.get(k)}
                    for a in attachments
                ]
            }
        self.store.add(user_node)

        # Assistant node created up front so the UI has a stable, clickable id.
        assistant_node = Node.new(role="assistant", content="", parent_id=user_node.id)
        self.store.add(assistant_node)

        turn = self.live.create(assistant_node.id)
        turn.append({"type": "user_node", "id": user_node.id, "parent_id": parent.id})
        turn.append({"type": "assistant_node", "id": assistant_node.id, "parent_id": user_node.id})

        thread = threading.Thread(
            target=self._run_turn, args=(turn, user_node, assistant_node), daemon=True,
        )
        thread.start()

        return {
            "user_id": user_node.id,
            "assistant_id": assistant_node.id,
            "parent_id": parent.id,
        }

    def subscribe(self, assistant_id: str, from_index: int = 0) -> Optional[Iterator[dict]]:
        """Return a generator of events for an in-progress turn, or None if the
        turn is unknown (already finished and reaped, or never existed)."""
        turn = self.live.get(assistant_id)
        if turn is None:
            return None
        return turn.read_from(from_index)

    def request_stop(self, assistant_id: str) -> bool:
        turn = self.live.get(assistant_id)
        if turn is None:
            return False
        turn.request_stop()
        return True

    def active_stream_ids(self) -> list[str]:
        """Assistant node ids whose generation is still running (for reconnect)."""
        return self.live.active_ids()

    def send_stream(self, content: str, parent_id: Optional[str] = None) -> Iterator[dict]:
        """Back-compat: start a turn and stream its events in one call.

        Used by any caller that wants the old single-generator behaviour (e.g.
        tests). The turn still runs in the background, so even here a dropped
        consumer does not abort generation.
        """
        info = self.start_turn(content, parent_id)
        sub = self.subscribe(info["assistant_id"], 0)
        if sub is not None:
            yield from sub

    def _run_turn(self, turn: "_LiveTurn", user_node: Node, assistant_node: Node) -> None:
        """Background generation body: push events into ``turn`` until done."""
        try:
            self._generate(turn, user_node, assistant_node)
        except Exception as e:  # noqa: BLE001 - last-resort guard for the thread
            turn.append({"type": "error", "message": f"{type(e).__name__}: {e}", "id": assistant_node.id})
        finally:
            turn.finish()
            # Keep the finished buffer around briefly so a reconnect that lands
            # just after completion can still replay the tail (incl. the `done`
            # event). A later refresh that misses this window falls back to the
            # persisted node content via /api/path.
            def _reap(aid: str = assistant_node.id) -> None:
                self.live.remove(aid)
            timer = threading.Timer(RETAIN_FINISHED_TURN_S, _reap)
            timer.daemon = True
            timer.start()

    def _generate(self, turn: "_LiveTurn", user_node: Node, assistant_node: Node) -> None:
        if TOOL_DECISION_MODE in ("native", "openai"):
            return self._generate_unified(turn, user_node, assistant_node)
        return self._generate_legacy(turn, user_node, assistant_node)

    def _generate_unified(self, turn: "_LiveTurn", user_node: Node, assistant_node: Node) -> None:
        """Single streaming tool-use loop.

        Every round streams WITH the tool definitions, so the model may answer,
        call tools, or mix both at any point — there is no separate invisible
        decision phase. Fresh tool calls are executed and folded back as native
        OpenAI tool messages; the loop ends when the model stops calling tools.
        ``CHATTABLE_AUTO_TOOL_MAX_ROUNDS`` is only a fuse: once hit, the next
        round streams without tools so the model must write its final answer.
        """
        emit = turn.append
        # Tool results are NOT tree nodes; they are collected here and stored
        # inside the assistant node's metadata (see _finalize_turn).
        tool_results: list[dict] = []
        context = self._live_context(user_node.id, tool_results)
        loop_messages = list(context["messages"])

        emit({
            "type": "assistant_start",
            "id": assistant_node.id,
            "context": {
                "messages": len(loop_messages),
                "estimated_tokens": context["estimated_tokens"],
                "pinned": len(context["pinned_nodes"]),
                "shortened": len(context["omitted_nodes"]),
                "prefix": context["cache"]["prefix_before_latest_user_hash"][:10],
            },
        })

        usage: dict = {}
        reasoning_parts: list[str] = []
        segments: list[str] = []
        # The native tool exchange, collected for persistence — replayed
        # byte-identically on later turns to preserve the cache prefix.
        replay: list[dict] = []
        seen_tool_calls: set[str] = set()
        nudged = False
        round_no = 0
        t0 = time.monotonic()

        while True:
            if turn.stop_requested:
                break
            allow_tools = (
                AUTO_TOOLS_ENABLED and not nudged and round_no < AUTO_TOOL_MAX_ROUNDS
            )
            tools = TOOL_REGISTRY.openai_tools() if allow_tools else None

            # One stream iteration, with a single retry on transient upstream
            # failures (truncated stream or dropped connection; clients drop
            # partial text on `reset`, the done event's final content
            # self-heals the live view).
            text_parts: list[str] = []
            iter_calls: list[dict] = []
            iter_reasoning: list[str] = []
            for attempt in (1, 2):
                text_parts = []
                iter_calls = []
                iter_reasoning = []
                stripper = _ToolCallStripper()  # suppress hallucinated tool-call XML
                if attempt > 1:
                    emit({"type": "reset", "reason": "upstream stream truncated; regenerating"})
                try:
                    for kind, payload in stream_backend(loop_messages, tools=tools):
                        if turn.stop_requested:
                            break
                        if kind == "done":
                            for k, v in (payload or {}).items():
                                if isinstance(v, (int, float)):
                                    usage[k] = usage.get(k, 0) + v
                            break
                        if kind == "reasoning":
                            reasoning_parts.append(payload)
                            iter_reasoning.append(payload)
                            emit({"type": "reasoning", "text": payload})
                        elif kind == "delta":
                            text_parts.append(payload)
                            safe = stripper.feed(payload)
                            if safe:
                                emit({"type": "delta", "text": safe})
                        elif kind == "tool_calls":
                            iter_calls = payload
                    tail = stripper.flush()
                    if tail:
                        emit({"type": "delta", "text": tail})
                    break
                except Exception as e:  # noqa: BLE001 - surface stream failures on the node
                    if _is_transient_stream_error(e) and attempt == 1:
                        continue
                    err = f"{type(e).__name__}: {e}"
                    self.store.append_content(assistant_node.id, f"[error: {err}]")
                    emit({"type": "error", "message": err, "id": assistant_node.id})
                    return

            iter_text = "".join(text_parts)
            if _scrub_answer_text(iter_text).strip():
                segments.append(iter_text)

            # Dedup across the whole turn; cap 3 fresh calls per round.
            fresh: list[dict] = []
            for call in iter_calls:
                sig = json.dumps({"name": call["name"], "arguments": call["arguments"]},
                                 sort_keys=True, ensure_ascii=False)
                if sig in seen_tool_calls:
                    continue
                seen_tool_calls.add(sig)
                fresh.append(call)
                if len(fresh) >= 3:
                    break

            if not fresh or turn.stop_requested:
                # The model produced a final answer. If it scrubbed to nothing
                # (markup-only or empty), nudge once and retry without tools.
                joined = _scrub_answer_text("\n\n".join(segments))
                if turn.stop_requested or joined.strip() or nudged:
                    break
                nudged = True
                # Strict backends (Kimi) reject an empty assistant message with
                # a 400, and an empty iteration (reasoning only, or a deduped
                # repeat call) is exactly what lands here — so skip it.
                if iter_text.strip():
                    loop_messages.append({"role": "assistant", "content": iter_text})
                loop_messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue

            # Fold the exchange back in native OpenAI tool-message format.
            # Messages are built with sorted keys so a JSONL round-trip (the
            # store flushes with sort_keys=True) keeps them byte-identical —
            # next turn's prefix then matches this turn's, cache-wise.
            raw_calls = []
            for i, call in enumerate(fresh):
                cid = call["id"] or f"call_{round_no}_{i}"
                call["_cid"] = cid
                raw_calls.append(_canon_raw_call(cid, call["name"], call["arguments"]))
            assistant_msg = _canon_assistant_tool_msg(iter_text, raw_calls, "".join(iter_reasoning))
            loop_messages.append(assistant_msg)
            replay.append(assistant_msg)
            # A tool_calls block must be answered by an unbroken run of
            # role=tool messages: anything interposed (e.g. the synthetic
            # image-carrying user message below) makes the API treat the
            # remaining ids as unanswered and reject the request. So images are
            # parked here and flushed once the block is closed.
            image_msgs: list[dict] = []
            answered: set[str] = set()
            for call in fresh:
                if turn.stop_requested:
                    break
                emit({"type": "tool_call", "name": call["name"], "arguments": call["arguments"]})
                result = TOOL_REGISTRY.run(call["name"], call["arguments"])
                record = self._tool_record(call, result)
                tool_results.append(record)
                emit({
                    "type": "tool_result",
                    "name": record["name"],
                    "arguments": record["arguments"],
                    "ok": record["ok"],
                    "error": record["error"],
                    "duration_ms": record["duration_ms"],
                    "output": record["output"],
                    "images": len(record.get("images") or []),
                })
                tool_msg = _canon_tool_result_msg(call["_cid"], record["output"])
                loop_messages.append(tool_msg)
                replay.append(tool_msg)
                answered.add(call["_cid"])
                # Vision: hand captured images to the model as a synthetic user
                # message (tool messages can't carry image parts). Deferred to
                # after the loop — see `image_msgs` above. NOT added to
                # `replay`: history stays text-only.
                images = record.get("images") or []
                if images and VISION_ENABLED and main.VISION_SUPPORTED:
                    parts: list[dict] = [{
                        "type": "text",
                        "text": f"（以下是 {record['name']} 抓取的 {len(images)} 张图片）",
                    }]
                    for im in images:
                        parts.append({"type": "image_url",
                                      "image_url": {"url": im["data_url"]}})
                    image_msgs.append({"content": parts, "role": "user"})

            # A stop breaks the loop above with calls still unexecuted. Their
            # ids were already issued in assistant_msg, so answer them with a
            # stub: an unanswered id is rejected on the next request, and the
            # gap would be baked into the node's persisted replay, poisoning
            # every later turn on this branch.
            for call in fresh:
                if call["_cid"] in answered:
                    continue
                stub = _canon_tool_result_msg(call["_cid"], _UNEXECUTED_TOOL_OUTPUT)
                loop_messages.append(stub)
                replay.append(stub)

            # Safe now that the tool block is closed.
            loop_messages.extend(image_msgs)
            round_no += 1

        full_text = _scrub_answer_text("\n\n".join(segments))
        self._finalize_turn(
            turn, user_node, assistant_node,
            context=context, full_text=full_text, usage=usage,
            reasoning="".join(reasoning_parts), tool_results=tool_results,
            tool_messages=replay, t0=t0,
        )

    def _finalize_turn(
        self,
        turn: "_LiveTurn",
        user_node: Node,
        assistant_node: Node,
        *,
        context: dict,
        full_text: str,
        usage: dict,
        reasoning: str,
        tool_results: list[dict],
        tool_messages: Optional[list[dict]] = None,
        t0: float,
    ) -> None:
        """Persist the assistant node and emit the closing ``done`` event."""
        # If the client interrupted and deleted this turn mid-stream, the user
        # node will be gone. Clean up our assistant node (which may now be an
        # orphan) and stop without persisting anything further.
        if not self.store.exists(user_node.id):
            self.store.delete_subtree(assistant_node.id)
            return

        elapsed = time.monotonic() - t0
        if not full_text.strip() and not turn.stop_requested:
            # Never persist an invisible reply: the model produced no usable
            # answer even after the nudge retry — say so, don't leave a blank.
            full_text = "[empty reply: the model produced no visible answer — it may have tried to call a tool in its final response; please retry]"
        metadata = {
            "cache": context["cache"],
            "reasoning_chars": len(reasoning),
            "reasoning": reasoning,
            "tool_results": tool_results,
        }
        if tool_messages:
            # Native tool exchange for byte-identical replay on later turns.
            metadata["tool_messages"] = tool_messages
        # One locked write appending a single line (was 3 separate flushes).
        self.store.finalize_node(assistant_node.id, full_text, usage, metadata)

        turn.append({
            "type": "done",
            "id": assistant_node.id,
            # Scrubbed final text, so clients can replace the raw streamed
            # accumulation (which may contain stripped tool-call markup).
            "content": full_text,
            "usage": usage,
            "stats": _fmt_stats(usage, elapsed),
            "elapsed": elapsed,
            "tool_results": [
                {k: r[k] for k in ("name", "arguments", "ok", "error", "duration_ms", "output")}
                for r in tool_results
            ],
        })

    def _generate_legacy(self, turn: "_LiveTurn", user_node: Node, assistant_node: Node) -> None:
        """Two-phase generation (invisible decision rounds, then a final stream
        without tools). Kept for CHATTABLE_TOOL_DECISION_MODE=prompt — use it on
        backends that don't support streaming tool_calls."""
        emit = turn.append
        decision_usage = {"input": 0, "output": 0, "total": 0, "cache_read": 0, "cache_creation": 0}
        # Tool results are NOT tree nodes; they are collected here and later
        # stored inside the assistant node's metadata, then surfaced to the
        # model as ephemeral context via _live_context / _expand_path.
        tool_results: list[dict] = []

        if AUTO_TOOLS_ENABLED:
            seen_tool_calls: set[str] = set()
            for _round in range(max(0, AUTO_TOOL_MAX_ROUNDS)):
                context = self._live_context(user_node.id, tool_results)
                # The tool-decision call is non-streaming, so the client would
                # otherwise sit with no feedback. Signal that we're deciding the
                # next move; cleared implicitly when a tool_call / delta arrives.
                emit({"type": "planning", "round": _round})
                try:
                    tool_calls, usage = self._decide_tool_calls(context["messages"])
                except Exception as e:  # noqa: BLE001
                    emit({"type": "tool_skipped", "error": f"{type(e).__name__}: {e}"})
                    break
                for key in decision_usage:
                    decision_usage[key] += usage.get(key, 0)
                if not tool_calls:
                    break
                fresh_calls = []
                for call in tool_calls:
                    sig = json.dumps(call, sort_keys=True, ensure_ascii=False)
                    if sig in seen_tool_calls:
                        continue
                    seen_tool_calls.add(sig)
                    fresh_calls.append(call)
                if not fresh_calls:
                    break
                for call in fresh_calls:
                    emit({"type": "tool_call", "name": call["name"], "arguments": call["arguments"]})
                    result = TOOL_REGISTRY.run(call["name"], call["arguments"])
                    record = self._tool_record(call, result)
                    tool_results.append(record)
                    emit({
                        "type": "tool_result",
                        "name": record["name"],
                        "arguments": record["arguments"],
                        "ok": record["ok"],
                        "error": record["error"],
                        "duration_ms": record["duration_ms"],
                        "output": record["output"],
                    })

        # Assistant attaches DIRECTLY to the user node — tool results are folded
        # into the assistant, not interposed as tree nodes. (The node itself was
        # created up front, above, so the UI already has its id.)
        context = self._live_context(user_node.id, tool_results)
        messages = context["messages"]

        emit({
            "type": "assistant_start",
            "id": assistant_node.id,
            "context": {
                "messages": len(messages),
                "estimated_tokens": context["estimated_tokens"],
                "pinned": len(context["pinned_nodes"]),
                "shortened": len(context["omitted_nodes"]),
                "prefix": context["cache"]["prefix_before_latest_user_hash"][:10],
            },
        })

        usage: dict = {}
        t0 = time.monotonic()
        full_text = ""

        # Two failure modes are retried once from scratch (clients drop partial
        # text on the `reset` event):
        #  1. the relay silently closed the SSE stream mid-answer — raised as
        #     StreamTruncatedError by stream_backend, and
        #  2. the model's final answer contained only tool-call markup
        #     (scrubbed below, leaving nothing) or no content at all — common
        #     when the tool-round cap cut its investigation short.
        # The retry appends an explicit "no more tool calls" nudge.
        turn_messages = messages
        for attempt in (1, 2):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            stripper = _ToolCallStripper()  # suppress hallucinated tool-call XML blocks
            if attempt > 1:
                emit({"type": "reset", "reason": "regenerating an empty or truncated reply"})
                turn_messages = messages + [{
                    "role": "user",
                    "content": _NO_TOOL_NUDGE,
                }]
            try:
                for kind, payload in stream_backend(turn_messages):
                    if turn.stop_requested:
                        break
                    if kind == "done":
                        # Numeric fields only: the payload also carries the
                        # upstream finish_reason string.
                        usage = {k: v for k, v in (payload or {}).items()
                                 if isinstance(v, (int, float))}
                        break
                    if kind == "reasoning":
                        reasoning_parts.append(payload)
                        emit({"type": "reasoning", "text": payload})
                    elif kind == "delta":
                        text_parts.append(payload)
                        safe = stripper.feed(payload)
                        if safe:
                            emit({"type": "delta", "text": safe})
            except Exception as e:  # noqa: BLE001 - surface stream failures on the node
                if _is_transient_stream_error(e) and attempt == 1:
                    continue
                err = f"{type(e).__name__}: {e}"
                self.store.append_content(assistant_node.id, f"[error: {err}]")
                emit({"type": "error", "message": err, "id": assistant_node.id})
                return

            # Normalise DSML markers, then scrub any tool-call markup that
            # survived streaming (incl. an unterminated trailing block).
            candidate = _scrub_answer_text("".join(text_parts))
            if candidate.strip() or turn.stop_requested:
                full_text = candidate
                break

        # Flush any remaining buffered safe text.
        tail = stripper.flush()
        if tail:
            emit({"type": "delta", "text": tail})

        for key in ("input", "output", "total", "cache_read", "cache_creation"):
            usage[key] = usage.get(key, 0) + decision_usage.get(key, 0)
        self._finalize_turn(
            turn, user_node, assistant_node,
            context=context, full_text=full_text, usage=usage,
            reasoning="".join(reasoning_parts),
            tool_results=tool_results, t0=t0,
        )
