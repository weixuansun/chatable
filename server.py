"""FastAPI web server for chatable.

Reuses ``service.ChatService`` (which in turn reuses store/main/tools unchanged)
and exposes a small JSON + SSE API consumed by ``static/index.html``.

Run:
    chatable-web                      # console script (see pyproject.toml)
    python -m uvicorn server:app      # or directly
    python server.py                  # convenience launcher

Honours the same env vars as the CLI (CHATTABLE_DB, DEEPSEEK_API_KEY, …).
Set CHATTABLE_WEB_HOST / CHATTABLE_WEB_PORT to change the bind address.

Security (see SecurityMiddleware):
  CHATTABLE_ALLOWED_HOSTS — comma-separated extra Host header values to accept
      (localhost, 127.0.0.1, ::1 and the bind host are always allowed).
      Anything else gets a 400 — this blocks DNS-rebinding access.
  CHATTABLE_AUTH_TOKEN — when set, every request must authenticate with
      "Authorization: Bearer <token>" or a cookie obtained by opening
      "/?token=<token>" once (the browser UI, SSE and uploads then just work).
  Cross-origin writes (and sensitive reads like /api/fs/list, /api/files/*)
  are rejected with 403 based on the Origin/Referer host. Requests with no
  Origin/Referer at all (curl, scripts) are only accepted from a loopback
  client when no token is configured; with a token, the token is the check.
  (The single-T spellings CHATABLE_ALLOWED_HOSTS / CHATABLE_AUTH_TOKEN are
  accepted as aliases.)
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
from http.cookies import SimpleCookie
from typing import Optional
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import main as llm
from service import ChatService
from store import NodeNotFoundError, resolve_db_path, TreeStore
from bookmark_store import BookmarkStore, resolve_bookmarks_path, BookmarkNotFoundError

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="chatable web")
# The data folder may be overridden at launch via `chatable-web --data-dir DIR`
# (see main()); the CHATTABLE_DATA_DIR env var covers the `uvicorn server:app`
# path where main() never runs. Next fallback is the folder persisted by the
# settings UI (settings.json), then CHATTABLE_DB/CHATTABLE_HOME/default.
_env_data_dir = os.environ.get("CHATTABLE_DATA_DIR") or llm.load_persisted_data_dir() or None
service = ChatService(TreeStore(db_path=resolve_db_path(_env_data_dir)))
# Resolve vision support from the settings cache (or background-probe it).
llm.init_vision_flag()


# ---------------------------------------------------------------------------
# Security middleware: Host allowlist + Origin/Referer CSRF check + opt-in token
# ---------------------------------------------------------------------------

_TOKEN_COOKIE = "chatable_token"
# Sensitive GETs (expose local files / filesystem layout) get the same
# Origin/Referer scrutiny as writes. Everything else is protected against
# cross-origin reads by the browser's same-origin policy and against DNS
# rebinding by the Host allowlist.
_SENSITIVE_GET_PREFIXES = ("/api/fs/list", "/api/files/")


def _normalize_host(value: str) -> str:
    """Lowercase host without port or IPv6 brackets ("[::1]:8000" → "::1")."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[1:end] if end != -1 else v.strip("[]")
    if v.count(":") == 1:  # host:port
        v = v.rsplit(":", 1)[0]
    return v


def _env(primary: str, alias: str = "") -> str:
    """First non-empty of the CHATTABLE_* name and its single-T alias."""
    value = os.environ.get(primary, "").strip()
    if not value and alias:
        value = os.environ.get(alias, "").strip()
    return value


# The actual bind host, set by main() from --host (whose default is
# CHATTABLE_WEB_HOST). The middleware allowlists it so the server never
# rejects its own Host header; the import-time env value covers the
# `uvicorn server:app` path where main() never runs.
_BIND_HOST = _env("CHATTABLE_WEB_HOST") or "127.0.0.1"


def _allowed_hosts() -> frozenset[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    bind = _normalize_host(_BIND_HOST)
    if bind:
        hosts.add(bind)
    for entry in _env("CHATTABLE_ALLOWED_HOSTS", "CHATABLE_ALLOWED_HOSTS").split(","):
        entry = _normalize_host(entry)
        if entry:
            hosts.add(entry)
    return frozenset(hosts)


def _auth_token() -> str:
    return _env("CHATTABLE_AUTH_TOKEN", "CHATABLE_AUTH_TOKEN")


def _is_loopback_client(scope: dict) -> bool:
    client = scope.get("client")
    if not client:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


async def _reject(send, status: int, message: str) -> None:
    body = json.dumps({"detail": message}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})


class SecurityMiddleware:
    """Pure-ASGI guard in front of everything (API, SSE, static):

    1. Host header allowlist (DNS-rebinding defence) → 400.
    2. Optional bearer/cookie token (CHATABLE_AUTH_TOKEN) → 401. A GET with
       a valid ``?token=`` sets an HttpOnly SameSite=Strict cookie and
       redirects to the same URL without the token, so the browser UI,
       EventSource and uploads authenticate via the cookie.
    3. Origin/Referer check (CSRF) on non-GET/HEAD/OPTIONS and on sensitive
       GETs: a present Origin/Referer host must be in the allowlist → 403.
       Header-less requests (curl, scripts) pass only from a loopback client
       when no token is configured — with a token, rule 2 already gated them.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        allowed = _allowed_hosts()
        token = _auth_token()
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()

        # 1. Host allowlist.
        host = _normalize_host(headers.get("host", ""))
        if host not in allowed:
            await _reject(send, 400, f"host not allowed: {host or '(missing)'}")
            return

        # 2. Token login / token check.
        authed = False
        if token:
            query = dict(parse_qsl(scope.get("query_string", b"").decode("latin-1")))
            if "token" in query:
                if hmac.compare_digest(query.pop("token"), token):
                    location = path + (("?" + urlencode(query)) if query else "")
                    cookie = (f"{_TOKEN_COOKIE}={quote(token, safe='')}; Path=/; "
                              f"HttpOnly; SameSite=Strict")
                    await send({
                        "type": "http.response.start",
                        "status": 302,
                        "headers": [(b"location", location.encode("latin-1")),
                                    (b"set-cookie", cookie.encode("latin-1")),
                                    (b"content-length", b"0")],
                    })
                    await send({"type": "http.response.body", "body": b""})
                else:
                    await _reject(send, 403, "invalid token")
                return
            auth = headers.get("authorization", "")
            if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), token):
                authed = True
            else:
                jar: SimpleCookie = SimpleCookie()
                try:
                    jar.load(headers.get("cookie", ""))
                except Exception:  # noqa: BLE001 - a malformed cookie header = no cookie
                    jar = SimpleCookie()
                morsel = jar.get(_TOKEN_COOKIE)
                if morsel is not None and hmac.compare_digest(
                        unquote(morsel.value), token):
                    authed = True
            if not authed:
                await _reject(send, 401, "authentication required "
                                         "(CHATTABLE_AUTH_TOKEN is set)")
                return

        # 3. Origin/Referer CSRF check.
        if method not in ("GET", "HEAD", "OPTIONS") or path.startswith(_SENSITIVE_GET_PREFIXES):
            source = headers.get("origin") or headers.get("referer")
            if source:
                if _normalize_host(urlsplit(source).netloc) not in allowed:
                    await _reject(send, 403, f"cross-origin request rejected: {source}")
                    return
            elif token:
                # A valid token out-of-band (curl with Bearer) is the defence.
                if not authed:
                    await _reject(send, 403, "origin/referer required")
                    return
            elif not _is_loopback_client(scope):
                await _reject(send, 403, "origin/referer required from non-local clients")
                return

        await self.app(scope, receive, send)


app.add_middleware(SecurityMiddleware)


class SendBody(BaseModel):
    content: str
    parent_id: Optional[str] = None
    # Uploaded attachment descriptors (filename/path/mime), images included.
    attachments: Optional[list[dict]] = None


class SettingsBody(BaseModel):
    # LLM settings (all optional — only apply if `model` is given).
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    # Storage folder (optional — only rebind if given).
    data_dir: Optional[str] = None


class EffortBody(BaseModel):
    level: str


class ReportBody(BaseModel):
    # Optional angle for the survey report ("侧重点").
    focus: str = ""


class BookmarkBody(BaseModel):
    url: str


class BookmarkOpenBody(BaseModel):
    parent_id: Optional[str] = None


@app.get("/api/config")
def api_config() -> dict:
    return service.config()


@app.post("/api/effort")
def api_effort(body: EffortBody) -> dict:
    """Set the reasoning effort (off/low/medium/high) from the composer.

    Allowed while a reply is generating: it applies to the next model request.
    """
    try:
        llm.set_reasoning_effort(body.level)
    except llm.ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return service.config()


@app.post("/api/settings")
def api_settings(body: SettingsBody) -> dict:
    """Apply model/storage settings entered in the web UI.

    The model config (base_url/api_key/model) is persisted to
    ``main.SETTINGS_PATH`` and reloaded on the next launch; the storage
    folder switch remains session-only. Refuses (409) while any reply or
    survey report is generating, so a switch never orphans a background job. Applies storage
    first, then the model config; either is optional. Returns the fresh
    config on success.
    """
    if service.has_active_streams():
        raise HTTPException(
            status_code=409,
            detail="a reply or survey report is generating; stop it before changing settings")

    # Storage folder
    if body.data_dir is not None and body.data_dir.strip():
        try:
            service.rebind_store(body.data_dir.strip())
            # Persist so the next launch opens this folder without re-selecting.
            llm.persist_config(
                data_dir=os.path.abspath(os.path.expanduser(body.data_dir.strip()))
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"storage: {e}")

    # Model / credentials.
    if body.model is not None and body.model.strip():
        try:
            llm.apply_manual_config(
                base_url=body.base_url or "",
                api_key=body.api_key or "",
                model=body.model,
            )
        except llm.ConfigError as e:
            raise HTTPException(status_code=400, detail=f"model: {e}")

    return service.config()


@app.get("/api/tree")
def api_tree() -> dict:
    return service.tree()


@app.get("/api/node/{node_id}")
def api_node(node_id: str) -> dict:
    try:
        return service.node_detail(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")


@app.get("/api/path/{node_id}")
def api_path(node_id: str) -> dict:
    try:
        return {"messages": service.path_messages(node_id)}
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")


@app.get("/api/export/{trunk_id}")
def api_export_tree(trunk_id: str) -> Response:
    """Download one whole conversation tree (trunk + all branches) as Markdown."""
    try:
        filename, content = service.export_tree_markdown(trunk_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {trunk_id}")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"not a trunk node: {trunk_id}")
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    return Response(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


def _report_call(fn, trunk_id: str, *args):
    """Run a report service call, mapping its errors to HTTP statuses."""
    try:
        return fn(trunk_id, *args)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {trunk_id}")
    except ValueError as e:
        # Not a trunk, nothing to report on, or over the input budget — the
        # message is written for the user.
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/report/{trunk_id}")
def api_report(trunk_id: str) -> dict:
    """The saved survey report of a conversation, plus whether one is being
    generated (``running``) and whether the tree changed since (``stale``)."""
    return _report_call(service.get_report, trunk_id)


@app.post("/api/report/{trunk_id}")
def api_report_start(trunk_id: str, body: ReportBody) -> StreamingResponse:
    """Generate a survey report over the whole conversation tree and stream it.

    Runs in the background like a chat turn, so closing the connection does not
    abort it; if a report is already being generated for this conversation the
    stream attaches to that job instead of starting another.
    """
    _report_call(service.start_report, trunk_id, body.focus)
    sub = service.subscribe_report(trunk_id, 0)
    return _sse(sub if sub is not None else iter(()))


@app.get("/api/report/{trunk_id}/stream")
def api_report_stream(trunk_id: str, from_index: int = 0) -> StreamingResponse:
    """Reattach to a report job, replaying events from ``from_index``. Sends a
    single ``gone`` event when no job is known (read the saved report instead)."""
    sub = service.subscribe_report(trunk_id, from_index)
    if sub is None:
        return _sse(iter(({"type": "gone", "id": trunk_id},)))
    return _sse(sub)


@app.post("/api/report/{trunk_id}/stop")
def api_report_stop(trunk_id: str) -> dict:
    """Stop a report job. Nothing is saved; an earlier report is kept."""
    return {"ok": service.stop_report(trunk_id)}


@app.get("/api/report/{trunk_id}/download")
def api_report_download(trunk_id: str) -> Response:
    """Download the saved survey report as Markdown."""
    result = _report_call(service.report_markdown, trunk_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no survey report for this conversation yet")
    filename, content = result
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    return Response(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disposition},
    )


@app.post("/api/delete/{node_id}")
def api_delete(node_id: str) -> dict:
    """Delete a node and its descendants — used to discard an interrupted turn."""
    removed = service.delete_subtree(node_id)
    return {"ok": True, "removed": removed}


class RenameBody(BaseModel):
    title: str = ""


@app.post("/api/node/{node_id}/rename")
def api_rename(node_id: str, body: RenameBody) -> dict:
    """Manually name a node (locks it against auto-titling; empty = unlock)."""
    try:
        return service.rename_node(node_id, body.title)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/suggest/{node_id}")
def api_suggest(node_id: str, force: bool = False) -> dict:
    """Generate candidate follow-up questions for an assistant reply.

    Called by the UI after a turn completes (and on demand). Keeps existing
    suggestions unless ``force=true``; LLM failures come back as
    ``{"ok": false, "error"}`` rather than an HTTP error.
    """
    try:
        return service.generate_suggestions(node_id, force=force)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")


@app.post("/api/title/{node_id}")
def api_title(node_id: str, force: bool = False) -> dict:
    """Generate a short title for an assistant node from its Q/A turn.

    Called by the UI after a turn completes; uses the summary profile (dsv4 by
    default) independently of the conversation model. Only fills an empty title
    unless ``force=true``.
    """
    try:
        return service.generate_title(node_id, force=force)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")


def _sse(gen) -> StreamingResponse:
    """Wrap an event iterator as a Server-Sent-Events response."""
    def event_stream():
        try:
            for event in gen:
                if event.get("type") == "_ping":
                    # Keepalive as an SSE comment — clients ignore non-data
                    # lines, and replay offsets are unaffected.
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001 - last-resort guard for the stream
            err = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/files/{filename}")
def api_uploaded_file(filename: str) -> FileResponse:
    """Serve an uploaded file (e.g. open a PDF in a new browser tab).

    Only files inside the uploads dir are reachable: the name is reduced to
    its basename and must resolve within that directory.
    """
    base = os.path.abspath(service.uploads_dir())
    path = os.path.abspath(os.path.join(base, os.path.basename(filename)))
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    # inline + filename so browsers render PDFs in the tab (no forced download)
    return FileResponse(path, filename=os.path.basename(path),
                        content_disposition_type="inline")


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict:
    """Store a browser-uploaded file (drag-and-drop) server-side.

    Returns the server path; the client references it in the next message so the
    existing web_fetch tool reads it (PDF text extraction, plain text, …). No
    new parsing is needed — uploads reuse the local-file path in web_tools.
    """
    MAX = 50 * 1024 * 1024  # 50 MB guard
    # Stream the body with a hard cap instead of buffering it whole first.
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX:
            raise HTTPException(status_code=413, detail="file too large (max 50 MB)")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    return service.save_upload(file.filename or "upload", data)


@app.post("/api/send")
def api_send(body: SendBody) -> StreamingResponse:
    """Start a turn (generation runs in the background) and stream its events.

    Because generation is detached from this connection, dropping it — e.g. a
    browser refresh — does not abort the reply. The client reconnects to the
    in-progress turn via /api/stream/{assistant_id}.
    """
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    info = service.start_turn(content, body.parent_id, attachments=body.attachments)
    sub = service.subscribe(info["assistant_id"], 0)
    # `sub` is None only if the turn was reaped before we subscribed (not
    # possible here since we just created it), but guard anyway.
    return _sse(sub if sub is not None else iter(()))


@app.get("/api/stream/{assistant_id}")
def api_stream(assistant_id: str, from_index: int = 0) -> StreamingResponse:
    """Reconnect to an in-progress turn, replaying events from ``from_index``.

    Used by the UI after a refresh to resume watching a reply that is still
    generating. Returns an immediate ``gone`` event if the turn is unknown
    (finished and reaped, or never existed) so the client falls back to the
    persisted node content.
    """
    sub = service.subscribe(assistant_id, from_index)
    if sub is None:
        return _sse(iter(({"type": "gone", "id": assistant_id},)))
    return _sse(sub)


@app.get("/api/active")
def api_active() -> dict:
    """Assistant node ids whose generation is still running (for reconnect)."""
    return {"active": service.active_stream_ids()}


@app.post("/api/stop/{assistant_id}")
def api_stop(assistant_id: str) -> dict:
    """Request that an in-progress turn stop generating."""
    return {"ok": service.request_stop(assistant_id)}


@app.get("/api/fs/list")
def api_fs_list(path: Optional[str] = None, hidden: bool = False) -> dict:
    """List subdirectories for the settings storage-folder picker.

    The UI is served locally (127.0.0.1 by default), so browsing the server's
    filesystem is the only way to offer a real folder picker — browsers never
    expose absolute paths. Directories only, dotfolders hidden unless asked.
    """
    p = os.path.abspath(os.path.expanduser(path or "~"))
    if not os.path.isdir(p):
        raise HTTPException(status_code=400, detail=f"not a directory: {p}")
    try:
        names = sorted(os.listdir(p), key=str.casefold)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {p}")
    dirs = [
        {"name": name, "path": os.path.join(p, name)}
        for name in names
        if (hidden or not name.startswith(".")) and os.path.isdir(os.path.join(p, name))
    ]
    parent = os.path.dirname(p)
    return {"path": p, "parent": parent if parent != p else None, "dirs": dirs}


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

@app.get("/api/bookmarks")
def api_bookmarks() -> list[dict]:
    return service.list_bookmarks()


@app.post("/api/bookmarks")
def api_bookmarks_add(body: BookmarkBody) -> dict:
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        return service.add_bookmark(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/bookmarks/{bookmark_id}/refresh")
def api_bookmarks_refresh(bookmark_id: str) -> dict:
    try:
        return service.refresh_bookmark(bookmark_id)
    except BookmarkNotFoundError:
        raise HTTPException(status_code=404, detail=f"bookmark not found: {bookmark_id}")


@app.delete("/api/bookmarks/{bookmark_id}")
def api_bookmarks_delete(bookmark_id: str) -> dict:
    return service.delete_bookmark(bookmark_id)


class BookmarkRenameBody(BaseModel):
    title: str = ""


@app.post("/api/bookmarks/{bookmark_id}/rename")
def api_bookmarks_rename(bookmark_id: str, body: BookmarkRenameBody) -> dict:
    """Manually name a bookmark (locks it against auto-extracted titles)."""
    try:
        return service.rename_bookmark(bookmark_id, body.title)
    except BookmarkNotFoundError:
        raise HTTPException(status_code=404, detail=f"bookmark not found: {bookmark_id}")


@app.post("/api/bookmarks/{bookmark_id}/open")
def api_bookmarks_open(bookmark_id: str, body: BookmarkOpenBody) -> StreamingResponse:
    """Seed a chat turn from a bookmark and stream its events."""
    try:
        info = service.open_bookmark(bookmark_id, parent_id=body.parent_id)
    except BookmarkNotFoundError:
        raise HTTPException(status_code=404, detail=f"bookmark not found: {bookmark_id}")
    sub = service.subscribe(info["assistant_id"], 0)
    return _sse(sub if sub is not None else iter(()))


@app.get("/")
def index() -> FileResponse:
    # no-cache: always revalidate, so UI edits show up on a plain refresh
    # (FileResponse is otherwise heuristically cached by browsers).
    return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                        headers={"Cache-Control": "no-cache"})


# Serve any other static assets (none required for MVP, but future-proof).
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="chatable-web",
        description="Web UI for tree-structured research chat.",
    )
    parser.add_argument(
        "-d", "--data-dir", metavar="PATH", default=None,
        help="Path to a chatable.jsonl file, or a folder (→ <folder>/chatable.jsonl). "
             "Defaults to $CHATTABLE_HOME or ~/.chatable.",
    )
    parser.add_argument("--host", default=os.environ.get("CHATTABLE_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHATTABLE_WEB_PORT", "8000")))
    args = parser.parse_args()

    if args.data_dir:
        # Rebind the module-global service onto the chosen folder before serving.
        global service
        service = ChatService(TreeStore(db_path=resolve_db_path(args.data_dir)))

    # The Host allowlist must know the ACTUAL bind host (CLI --host wins over
    # the CHATTABLE_WEB_HOST env fallback captured at import time).
    global _BIND_HOST
    _BIND_HOST = args.host

    bind = _normalize_host(args.host)
    try:
        bind_ip = ipaddress.ip_address(bind)
        loopback = bind_ip.is_loopback
        wildcard = bind_ip.is_unspecified
    except ValueError:
        loopback = bind == "localhost"
        wildcard = False
    if not loopback:
        print("=" * 72)
        if wildcard:
            print(f"WARNING: binding to the wildcard address {args.host}.")
            print("  The Host allowlist only accepts localhost/127.0.0.1/[::1] "
                  "by default, so requests")
            print("  arriving with a LAN IP or hostname will be REJECTED (400) "
                  "until you allow them:")
            print("    CHATTABLE_ALLOWED_HOSTS=<this-machine-hostname-or-ip>  "
                  "# comma-separated")
        else:
            print(f"WARNING: binding to a non-loopback address ({args.host}).")
            print(f"  Only Host: {bind} (+ localhost names) is accepted; other "
                  "names for this machine")
            print("  need CHATTABLE_ALLOWED_HOSTS=<name1>,<name2>.")
        if not _auth_token():
            print("  CHATTABLE_AUTH_TOKEN is NOT set: anyone who can reach this "
                  "port has FULL access")
            print("  to your chat history, bookmarks, uploaded files and the "
                  "filesystem picker.")
            print("  Set CHATTABLE_AUTH_TOKEN (open "
                  f"http://<host>:{args.port}/?token=<token> once) unless you "
                  "know exactly")
            print("  what you are doing.")
        print("=" * 72)

    print(f"chatable web → http://{args.host}:{args.port}")
    print(f"  backend: {llm.backend_description()}")
    print(f"  history file: {service.store.db_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
