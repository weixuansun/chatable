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
import shutil
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
from workspace import WorkspaceError, default_workspace_path
from orch.service import OrchService

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

# The orchestrator (external coding agents over the snapshot DAG).
orch = OrchService(service)
# Stop on an agent-run reply cancels the queued/running orchestrator run.
service.run_stop_hook = orch.stop_chat_run
# Deleting a chat node cleans up (or cancels) the orchestrator run behind it.
service.subtree_deleted_hook = orch.chat_nodes_deleted


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


class AdoptBody(BaseModel):
    node_id: str


class UndoAdoptBody(BaseModel):
    previous_commit: str


class SettingsBody(BaseModel):
    # LLM settings (all optional — only apply if `model` is given).
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    # Storage folder (optional — only rebind if given).
    data_dir: Optional[str] = None
    # Agent project directory (optional; "" restores <data_dir>/workspace).
    workspace_dir: Optional[str] = None


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
    """Apply model/storage/workspace settings entered in the web UI.

    The model config (base_url/api_key/model) is persisted to
    ``main.SETTINGS_PATH`` and reloaded on the next launch; the storage
    folder and the agent project directory are persisted the same way.
    Refuses (409) while any reply or survey report is generating — or, for a
    workspace switch, while an agent run is queued/running — so a switch
    never orphans a background job. Returns the fresh config on success.
    """
    if service.has_active_streams():
        raise HTTPException(
            status_code=409,
            detail="a reply or survey report is generating; stop it before changing settings")

    # Storage folder
    if body.data_dir is not None and body.data_dir.strip():
        if orch.busy():
            raise HTTPException(
                status_code=409,
                detail="an agent run is queued or running; stop it before changing storage")
        try:
            service.rebind_store(body.data_dir.strip())
            orch.rebind()
            # Persist so the next launch opens this folder without re-selecting.
            llm.persist_config(
                data_dir=os.path.abspath(os.path.expanduser(body.data_dir.strip()))
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"storage: {e}")

    # Agent project directory ("" restores <data_dir>/workspace). Ignored when
    # CHATTABLE_WORKSPACE pins the workspace — the env var wins.
    if body.workspace_dir is not None:
        if os.environ.get("CHATTABLE_WORKSPACE", "").strip():
            raise HTTPException(
                status_code=400,
                detail="workspace: pinned by CHATTABLE_WORKSPACE; unset it to choose one here")
        try:
            target = service.validate_workspace_dir(body.workspace_dir)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"workspace: {e}")
        new_root = target or os.path.realpath(
            os.path.abspath(default_workspace_path(service.data_dir())))
        if new_root != service.workspace().root and orch.busy():
            raise HTTPException(
                status_code=409,
                detail="an agent run is queued or running; stop it before changing "
                       "the project directory")
        llm.persist_config(workspace_dir=target)
        try:
            orch.rebind()
        except RuntimeError as e:  # a run slipped in between the check and now
            raise HTTPException(status_code=409, detail=str(e))

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


@app.post("/api/sample-project")
def api_sample_project() -> dict:
    """Copy the bundled sample project into <data_dir>/sample-project (never
    overwriting an existing copy) and make it the agent project directory —
    the onboarding "用示例项目试一试" button.

    Follows the same rules as the workspace branch of /api/settings: the
    CHATTABLE_WORKSPACE env var pins the directory (400), and a switch is
    refused while an agent run is queued/running (409).
    """
    if os.environ.get("CHATTABLE_WORKSPACE", "").strip():
        raise HTTPException(
            status_code=400,
            detail="workspace: pinned by CHATTABLE_WORKSPACE; unset it to choose one here")
    bundled = os.path.join(HERE, "examples", "sample-project")
    if not os.path.isdir(bundled):
        raise HTTPException(status_code=404, detail="bundled sample project not found")
    dest = os.path.join(service.data_dir(), "sample-project")
    created = not os.path.exists(dest)
    if created:
        shutil.copytree(bundled, dest)
    try:
        target = service.validate_workspace_dir(dest)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"workspace: {e}")
    if target != service.workspace().root and orch.busy():
        raise HTTPException(
            status_code=409,
            detail="an agent run is queued or running; stop it before changing "
                   "the project directory")
    llm.persist_config(workspace_dir=target)
    try:
        orch.rebind()
    except RuntimeError as e:  # a run slipped in between the check and now
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True, "path": target, "created": created, "config": service.config()}


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
    try:
        info = service.start_turn(content, body.parent_id,
                                  attachments=body.attachments)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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


# ---------------------------------------------------------------------------
# Agent snapshots (view diff / compare / files / adopt of run-result nodes)
# ---------------------------------------------------------------------------

def _require_agent() -> None:
    """Agent endpoints are dead when agent mode is unsafe: the server listens
    beyond loopback without CHATTABLE_AUTH_TOKEN (see service.agent_available)."""
    import service as _svc_mod
    if not _svc_mod.agent_available():
        raise HTTPException(status_code=403, detail=_svc_mod.AGENT_TOKEN_MESSAGE)


@app.get("/api/agent/status")
def api_agent_status() -> dict:
    """Whether agent runs can be used, and where the workspace lives."""
    return service.agent_status()


@app.post("/api/agent/adopt")
def api_agent_adopt(body: AdoptBody) -> dict:
    """Adopt a node's workspace snapshot: snapshot the current state first
    (undoable), restore the target snapshot, record the adoption."""
    _require_agent()
    try:
        return service.adopt_snapshot(body.node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {body.node_id}")
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/agent/undo-adopt")
def api_agent_undo_adopt(body: UndoAdoptBody) -> dict:
    """Undo an adoption: restore the workspace to its pre-adopt snapshot."""
    _require_agent()
    try:
        return service.undo_adopt(body.previous_commit)
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agent/node/{node_id}/diff")
def api_agent_node_diff(node_id: str) -> dict:
    """View diff for a node: everything its agent turn changed (the node's
    end-of-turn snapshot vs the workspace state the turn started from)."""
    try:
        return service.node_diff(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agent/node/{node_id}/adopt-preview")
def api_agent_node_adopt_preview(node_id: str) -> dict:
    """Preview of adopting a node's snapshot: the file changes it would make
    to the CURRENT workspace, flagging files it would delete and files the
    user edited locally that the adopt would overwrite."""
    _require_agent()
    try:
        return service.adopt_preview(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agent/compare")
def api_agent_compare(a: str, b: str) -> dict:
    """Diff the workspace snapshots nearest nodes ``a`` and ``b`` (branch
    compare): unified diff plus a per-file additions/deletions summary."""
    _require_agent()
    try:
        return service.agent_compare(a, b)
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agent/files")
def api_agent_files(node: str) -> dict:
    """Files tracked in the snapshot nearest ``node``."""
    _require_agent()
    try:
        return service.agent_files(node)
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/agent/file")
def api_agent_file(node: str, path: str) -> dict:
    """One file's content at the snapshot nearest ``node`` (200KB cap)."""
    _require_agent()
    try:
        return service.agent_file(node, path)
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Orchestrator (external coding agents over a snapshot DAG)
# ---------------------------------------------------------------------------

class OrchRunsBody(BaseModel):
    from_node: str
    items: list[dict]


class OrchAdoptBody(BaseModel):
    node_id: str


class OrchUndoBody(BaseModel):
    previous_commit: str


class OrchMergeBody(BaseModel):
    nodes: list[str]
    agent: str
    test: bool = True


class OrchProjectBody(BaseModel):
    notes: Optional[str] = None
    test_cmd: Optional[str] = None


class RunBody(BaseModel):
    content: str
    parent_id: Optional[str] = None
    agents: list[str]


class RunMergeBody(BaseModel):
    node_ids: list[str]
    agent: str = "builtin"
    test: bool = True


@app.get("/api/run/executors")
def api_run_executors() -> dict:
    """Which agents can take an instruction (unavailable ones say why)."""
    return {"executors": orch.executors_status(), "agent_ok": __import__("service").agent_available(),
            "workspace": orch.workspace.root}


@app.post("/api/run")
def api_run(body: RunBody) -> dict:
    """交给 agent: run the instruction from a conversation node, once per
    selected agent, each in its own worktree. Progress streams through
    /api/stream/{assistant_id}; the result is a snapshot node in the tree."""
    _require_agent()
    try:
        return {"runs": orch.start_chat_runs(body.content, body.parent_id, body.agents)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"node not found: {body.parent_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/run/merge")
def api_run_merge(body: RunMergeBody) -> dict:
    """Merge ≥2 result nodes (3-way merge of their snapshots) into a new node."""
    _require_agent()
    try:
        return orch.start_chat_merge(body.node_ids, body.agent, run_tests=body.test)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"node not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/run/{run_id}")
def api_run_detail(run_id: str) -> dict:
    """Single run, queued/running/finished: status, timing, step trail,
    changed files (snapshot diff) and the failure reason."""
    _require_agent()
    try:
        return orch.run_detail(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")


@app.get("/api/orch/state")
def api_orch_state() -> dict:
    """DAG nodes, runs, executor availability, limits for this project."""
    _require_agent()
    return orch.state()


@app.post("/api/orch/runs")
def api_orch_runs(body: OrchRunsBody) -> dict:
    """Queue runs: each item is {instruction, agents: [...], after?: run_id}."""
    _require_agent()
    try:
        return {"runs": orch.submit_runs(body.from_node, body.items)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"node not found: {body.from_node}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/orch/runs/{run_id}/cancel")
def api_orch_cancel(run_id: str) -> dict:
    _require_agent()
    if not orch.cancel_run(run_id):
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return {"ok": True}


@app.get("/api/orch/node/{node_id}")
def api_orch_node(node_id: str) -> dict:
    _require_agent()
    try:
        return orch.node_detail(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")


@app.get("/api/orch/node/{node_id}/adopt-preview")
def api_orch_node_adopt_preview(node_id: str) -> dict:
    """Preview of adopting an orchestrator node's snapshot: the file changes
    it would make to the CURRENT workspace, with will_delete / local_edit
    flags and summary counts."""
    _require_agent()
    try:
        return orch.adopt_preview(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/orch/compare")
def api_orch_compare(a: str, b: str) -> dict:
    """Diff two orchestrator nodes' snapshots."""
    _require_agent()
    return orch.compare(a, b)


@app.post("/api/orch/adopt")
def api_orch_adopt(body: OrchAdoptBody) -> dict:
    """Adopt an orchestrator node's snapshot into the workspace (undoable)."""
    _require_agent()
    try:
        return orch.adopt(body.node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"node not found: {body.node_id}")
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/orch/undo-adopt")
def api_orch_undo_adopt(body: OrchUndoBody) -> dict:
    _require_agent()
    try:
        return orch.undo_adopt(body.previous_commit)
    except WorkspaceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/orch/stream")
def api_orch_stream(from_index: int = 0) -> StreamingResponse:
    """SSE: run_started / run_event / run_finished / node_created."""
    _require_agent()
    return _sse(orch.subscribe(from_index))


@app.post("/api/orch/merge")
def api_orch_merge(body: OrchMergeBody) -> dict:
    """3-way merge of ≥2 nodes; conflict/red tests auto-dispatch a resolver run."""
    _require_agent()
    try:
        return orch.merge(body.nodes, body.agent, run_tests=body.test)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/orch/project")
def api_orch_project_get() -> dict:
    """Project notes + test command (used in CONTEXT.md)."""
    _require_agent()
    return orch.project_info()


@app.put("/api/orch/project")
def api_orch_project_put(body: OrchProjectBody) -> dict:
    _require_agent()
    return orch.update_project(body.notes, body.test_cmd)


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
        # Rebind in place: `orch` (and its queue thread) hold this same
        # service object, so swapping the object would leave them behind.
        service.rebind_store(args.data_dir)
        orch.rebind()

    # The Host allowlist must know the ACTUAL bind host (CLI --host wins over
    # the CHATTABLE_WEB_HOST env fallback captured at import time).
    global _BIND_HOST
    _BIND_HOST = args.host
    os.environ["CHATTABLE_BIND_HOST"] = args.host  # read by agent_available()

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
