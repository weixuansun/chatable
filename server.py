"""FastAPI web server for chatable.

Reuses ``service.ChatService`` (which in turn reuses store/main/tools unchanged)
and exposes a small JSON + SSE API consumed by ``static/index.html``.

Run:
    chatable-web                      # console script (see pyproject.toml)
    python -m uvicorn server:app      # or directly
    python server.py                  # convenience launcher

Honours the same env vars as the CLI (CHATTABLE_DB, DEEPSEEK_API_KEY, …).
Set CHATTABLE_WEB_HOST / CHATTABLE_WEB_PORT to change the bind address.
"""
from __future__ import annotations

import json
import os
from typing import Optional
from urllib.parse import quote

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


class BookmarkBody(BaseModel):
    url: str


class BookmarkOpenBody(BaseModel):
    parent_id: Optional[str] = None


@app.get("/api/config")
def api_config() -> dict:
    return service.config()


@app.post("/api/settings")
def api_settings(body: SettingsBody) -> dict:
    """Apply model/storage settings entered in the web UI.

    The model config (base_url/api_key/model) is persisted to
    ``main.SETTINGS_PATH`` and reloaded on the next launch; the storage
    folder switch remains session-only. Refuses (409) while any reply is
    generating, so a switch never orphans a background turn. Applies storage
    first, then the model config; either is optional. Returns the fresh
    config on success.
    """
    if service.has_active_streams():
        raise HTTPException(status_code=409,
                            detail="a reply is generating; stop it before changing settings")

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
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    MAX = 50 * 1024 * 1024  # 50 MB guard
    if len(data) > MAX:
        raise HTTPException(status_code=413, detail="file too large (max 50 MB)")
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

    print(f"chatable web → http://{args.host}:{args.port}")
    print(f"  backend: {llm.backend_description()}")
    print(f"  history file: {service.store.db_path}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
