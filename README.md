# chatable

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/weixuansun/chatable/actions/workflows/ci.yml/badge.svg)](https://github.com/weixuansun/chatable/actions/workflows/ci.yml)

A tree-structured research chat web UI for DeepSeek / OpenAI-compatible models.
Any message in your conversation can be a fork point: click an assistant node in
the tree pane, ask a new question, and a new branch appears in the persisted
conversation tree.

The goal is paper reading, math/proof discussion, and algorithm idea
exploration where later turns can pollute the context and you often want to
return to an earlier definition, claim, or answer.

![screenshot](docs/screenshot.png)

## Run

```bash
pip install -e ".[web]"
DEEPSEEK_API_KEY=sk-... chatable-web
# then open http://127.0.0.1:8000
```

The web server honours environment variables for the LLM backend
(`DEEPSEEK_API_KEY`, `TAVILY_API_KEY`, the auto-tool and cache settings, …).
Override the bind address with `CHATTABLE_WEB_HOST` / `CHATTABLE_WEB_PORT`.

## Web UI

The UI is a single static page (`static/index.html`, no build step) talking to
a small FastAPI backend:

- **Tree pane (left).** The conversation tree is drawn with `├─ └─ │` guide
  lines. Only **assistant** nodes are selectable — click one to make it the
  fork point for the next message; user and tool nodes are shown but not
  navigable. After a turn finishes, the assistant node is **auto-titled**: a
  short summary of the Q/A is generated in the background by a dedicated
  model (`CHATTABLE_SUMMARY_MODEL`, empty by default) — independent of the
  conversation model — and shown as the node's label. Only empty titles are
  filled, so manual titles are never overwritten; leave the env var empty to
  disable.
- **Conversation pane (right).** Renders the root → selected branch with
  Markdown **and LaTeX math** (KaTeX, supporting `$…$`, `$$…$$`, `\(…\)`,
  `\[…\]`). Reasoning/thinking tokens stream in a dimmed block, separate from
  the final answer.
- **Tool calls fold into the assistant turn.** Automatic `web_search` /
  `web_fetch` / `grep` calls are no longer separate tree nodes; they are shown
  as collapsible cards inside the assistant reply (tool name, arguments,
  status, duration, and full output), stream live while running, and are stored
  in the assistant node's metadata. They are reconstructed as ephemeral
  context for later turns, so byte-stable prefixes (and prefix caching) are
  preserved.
- **Stop button.** During thinking, tool use, or the streamed answer, the send
  button becomes a stop button. Stopping aborts the request, discards that
  turn's nodes, and restores your message to the composer for re-editing.
- **File uploads (drag-and-drop).** Drop a file anywhere on the page — or use
  the paperclip button — to attach it (PDF, plain text, …). The file is
  uploaded to the server's data folder (`uploads/`) and referenced by path in
  your message, so the model reads it with `web_fetch` (full PDF text
  extraction, then `grep` for sections) just like any other local file.
- **Export.** Hover a top-level (trunk) node in the tree pane — or the center
  card in the mind map — and click the download button to export that whole
  conversation tree as a readable Markdown file: depth-first hierarchical
  sections (forks become sub-sections), code/math/quotes preserved verbatim.
  Tool-call records and reasoning are not exported.
- **Settings panel.** The gear button opens a panel to change the LLM model,
  base URL, API key, and storage folder *after* launch — no restart needed.
  Changes apply to the running process only (not persisted; the command-line/env
  config still sets the startup defaults). Secrets are never sent back to the
  browser, and leaving the API key field blank keeps the existing one. Switching
  is refused (409) while a reply is streaming.
- **Bookmarks sidebar.** The bookmark button toggles a right sidebar where you
  can save URLs (blogs, papers, articles). The server fetches the page,
  extracts a title, and asks the LLM for a concise summary. Click a bookmark to
  start a new chat turn seeded with its URL, title, and summary, then discuss
  or analyze it with the model. Bookmarks are stored in `bookmarks.jsonl` next
  to the chat history file.

Conversation state is persisted as JSONL — one file per conversation under
`<data-dir>/trees/<trunk_id>.jsonl`. Mutations append a single line to the
affected conversation's file, so cloud-sync folders (iCloud/Dropbox) only
ship the conversation that changed. A legacy single `chatable.jsonl` (or
`chatable.sqlite3` via `sqlite_to_jsonl.py`) is auto-migrated into `trees/`
on first launch and kept as a `.bak` backup. By default the data directory
is `~/.chatable`.

To keep history in a specific local folder, pass `--data-dir` (alias `-d`) at
launch. The folder is created if missing, its history is loaded on
startup, and new turns are written there:

```bash
chatable-web --data-dir ~/research/transformers
```

The active history file is resolved by precedence:
`--data-dir` → `$CHATTABLE_DB` → `$CHATTABLE_HOME` → `~/.chatable`. The
equivalent env-var overrides still work:

```bash
CHATTABLE_DB=/path/to/chatable.jsonl chatable-web       # exact legacy file
CHATTABLE_HOME=/path/to/chatable-home chatable-web       # whole data dir
CHATTABLE_DATA_DIR=/path/to/data chatable-web             # uvicorn server:app path
```

By default, chatable uses the OpenAI SDK with DeepSeek's OpenAI-compatible API:

```python
OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
```

and sends `model="deepseek-v4-pro"`, `reasoning_effort="high"`, and
`extra_body={"thinking":{"type":"enabled"}}`.

Override any of these via env vars:

| Var                  | Default                                              |
|----------------------|------------------------------------------------------|
| `DEEPSEEK_API_KEY`   | required for default backend                         |
| `DEEPSEEK_BASE_URL`  | `https://api.deepseek.com`                           |
| `DEEPSEEK_MODEL`     | `deepseek-v4-pro`                                    |
| `DEEPSEEK_REASONING_EFFORT` | `high`                                      |
| `DEEPSEEK_THINKING_TYPE` | `enabled`                                      |
| `CLAUDE_MAX_TOKENS`  | `32768`                                              |
| `CHATTABLE_AUTO_TOOLS` | `1`                                                |
| `CHATTABLE_AUTO_TOOL_MAX_ROUNDS` | `10`                                     |
| `CHATTABLE_TOOL_DECISION_MODE` | `native`                                  |
| `CHATTABLE_CACHE_POLICY` | `auto` (`explicit` for Anthropic-style cache control) |
| `CHATTABLE_CACHE_CONTROL_MAX_BREAKPOINTS` | `4`                            |
| `TAVILY_API_KEY`     | optional; enables Tavily web search (else DuckDuckGo)|
| `CHATTABLE_SUMMARY_MODEL` | (empty); model used to auto-title nodes (empty = off) |
| `CHATTABLE_SUMMARY_TITLE_MAX_CHARS` | `48`                                 |
| `CHATTABLE_WEB_HOST` | `127.0.0.1`                                          |
| `CHATTABLE_WEB_PORT` | `8000`                                               |

## Automatic Web Tools

Normal chat turns can automatically use read-only web tools. Generation runs
as a single streaming loop: every round streams with the tool definitions
attached, so the model may answer, call tools, or mix both at any point:

- `web_search` for current web/source discovery.
- `web_fetch` for reading a specific URL.
- `grep` for searching within previously fetched content.

Tool calls are executed as they arrive (rendered as collapsible cards in the
UI, stored in the assistant node's metadata), folded back into the
conversation as native tool messages, and the model continues — you see
"search → think → answer → maybe search more" as one interleaved reply. The
loop ends when the model stops calling tools; `CHATTABLE_AUTO_TOOL_MAX_ROUNDS`
(default `10`) is only a fuse — once hit, the final round streams without
tools so the model must wrap up.

Prefix-cache note: a turn's tool exchange is persisted as native OpenAI tool
messages (`metadata.tool_messages`, key order canonicalized) and replayed
byte-identically on later turns, so the next request shares its prefix with
the previous turn's and the whole investigation stays cache-hittable. Set
`CHATTABLE_NATIVE_TOOL_HISTORY=0` to fall back to folding results into
system text (also the automatic behavior for older nodes).

## Node titles & summaries

After each turn, the backend generates two labels for the assistant node in a
single cheap call (thinking disabled, inputs truncated):

- **title** — a ≤48-char name shown in the tree and mind map;
- **summary** — a one-sentence gist shown as the tree row's second line and
  the mind map card's answer area, and included in tree search (title +
  summary + content preview, multi-word AND).

Generation is **on by default** using the main conversation model. Set
`CHATTABLE_SUMMARY_MODEL` to a cheaper model to cut cost, or
`CHATTABLE_AUTO_TITLE=0` to disable. Manual rename: hover a tree row and
click the pencil — a manual name locks the node against auto-titling
(`metadata.title_locked`); submit an empty name to unlock and restore the
automatic flow (`POST /api/node/{id}/rename`).

This requires a backend that streams `tool_calls` deltas (verified against
DeepSeek). On backends without that support, set
`CHATTABLE_TOOL_DECISION_MODE=prompt` to fall back to the legacy two-phase
flow (invisible decision rounds, then a final answer stream).

Disable automatic tools:

```bash
CHATTABLE_AUTO_TOOLS=0 chatable-web
```

By default, automatic tool selection uses native chat-completions `tools` /
`tool_calls`. If your backend doesn't support native tool calling, the service
falls back to a JSON decision prompt. Override the mode:

```bash
CHATTABLE_TOOL_DECISION_MODE=prompt chatable-web
```

## Context model

Each request is built from:

1. A frozen system prompt.
2. Pinned nodes that are not already on the selected path.
3. The root → selected branch path, including any `system_note` tool results,
   plus the new user turn.

If a path grows past `CHATTABLE_CONTEXT_MAX_PATH_MESSAGES` (default `80`), the
backend keeps the most recent `CHATTABLE_CONTEXT_KEEP_RECENT_MESSAGES`
(default `24`) as exact messages and inserts short excerpts for earlier turns.

## Caching

The system prompt is a frozen literal — no timestamps, UUIDs, or session IDs.
`messages` bytes are stable along a branch — forks from the same ancestor send
byte-identical prefixes. Any opaque upstream prefix caching has the best
possible chance of landing.

If your backend accepts Anthropic-style content blocks with `cache_control`, opt
in explicitly:

```bash
CHATTABLE_CACHE_POLICY=explicit chatable-web
```

In explicit mode, chatable marks up to four stable system-context messages
before the latest user turn with `{"cache_control":{"type":"ephemeral"}}`.
Leave this disabled for providers that reject non-standard content block
fields.

## Architecture

- `main.py` — core LLM backend calls, stream parsing, context construction.
- `models.py` — `Node` dataclass.
- `store.py` / `jsonl_store.py` — JSONL-backed `TreeStore`. One file per
  conversation under `<data-dir>/trees/`, append-only writes with
  last-write-wins on load, atomic rewrites for compaction/deletes.
  Parent-pointer tree with pinned nodes and persistent metadata.
  `path_to_root` is the function that defines the selected branch.
  `delete_subtree` removes a node and its descendants (used to discard an
  interrupted turn).
- `tools.py` — read-only tool registry shared by automatic and manual tool
  execution.
- `web_tools.py` — `web_search` (Tavily, DuckDuckGo fallback), `web_fetch`,
  and `grep` implementations.
- `service.py` — print-free orchestration layer for the web UI. Wraps
  the store/`main`/tools, exposes the tree/branch as JSON, streams a turn as
  structured events, suppresses hallucinated tool-call XML, folds tool results
  into the assistant node, and renders per-tree Markdown exports.
- `server.py` — FastAPI app and JSON/SSE endpoints serving `static/index.html`
  (run via the `chatable-web` console script).

## Research workflow

The tree structure is central: fork from any earlier assistant reply to explore
an alternative direction without polluting the main branch. Pin important
definitions or lemmas so they stay in context across branches.

## License

[Apache License 2.0](LICENSE)
