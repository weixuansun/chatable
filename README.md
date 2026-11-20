# chatable

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/weixuansun/chatable/actions/workflows/ci.yml/badge.svg)](https://github.com/weixuansun/chatable/actions/workflows/ci.yml)

![Concept: fork to explore, prune to focus, build your mindmap](image/concept.png)

A tree-structured research chat web UI for paper and blog reading with LLMs.
Ask a question at any point, fork the conversation, and come back to a mind
map of where you've been.
When you talk to LLM, not all context is neccesary, you can control your context 
by forking at any conversation turn.

It's built for paper/blog reading, math/proof discussion, and algorithm idea
exploration — workflows that have:

- **Natural mind map.** Your conversation is already a tree, so you get a
visual overview of your reading for free.
- **No Context pollution.** Later turns can derail the context; jump back to any
earlier definition, claim, or answer instead of scrolling and re-explaining.
- **Fork from anywhere.** Any message can be a fork point — ask a side
question about a single sentence without touching the main-line context.
- **Cheaper context.** Forking reuses the shared prefix instead of repeating
it, saving context length (and staying cache-friendly).
- **Private by default.** It runs against your own LLM API key and stores
everything on your local filesystem — nothing is uploaded to any
third-party service.
- **Configurable in the browser.** Model, base URL, and API key can be
switched from the settings panel without restarting, and the UI appearance
(theme color, font size) is adjustable too.
- **Bookmarks / read-it-later.** Save URLs (blogs, papers, articles) in the
bookmark sidebar; the server fetches a title and summary, and you can start
a chat from any bookmark to discuss it with the model.
- **Survey report.** From a conversation's "⋯" menu, have the model read the
whole tree — every branch — and write one report on the topic, organized by
theme rather than by turn. An optional focus steers it; the report is saved
alongside your data and can be copied or downloaded as Markdown.



## Screenshots

**Start screen** — pick a suggested prompt or ask anything to open a new trunk.

![Start screen](image/开始界面.png)

**Tree + chat** — the conversation tree on the left; click any assistant node to fork a new branch from it.

![Chat view](image/chat页面.png)

**Mind map** — the whole conversation tree as an interactive map.

![Mind map](image/mindmap.png)

## Installation

Requires Python 3.10+. Dependencies are declared in `pyproject.toml`
(`requirements.txt` is just a `-e .` shim over it).

```bash
pip install -e .
```

For development (runs the test suite from the repo root):

```bash
pip install -e ".[dev]"
pytest
```



## Usage

Set your API key and start the web server:

```bash
DEEPSEEK_API_KEY=sk-... chatable-web
# then open http://127.0.0.1:8000
```

- Click an assistant node in the tree pane to fork the conversation from that
point.
- Conversation state is persisted as JSONL under `~/.chatable`; pass
`--data-dir <path>` to keep history in a specific folder.
- Common overrides via environment variables: `DEEPSEEK_BASE_URL`,
`DEEPSEEK_MODEL`, `CHATTABLE_WEB_HOST` / `CHATTABLE_WEB_PORT` (bind address),
and more — the in-app settings panel covers model, base URL, API key, and
storage folder.

### Exposing the server (optional)

By default the server only accepts requests whose `Host` header is localhost,
`127.0.0.1` or `[::1]` (plus the bind host), and rejects cross-origin
writes — so other devices and web pages you visit cannot reach your chats.
If you deliberately serve beyond localhost:

- `CHATABLE_ALLOWED_HOSTS` — comma-separated extra hostnames to accept
  (e.g. `CHATABLE_ALLOWED_HOSTS=chat.lan,myhost`).
- `CHATABLE_AUTH_TOKEN` — require this token for every request. Open
  `http://<host>:<port>/?token=<token>` once to set the login cookie;
  scripts can use `Authorization: Bearer <token>`.
  Binding to a non-loopback address without a token prints a loud warning.

### Web search (optional)

Automatic `web_search` works out of the box via DuckDuckGo (free, no key).
For higher-quality results, add one or both provider keys — they are tried in
the order **Tavily → Exa → DuckDuckGo**, falling back on failure:

```bash
TAVILY_API_KEY=tvly-... EXA_API_KEY=... chatable-web
```

- `TAVILY_API_KEY` — create a key at [tavily.com](https://tavily.com);
  the free tier includes 1,000 searches/month.
- `EXA_API_KEY` — create a key at [exa.ai](https://exa.ai);
  the free tier adds $10 of credits every month (no card required).

Both are read from the environment at request time, so exporting them in your
shell rc file (e.g. `~/.zshrc`) also works.



## License

[Apache License 2.0](LICENSE)
