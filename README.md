# chatable

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/weixuansun/chatable/actions/workflows/ci.yml/badge.svg)](https://github.com/weixuansun/chatable/actions/workflows/ci.yml)

![Concept: fork to explore, prune to focus, build your mindmap](image/concept.png)

**Ask once, let AI try several approaches side by side, compare them, and
keep the best one.** chatable is a conversation tree where chat and code
changes live on the same map: fork at any turn, hand a node to one or more
coding agents, and every reply node remembers both the conversation and the
files at that moment. Agent changes stay in an isolated copy until you click
**采用 (Adopt)**, and adopting is undoable.

## Why chatable

The pain points it targets:

1. **Chats only go one way.** Trying a different idea means a new chat (and
   lost context) or rewriting the old one until it is a mess.
2. **You can't try several approaches — or several agents — at once.** An AI
   coding session usually commits to one approach; if it's wrong you start
   over.
3. **Plans discussed in a chat get copy-pasted into the agent by hand.**
4. **Agents edit your real project.** When they get it wrong, rolling back is
   on you.
5. **Attempts are hard to compare or combine.** Results end up scattered
   across chats and branches.
6. **You can't go back to a midpoint.** Returning to turn 5 of the chat
   leaves the code at turn 10.

How chatable answers them:

- **Fork from anywhere (1).** Any message is a fork point; side questions
  don't pollute the main line, and the tree doubles as a mind map.
- **Race approaches and agents (2).** One 交给 agent instruction can go to the
  built-in DeepSeek agent, Claude Code, Codex or Kimi Code at once — each runs
  on its own branch, and the results sit next to each other on the map.
- **Chat, then hand off (3).** The same node can switch from 对话 (chat) to
  交给 agent; the branch's discussion is written into the agent's context
  file automatically.
- **Changes stay in an isolated copy until you adopt them (4).** Every run
  works in a throwaway worktree checked out from its parent node's snapshot;
  your project directory changes only on Adopt, which backs it up first and
  can be undone.
- **Compare and merge (5).** Diff any two nodes, or Shift+click several and
  merge them (3-way); conflicts and new test failures are handed to a
  resolver agent.
- **Conversation and code go back together (6).** Each node stores a file
  snapshot, so forking from an old node continues from both its context and
  its files.

Also included, from chatable's research-chat roots:

- **Private by default.** Your own LLM API key, everything stored on your
  local filesystem.
- **Configurable in the browser.** Model, base URL, API key and appearance
  without restarting.
- **Bookmarks / read-it-later** and **survey reports** that read a whole
  tree and write one themed report.

### Status

chatable is a local, single-user prototype. Known gaps before the
"isolated until Adopt" promise holds everywhere:

- The legacy agent-mode toggle still exists; continuing a branch there
  restores your workspace directory in place instead of using a separate copy.
- External CLIs (Claude Code, Codex, Kimi Code) run in the isolated worktree
  but are **not** sandboxed, so they could still write outside it.
- The macOS Seatbelt sandbox is covered only by unit tests on Linux, and
  full Claude Code / Codex runs have not been verified end to end yet.

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
`127.0.0.1` or `[::1]` (plus the bind host from `--host` /
`CHATTABLE_WEB_HOST`), and rejects cross-origin writes — so other devices and
web pages you visit cannot reach your chats. If you deliberately serve beyond
localhost:

- `CHATTABLE_ALLOWED_HOSTS` — comma-separated extra hostnames to accept
  (e.g. `CHATTABLE_ALLOWED_HOSTS=chat.lan,myhost`). **Required** when binding
  to `0.0.0.0` / `::` or a LAN address: requests arriving with this machine's
  LAN IP or hostname are otherwise rejected with 400.
- `CHATTABLE_AUTH_TOKEN` — require this token for every request. Open
  `http://<host>:<port>/?token=<token>` once to set the login cookie;
  scripts can use `Authorization: Bearer <token>`.
  Binding to a non-loopback address without a token prints a loud warning.

(The single-T spellings `CHATABLE_ALLOWED_HOSTS` / `CHATABLE_AUTH_TOKEN` are
accepted as aliases.)

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

`web_fetch` refuses non-public addresses (SSRF guard). If you run a TUN-mode
proxy (Clash/Mihomo "fake-IP"), hostnames resolve into 198.18.0.0/15 — that
range is allowed by default so fetches keep working. Override with
`CHATTABLE_FETCH_ALLOW_NETS` (comma-separated CIDRs, e.g. to allow an
intranet range; set it empty to disable the default).

### Agent mode (prototype, optional)

Agent mode lets the model work autonomously in a dedicated workspace
(workspace file tools + shell, per-step tree nodes, shadow-git snapshots,
approvals). Steps stream live with their own diff / output (shell steps also
list the files they touched) but have no buttons; when the task finishes the
reply becomes a **node** with exactly one workspace snapshot and five
actions: **View diff** (改动, the whole task's change), **Compare** (对比),
**Files** (文件), **Adopt** (采用 — make the workspace match this node,
backed up and undoable) and **Fork from here** (从此分叉 — continuing from an
older node restores its files first). The mind map and the sidebar tree show only
these nodes (question + reply, with status and files-changed count); the
individual steps appear only in the main chat, and branches hang off the
reply node. It is **off by default** and only available when
`CHATTABLE_AUTH_TOKEN` is set — the toggle lives in the composer. See
[docs/agent-api.md](docs/agent-api.md) for the full API. After an agent
reply, typing again continues the same branch by default (the ✕ on the
composer pill starts a new conversation); "always allow" approvals are
scoped to the branch they were granted on.

Shell commands run through two safety layers:

- **Sandbox (real boundary).** When `bwrap` (bubblewrap) is installed,
  `run_shell` commands execute confined: read-only system dirs, the workspace
  read-write at its real path, private `/tmp`, hidden home and data
  directories, a synthetic `HOME`, no network by default. PATH directories
  stay available read-only — narrowly: a PATH entry's parent is bound only
  for real venv/conda roots (`pyvenv.cfg` / `conda-meta/`), and non-venv
  PATH dirs under `$HOME` are skipped unless
  `CHATTABLE_AGENT_SANDBOX_PATH_HOME=1`.
  - **macOS:** the same policy via Seatbelt (`/usr/bin/sandbox-exec`, built
    in): writes only to the workspace, a per-run temp dir (`TMPDIR`) and a
    synthetic `HOME`; `/Users`, `$HOME` and the data dir are unreadable except
    the workspace and the toolchain (venv roots on PATH + the base Python they
    point at); no network at all (IP and unix sockets) by default; clean env.
    The badge shows `sandbox: seatbelt · no net`. **The Seatbelt profile is
    covered only by mocked unit tests on Linux CI — please report anything
    that breaks on macOS.** Quick check from the repo dir (expect `ok` and
    `done` with no `LEAK` lines):

    ```
    python3 -c "import os,subprocess,shell_sandbox as s;H=os.environ['HOME'];rt=s.make_run_tmp();a,e=s.build_seatbelt_argv(f'echo ok; touch {H}/sb-x 2>/dev/null && echo WRITE-LEAK; ls {H}/Desktop >/dev/null 2>&1 && echo READ-LEAK; curl -sI -m5 https://example.com >/dev/null 2>&1 && echo NET-LEAK; touch sb-ok && rm sb-ok && echo done',os.getcwd(),rt);print(subprocess.run(a,env=e,capture_output=True,text=True).stdout)"
    ```
  - `CHATTABLE_AGENT_SANDBOX=auto|bwrap|seatbelt|off` (default `auto`: bwrap
    on Linux, seatbelt on macOS, when available; `/api/agent/status` and the
    composer badge warn when none is in effect).
  - `CHATTABLE_AGENT_SHELL_NET=1` allows network access inside the sandbox.
- **Denials are explained.** When a sandboxed command fails because the
  sandbox blocked it (no network, a path outside the workspace, a tool under
  your home dir), the agent gets a `[sandbox]` note telling it not to retry,
  and the chat shows which switch allows it (`CHATTABLE_AGENT_SHELL_NET=1`,
  `CHATTABLE_AGENT_SANDBOX_PATH_HOME=1`). Defaults are not relaxed.
- **Nested repositories are snapshotted as files.** A `git clone` / `git init`
  the agent runs inside the workspace is stored file-by-file (honouring that
  repo's own `.gitignore`; its `.git` is not stored), so follow-ups, forks and
  parallel runs see the code instead of an empty directory.
- **Deny-list (advisory tripwire).** Obviously destructive commands
  (`rm -rf /`, `sudo`, `curl | sh`, `dd of=/dev/…`, `mkfs`, fork bombs,
  `git push --force`, writes to `~/.ssh`…) are refused outright.
  `CHATTABLE_AGENT_SHELL_DENYLIST=off` disables it — note the sandbox, not
  the deny-list, is the real protection.

### 交给 agent: external coding agents in the conversation tree

One switch above the composer decides what a message does:

- **对话** — a normal chatable reply; nothing touches your files.
- **交给 agent** — the message is an instruction for a coding agent
  (built-in DeepSeek agent, Kimi Code, Claude Code or Codex; pick one or
  several — several = race the same instruction). It runs in an isolated git
  worktree from the current node's file snapshot, its steps stream into the
  main chat, and the result becomes a normal reply node with status, files
  changed and test result. The real workspace changes only when you click
  **采用 (Adopt)**.

After sending, the composer stays on the same node, so sending another
instruction starts a parallel task; the 「从这里发出的 agent 任务」 list under
the conversation shows each one's live status (查看 / 取消). In the mind
map, run cards show the agent and status; **Shift+click** (or the ☐ button)
two or more cards with snapshots to **合并** (3-way merge under their common
ancestor; conflicts or new test failures are handed to a resolver agent
automatically) or **对比**. The agent's context (`.chatable/CONTEXT.md`) is
built from the conversation path. API: see "Agent runs in the conversation"
in [docs/agent-api.md](docs/agent-api.md); design notes in
[docs/orchestrator-design.md](docs/orchestrator-design.md).

- Executors are auto-detected (`CHATTABLE_KIMI_BIN` / `CHATTABLE_CLAUDE_BIN`
  / `CHATTABLE_CODEX_BIN` to override); unavailable ones show greyed out.
- Per-agent concurrency: `CHATTABLE_ORCH_LIMITS` (default
  `kimi=2,claude=2,codex=2,builtin=3`). Run/test timeouts:
  `CHATTABLE_ORCH_RUN_TIMEOUT_S` (1800), `CHATTABLE_ORCH_TEST_TIMEOUT_S` (300).
- Requires `CHATTABLE_AUTH_TOKEN` like agent mode.
- Sandboxing: `builtin` runs use the agent shell sandbox (bwrap on Linux,
  Seatbelt on macOS) confined to the run's worktree. External CLIs (kimi,
  claude, codex) are **not** wrapped — they need network to their APIs and
  use their own permission modes; they are isolated only by running in a
  throwaway worktree (the real workspace changes only on Adopt).





## License

[Apache License 2.0](LICENSE)
