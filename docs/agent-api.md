# Agent API (prototype)

Agent work in chatable runs as **agent runs** (the UI's 「交给 agent」): each
run executes a coding-agent loop in an isolated git worktree checked out from
a conversation node's file snapshot, and the result is stored as a snapshot
on the run's reply node. The real workspace changes only on **Adopt**. The
built-in agent loop (workspace-confined file and shell tools, a persisted
**step node** per tool call, shadow-git snapshots) is retained as the engine
of the orchestrator's `builtin` executor, running inside the run's isolated
worktree. The legacy in-place agent mode — toggling the loop to run directly
in the main workspace from a chat message, restoring that workspace in place
on follow-ups — has been removed; agent turns are internal to the
orchestrator and always work on a throwaway copy.

No access token is needed while the server listens on loopback (the default
`127.0.0.1`); when it is bound beyond loopback (`--host 0.0.0.0`, a LAN
address…) without `CHATTABLE_AUTH_TOKEN`, agent runs and the `/api/agent/*`
endpoints refuse (`403`, reason "Agent mode is off because the server
listens beyond localhost without CHATTABLE_AUTH_TOKEN…").

## Endpoints

The `/api/agent/*` endpoints below serve the **snapshot nodes** produced by
agent runs: viewing a node's changes, comparing two nodes, browsing the
files at a snapshot, and adopting a snapshot into the real workspace. All
endpoints sit behind the existing `SecurityMiddleware` (Host allowlist,
Origin/Referer CSRF check, bearer/cookie token). When a token is configured
it is required for these routes like any other.

| Method & path | Body / query | Description |
|---|---|---|
| `GET /api/agent/status` | — | `{available, reason, workspace, auto_approve, git}` — `available=false` (with `reason`) when no token is set; `workspace` is the resolved workspace root (created on demand); `git` reports shadow-snapshot availability. |
| `GET /api/agent/node/{node_id}/diff` | — | Node **View diff**: `{ok, node_id, commit, base_commit, diff, files: [{path, status, additions, deletions}]}` — everything the agent turn changed: the node's end-of-turn snapshot vs `ws_parent_commit` (the workspace when the turn started). |
| `GET /api/agent/compare?a=&b=` | — | Branch compare: `{ok, a_commit, b_commit, diff, files: [{path, status, additions, deletions}]}` between the snapshots nearest the two nodes. |
| `GET /api/agent/files?node=` | — | `{ok, commit, files: [path…]}` — files tracked in the snapshot nearest the node. |
| `GET /api/agent/file?node=&path=` | — | `{ok, commit, path, content, truncated}` — one file at that snapshot (200KB cap, path confined). |
| `POST /api/agent/adopt` | `{node_id}` | Adopt the node's snapshot into the workspace: snapshots the current state first, restores, records `adopted_at`/`adopted_commit`/`previous_commit` on the node. Returns `{ok, adopted_commit, previous_commit, files}`. |
| `POST /api/agent/undo-adopt` | `{previous_commit}` | Restore the pre-adopt snapshot (commit must exist in the shadow repo). |

### Agent runs in the conversation (CHA-5, the UI's 「交给 agent」 mode)

| Method & path | Body / query | Description |
|---|---|---|
| `GET /api/run/executors` | — | `{executors: {id: {available, reason}}, agent_ok, workspace}` — ids `builtin`, `kimi`, `claude`, `codex`; unavailable ones carry the reason. |
| `POST /api/run` | `{content, parent_id?, agents: [id…]}` | Run the instruction from conversation node `parent_id` (none = new conversation), once per agent (several = race), each in its own worktree from the parent's snapshot. Creates a user node (`orch_run`, `executor`) and an assistant node (`run_status` queued→running→ok/failed/cancelled) per agent and returns `{runs: [{run_id, user_id, assistant_id, agent}]}`. Progress streams like a chat turn on `/api/stream/{assistant_id}` (`assistant_start` with `run: true`, `notice`, `step_start`/`step_done`, `plan`, `done`). The finished assistant node carries `ws_commit` / `ws_parent_commit`, `files_changed`, `facts.test` (baseline-relative) and the agent's conclusion as content, so the node buttons (改动/对比/文件/采用/从此分叉) work unchanged. `.chatable/CONTEXT.md` is built from the conversation path; the agent session is resumed when the same agent continues its own run. Several sends from the same node = parallel siblings. |
| `POST /api/run/merge` | `{node_ids: [id…≥2], agent?, test?}` | 3-way merge of the nodes' snapshots under their lowest common ancestor → a new merge node (`orch_merge`, `merged_from`, `run_status` ok/conflict/tests_failed/failed). On conflict or new test failures a resolver run with `agent` starts automatically from the merge node. Returns `{user_id, assistant_id, parent_id}`. |
| `GET /api/run/{run_id}` | — | Single run, queued/running/finished (404 when unknown): `{id, agent, status, instruction, created_at, started_at, finished_at, duration_s, user_id, assistant_id, node_id, steps_done, current_step, steps, files, error}`. `steps` is the compact, persisted tool trail (`{name, args_summary, at, ok, output?}`, capped at the latest 200 — written as the run progresses, so it survives a restart); `current_step` is the open (or last) one. `files` is the snapshot diff `[{path, status: A/M/D…, added, removed}]` — live from the run's worktree while running, final once finished. On failed/cancelled runs `error` carries the reason and `output_tail` the last ≤3 step outputs. The assistant node's `run_id` metadata links a chat node to its run. |
| `POST /api/stop/{assistant_id}` | — | On a run node: cancel it (queued → cancelled at once; running → stopped, partial work is still snapshotted). |

`POST /api/send` is always plain chat — it no longer accepts an `agent`
flag; agent work goes through `POST /api/run` above. The old orchestrator
overlay is gone from the UI; `/api/orch/*` is kept for API use.

`GET /api/agent/compare?a=&b=` also accepts `a=workspace` (current workspace
state, snapshotted on the spot) — used by the adopt dialog's preview.

## Shell safety

`run_shell` is the dangerous one; it has two layers:

1. **Sandbox (the real boundary).** With `bwrap` installed and
   `CHATTABLE_AGENT_SANDBOX=auto` (default), commands run confined:
   read-only `/usr`/`/bin`/`lib*`/`/etc`/`/opt`, workspace read-write at the
   same path, private tmpfs `/tmp`, tmpfs over `/home` and `/root`, the
   chatable data dir hidden, synthetic `HOME` (`/tmp/agent-home`),
   `--unshare-pid/ipc/uts`, `--die-with-parent`, and **no network** unless
   `CHATTABLE_AGENT_SHELL_NET=1`. PATH directories are re-mounted read-only
   so an existing toolchain outside the workspace keeps working — narrowly:
   a PATH entry's parent is bound only when it is a real venv/conda root
   (`pyvenv.cfg` / `conda-meta/`, never a home dir or the data dir), and
   PATH entries under `$HOME` that are not venv bins are skipped entirely
   unless `CHATTABLE_AGENT_SANDBOX_PATH_HOME=1`. `CHATTABLE_AGENT_SANDBOX=
   off` disables. **macOS:** `seatbelt` backend — `sandbox-exec -p <profile>
   -D WORKSPACE=… -D RUN_TMP=…` with `(allow default)`, `(deny file-write*)`
   + re-allow the workspace, a per-run temp dir (TMPDIR, synthetic HOME) and
   tty/null devices; `(deny file-read-data)` on `/Users`, `/var/root`,
   `$HOME` and the data dir + re-allow the workspace, run temp and venv
   toolchain dirs; `(deny network*)` unless `CHATTABLE_AGENT_SHELL_NET=1`;
   paths realpath'd (`/var`→`/private/var`); clean env (PATH/HOME/TMPDIR/
   LANG/TERM). Mode `auto` = bwrap on Linux, seatbelt on macOS. When no
   backend is available, `/api/agent/status` reports
   `sandbox: "off"` plus a loud `sandbox_warning` (the composer shows a red
   "no sandbox" badge).
2. **Deny-list (advisory).** `rm -rf /|~`, `sudo`, `curl|sh`, recursive
   `chmod 777` on `/`/`~`, `dd of=/dev/…`, `mkfs`, fork bombs,
   `git push --force`, and writes into `.ssh`/`.aws`/`.gnupg` are refused
   outright, before the command runs. `CHATTABLE_AGENT_SHELL_DENYLIST=off`
   disables it. It is a tripwire, not a boundary — the sandbox is.

Every shell runs with a scrubbed environment: credentials
(`*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL(S)`,
`*_AUTH`, `AWS_*`, `GH_*`/`GITHUB_*`, `OPENAI_*`, `DEEPSEEK_*`,
`ANTHROPIC_*`, `CHATTABLE_AUTH_TOKEN`) are dropped.

`/api/agent/status` gains `{sandbox, sandbox_warning, shell_network,
shell_denylist}`.

Existing tree endpoints (`GET /api/tree`, `GET /api/node/{id}`,
`GET /api/path/{id}`) return step nodes like any other node. The
conversation view renders them; the **sidebar tree and the mind map do not**
— they show only conversation nodes (question + reply, with the reply's
status: running / done / error / stopped / round-limit and `files_changed`),
lifting any children of step nodes (legacy follow-ups) onto the owning reply
so branches always hang off reply nodes. Agent assistant nodes in
`/api/tree` carry `agent`, `stop_reason` and `files_changed`.

## SSE events (agent run turns only, in addition to the existing ones)

Existing events (`user_node`, `assistant_node`, `assistant_start`,
`reasoning`, `delta`, `tool_call`, `tool_result`, `reset`, `error`, `done`)
are unchanged. `assistant_start` gains `agent: true` and `workspace`.

```json
{"type": "step_start", "id", "parent_id", "tool", "args", "args_summary"}
{"type": "step_output", "step_id", "chunk"}
{"type": "step_done", "id", "status", "tool", "ok", "duration_ms", "touched_files"}   // no per-step ws_commit (node-level snapshots)
{"type": "plan", "step_id", "items": [{"content", "status"}]}
{"type": "notice", "text"}
```

- `step_output` streams `run_shell` stdout/stderr incrementally while the
  command runs (the persisted step keeps head+tail 8KB with an omission
  note). Stop kills the shell's whole process group.
- Run turns auto-approve every tool call, so no approval events are emitted.
- `plan` carries the latest `update_plan` checklist; the final plan is also
  persisted on the assistant node metadata as `plan`.
- `notice` signals loop-level events — currently the round-limit fuse: after
  `CHATTABLE_AGENT_MAX_ROUNDS` tool rounds the model gets one final
  no-tools round to write its summary.

## Step node schema

Each executed agent tool call is one persisted node, written **as it
happens** and chained: first step's parent is the assistant node of the
turn, each next step's parent is the previous step — the trajectory is a
path in the tree (`user → assistant → step → step → …`). Steps carry no
snapshot; the turn's single end-of-turn `ws_commit` (+ `ws_parent_commit` =
the base) lives on the assistant node's metadata.

```json
{
  "role": "step",
  "content": "write_file src/a.py",           // = args_summary, one line
  "metadata": {
    "step": true,
    "tool": "write_file",
    "call_id": "call_0_1",
    "args": {"path": "src/a.py", "content": "…"},
    "args_summary": "write_file src/a.py",
    "status": "ok | error",      // "pending_approval"/"denied" only in legacy conversations
    "result": "…tool output, truncated to 4000 chars…",
    "error": "…only when status == error…",
    "diff": "…unified diff, write/edit only…",
    "duration_ms": 42,
    "touched_files": [{"status": "A", "path": "out.txt"}]  // run_shell only, when any
    // legacy conversations may also have "ws_commit"/"ws_parent_commit" here
  }
}
```

Context building excludes step nodes from the model messages (they are
display/snapshot anchors); the assistant node's persisted `tool_messages`
replay keeps the OpenAI message sequence valid and byte-stable. **Forking
from a step truncates the replay at that step**: only tool calls whose step
nodes are on the fork's path are replayed (a round's `tool_calls` list is
split when the fork lands mid-round; unanswered ids get a stub result), the
turn's final answer is dropped (its work was rolled back), and a system note
tells the model it is continuing from step k with the workspace restored to
that point. When the path leaves the turn through its reply node (a normal
follow-up/fork) or covers the turn's WHOLE chain (legacy follow-ups stored
under the step-chain tail), the full replay and the final answer are
kept — truncation only happens on mid-turn forks.

## Agent tools (model-facing, builtin agent loop)

| Tool | read_only | Notes |
|---|---|---|
| `list_dir(path?, depth?)` | yes | flat workspace-relative paths (dirs end with `/`), default depth 2, skips noise dirs (`.git`, `__pycache__`, `.pytest_cache`, `node_modules`, `.venv`, `dist`, `build`, …) and root `.gitignore`, capped at 500 entries |
| `glob(pattern, path?)` | yes | `**/*.py`-style file finding, newest first, capped with a "N more matches" note |
| `read_file(path, offset?, limit?)` | yes | line numbers, 200KB cap with truncation note, binary refused |
| `grep(pattern, path?, glob?, ignore_case?, context?)` | yes | pure-Python regex; `path:line: text` output (context lines use `-`), capped with a "N more matches" note |
| `write_file(path, content)` | no | creates parents, ≤1MB, returns unified diff |
| `edit_file(path, old_string, new_string, replace_all?, expected_replacements?)` | no | not-found errors show the most similar region with line numbers; non-unique errors list occurrence line numbers; returns unified diff |
| `run_shell(command, timeout?)` | no | cwd=workspace, streams output as `step_output` events, head+tail 8KB kept, timeout default 120s / max 600s, stop kills the whole process group, credentials env vars scrubbed, **sandboxed when available (bwrap on Linux, Seatbelt on macOS)** (no network by default), deny-list refuses destructive commands outright |
| `update_plan(items)` | yes | TodoWrite-style plan checklist; latest plan emitted as `plan` SSE and persisted on the assistant node |

All paths are confined to the workspace: real paths must stay under the
workspace realpath (`..`, absolute paths outside, and symlink escapes are
rejected).

The agent system prompt instructs explore-before-edit, read-before-edit,
minimal edits, plan tracking, test-after-change and verify-before-claiming.
If `AGENTS.md`, `CLAUDE.md` or `.chatable/instructions.md` exists at the
workspace root it is appended to that prompt (20KB cap).

Agent loop limits: up to 40 rounds (`CHATTABLE_AGENT_MAX_ROUNDS`), up to 8
tool calls per round (`CHATTABLE_AGENT_MAX_CALLS_PER_ROUND`), and **no**
cross-round dedupe of repeated calls. Consecutive read-only calls of a round
execute concurrently (`CHATTABLE_AGENT_PARALLEL_READS`, default 4); mutating
calls stay sequential, and results always fold back in call order. Normal mode keeps its 3-calls/round, 10-round, dedupe behaviour
untouched. When the round limit is hit the model gets one final no-tools
round to summarize (plus a `notice` event): the wrap-up prompt contains a
**server-computed factual record** — the files actually changed this turn
(shadow-git diff of the turn's base snapshot vs now, with +/- counts or
"NONE") and the last shell commands with exit status — and the assistant
node is marked `stop_reason: "round_limit", incomplete: true` so the UI can
offer a Continue affordance on that branch.

Long turns: past ~60k estimated tokens of loop context
(`CHATTABLE_AGENT_CTX_BUDGET_TOKENS`), tool results older than the last 6
are shrunk to head/tail excerpts with a note. Only the live loop context is
affected; persisted step results stay full.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CHATTABLE_AUTH_TOKEN` | (unset) | Optional on loopback; **required for agent runs and the `/api/agent/*` endpoints when the server binds beyond loopback.** Also the existing server auth token. |
| `CHATTABLE_WORKSPACE` | `<data_dir>/workspace` | Agent workspace root (created if missing). |
| `CHATTABLE_AGENT_AUTO_APPROVE` | (unset) | Comma-separated: `writes` (write_file+edit_file), `shell` (run_shell), `all`. |
| `CHATTABLE_AGENT_MAX_ROUNDS` | `40` | Agent loop round cap. |
| `CHATTABLE_AGENT_PARALLEL_READS` | `4` | Max concurrent read-only tool calls per round. |
| `CHATTABLE_AGENT_MAX_CALLS_PER_ROUND` | `8` | Agent loop per-round tool-call cap. |
| `CHATTABLE_AGENT_APPROVAL_TIMEOUT_S` | `600` | Approval wait before treating as denied. |
| `CHATTABLE_AGENT_CTX_BUDGET_TOKENS` | `60000` | Estimated-token budget before old tool results are shrunk. |
| `CHATTABLE_AGENT_SANDBOX` | `auto` | `auto` (bwrap on Linux / seatbelt on macOS, if available) / `bwrap` / `seatbelt` / `off`. |
| `CHATTABLE_AGENT_SHELL_NET` | (unset) | `1` allows network inside the shell sandbox. |
| `CHATTABLE_AGENT_SHELL_DENYLIST` | (unset) | `off` disables the advisory command deny-list. |
| `CHATTABLE_AGENT_SANDBOX_PATH_HOME` | (unset) | `1` also re-mounts non-venv PATH dirs under `$HOME` (skipped by default). |

## Shadow git snapshots

Snapshots live in `<data_dir>/agent_shadow.git` and operate with
`git --git-dir=<shadow> --work-tree=<workspace>` — a `.git` inside the
workspace is never read, written, or cleaned (it is excluded from tracking,
and `git clean` skips it). **Snapshots are node-granular.** A **base snapshot** of the pre-turn
workspace is committed at the start of every agent turn and recorded on the
user node (`ws_commit`). Steps do **not** snapshot: write/edit steps keep
their inline `diff`, shell steps keep their output plus `touched_files`
(`[{status: A|M|D, path}]`, computed by staging the shadow index around the
command — no commit). When the turn ends — finished, stopped, round-capped
or crashed — exactly **one** snapshot is committed and recorded on the
assistant node as `ws_commit`, with `ws_parent_commit` = the turn's base.
A turn that changed nothing reuses the existing HEAD (no empty commit).

A node's file state is its own `ws_commit`, else the nearest
ancestor-or-self one (a step resolves to its turn's base: the assistant
node's end-of-turn commit is in its steps' future). **Backward
compatibility:** conversations saved with per-step snapshots still load —
an assistant node without `ws_commit` falls back to its last step's commit
(also surfaced as `ws_commit` in `/api/path` and the mind-map data). If git
is not installed, snapshots degrade to no-ops (`ws_commit` absent) and
restore/diff calls return a clear error.

**UI.** Steps stream live with their inline diff / output but carry no
buttons. Under a finished agent reply (and on its mind-map card) the node
offers: **View diff** (改动 — the whole turn vs its base,
`/api/agent/node/{id}/diff`), **Compare** (对比), **Files** (文件),
**Adopt** (采用 — restore the workspace to this node, backed up first and
undoable) and **Fork from here** (从此分叉 — continue the conversation from
this node; a new agent run starts from its snapshot in its own worktree).

Shell commands can still cause effects outside the workspace (network,
global installs); those are **not** rolled back by restores.


## CHA-6 notes

- `notice` events may carry `kind: "sandbox"` — a sandboxed shell command was
  blocked (network off / path outside the workspace / tool under `$HOME`);
  `text` says which env var allows it. Emitted once per kind per turn. The
  step's `sandbox_block` (`network` | `filesystem` | `path`) is persisted and
  the tool output ends with a `[sandbox]` note for the model.
- Snapshots store nested git repositories (e.g. a clone inside the workspace)
  as plain files instead of gitlinks (`workspace.add_all_with_nested`).
- Tool calls a model leaks into its text (DSML markers, `<invoke>` XML, or the
  `<|open|>tools<|sep|><|open|>call tool="…"…` template) are hidden from the
  answer and, when they name a tool offered that round, executed.
- 对话 turns get a system note that they have web tools only (no local
  files); the UI shows a hint with a one-click switch to 交给 agent on a branch
  that has agent work. Older tool results in the chat tool loop shrink past
  `CHATTABLE_CHAT_CTX_BUDGET_TOKENS` (default 120000 est. tokens; the last 2
  stay whole).
- On a branch that has agent runs, a 对话 turn also gets a reminder right
  before the latest user message: do not fetch the user's code from GitHub,
  ask them to switch to 交给 agent instead. Live check: without it the model
  spent 10 web_fetch rounds on raw.githubusercontent.com (206k input tokens,
  empty reply); with it, it answered in one round (4k input) and pointed to
  交给 agent.
