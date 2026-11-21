# Agent Mode API (prototype)

Agent mode gives a conversation a coding-agent loop: workspace-confined file
and shell tools, a persisted **step node** per tool call, shadow-git
workspace snapshots, and human approval for mutating calls. It is **off by
default** and only available when the server is started with
`CHATTABLE_AUTH_TOKEN` — without a token every agent entry point refuses
(`400`/`403` with `"Agent mode requires CHATTABLE_AUTH_TOKEN to be set"`).
When agent mode is off, chat behaviour, tool limits and dedupe are exactly
the pre-agent behaviour.

## Enabling agent mode

Two ways, either is enough:

- **Per send:** `POST /api/send` with `"agent": true`.
- **Per conversation:** `POST /api/agent/mode {"node_id": <any node in the
  conversation>, "enabled": true}`. Stored as `metadata.agent_mode` on the
  conversation's trunk (root) node, so it survives reloads and applies to
  every later send in that conversation until disabled.

## Endpoints

All endpoints sit behind the existing `SecurityMiddleware` (Host allowlist,
Origin/Referer CSRF check, bearer/cookie token). When a token is configured
it is required for these routes like any other.

| Method & path | Body / query | Description |
|---|---|---|
| `GET /api/agent/status` | — | `{available, reason, workspace, auto_approve, git}` — `available=false` (with `reason`) when no token is set; `workspace` is the resolved workspace root (created on demand); `git` reports shadow-snapshot availability. |
| `POST /api/agent/mode` | `{node_id, enabled}` | Toggle agent mode for the conversation owning `node_id`. Returns `{ok, trunk_id, agent_mode}`. |
| `POST /api/send` | `{content, parent_id?, attachments?, agent?}` | Unchanged, plus the `agent` flag. Agent turns stream the extra SSE events below. **Sending from a node in agent mode first restores the workspace to that node's nearest snapshot** (restore-on-checkout), so files always match the branch being continued. |
| `POST /api/agent/approve` | `{step_id, decision, always?, scope?}` | `decision`: `"approve"` \| `"deny"`. `always: true` (with approve) remembers the choice **for the current conversation** (in-memory per trunk): `scope: "tool"` (default) auto-approves this tool, `scope: "writes"` auto-approves write_file+edit_file. `404` when no approval is pending for `step_id` (e.g. it timed out or the turn was stopped — a stop resolves pending approvals as denied). |
| `GET /api/agent/approvals?node=` | — | `{always: [tool…]}` — this conversation's session auto-approvals (for the composer pill). |
| `POST /api/agent/approvals/reset` | `{node_id}` | Clear them. |
| `POST /api/agent/restore` | `{node_id}` | Reset the workspace to the snapshot of the nearest ancestor-or-self node with `ws_commit`. Files absent from the snapshot are deleted; any `.git` directory inside the workspace is left untouched. Returns `{ok, restored, commit}` or `{ok: false, restored: false, reason}` when the path has no snapshot. |
| `GET /api/agent/step/{step_id}/diff` | — | `{id, tool, status, ws_commit, diff}` — the step's stored edit diff for write/edit, else the shadow-git diff between `ws_parent_commit..ws_commit`. |
| `GET /api/agent/compare?a=&b=` | — | Branch compare: `{ok, a_commit, b_commit, diff, files: [{path, status, additions, deletions}]}` between the snapshots nearest the two nodes. |
| `GET /api/agent/files?node=` | — | `{ok, commit, files: [path…]}` — files tracked in the snapshot nearest the node. |
| `GET /api/agent/file?node=&path=` | — | `{ok, commit, path, content, truncated}` — one file at that snapshot (200KB cap, path confined). |
| `POST /api/agent/adopt` | `{node_id}` | Adopt the node's snapshot into the workspace: snapshots the current state first, restores, records `adopted_at`/`adopted_commit`/`previous_commit` on the node. Returns `{ok, adopted_commit, previous_commit, files}`. |
| `POST /api/agent/undo-adopt` | `{previous_commit}` | Restore the pre-adopt snapshot (commit must exist in the shadow repo). |

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
   off` disables; when auto can't find bwrap, `/api/agent/status` reports
   `sandbox: "off"` plus a loud `sandbox_warning` (the composer shows a red
   "no sandbox" badge).
2. **Deny-list (advisory).** `rm -rf /|~`, `sudo`, `curl|sh`, recursive
   `chmod 777` on `/`/`~`, `dd of=/dev/…`, `mkfs`, fork bombs,
   `git push --force`, and writes into `.ssh`/`.aws`/`.gnupg` are refused
   outright (before any approval prompt). `CHATTABLE_AGENT_SHELL_DENYLIST=off`
   disables it. It is a tripwire, not a boundary — the sandbox is.

Every shell runs with a scrubbed environment: credentials
(`*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL(S)`,
`*_AUTH`, `AWS_*`, `GH_*`/`GITHUB_*`, `OPENAI_*`, `DEEPSEEK_*`,
`ANTHROPIC_*`, `CHATTABLE_AUTH_TOKEN`) are dropped.

`/api/agent/status` gains `{sandbox, sandbox_warning, shell_network,
shell_denylist}`.

Existing tree endpoints (`GET /api/tree`, `GET /api/node/{id}`,
`GET /api/path/{id}`) return step nodes like any other node, so the mind map
and conversation view can render them.

## SSE events (agent turns only, in addition to the existing ones)

Existing events (`user_node`, `assistant_node`, `assistant_start`,
`reasoning`, `delta`, `tool_call`, `tool_result`, `reset`, `error`, `done`)
are unchanged. `assistant_start` gains `agent: true` and `workspace`.

```json
{"type": "step_start", "id", "parent_id", "tool", "args", "args_summary"}
{"type": "step_output", "step_id", "chunk"}
{"type": "approval_required", "step_id", "tool", "args", "preview_diff"}
{"type": "approval_resolved", "step_id", "decision"}
{"type": "step_done", "id", "status", "tool", "ok", "duration_ms", "ws_commit"}
{"type": "plan", "step_id", "items": [{"content", "status"}]}
{"type": "notice", "text"}
```

- `step_output` streams `run_shell` stdout/stderr incrementally while the
  command runs (the persisted step keeps head+tail 8KB with an omission
  note). Stop kills the shell's whole process group.
- `approval_required` is emitted for `write_file`/`edit_file`/`run_shell`
  unless policy auto-approves them; `preview_diff` carries the unified diff
  the write/edit *would* produce (empty for shell). The turn thread then
  blocks until `POST /api/agent/approve` arrives or the 10-minute timeout
  treats it as denied.
- Denied calls are folded back to the model as a tool result saying the user
  denied it; the loop continues.
- `plan` carries the latest `update_plan` checklist; the final plan is also
  persisted on the assistant node metadata as `plan`.
- `notice` signals loop-level events — currently the round-limit fuse: after
  `CHATTABLE_AGENT_MAX_ROUNDS` tool rounds the model gets one final
  no-tools round to write its summary.

## Step node schema

Each executed agent tool call is one persisted node, written **as it
happens** and chained: first step's parent is the assistant node of the
turn, each next step's parent is the previous step — the trajectory is a
path in the tree (`user → assistant → step → step → …`). The turn's final
`ws_commit` is also mirrored onto the assistant node's metadata.

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
    "status": "pending_approval | denied | ok | error",
    "result": "…tool output, truncated to 4000 chars…",
    "error": "…only when status == error…",
    "diff": "…unified diff, write/edit only…",
    "duration_ms": 42,
    "ws_commit": "<shadow git commit>",        // mutating steps that succeeded
    "ws_parent_commit": "<commit before this step>"
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
that point.

## Agent tools (model-facing, agent mode only)

| Tool | read_only | Approval default | Notes |
|---|---|---|---|
| `list_dir(path?, depth?)` | yes | auto | flat workspace-relative paths (dirs end with `/`), default depth 2, skips noise dirs (`.git`, `__pycache__`, `.pytest_cache`, `node_modules`, `.venv`, `dist`, `build`, …) and root `.gitignore`, capped at 500 entries |
| `glob(pattern, path?)` | yes | auto | `**/*.py`-style file finding, newest first, capped with a "N more matches" note |
| `read_file(path, offset?, limit?)` | yes | auto | line numbers, 200KB cap with truncation note, binary refused |
| `grep(pattern, path?, glob?, ignore_case?, context?)` | yes | auto | pure-Python regex; `path:line: text` output (context lines use `-`), capped with a "N more matches" note |
| `write_file(path, content)` | no | ask | creates parents, ≤1MB, returns unified diff |
| `edit_file(path, old_string, new_string, replace_all?, expected_replacements?)` | no | ask | not-found errors show the most similar region with line numbers; non-unique errors list occurrence line numbers; returns unified diff |
| `run_shell(command, timeout?)` | no | ask | cwd=workspace, streams output as `step_output` events, head+tail 8KB kept, timeout default 120s / max 600s, stop kills the whole process group, credentials env vars scrubbed, **bwrap-sandboxed when available** (no network by default), deny-list refuses destructive commands outright |
| `update_plan(items)` | yes | auto | TodoWrite-style plan checklist; latest plan emitted as `plan` SSE and persisted on the assistant node |

All paths are confined to the workspace: real paths must stay under the
workspace realpath (`..`, absolute paths outside, and symlink escapes are
rejected).

The agent system prompt instructs explore-before-edit, read-before-edit,
minimal edits, plan tracking, test-after-change and verify-before-claiming.
If `AGENTS.md`, `CLAUDE.md` or `.chatable/instructions.md` exists at the
workspace root it is appended to that prompt (20KB cap).

Agent loop limits: up to 25 rounds (`CHATTABLE_AGENT_MAX_ROUNDS`), up to 8
tool calls per round (`CHATTABLE_AGENT_MAX_CALLS_PER_ROUND`), and **no**
cross-round dedupe of repeated calls. Normal mode keeps its 3-calls/round,
10-round, dedupe behaviour untouched. When the round limit is hit the model
gets one final no-tools round to summarize (plus a `notice` event).

Long turns: past ~60k estimated tokens of loop context
(`CHATTABLE_AGENT_CTX_BUDGET_TOKENS`), tool results older than the last 6
are shrunk to head/tail excerpts with a note. Only the live loop context is
affected; persisted step results stay full.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CHATTABLE_AUTH_TOKEN` | (unset) | **Required for agent mode.** Also the existing server auth token. |
| `CHATTABLE_WORKSPACE` | `<data_dir>/workspace` | Agent workspace root (created if missing). |
| `CHATTABLE_AGENT_AUTO_APPROVE` | (unset) | Comma-separated: `writes` (write_file+edit_file), `shell` (run_shell), `all`. |
| `CHATTABLE_AGENT_MAX_ROUNDS` | `25` | Agent loop round cap. |
| `CHATTABLE_AGENT_MAX_CALLS_PER_ROUND` | `8` | Agent loop per-round tool-call cap. |
| `CHATTABLE_AGENT_APPROVAL_TIMEOUT_S` | `600` | Approval wait before treating as denied. |
| `CHATTABLE_AGENT_CTX_BUDGET_TOKENS` | `60000` | Estimated-token budget before old tool results are shrunk. |
| `CHATTABLE_AGENT_SANDBOX` | `auto` | `auto` (bwrap if available) / `bwrap` / `off`. |
| `CHATTABLE_AGENT_SHELL_NET` | (unset) | `1` allows network inside the shell sandbox. |
| `CHATTABLE_AGENT_SHELL_DENYLIST` | (unset) | `off` disables the advisory command deny-list. |
| `CHATTABLE_AGENT_SANDBOX_PATH_HOME` | (unset) | `1` also re-mounts non-venv PATH dirs under `$HOME` (skipped by default). |

## Shadow git snapshots

Snapshots live in `<data_dir>/agent_shadow.git` and operate with
`git --git-dir=<shadow> --work-tree=<workspace>` — a `.git` inside the
workspace is never read, written, or cleaned (it is excluded from tracking,
and `git clean` skips it). A **base snapshot** of the pre-turn workspace is
committed at the start of every agent turn and recorded on the user node,
then one commit after every successful write/edit/shell step (recorded as
`ws_commit` on the step; the turn's final commit is mirrored onto the
assistant node). A node's file state is the nearest ancestor-or-self
`ws_commit` — where an assistant node's mirror counts only for the
assistant node itself, not for its own steps (their nearest snapshot is the
last earlier step, or the base). If git is not installed, snapshots degrade
to no-ops (`ws_commit` absent) and restore/diff calls return a clear error.

Shell commands can still cause effects outside the workspace (network,
global installs); those are **not** rolled back by restores.
