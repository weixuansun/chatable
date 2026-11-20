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
| `POST /api/agent/approve` | `{step_id, decision, always?}` | `decision`: `"approve"` \| `"deny"`. `always: true` (with approve) auto-approves that tool for the rest of the server session. `404` when no approval is pending for `step_id` (e.g. it timed out). |
| `POST /api/agent/restore` | `{node_id}` | Reset the workspace to the snapshot of the nearest ancestor-or-self node with `ws_commit`. Files absent from the snapshot are deleted; any `.git` directory inside the workspace is left untouched. Returns `{ok, restored, commit}` or `{ok: false, restored: false, reason}` when the path has no snapshot. |
| `GET /api/agent/step/{step_id}/diff` | — | `{id, tool, status, ws_commit, diff}` — the step's stored edit diff for write/edit, else the shadow-git diff between `ws_parent_commit..ws_commit`. |

Existing tree endpoints (`GET /api/tree`, `GET /api/node/{id}`,
`GET /api/path/{id}`) return step nodes like any other node, so the mind map
and conversation view can render them.

## SSE events (agent turns only, in addition to the existing ones)

Existing events (`user_node`, `assistant_node`, `assistant_start`,
`reasoning`, `delta`, `tool_call`, `tool_result`, `reset`, `error`, `done`)
are unchanged. `assistant_start` gains `agent: true` and `workspace`.

```json
{"type": "step_start", "id", "parent_id", "tool", "args", "args_summary"}
{"type": "approval_required", "step_id", "tool", "args", "preview_diff"}
{"type": "approval_resolved", "step_id", "decision"}
{"type": "step_done", "id", "status", "tool", "ok", "duration_ms", "ws_commit"}
```

- `approval_required` is emitted for `write_file`/`edit_file`/`run_shell`
  unless policy auto-approves them; `preview_diff` carries the unified diff
  the write/edit *would* produce (empty for shell). The turn thread then
  blocks until `POST /api/agent/approve` arrives or the 10-minute timeout
  treats it as denied.
- Denied calls are folded back to the model as a tool result saying the user
  denied it; the loop continues.

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
    "ws_commit": "<shadow git commit>",        // mutating steps that succeeded
    "ws_parent_commit": "<commit before this step>"
  }
}
```

Context building excludes step nodes from the model messages (they are
display/snapshot anchors); the assistant node's persisted `tool_messages`
replay keeps the OpenAI message sequence valid and byte-stable, so forking
from any step still produces a valid, cache-friendly request.

## Agent tools (model-facing, agent mode only)

| Tool | read_only | Approval default | Notes |
|---|---|---|---|
| `list_dir(path?, depth?)` | yes | auto | tree listing, skips `.git`, capped at 500 entries |
| `read_file(path, offset?, limit?)` | yes | auto | line numbers, 200KB cap with truncation note, binary refused |
| `grep(pattern, path?, glob?)` | yes | auto | pure-Python regex, capped at 100 matches, skips `.git`/binary |
| `write_file(path, content)` | no | ask | creates parents, ≤1MB, returns unified diff |
| `edit_file(path, old_string, new_string, replace_all?)` | no | ask | errors unless `old_string` is found and unique (or `replace_all`); returns unified diff |
| `run_shell(command, timeout?)` | no | ask | cwd=workspace, ≤60s, output capped, `*_API_KEY`/`*_TOKEN`/`*_SECRET` env vars scrubbed |

All paths are confined to the workspace: real paths must stay under the
workspace realpath (`..`, absolute paths outside, and symlink escapes are
rejected).

Agent loop limits: up to 25 rounds (`CHATTABLE_AGENT_MAX_ROUNDS`), up to 8
tool calls per round (`CHATTABLE_AGENT_MAX_CALLS_PER_ROUND`), and **no**
cross-round dedupe of repeated calls. Normal mode keeps its 3-calls/round,
10-round, dedupe behaviour untouched.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CHATTABLE_AUTH_TOKEN` | (unset) | **Required for agent mode.** Also the existing server auth token. |
| `CHATTABLE_WORKSPACE` | `<data_dir>/workspace` | Agent workspace root (created if missing). |
| `CHATTABLE_AGENT_AUTO_APPROVE` | (unset) | Comma-separated: `writes` (write_file+edit_file), `shell` (run_shell), `all`. |
| `CHATTABLE_AGENT_MAX_ROUNDS` | `25` | Agent loop round cap. |
| `CHATTABLE_AGENT_MAX_CALLS_PER_ROUND` | `8` | Agent loop per-round tool-call cap. |
| `CHATTABLE_AGENT_APPROVAL_TIMEOUT_S` | `600` | Approval wait before treating as denied. |

## Shadow git snapshots

Snapshots live in `<data_dir>/agent_shadow.git` and operate with
`git --git-dir=<shadow> --work-tree=<workspace>` — a `.git` inside the
workspace is never read, written, or cleaned (it is excluded from tracking,
and `git clean` skips it). The workspace state is committed after every
successful write/edit/shell step, plus one base snapshot before the first
mutation of a turn. A node's file state is the nearest ancestor-or-self
`ws_commit`. If git is not installed, snapshots degrade to no-ops
(`ws_commit` absent) and restore/diff calls return a clear error.

Shell commands can still cause effects outside the workspace (network,
global installs); those are **not** rolled back by restores.
