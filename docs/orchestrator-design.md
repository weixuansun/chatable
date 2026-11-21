# chatable orchestrator — design (branch `orchestrator`)

> 定位：chatable 不再自己实现 agent 内核（工具、压缩、子 agent、审批细节），而是做**上层**：
> 任务树/DAG、快照、并行派发、上下文交接、对比、合并、采用。每个节点里的具体工作交给成熟的外部
> coding agent（Kimi Code、Claude Code、Codex……），内置的 DeepSeek agent 保留为便宜的 `builtin` 执行器。

## 1. Core model

**Node = frozen snapshot + result.**

| field | meaning |
|---|---|
| `snapshot` | commit in the shadow git (`<data>/agent_shadow.git`), immutable |
| `instruction` | the instruction that produced this node (root: none) |
| `parents` | list of node ids (DAG: merge nodes have ≥2 parents) |
| `agent` / `model` | executor id (`kimi`, `claude`, `codex`, `builtin`) and model if known |
| `status` | `queued` · `running` · `ok` · `failed` · `cancelled` (+ `adopted` flag) |
| `conclusion` | structured: `done`, `decisions`, `not_done`, `next` (from the agent's final reply) |
| `facts` | system-computed: changed files with +/-, test command + result, exit code, duration, tokens/cost if reported |
| `session_id` | agent session id (for optional resume) |
| `events` | run transcript (tool calls/results, messages) — **shown, not forkable** |

Forking granularity is **one agent run** (not one tool call). The root node is the
initial workspace snapshot.

## 2. Runs (one instruction = one run = one child node)

From any node N the user sends **many different instructions in parallel** (primary use
case) and optionally the same instruction to several agents (race). For each instruction:

1. queue a run (global queue, per-agent concurrency limits, e.g. kimi=2, builtin=3);
2. create an isolated worktree from N's snapshot (`git worktree add` on the shadow repo
   or export the tree into `<data>/runs/<run_id>/wt`);
3. write `.chatable/CONTEXT.md` (handoff, §4) into the worktree;
4. start the executor headless with full-auto permissions *inside the worktree only*;
   stream its events to the UI (SSE) and persist them on the run;
5. on exit: snapshot the worktree (excluding `.chatable/`) → child node with
   conclusion + facts; remove the worktree.

Failed or cancelled runs still produce a node (status failed/cancelled) with whatever
changes were made, so they can be inspected, continued or discarded.

Optional dependencies: an instruction may say "after X" → it is queued until X's node
exists and then starts **from X's node**.

The real project directory is only touched by **adopt** (existing adopt/undo).

## 3. Merge (DAG) — implemented in o2

`POST /api/orch/merge {nodes: [N1..Nk], agent, test?}`:

* Sequential 3-way merges in a throwaway worktree on the shadow repo
  (`git merge --no-ff`, conflicts committed as intermediate merge commits with
  markers left in the files); the final tree is committed with ALL input
  snapshots as parents. `.chatable/` is never merged (repo exclude).
* base = merge-base of the snapshots; `facts.files` = changed files vs base.
* Clean merge → merge node M (`parents=[N1..Nk]`, `agent="merge"`,
  `status="ok"`); the project test command runs when available
  (`test: false` skips) → result in `facts.test`.
* Conflict → M with `status="conflict"` (markers committed,
  `facts.conflict_files`); red tests → `status="tests_failed"`. Both
  AUTO-DISPATCH a resolver run from M: its CONTEXT.md gets a "Merge task"
  section (each parent's instruction + conclusion, the conflicting file list,
  capped failing test output) and the instruction to resolve while keeping
  every parent's intent. Its result is M′ (child of M).
* Context path layer below a merge walks ALL parents (dedupe, creation order).
* SSE events: `merge_started`, `merge_finished`.

## 4. Context handoff (`.chatable/CONTEXT.md`)

chatable does not rely on any agent's own memory; every run gets an assembled handoff:

1. **Project layer** — AGENTS.md / CLAUDE.md / README excerpt (capped), user project notes,
   detected test command (pytest / npm test / …, user-editable).
2. **Path layer** — for each ancestor from root to N: instruction + frozen structured
   conclusion. The nearest ~3 in full (~200 words each), older ones compressed to one line.
   Plus `git diff --stat` per step and the full diff of N vs its parent (capped).
3. **Sibling layer** (v2) — other instructions running from the same node.
4. **Task layer** — the instruction, constraints (e.g. "tests must pass", "don't touch X"),
   user-attached material (a node's diff, a file, an error).

Prompt to the agent: `Read .chatable/CONTEXT.md first, then do the following instruction:
<instruction>. When finished, end your reply with a "## Conclusion" section with
Done / Key decisions / Not done / Next steps.` The conclusion is parsed and stored with the
facts. Session resume (same agent, linear child) is an optional accelerator; the handoff
is always written.

`.chatable/` is excluded from snapshots, compare and adopt.

## 5. Executor interface

```python
class Executor:
    id: str                      # "kimi" | "claude" | "codex" | "builtin"
    def available(self) -> tuple[bool, str]   # binary present / auth or key present
    def start(self, workdir, prompt, session_id=None, env=None) -> Run
class Run:
    def events(self) -> Iterator[Event]       # normalized: message, tool_call, tool_result, error, meta
    def cancel(self) -> None
    result: final_text, exit_code, session_id, usage (tokens/cost if known)
```

Adapters (each a small module that maps the CLI's JSON lines to normalized events):

| agent | command | notes |
|---|---|---|
| kimi | `kimi --output-format stream-json -p <prompt>` (resume: `-S <id>`) | JSON lines `{role: assistant/tool/meta, tool_calls, content}`; `meta.session.resume_hint` carries `session_id`. Sample: `tests/fixtures/kimi_stream.jsonl` |
| claude | `claude -p --output-format stream-json --verbose --permission-mode acceptEdits` (resume `--resume <id>`) | events `system/init`, `assistant`, `user` (tool_result), `result` (cost, session_id) |
| codex | `codex exec --json --full-auto` (resume `codex exec resume <id>`) | JSON events (thread/item started/completed, turn completed) |
| builtin | in-process DeepSeek agent loop from `agent-proto` | runs confined to the worktree |

Keys stay in each agent's own env; the builtin executor never sees others' keys.
Sandbox: the external CLIs need network to their APIs, so v1 runs them confined by
working directory (bwrap with network where feasible is a v2 item).

## 6. UI — implemented in o3

Full-screen overlay (workflow icon above the conversation list):

* DAG canvas of snapshot cards (layered by depth, pan/zoom/fit): instruction,
  agent icon, status chip (queued/running/ok/failed/cancelled/conflict/
  tests_failed, adopted badge), +/- lines, test chip, duration. Edges
  parent→child; merge nodes draw an edge from every parent. Queued/running
  runs render as dashed placeholder cards with a live last-event line (SSE).
* Node panel: instruction, structured conclusion, facts (files +/-, test
  summary, exit code, duration, tokens), collapsible event transcript, and
  Compare / Adopt actions.
* New-instructions composer on the selected node: multiple rows, each with
  instruction + agent multi-select (unavailable executors greyed with the
  reason) + optional "after". Runs appear immediately via SSE.
* Merge: checkbox-select ≥2 cards → agent picker → merge; conflict or red
  tests auto-dispatch the resolver run.
* Compare modal (file list + colored diff), Adopt confirm with file preview
  and Undo toast (same patterns as the agent UI).
* Project bar: workspace, executor availability badges, queue counts,
  editable project notes + test command (GET/PUT /api/orch/project).

## 7. Reuse vs new

Reuse: shadow git snapshots, compare, adopt/undo, tree store, mind-map rendering, SSE.
New: executor interface + adapters, worktree manager, run queue, context assembly,
conclusion parsing, DAG parents + merge, orchestrator UI.

## 8. v1 scope / environment

v1 = kimi + builtin exercised live; claude/codex adapters implemented and unit-tested
against recorded/mock streams (the box has no claude/codex CLI or keys). Demo:
more-itertools, three different instructions in parallel from the root (fix a bug from an
issue description, add a function, improve docs) → merge → tests green → adopt.

## 9. API (phase o1)

All endpoints require `CHATTABLE_AUTH_TOKEN` (same middleware as the rest).
Events stream over SSE: `run_started`, `run_event`, `run_finished`,
`node_created` (plus `_ping` keepalives; replay via `?from_index=`).

| Endpoint | Body / query | Description |
|---|---|---|
| `GET /api/orch/state` | — | `{project: {id, workspace, test_cmd, root}, nodes: [...], runs: [...], executors: {id: {available, reason}}, limits}` |
| `POST /api/orch/runs` | `{from_node, items: [{instruction, agents: [...], after?: run_id}]}` | Queue runs; returns `{runs: [run_id]}`. `after` waits for the dependency run and rebases onto its result node. |
| `POST /api/orch/runs/{id}/cancel` | — | Queued → cancelled; running → kill the executor's process group. |
| `GET /api/orch/node/{id}` | — | One node with its frozen event transcript. |
| `GET /api/orch/compare?a=&b=` | — | `{ok, a_commit, b_commit, diff, files:[{path,status,additions,deletions,binary}]}` between two nodes' snapshots. |
| `POST /api/orch/adopt` | `{node_id}` | Snapshot the workspace first (undoable), restore the node's snapshot, mark the node adopted. Returns `{ok, adopted_commit, previous_commit, files}`. |
| `POST /api/orch/undo-adopt` | `{previous_commit}` | Restore the pre-adopt snapshot. |
| `POST /api/orch/merge` | `{nodes: [...], agent, test?}` | 3-way merge (see §3); returns `{ok, node_id, status, run_id?}`. |
| `GET /api/orch/project` | — | `{id, workspace, notes, test_cmd, detected_test_cmd}` |
| `PUT /api/orch/project` | `{notes?, test_cmd?}` | Persist project notes (CONTEXT.md project layer) and a test-command override. |
| `GET /api/orch/stream` | `?from_index=` | SSE of all orchestrator events. |

Run/node model, executors, and environment:

- Node fields: `id, root?, snapshot, instruction, parents[], agent, model,
  status (ok|failed|cancelled), adopted, conclusion {done, decisions,
  not_done, next}, facts {files:[{path,added,removed}], test?: {cmd,ok,
  summary}, exit_code, duration_s, usage, error?, killed?}, session_id,
  events[]`. Runs: `id, from_node, instruction, agent, after?, status
  (queued|running|ok|failed|cancelled), node_id, events[], error`.
- Executors: `kimi` (`kimi --output-format stream-json -p`, resume `-S`),
  `claude` (`claude -p --output-format stream-json --verbose
  --permission-mode acceptEdits`, resume `--resume`), `codex`
  (`codex exec --json --full-auto`, resume `codex exec resume`),
  `builtin` (the in-process DeepSeek agent loop, confined to the run
  worktree with auto-approve). Binaries: `CHATTABLE_KIMI_BIN` /
  `CHATTABLE_CLAUDE_BIN` / `CHATTABLE_CODEX_BIN`, else PATH conventions.
  `available()` checks the binary (plus an auth hint for claude/codex).
  Other providers' API keys are scrubbed from each executor's env.
- Limits: `CHATTABLE_ORCH_LIMITS` (default `kimi=2,claude=2,codex=2,
  builtin=3`). Run timeout `CHATTABLE_ORCH_RUN_TIMEOUT_S` (default 1800);
  post-run test timeout `CHATTABLE_ORCH_TEST_TIMEOUT_S` (default 300).
- Each run: worktree from the parent snapshot → `.chatable/CONTEXT.md`
  (project/path/task layers) → executor → snapshot back into the shadow
  repo → post-run test command (auto-detected) → node with conclusion +
  facts. Failed/cancelled runs still produce a node.
- Merge: phase o2 (hook: `parents` is already a list).
