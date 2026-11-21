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

## 3. Merge (DAG)

User selects nodes N1..Nk (typically siblings) → **Merge**:

* base = common ancestor (merge-base) of their snapshots; apply 3-way merges in order
  (`git merge-tree` / `git merge` in a temp worktree);
* clean merge → merge node M (parents N1..Nk), auto-run the project test command, store
  result in facts;
* conflict **or** red tests → automatically dispatch a run on M (agent chosen by the user,
  default the first available) with a context that contains the parents' instructions and
  conclusions, the conflicting files (with markers left in the worktree) and the failing
  test output → its result is M′ (child of M).

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

## 6. UI

* Mind map of **snapshot cards**: instruction, agent icon, status, +/- lines, tests.
* On a node: **New instructions** — several rows, each with instruction text + agent(s).
* Running edges show the live event stream; a node's panel shows the transcript.
* Compare any two nodes; select siblings → **Merge**; **Adopt** / Undo.

## 7. Reuse vs new

Reuse: shadow git snapshots, compare, adopt/undo, tree store, mind-map rendering, SSE.
New: executor interface + adapters, worktree manager, run queue, context assembly,
conclusion parsing, DAG parents + merge, orchestrator UI.

## 8. v1 scope / environment

v1 = kimi + builtin exercised live; claude/codex adapters implemented and unit-tested
against recorded/mock streams (the box has no claude/codex CLI or keys). Demo:
more-itertools, three different instructions in parallel from the root (fix a bug from an
issue description, add a function, improve docs) → merge → tests green → adopt.
