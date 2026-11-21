"""CHA-5: agent runs that live in the conversation tree.

A run started from a chat node is an ordinary user→assistant pair in the
chat store; progress streams through the chat live-turn buffer (so the
main chat renders it like agent steps) and the result snapshot lands on
the assistant node as ``ws_commit`` — the CHA-4 node actions apply as is.
Mixed into OrchService (needs: chat, shadow, store, executors, runs_dir,
_cond, _cancelled, _build_test_fact).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from models import Node
from orch.context import detect_test_cmd
from orch.store import new_id
from orch.worktree import commit_worktree_tree, merge_into_worktree, worktree_remove

STEP_RESULT_CAP = 4000


def _args_summary(name: str, args) -> str:
    if isinstance(args, dict):
        for key in ("command", "cmd", "path", "file_path", "file", "pattern", "query", "url"):
            if args.get(key):
                return f"{name} {str(args[key])[:160]}"
    return name or "tool"


class ChatRunsMixin:
    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def executors_status(self) -> dict:
        out = {}
        for eid, ex in self.executors.items():
            try:
                ok, reason = ex.available()
            except Exception as e:  # noqa: BLE001
                ok, reason = False, str(e)
            out[eid] = {"available": bool(ok), "reason": reason or ""}
        return out

    def start_chat_runs(self, content: str, parent_id: Optional[str], agents: list[str],
                        extra_context: str = "") -> list[dict]:
        """Create one run per agent from chat node *parent_id* (None/root =
        a new conversation). Returns [{run_id, user_id, assistant_id, agent}]."""
        from store import ROOT_ID
        chat = self.chat
        content = (content or "").strip()
        if not content:
            raise ValueError("instruction is required")
        agents = [a for a in (agents or []) if a]
        if not agents:
            raise ValueError("pick at least one agent")
        for a in agents:
            if a not in self.executors:
                raise ValueError(f"unknown agent: {a}")
        parent_id = parent_id or ROOT_ID
        if parent_id != ROOT_ID and not chat.store.exists(parent_id):
            raise KeyError(parent_id)
        base = chat.nearest_ws_commit(parent_id) if parent_id != ROOT_ID else None
        if not base:
            base = self.shadow.snapshot("workspace before agent run", allow_empty=True)
        if not base:
            raise ValueError("workspace snapshots unavailable (is git installed?)")
        out = []
        for agent in agents:
            user = Node.new(role="user", content=content, parent_id=parent_id)
            user.metadata = {"orch_run": True, "executor": agent, "ws_commit": base}
            chat.store.add(user)
            asst = Node.new(role="assistant", content="", parent_id=user.id)
            asst.metadata = {"agent": True, "orch_run": True, "executor": agent,
                             "run_status": "queued", "ws_parent_commit": base}
            chat.store.add(asst)
            turn = chat.live.create(asst.id)
            turn.append({"type": "user_node", "id": user.id, "parent_id": parent_id})
            turn.append({"type": "assistant_node", "id": asst.id, "parent_id": user.id})
            turn.append({"type": "assistant_start", "id": asst.id, "agent": True,
                         "run": True, "executor": agent, "workspace": self.workspace.root,
                         "context": {"messages": 0, "estimated_tokens": 0,
                                     "pinned": 0, "shortened": 0, "prefix": ""}})
            run = {
                "id": new_id(), "from_node": None, "instruction": content,
                "agent": agent, "after": None, "status": "queued",
                "created_at": time.time(), "started_at": None, "finished_at": None,
                "node_id": None, "events": [], "error": "",
                "chat": {"user_id": user.id, "assistant_id": asst.id,
                         "parent_id": parent_id, "base_commit": base,
                         "extra_context": extra_context or ""},
            }
            self._chat_turns[run["id"]] = turn
            self._chat_run_by_asst[asst.id] = run["id"]
            self.store.add_run(run)
            turn.append({"type": "notice", "text": f"已排队：{agent}"})
            out.append({"run_id": run["id"], "user_id": user.id,
                        "assistant_id": asst.id, "agent": agent})
        with self._cond:
            self._cond.notify_all()
        return out

    def stop_chat_run(self, assistant_id: str) -> bool:
        run_id = self._chat_run_by_asst.get(assistant_id)
        if not run_id:
            return False
        return self.cancel_run(run_id)

    # ------------------------------------------------------------------ #
    # run hooks (called from OrchService._run_one)
    # ------------------------------------------------------------------ #

    def _chat_path_nodes(self, parent_id: str) -> list:
        from store import ROOT_ID
        if parent_id == ROOT_ID:
            return []
        return [n for n in self.chat.store.path_to_root(parent_id) if n.role != "step"]

    def _chat_parent_diff(self, parent_id: str) -> str:
        from store import ROOT_ID
        if parent_id == ROOT_ID or not self.chat.store.exists(parent_id):
            return ""
        node = self.chat.store.get(parent_id)
        md = node.metadata if isinstance(node.metadata, dict) else {}
        head, base = md.get("ws_commit"), md.get("ws_parent_commit")
        if not head or not base or head == base:
            return ""
        try:
            return self.shadow.diff(head, base)
        except Exception:  # noqa: BLE001
            return ""

    def _chat_resume_session(self, parent_id: str, agent: str) -> Optional[str]:
        from store import ROOT_ID
        if parent_id == ROOT_ID or not self.chat.store.exists(parent_id):
            return None
        md = self.chat.store.get(parent_id).metadata or {}
        if md.get("orch_run") and md.get("executor") == agent:
            return md.get("session_id")
        return None

    def _chat_run_started(self, run: dict) -> None:
        aid = run["chat"]["assistant_id"]
        self.chat.store.update_metadata(aid, {"run_status": "running"})
        turn = self._chat_turns.get(run["id"])
        if turn:
            turn.append({"type": "notice", "text": f"开始运行：{run['agent']}（独立工作区）"})

    def _chat_event(self, run: dict, ev: dict, state: dict) -> None:
        """Map one normalized executor event onto chat step nodes/events."""
        turn = self._chat_turns.get(run["id"])
        if turn is None:
            return
        typ = ev.get("type")
        store = self.chat.store
        if typ == "tool_call":
            name = str(ev.get("name") or "tool")
            args = ev.get("arguments") if isinstance(ev.get("arguments"), dict) else {}
            summary = _args_summary(name, args)
            parent = state.get("tail") or run["chat"]["assistant_id"]
            step = Node.new(role="step", content=summary, parent_id=parent)
            step.metadata = {"step": True, "tool": name, "args": args,
                             "args_summary": summary, "status": "running",
                             "call_id": ev.get("id"), "external": True}
            store.add(step)
            state["tail"] = step.id
            state.setdefault("open", []).append((ev.get("id"), step.id, name, time.monotonic()))
            turn.append({"type": "step_start", "id": step.id, "parent_id": parent,
                         "tool": name, "args": args, "args_summary": summary})
        elif typ == "tool_result":
            opened = state.get("open") or []
            match = None
            for i, (cid, sid, name, t0) in enumerate(opened):
                if cid is not None and cid == ev.get("id"):
                    match = opened.pop(i)
                    break
            if match is None and opened:
                match = opened.pop(0)
            if match is None:
                return
            _cid, sid, name, t0 = match
            ok = ev.get("ok") is not False
            output = str(ev.get("output") or "")
            dur = int((time.monotonic() - t0) * 1000)
            store.update_metadata(sid, {"status": "ok" if ok else "error",
                                        "result": output[:STEP_RESULT_CAP],
                                        "duration_ms": dur})
            turn.append({"type": "step_done", "id": sid, "status": "ok" if ok else "error",
                         "tool": name, "ok": ok, "duration_ms": dur})
            turn.append({"type": "tool_result", "name": name, "arguments": {},
                         "ok": ok, "error": "" if ok else "failed",
                         "duration_ms": dur, "output": output[:STEP_RESULT_CAP]})
        elif typ == "meta" and ev.get("kind") == "plan":
            turn.append({"type": "plan", "step_id": None, "items": ev.get("items") or []})
        elif typ == "error":
            turn.append({"type": "notice", "text": str(ev.get("text") or "error")[:300]})

    def _chat_notice(self, run: dict, text: str) -> None:
        turn = self._chat_turns.get(run["id"])
        if turn:
            turn.append({"type": "notice", "text": text})

    def _chat_should_stop(self, run: dict) -> bool:
        turn = self._chat_turns.get(run["id"])
        return bool(turn and turn.stop_requested)

    def _chat_run_finished(self, run: dict, *, status: str, end_commit: str, base: str,
                           files: list, facts: dict, conclusion: dict,
                           final_text: str, session_id: Optional[str],
                           usage: dict, error: str) -> None:
        chat = self.chat
        aid = run["chat"]["assistant_id"]
        # Close steps that never got a result (killed/cancelled runs).
        for sid in [s for s in self._open_steps(aid)]:
            chat.store.update_metadata(sid, {"status": "error", "result": "(no result)"})
        content = (final_text or "").strip()
        if not content:
            content = ("已取消。" if status == "cancelled"
                       else f"运行失败：{error or 'agent 没有给出回复'}")
        md = {
            "agent": True, "orch_run": True, "executor": run["agent"],
            "run_id": run["id"], "run_status": status,
            "ws_commit": end_commit, "ws_parent_commit": base,
            "files_changed": len(files or []), "facts": facts,
            "conclusion": conclusion, "session_id": session_id,
            "stop_reason": ("stopped" if status == "cancelled"
                            else "error" if status == "failed" else None),
        }
        if run["chat"].get("merged_from"):
            md["merged_from"] = run["chat"]["merged_from"]
        chat.store.finalize_node(aid, content, usage or {}, md)
        turn = self._chat_turns.pop(run["id"], None)
        self._chat_run_by_asst.pop(aid, None)
        if turn is not None:
            turn.append({"type": "done", "id": aid, "content": content,
                         "usage": usage or {}, "stats": "", "elapsed": facts.get("duration_s"),
                         "ws_commit": end_commit, "stop_reason": md["stop_reason"],
                         "run_status": status, "tool_results": []})
            self._retire_turn(turn, aid)

    def _open_steps(self, aid: str) -> list[str]:
        out = []
        cmap = self.chat.store.children_map()
        cur = aid
        while True:
            kids = [c for c in cmap.get(cur, []) if self.chat.store.get(c).role == "step"]
            if not kids:
                break
            cur = kids[0]
            if (self.chat.store.get(cur).metadata or {}).get("status") == "running":
                out.append(cur)
        return out

    def _retire_turn(self, turn, aid: str) -> None:
        import service as service_mod
        turn.finish()
        timer = threading.Timer(service_mod.RETAIN_FINISHED_TURN_S,
                                lambda: self.chat.live.remove(aid))
        timer.daemon = True
        timer.start()

    # ------------------------------------------------------------------ #
    # merge of chat nodes
    # ------------------------------------------------------------------ #

    def _lca(self, node_ids: list[str]) -> Optional[str]:
        paths = [[n.id for n in self.chat.store.path_to_root(i)] for i in node_ids]
        common = None
        for ids in zip(*paths):
            if all(x == ids[0] for x in ids):
                common = ids[0]
            else:
                break
        return common

    def start_chat_merge(self, node_ids: list[str], agent: str = "builtin",
                         run_tests: bool = True) -> dict:
        chat = self.chat
        node_ids = list(dict.fromkeys(node_ids or []))
        if len(node_ids) < 2:
            raise ValueError("pick at least two nodes to merge")
        for i in node_ids:
            if not chat.store.exists(i):
                raise KeyError(i)
        if agent not in self.executors:
            raise ValueError(f"unknown agent: {agent}")
        commits = [chat.nearest_ws_commit(i) for i in node_ids]
        if any(not c for c in commits):
            raise ValueError("every merged node needs a workspace snapshot")
        lca = self._lca(node_ids)
        if not lca:
            raise ValueError("the nodes are in different conversations")
        if lca in node_ids:
            raise ValueError("one node is an ancestor of another — nothing to merge")
        base_commit = chat.nearest_ws_commit(lca)
        nodes = [chat.store.get(i) for i in node_ids]

        def instr(n) -> str:
            p = chat.store.get(n.parent_id) if n.parent_id and chat.store.exists(n.parent_id) else None
            return " ".join(((p.content if p is not None and p.role == "user" else n.content) or "").split())[:60]

        title = "合并：" + " + ".join(f"#{n.id[:6]}「{instr(n)}」" for n in nodes)
        user = Node.new(role="user", content=title, parent_id=lca)
        user.metadata = {"orch_merge": True, "merged_from": node_ids,
                         "executor": "merge", "ws_commit": base_commit}
        chat.store.add(user)
        asst = Node.new(role="assistant", content="", parent_id=user.id)
        asst.metadata = {"agent": True, "orch_run": True, "orch_merge": True,
                         "executor": "merge", "run_status": "running",
                         "merged_from": node_ids, "ws_parent_commit": base_commit}
        chat.store.add(asst)
        turn = chat.live.create(asst.id)
        turn.append({"type": "user_node", "id": user.id, "parent_id": lca})
        turn.append({"type": "assistant_node", "id": asst.id, "parent_id": user.id})
        turn.append({"type": "assistant_start", "id": asst.id, "agent": True, "run": True,
                     "executor": "merge", "workspace": self.workspace.root,
                     "context": {"messages": 0, "estimated_tokens": 0,
                                 "pinned": 0, "shortened": 0, "prefix": ""}})
        turn.append({"type": "notice", "text": f"正在合并 {len(nodes)} 个节点…"})
        threading.Thread(target=self._do_chat_merge,
                         args=(turn, user, asst, nodes, commits, base_commit, agent, run_tests),
                         daemon=True).start()
        return {"user_id": user.id, "assistant_id": asst.id, "parent_id": lca}

    def _do_chat_merge(self, turn, user, asst, nodes, commits, base_commit,
                       agent, run_tests) -> None:
        chat = self.chat
        t0 = time.monotonic()
        wt = os.path.join(self.runs_dir, f"merge-{new_id()}", "wt")
        status, conflict_files, test_fact, merge_commit, error = "failed", [], None, None, ""
        try:
            try:
                merged = merge_into_worktree(self.shadow, commits, wt)
                conflict_files = merged["conflict_files"]
                status = "conflict" if conflict_files else "ok"
                if run_tests and not conflict_files:
                    test_cmd = self.store.meta.get("test_cmd") or detect_test_cmd(wt)
                    if test_cmd:
                        turn.append({"type": "notice", "text": f"运行测试：{test_cmd}"})
                        test_fact = self._build_test_fact(wt, test_cmd, base_commit)
                        if not test_fact["ok"]:
                            status = "tests_failed"
                merge_commit = commit_worktree_tree(
                    self.shadow, wt, commits, f"chat merge of {len(commits)} nodes")
            finally:
                worktree_remove(self.shadow, wt)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            status = "failed"
        files = []
        if merge_commit and base_commit:
            try:
                files = self.shadow.diff_numstat(base_commit, merge_commit)
            except Exception:  # noqa: BLE001
                files = []
        lines = []
        for n in nodes:
            p = chat.store.get(n.parent_id) if n.parent_id and chat.store.exists(n.parent_id) else None
            instruction = (p.content if p is not None and p.role == "user" else "") or "(对话)"
            md = n.metadata if isinstance(n.metadata, dict) else {}
            done = ((md.get("conclusion") or {}).get("done") or n.content or "").strip()
            lines.append(f"### #{n.id[:6]} · {instruction.strip()[:200]}\n{done[:800]}")
        head = {"ok": f"已干净地合并 {len(nodes)} 个分支。",
                "conflict": f"合并 {len(nodes)} 个分支时有冲突：" + ", ".join(conflict_files)
                            + "。已自动派一个 agent 去解决。",
                "tests_failed": f"已合并 {len(nodes)} 个分支，但测试出现新的失败。已自动派一个 agent 去修。",
                "failed": f"合并失败：{error}"}[status]
        if test_fact:
            head += f"\n\n测试：{test_fact['passed']} 通过，{test_fact['failed']} 失败" + (
                f"（新失败 {len(test_fact['new_failures'])}）" if test_fact["new_failures"] else "")
        content = head + "\n\n## 合并的分支\n\n" + "\n\n".join(lines)
        facts = {"files": [{"path": f["path"], "added": f["additions"],
                            "removed": f["deletions"]} for f in files],
                 "duration_s": round(time.monotonic() - t0, 1)}
        if conflict_files:
            facts["conflict_files"] = conflict_files
        if test_fact:
            facts["test"] = test_fact
        if error:
            facts["error"] = error
        md = {"agent": True, "orch_run": True, "orch_merge": True, "executor": "merge",
              "run_status": status, "merged_from": [n.id for n in nodes],
              "ws_commit": merge_commit or base_commit, "ws_parent_commit": base_commit,
              "files_changed": len(files), "facts": facts,
              "stop_reason": "error" if status == "failed" else None}
        chat.store.finalize_node(asst.id, content, {}, md)
        turn.append({"type": "done", "id": asst.id, "content": content, "usage": {},
                     "stats": "", "ws_commit": md["ws_commit"], "stop_reason": md["stop_reason"],
                     "run_status": status, "tool_results": []})
        self._retire_turn(turn, asst.id)
        if status in ("conflict", "tests_failed") and merge_commit:
            if status == "conflict":
                instruction = ("解决这些文件里的合并冲突标记，保留每个被合并分支的改动意图；"
                               "如果有测试命令，最后跑一遍测试。冲突文件：" + ", ".join(conflict_files))
            else:
                instruction = ("合并后的结果出现了新的测试失败。修复它们，同时保留每个被合并分支的改动意图。")
            extra = "## Merge task — the branch changes to preserve\n\n" + "\n\n".join(lines)
            if test_fact and test_fact.get("summary"):
                extra += "\n\n### Failing test output\n\n```\n" + str(test_fact["summary"])[:2000] + "\n```"
            try:
                self.start_chat_runs(instruction, asst.id, [agent], extra_context=extra)
            except Exception:  # noqa: BLE001 - the merge node itself is still useful
                pass
