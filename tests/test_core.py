import json
import os
import tempfile
import types
import unittest

import main
from main import (
    apply_explicit_cache_control,
    build_context,
    explicit_cache_control_indexes,
    extract_tool_calls,
)
from models import Node
from store import ROOT_ID, TreeStore
from tools import ToolRegistry, ToolResult, ToolSpec
from web_tools import WebToolResult


class StoreContextTests(unittest.TestCase):
    def test_path_to_root_returns_selected_branch_only(self):
        with tempfile.TemporaryDirectory() as d:
            store = TreeStore(os.path.join(d, "chat.jsonl"))
            first = Node.new("user", "root question", ROOT_ID)
            store.add(first)
            answer = Node.new("assistant", "root answer", first.id)
            store.add(answer)
            fork = Node.new("user", "fork question", first.id)
            store.add(fork)

            self.assertEqual([n.id for n in store.path_to_root(fork.id)], [first.id, fork.id])
            self.assertEqual([n.id for n in store.path_to_root(answer.id)], [first.id, answer.id])

    def test_build_context_has_stable_cache_hashes(self):
        user = Node.new("user", "定义一个不变量", ROOT_ID)
        assistant = Node.new("assistant", "不变量是 I。", user.id)
        path = [user, assistant]

        c1 = build_context(path)
        c2 = build_context(path)

        self.assertEqual(c1["cache"]["request_hash"], c2["cache"]["request_hash"])
        self.assertEqual(
            c1["cache"]["next_turn_prefix_hash"],
            c2["cache"]["next_turn_prefix_hash"],
        )
        self.assertEqual(c1["cache"]["request_message_count"], len(c1["messages"]))

    def test_store_title_and_tags_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "chat.jsonl")
            store = TreeStore(db_path)
            node = Node.new("user", "prove lemma", ROOT_ID)
            store.add(node)

            titled = store.set_title(node.id, "Main Lemma")
            tagged = store.add_tags(node.id, ["math", "paper", "math"])
            untagged = store.remove_tags(node.id, ["paper"])

            self.assertEqual(titled.title, "Main Lemma")
            self.assertEqual(tagged.metadata["tags"], ["math", "paper"])
            self.assertEqual(untagged.metadata["tags"], ["math"])

    def test_summary_nodes_are_sent_as_system_context(self):
        user = Node.new("user", "question", ROOT_ID)
        summary = Node.new("summary", "Key invariant: I stays positive.", user.id)

        context = build_context([user, summary])

        self.assertEqual(context["messages"][-1]["role"], "system")
        self.assertIn("Key invariant", context["messages"][-1]["content"])

    def test_truncation_never_splits_tool_exchange(self):
        # Long tool chains push the branch past the message cap; the naive
        # tail cut would start `kept` with an orphan role=tool message (its
        # assistant tool_calls parent fell into the omitted range) — the API
        # rejects that with "tool_call_id is not found".
        nodes = [Node.new("user", "q", ROOT_ID)]
        for i in range(41):
            call = {"function": {"arguments": "{}", "name": "web_search"},
                    "id": f"c{i}", "type": "function"}
            am = Node.new("raw_message", json.dumps(
                {"content": "", "role": "assistant", "tool_calls": [call]}), nodes[-1].id)
            tm = Node.new("raw_message", json.dumps(
                {"content": "out", "role": "tool", "tool_call_id": f"c{i}"}), am.id)
            nodes += [am, tm]
        nodes.append(Node.new("user", "q2", nodes[-1].id))

        ctx = build_context(nodes)
        self.assertGreater(len(nodes), 80)  # cap exceeded → truncation active
        seen_call_ids = set()
        for m in ctx["messages"]:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    seen_call_ids.add(c["id"])
            elif m["role"] == "tool":
                self.assertIn(m["tool_call_id"], seen_call_ids)

    def test_store_repairs_orphan_nodes_on_load(self):
        """Nodes whose parent_id points nowhere are re-parented to root."""
        with tempfile.TemporaryDirectory() as d:
            jsonl_path = os.path.join(d, "chatable.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                # Write two nodes: one whose parent doesn't exist (orphan).
                fh.write(json.dumps({
                    "id": "node1", "role": "user", "content": "question",
                    "parent_id": "root", "created_at": 1.0,
                    "usage": None, "pinned": False, "token_count": 3,
                    "title": "", "summary": "", "metadata": None,
                }) + "\n")
                fh.write(json.dumps({
                    "id": "node2", "role": "assistant", "content": "answer",
                    "parent_id": "missing_parent", "created_at": 2.0,
                    "usage": None, "pinned": False, "token_count": 3,
                    "title": "", "summary": "", "metadata": None,
                }) + "\n")
            store = TreeStore(jsonl_path)
            self.assertTrue(store.exists("node1"))
            self.assertTrue(store.exists("node2"))
            self.assertEqual(store.get("node2").parent_id, ROOT_ID)  # repaired

    def test_store_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "nested", "chatable.jsonl")
            store = TreeStore(db_path)
            # Tree files are created lazily on first write.
            node = Node.new("user", "test", ROOT_ID)
            store.add(node)
            trunk_file = os.path.join(d, "nested", "trees", node.id + ".jsonl")
            self.assertTrue(os.path.exists(trunk_file))
            self.assertTrue(store.schema_info()["ok"])


class PerTreeStoreTests(unittest.TestCase):
    """The per-conversation file layout: one trees/<trunk_id>.jsonl each."""

    @staticmethod
    def _tree_path(d, nid):
        return os.path.join(d, "trees", nid + ".jsonl")

    def test_each_trunk_gets_its_own_file_and_writes_are_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            store = TreeStore(os.path.join(d, "chat.jsonl"))
            a = Node.new("user", "conversation A", ROOT_ID)
            b = Node.new("user", "conversation B", ROOT_ID)
            store.add(a)
            store.add(b)
            path_a = self._tree_path(d, a.id)
            path_b = self._tree_path(d, b.id)
            self.assertTrue(os.path.exists(path_a))
            self.assertTrue(os.path.exists(path_b))

            with open(path_b, encoding="utf-8") as fh:
                before = fh.read()
            store.set_title(a.id, "title A")  # must only touch A's file
            with open(path_b, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), before)

            reply = Node.new("assistant", "answer", a.id)
            store.add(reply)  # a child lands in its trunk's file
            with open(path_a, encoding="utf-8") as fh:
                self.assertIn(reply.id, fh.read())
            with open(path_b, encoding="utf-8") as fh:
                self.assertNotIn(reply.id, fh.read())

    def test_mutations_append_and_last_write_wins(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "chat.jsonl")
            store = TreeStore(db_path)
            n = Node.new("user", "q", ROOT_ID)
            store.add(n)
            store.set_title(n.id, "first")
            store.set_title(n.id, "second")

            with open(self._tree_path(d, n.id), encoding="utf-8") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            self.assertEqual(len(lines), 3)  # add + 2 renames, all appended

            fresh = TreeStore(db_path)
            self.assertEqual(fresh.get(n.id).title, "second")

    def test_finalize_node_writes_all_fields_in_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            store = TreeStore(os.path.join(d, "chat.jsonl"))
            n = Node.new("assistant", "", ROOT_ID)
            store.add(n)
            store.finalize_node(n.id, "full text", {"input": 1}, {"k": "v"})

            node = store.get(n.id)
            self.assertEqual(node.content, "full text")
            self.assertEqual(node.usage, {"input": 1})
            self.assertEqual(node.metadata, {"k": "v"})

            with open(self._tree_path(d, n.id), encoding="utf-8") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)  # add + finalize, nothing more

    def test_delete_trunk_removes_file(self):
        with tempfile.TemporaryDirectory() as d:
            store = TreeStore(os.path.join(d, "chat.jsonl"))
            t = Node.new("user", "conv", ROOT_ID)
            store.add(t)
            child = Node.new("assistant", "ans", t.id)
            store.add(child)
            self.assertTrue(os.path.exists(self._tree_path(d, t.id)))

            removed = store.delete_subtree(t.id)
            self.assertEqual(removed, 2)
            self.assertFalse(os.path.exists(self._tree_path(d, t.id)))
            self.assertFalse(store.exists(t.id))
            self.assertFalse(store.exists(child.id))

    def test_delete_branch_rewrites_trunk_file(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "chat.jsonl")
            store = TreeStore(db_path)
            t = Node.new("user", "conv", ROOT_ID)
            store.add(t)
            c1 = Node.new("assistant", "ans 1", t.id)
            c2 = Node.new("assistant", "ans 2", t.id)
            store.add(c1)
            store.add(c2)

            removed = store.delete_subtree(c1.id)
            self.assertEqual(removed, 1)

            with open(self._tree_path(d, t.id), encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn(t.id, content)
            self.assertIn(c2.id, content)
            self.assertNotIn(c1.id, content)

            fresh = TreeStore(db_path)
            self.assertFalse(fresh.exists(c1.id))
            self.assertTrue(fresh.exists(c2.id))

    def test_legacy_single_file_migrates_to_per_trunk_trees(self):
        with tempfile.TemporaryDirectory() as d:
            legacy = os.path.join(d, "chatable.jsonl")
            rows = [
                {"id": "t1", "role": "user", "content": "conv one",
                 "parent_id": "root", "created_at": 1.0},
                {"id": "r1", "role": "assistant", "content": "reply one",
                 "parent_id": "t1", "created_at": 2.0},
                {"id": "t2", "role": "user", "content": "conv two",
                 "parent_id": "root", "created_at": 3.0},
            ]
            with open(legacy, "w", encoding="utf-8") as fh:
                for row in rows:
                    row = dict(row, usage=None, pinned=False, token_count=3,
                               title="", summary="", metadata=None)
                    fh.write(json.dumps(row) + "\n")

            store = TreeStore(legacy)
            self.assertTrue(os.path.exists(self._tree_path(d, "t1")))
            self.assertTrue(os.path.exists(self._tree_path(d, "t2")))
            self.assertFalse(os.path.exists(legacy))
            self.assertTrue(os.path.exists(legacy + ".bak"))

            with open(self._tree_path(d, "t1"), encoding="utf-8") as fh:
                t1_content = fh.read()
            self.assertIn("r1", t1_content)   # child grouped with its trunk
            self.assertNotIn("t2", t1_content)
            self.assertEqual(store.get("r1").parent_id, "t1")


class ExportTreeMarkdownTests(unittest.TestCase):
    def _service(self, d):
        from service import ChatService
        return ChatService(TreeStore(os.path.join(d, "chat.jsonl")))

    def _node(self, role, content, parent_id, created_at, **kw):
        n = Node.new(role, content, parent_id)
        n.created_at = created_at
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def _branched_tree(self, store):
        # trunk Q → A → two forked follow-ups, each with its own answer
        q = self._node("user", "root question", ROOT_ID, 1.0)
        store.add(q)
        a = self._node("assistant", "root answer", q.id, 2.0)
        store.add(a)
        f1 = self._node("user", "fork one", a.id, 3.0)
        f2 = self._node("user", "fork two", a.id, 4.0)
        store.add(f1)
        store.add(f2)
        store.add(self._node("assistant", "answer one", f1.id, 5.0))
        store.add(self._node("assistant", "answer two", f2.id, 6.0))
        return q, a, f1, f2

    def test_export_tree_hierarchical_sections(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q, a, f1, f2 = self._branched_tree(svc.store)

            fname, md = svc.export_tree_markdown(q.id)

            self.assertTrue(fname.endswith(".md"))
            self.assertIn("# ", md.splitlines()[0])     # document title
            self.assertIn("## root question", md)       # trunk user node at level 2
            self.assertIn("### fork one", md)           # forks one level deeper
            self.assertIn("### fork two", md)
            self.assertIn("root answer", md)
            self.assertIn("answer one", md)
            self.assertIn("answer two", md)
            # chronological: fork one before fork two
            self.assertLess(md.index("fork one"), md.index("fork two"))

    def test_export_preserves_code_math_and_quotes(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q = self._node("user", "格式问题", ROOT_ID, 1.0)
            svc.store.add(q)
            body = "代码:\n\n```python\nprint('hi')\n```\n\n数学: $x^2 + y^2 = z^2$\n\n> 一段引用"
            svc.store.add(self._node("assistant", body, q.id, 2.0))

            _, md = svc.export_tree_markdown(q.id)

            self.assertIn("```python\nprint('hi')\n```", md)   # code fence untouched
            self.assertIn("$x^2 + y^2 = z^2$", md)             # math untouched
            self.assertIn("> 一段引用", md)                     # quote untouched

    def test_export_omits_tool_results_and_reasoning(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q = self._node("user", "question", ROOT_ID, 1.0)
            svc.store.add(q)
            a = self._node("assistant", "clean answer", q.id, 2.0)
            a.metadata = {
                "reasoning": "secret thinking trace",
                "tool_results": [{"name": "web_search", "arguments": {"q": "x"},
                                  "output": "raw tool output"}],
            }
            svc.store.add(a)

            _, md = svc.export_tree_markdown(q.id)

            self.assertIn("clean answer", md)
            self.assertNotIn("secret thinking trace", md)
            self.assertNotIn("web_search", md)
            self.assertNotIn("raw tool output", md)

    def test_export_rejects_non_trunk_node(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q, a, _, _ = self._branched_tree(svc.store)
            with self.assertRaises(ValueError):
                svc.export_tree_markdown(a.id)

    def test_export_sanitizes_heading(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q = self._node("user", "# 注入标题\n第二行内容", ROOT_ID, 1.0)
            svc.store.add(q)

            _, md = svc.export_tree_markdown(q.id)

            for line in md.splitlines():
                if line.startswith("## "):
                    self.assertNotIn("\n", line)              # single line
                    self.assertFalse(line.startswith("## #"))  # no injected '#'

    def test_export_filename_uses_title_slug(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._service(d)
            q = self._node("user", "root question", ROOT_ID, 1.0, title="Transformer 笔记")
            svc.store.add(q)

            fname, _ = svc.export_tree_markdown(q.id)

            self.assertTrue(fname.startswith("chatable-Transformer-笔记-"))
            self.assertTrue(fname.endswith(".md"))


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_validates_required_arguments(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="echo input",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            handler=lambda args: WebToolResult.success(args["text"]),
        ))

        result = registry.run("echo", {})

        self.assertFalse(result.ok)
        self.assertIn("missing required argument", result.output)

    def test_tool_registry_runs_and_truncates_results(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="long",
            description="long output",
            input_schema={"type": "object", "properties": {}},
            handler=lambda args: WebToolResult.success("abcdef"),
            result_max_chars=3,
        ))

        result = registry.run("long", {})

        self.assertTrue(result.ok)
        self.assertIn("abc", result.output)
        self.assertIn("truncated", result.output)

    def test_decision_prompt_is_generated_from_registered_tools(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="echo input",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            handler=lambda args: WebToolResult.success(args["text"]),
        ))

        prompt = registry.decision_prompt()

        self.assertIn("echo", prompt)
        self.assertIn("echo input", prompt)
        self.assertIn('"required":["text"]', prompt)

    def test_openai_tools_schema_is_generated_from_registry(self):
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="echo input",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            handler=lambda args: WebToolResult.success(args["text"]),
        ))

        tools = registry.openai_tools()

        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "echo")
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["text"])


class BackendToolCallTests(unittest.TestCase):
    def test_deepseek_openai_sdk_defaults(self):
        # main.py applies ~/.chatable/settings.json at import time; point it at
        # a nonexistent file so the defaults are what we actually assert.
        import importlib

        old = os.environ.get("CHATTABLE_SETTINGS")
        os.environ["CHATTABLE_SETTINGS"] = "/nonexistent/chatable-settings.json"
        try:
            importlib.reload(main)
            self.assertEqual(main.OPENAI_BASE_URL, "https://api.deepseek.com")
            self.assertEqual(main.OPENAI_MODEL, "deepseek-v4-pro")
            self.assertEqual(main.OPENAI_REASONING_EFFORT, "high")
            self.assertEqual(main.OPENAI_THINKING_TYPE, "enabled")
        finally:
            if old is None:
                os.environ.pop("CHATTABLE_SETTINGS", None)
            else:
                os.environ["CHATTABLE_SETTINGS"] = old
            importlib.reload(main)

    def test_extract_tool_calls_parses_openai_shape(self):
        resp = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": "{\"query\":\"prefix cache\"}",
                        },
                    }]
                }
            }]
        }

        calls = extract_tool_calls(resp)

        self.assertEqual(calls, [{
            "id": "call_1",
            "name": "web_search",
            "arguments": {"query": "prefix cache"},
        }])

    def test_explicit_cache_control_marks_stable_system_messages(self):
        old_policy = main.CACHE_POLICY
        try:
            main.CACHE_POLICY = "explicit"
            messages = [
                {"role": "system", "content": "system"},
                {"role": "system", "content": "pinned"},
                {"role": "user", "content": "latest"},
            ]

            indexes = explicit_cache_control_indexes(messages)
            marked = apply_explicit_cache_control(messages)

            self.assertEqual(indexes, [0, 1])
            self.assertIsInstance(marked[0]["content"], list)
            self.assertEqual(marked[0]["content"][0]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(messages[0]["content"], "system")
        finally:
            main.CACHE_POLICY = old_policy


class StreamTruncationTests(unittest.TestCase):
    """_openai_stream must distinguish a clean end from a dropped connection."""

    def _run_with_chunks(self, chunks):
        class _Stream:
            def __iter__(self):
                return iter(chunks)

            def close(self):
                pass

        class _Completions:
            def create(self, **kwargs):
                return _Stream()

        fake = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        old = main._openai_client
        main._openai_client = fake
        try:
            return list(main._openai_stream([{"role": "user", "content": "hi"}]))
        finally:
            main._openai_client = old

    def test_clean_end_yields_done(self):
        events = self._run_with_chunks([
            {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        ])
        self.assertIn(("delta", "partial"), events)
        self.assertEqual(events[-1][0], "done")

    def test_silent_mid_stream_close_raises(self):
        # The relay dropped the connection: deltas arrived but neither a
        # finish_reason nor a usage chunk ever did.
        with self.assertRaises(main.StreamTruncatedError):
            self._run_with_chunks([
                {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]},
            ])

    def test_tool_call_deltas_assembled(self):
        # Mirrors the real DeepSeek stream shape: first chunk carries id/name,
        # later chunks carry arguments string fragments (see verification).
        events = self._run_with_chunks([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": ""}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"query\":"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": " \"hi\"}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        ])
        kinds = [k for k, _ in events]
        self.assertEqual(kinds, ["tool_calls", "done"])
        calls = events[0][1]
        self.assertEqual(calls, [{"id": "call_1", "name": "web_search",
                                  "arguments": {"query": "hi"}}])


class LiveTurnKeepaliveTests(unittest.TestCase):
    def test_read_from_emits_keepalive_when_idle(self):
        from service import _LiveTurn

        turn = _LiveTurn("a1")
        gen = turn.read_from(0, timeout=0.05)
        self.assertEqual(next(gen)["type"], "_ping")
        turn.finish()
        # After finish with no events buffered, the generator returns.
        self.assertIsNone(next(gen, None))


class RawToolCallJsonScrubTests(unittest.TestCase):
    DUMP = ('{\n  "tool_calls": [\n    {\n      "id": "call_cache",\n'
            '      "type": "function",\n'
            '      "function": {"name": "web_fetch", "arguments": "{\\"url\\":\\"https://x\\"}"}\n'
            '    }\n  ]\n}')

    def setUp(self):
        from service import _strip_raw_tool_call_json
        self.scrub = _strip_raw_tool_call_json

    def test_whole_message_dump_removed(self):
        self.assertEqual(self.scrub(self.DUMP), "")

    def test_fenced_dump_removed(self):
        text = "Let me fetch that.\n```json\n" + self.DUMP + "\n```\nDone."
        self.assertEqual(self.scrub(text), "Let me fetch that.\n\nDone.")

    def test_prose_around_dump_kept(self):
        text = "Trying the cache:\n" + self.DUMP + "\nThat failed, sorry."
        out = self.scrub(text)
        self.assertIn("Trying the cache:", out)
        self.assertIn("That failed, sorry.", out)
        self.assertNotIn("tool_calls", out)

    def test_ordinary_json_untouched(self):
        text = 'Example: {"result": [1, 2, {"nested": true}]} end.'
        self.assertEqual(self.scrub(text), text)


class GenerateRetryTests(unittest.TestCase):
    """The unified stream loop recovers from soft failures and runs tools."""

    def _run_turn(self, fake_stream, extra_patches=None):
        from contextlib import ExitStack
        from unittest.mock import patch

        import service as service_mod
        from service import ChatService

        d = tempfile.TemporaryDirectory()
        svc = ChatService(TreeStore(os.path.join(d.name, "chat.jsonl")))
        with ExitStack() as stack:
            stack.enter_context(patch.object(service_mod, "stream_backend", fake_stream))
            for p in (extra_patches or []):
                stack.enter_context(p)
            info = svc.start_turn("q", None)
            events = list(svc.subscribe(info["assistant_id"], 0))
        node = svc.store.get(info["assistant_id"])
        result = {
            "content": node.content,
            "metadata": node.metadata or {},
            "usage": node.usage or {},
            "events": events,
        }
        d.cleanup()
        return result

    def test_markup_only_answer_retries_with_nudge(self):
        calls = []

        def fake_stream(messages, tools=None):
            calls.append(messages)
            if len(calls) == 1:
                # First iteration: only a tool_calls dump (scrubbed to empty).
                yield ("delta", '{"tool_calls": [{"name": "web_search", "arguments": {}}]}')
                yield ("done", {"input": 1, "output": 1})
            else:
                yield ("delta", "real answer")
                yield ("done", {"input": 1, "output": 2})

        res = self._run_turn(fake_stream)

        self.assertEqual(res["content"], "real answer")
        self.assertEqual(len(calls), 2)
        self.assertIn("不要再输出任何工具调用", calls[1][-1]["content"])

    def test_truncated_stream_regenerates(self):
        calls = []

        def fake_stream(messages, tools=None):
            calls.append(messages)
            if len(calls) == 1:
                yield ("delta", "partial")
                raise main.StreamTruncatedError("cut")
            yield ("delta", "full answer")
            yield ("done", {"input": 1, "output": 1})

        res = self._run_turn(fake_stream)

        self.assertEqual(res["content"], "full answer")
        self.assertEqual(len(calls), 2)

    def test_double_failure_persists_guard_text(self):
        def fake_stream(messages, tools=None):
            yield ("done", {"input": 1, "output": 0})

        res = self._run_turn(fake_stream)

        self.assertIn("[empty reply:", res["content"])

    def test_unified_loop_executes_tools_and_continues(self):
        from unittest.mock import patch

        import service as service_mod

        calls = []

        def fake_stream(messages, tools=None):
            calls.append({"messages": messages, "tools": tools})
            if len(calls) == 1:
                yield ("reasoning", "让我想想…")
                yield ("delta", "先搜一下。")
                yield ("tool_calls", [{"id": "c1", "name": "web_search",
                                       "arguments": {"query": "kv"}}])
                yield ("done", {"input": 10, "output": 5})
            else:
                yield ("delta", "最终答案")
                yield ("done", {"input": 20, "output": 7})

        def fake_run(name, arguments):
            return ToolResult(call_id="t1", name=name, arguments=arguments,
                              ok=True, output="fake output")

        res = self._run_turn(fake_stream, [
            patch.object(service_mod.TOOL_REGISTRY, "run", fake_run),
        ])

        self.assertEqual(res["content"], "先搜一下。\n\n最终答案")
        self.assertEqual(res["metadata"]["reasoning"], "让我想想…")
        self.assertEqual(res["metadata"]["reasoning_chars"], len("让我想想…"))
        self.assertEqual(len(res["metadata"]["tool_results"]), 1)
        self.assertEqual(res["metadata"]["tool_results"][0]["name"], "web_search")
        self.assertEqual(res["usage"]["input"], 30)
        # The second iteration received the native tool exchange appended.
        second = calls[1]["messages"]
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertEqual(second[-2]["tool_calls"][0]["id"], "c1")
        # DeepSeek thinking mode requires reasoning_content to be passed back.
        self.assertEqual(second[-2]["reasoning_content"], "让我想想…")
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["tool_call_id"], "c1")
        self.assertEqual(second[-1]["content"], "fake output")

    def test_fuse_streams_final_round_without_tools(self):
        from unittest.mock import patch

        import service as service_mod

        round_tools = []

        def fake_stream(messages, tools=None):
            round_tools.append(tools)
            if tools is not None:
                n = len(round_tools)
                yield ("tool_calls", [{"id": f"c{n}", "name": "web_search",
                                       "arguments": {"query": f"q{n}"}}])
                yield ("done", {"input": 1, "output": 1})
            else:
                yield ("delta", "收尾回答")
                yield ("done", {"input": 1, "output": 1})

        def fake_run(name, arguments):
            return ToolResult(call_id="t", name=name, arguments=arguments,
                              ok=True, output="out")

        res = self._run_turn(fake_stream, [
            patch.object(service_mod, "AUTO_TOOL_MAX_ROUNDS", 2),
            patch.object(service_mod.TOOL_REGISTRY, "run", fake_run),
        ])

        self.assertEqual(res["content"], "收尾回答")
        self.assertEqual(len(res["metadata"]["tool_results"]), 2)
        # 2 tool rounds + 1 final no-tools round.
        self.assertEqual(len(round_tools), 3)
        self.assertIsNone(round_tools[-1])

    def test_tool_exchange_replayed_byte_identical_next_turn(self):
        from contextlib import ExitStack
        from unittest.mock import patch

        import service as service_mod
        from service import ChatService

        def canon(m):
            return json.dumps(m, ensure_ascii=False, sort_keys=True)

        def contains_subseq(haystack, needle):
            return any(
                haystack[i:i + len(needle)] == needle
                for i in range(len(haystack) - len(needle) + 1)
            )

        d = tempfile.TemporaryDirectory()
        svc = ChatService(TreeStore(os.path.join(d.name, "chat.jsonl")))
        calls = []

        def fake_stream(messages, tools=None):
            calls.append(list(messages))
            if len(calls) == 1:
                yield ("tool_calls", [{"id": "c1", "name": "web_search",
                                       "arguments": {"query": "kv"}}])
                yield ("done", {"input": 1, "output": 1})
            else:
                yield ("delta", "answer")
                yield ("done", {"input": 1, "output": 1})

        def fake_run(name, arguments):
            return ToolResult(call_id="t1", name=name, arguments=arguments,
                              ok=True, output="搜索结果")

        with ExitStack() as stack:
            stack.enter_context(patch.object(service_mod, "stream_backend", fake_stream))
            stack.enter_context(patch.object(service_mod.TOOL_REGISTRY, "run", fake_run))
            info1 = svc.start_turn("q1", None)
            list(svc.subscribe(info1["assistant_id"], 0))
            info2 = svc.start_turn("q2", info1["assistant_id"])
            list(svc.subscribe(info2["assistant_id"], 0))

        meta = svc.store.get(info1["assistant_id"]).metadata
        replay = [canon(m) for m in meta["tool_messages"]]
        self.assertTrue(replay)
        # The exchange as sent live during turn 1 matches what was persisted…
        turn1_second = [canon(m) for m in calls[1]]
        self.assertTrue(contains_subseq(turn1_second, replay))
        # …and turn 2 replays it byte-identically (the cache prefix carries over)…
        turn2_msgs = [canon(m) for m in calls[2]]
        self.assertTrue(contains_subseq(turn2_msgs, replay))
        # …with no folded system-text fallback.
        self.assertFalse(any(
            "Automatic tool result" in (m.get("content") or "")
            for m in calls[2]
        ))
        d.cleanup()


class NodeGistTests(unittest.TestCase):
    """generate_title fills title+summary; rename_node manages the lock."""

    def _svc(self, d):
        from service import ChatService
        return ChatService(TreeStore(os.path.join(d, "chat.jsonl")))

    def _qa(self, svc):
        user = Node.new("user", "什么是 RoPE?", ROOT_ID)
        svc.store.add(user)
        asst = Node.new("assistant", "RoPE 是旋转位置编码…", user.id)
        svc.store.add(asst)
        return user, asst

    def test_generate_title_fills_title_and_summary(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(d)
            _, asst = self._qa(svc)
            gist = {"title": "RoPE 简介", "summary": "RoPE 是一种相对位置编码。"}
            with patch.object(main, "summarize_turn", lambda u, a: gist):
                res = svc.generate_title(asst.id)
            self.assertTrue(res["ok"])
            node = svc.store.get(asst.id)
            self.assertEqual(node.title, "RoPE 简介")
            self.assertEqual(node.summary, "RoPE 是一种相对位置编码。")

    def test_generate_title_skips_titled_and_respects_lock(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(d)
            _, asst = self._qa(svc)
            svc.store.set_title(asst.id, "人工命名")
            svc.store.update_metadata(asst.id, {"title_locked": True})
            gist = {"title": "自动标题", "summary": "自动摘要。"}
            with patch.object(main, "summarize_turn", lambda u, a: gist):
                res = svc.generate_title(asst.id)
            node = svc.store.get(asst.id)
            # Locked: title untouched, but the empty summary still got filled.
            self.assertEqual(node.title, "人工命名")
            self.assertEqual(node.summary, "自动摘要。")
            self.assertNotIn("skipped", res)

    def test_generate_title_skips_when_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(d)
            _, asst = self._qa(svc)
            svc.store.set_title(asst.id, "已有标题")
            svc._set_node_summary(asst.id, "已有摘要。")
            res = svc.generate_title(asst.id)  # no backend call needed
            self.assertEqual(res.get("skipped"), "already titled")

    def test_rename_node_sets_and_clears_lock(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(d)
            user, _ = self._qa(svc)
            svc.rename_node(user.id, "我的问题")
            node = svc.store.get(user.id)
            self.assertEqual(node.title, "我的问题")
            self.assertTrue(node.metadata["title_locked"])
            # Empty title restores the automatic flow.
            svc.rename_node(user.id, "  ")
            node = svc.store.get(user.id)
            self.assertEqual(node.title, "")
            self.assertFalse(node.metadata["title_locked"])

    def test_rename_node_rejects_other_roles(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(d)
            note = Node.new("system_note", "tool output", ROOT_ID)
            svc.store.add(note)
            with self.assertRaises(ValueError):
                svc.rename_node(note.id, "x")


class InjectImagesTests(unittest.TestCase):
    """_inject_images turns the current turn's user message multimodal."""

    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    def _svc(self, d, attachments):
        from unittest.mock import patch

        from service import ChatService

        svc = ChatService(TreeStore(os.path.join(d, "chat.jsonl")))
        with patch.object(svc, "_run_turn", lambda *a, **k: None):
            info = svc.start_turn("看图说话", None, attachments=attachments)
        return svc, info

    def _write_img(self, d):
        img = os.path.join(d, "pic.png")
        with open(img, "wb") as f:
            f.write(self.PNG)
        return img

    def test_images_injected_into_last_user_message(self):
        from unittest.mock import patch

        import service as service_mod

        with tempfile.TemporaryDirectory() as d:
            img = self._write_img(d)
            svc, info = self._svc(d, [{"filename": "pic.png", "path": img, "mime": "image/png"}])
            with patch.object(service_mod.main, "VISION_SUPPORTED", True):
                ctx = svc._live_context(info["user_id"], [])
            last_user = [m for m in ctx["messages"] if m["role"] == "user"][-1]
            self.assertIsInstance(last_user["content"], list)
            self.assertEqual(last_user["content"][0], {"type": "text", "text": "看图说话"})
            part = last_user["content"][1]
            self.assertEqual(part["type"], "image_url")
            self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_no_attachments_keeps_plain_text(self):
        with tempfile.TemporaryDirectory() as d:
            svc, info = self._svc(d, [])
            ctx = svc._live_context(info["user_id"], [])
            last_user = [m for m in ctx["messages"] if m["role"] == "user"][-1]
            self.assertIsInstance(last_user["content"], str)

    def test_vision_disabled_keeps_plain_text(self):
        from unittest.mock import patch

        import service as service_mod

        with tempfile.TemporaryDirectory() as d:
            img = self._write_img(d)
            svc, info = self._svc(d, [{"filename": "pic.png", "path": img, "mime": "image/png"}])
            with patch.object(service_mod, "VISION_ENABLED", False):
                ctx = svc._live_context(info["user_id"], [])
            last_user = [m for m in ctx["messages"] if m["role"] == "user"][-1]
            self.assertIsInstance(last_user["content"], str)

    def test_attachments_persisted_on_user_node(self):
        with tempfile.TemporaryDirectory() as d:
            img = self._write_img(d)
            svc, info = self._svc(d, [{"filename": "pic.png", "path": img, "mime": "image/png"}])
            meta = svc.store.get(info["user_id"]).metadata
            self.assertEqual(meta["attachments"][0]["mime"], "image/png")


class PersistedDataDirTests(unittest.TestCase):
    def test_data_dir_merge_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            old = main.SETTINGS_PATH
            main.SETTINGS_PATH = path
            try:
                main.persist_config(data_dir="/tmp/mydata")
                self.assertEqual(main.load_persisted_data_dir(), "/tmp/mydata")
                # A later model-only save keeps the persisted folder.
                main.persist_config()
                self.assertEqual(main.load_persisted_data_dir(), "/tmp/mydata")
            finally:
                main.SETTINGS_PATH = old


class SummarizeTurnTests(unittest.TestCase):
    def _run_with_text(self, text):
        resp = {"choices": [{"message": {"content": text}}]}

        class _Completions:
            def create(self, **kwargs):
                return resp

        fake = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        old = main._openai_client
        main._openai_client = fake
        try:
            return main.summarize_turn("q", "a")
        finally:
            main._openai_client = old

    def test_parses_two_line_gist(self):
        gist = self._run_with_text("Title: RoPE 简介\nSummary: RoPE 是一种相对位置编码。")
        self.assertEqual(gist, {"title": "RoPE 简介",
                                "summary": "RoPE 是一种相对位置编码。"})

    def test_incomplete_output_raises(self):
        with self.assertRaises(main.SummaryUnavailable):
            self._run_with_text("Title: 只有标题")


class VisionProbeTests(unittest.TestCase):
    """probe_vision_support classifies the backend's image acceptance."""

    def _probe(self, create_outcome):
        class _Completions:
            def create(self, **kwargs):
                if isinstance(create_outcome, Exception):
                    raise create_outcome
                return create_outcome

        fake = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        old_client = main._openai_client
        old_flag = main.VISION_SUPPORTED
        old_path = main.SETTINGS_PATH
        main._openai_client = fake
        main.SETTINGS_PATH = os.path.join(tempfile.mkdtemp(), "settings.json")
        try:
            return main.probe_vision_support()
        finally:
            main._openai_client = old_client
            main.VISION_SUPPORTED = old_flag
            main.SETTINGS_PATH = old_path

    def test_probe_success(self):
        self.assertTrue(self._probe(object()))

    def test_probe_4xx_marks_unsupported(self):
        class _BadRequest(Exception):
            status_code = 400

        self.assertFalse(self._probe(_BadRequest("no images")))

    def test_probe_network_error_stays_unknown(self):
        main.VISION_SUPPORTED = None  # the finally in _probe restores this
        result = self._probe(ConnectionError("down"))
        self.assertIsNone(result)


class HtmlImageExtractionTests(unittest.TestCase):
    def test_filters_noise_and_collects_content_images(self):
        from web_tools import _extract_image_urls

        html = """
        <img src="/assets/logo.png">
        <img src="https://cdn.example.com/tracking-pixel.gif">
        <figure><img src="images/figure1.png"></figure>
        <img src="https://cdn.example.com/chart.jpg">
        <img src="data:image/png;base64,xxxx">
        <img srcset="https://cdn.example.com/wide-2x.jpg 2x">
        """
        urls = _extract_image_urls(html, "https://example.com/blog/post")

        self.assertIn("https://example.com/blog/images/figure1.png", urls)
        self.assertIn("https://cdn.example.com/chart.jpg", urls)
        self.assertIn("https://cdn.example.com/wide-2x.jpg", urls)
        self.assertFalse(any("logo" in u for u in urls))
        self.assertFalse(any("tracking" in u for u in urls))
        self.assertFalse(any(u.startswith("data:") for u in urls))


class PdfImageExtractionTests(unittest.TestCase):
    def test_scanned_pdf_renders_pages(self):
        import web_tools

        if web_tools.fitz is None:
            self.skipTest("pymupdf not installed")
        fitz = web_tools.fitz
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(10, 10, 200, 100), color=(1, 0, 0), fill=(0, 0, 1))
        pdf_bytes = doc.tobytes()
        doc.close()

        images = web_tools._extract_pdf_images(pdf_bytes)

        self.assertTrue(images)
        self.assertEqual(images[0]["url"], "page#1")
        self.assertTrue(images[0]["data_url"].startswith("data:image/jpeg;base64,"))


class ToolImageInjectionTests(unittest.TestCase):
    """Images captured by tools reach the model as a synthetic user message."""

    def _run(self, vision_supported):
        from unittest.mock import patch

        import service as service_mod

        calls = []

        def fake_stream(messages, tools=None):
            calls.append(list(messages))
            if len(calls) == 1:
                yield ("tool_calls", [{"id": "c1", "name": "web_fetch",
                                       "arguments": {"url": "https://x"}}])
                yield ("done", {"input": 1, "output": 1})
            else:
                yield ("delta", "图里是只猫")
                yield ("done", {"input": 1, "output": 1})

        def fake_run(name, arguments):
            return ToolResult(call_id="t1", name=name, arguments=arguments, ok=True,
                              output="Content from https://x: …",
                              metadata={"images": [{"url": "https://x/a.png",
                                                    "data_url": "data:image/png;base64,AAA"}]})

        helper = GenerateRetryTests()
        with patch.object(service_mod, "VISION_ENABLED", True), \
             patch.object(service_mod.main, "VISION_SUPPORTED", vision_supported):
            res = helper._run_turn(fake_stream, [
                patch.object(service_mod.TOOL_REGISTRY, "run", fake_run),
            ])
        return calls, res

    def test_images_injected_when_vision_supported(self):
        calls, res = self._run(True)
        second = calls[1]
        img_msgs = [m for m in second
                    if m.get("role") == "user" and isinstance(m.get("content"), list)]
        self.assertTrue(img_msgs, "synthetic user message with image parts")
        self.assertEqual(img_msgs[0]["content"][1]["type"], "image_url")
        # …and it is NOT persisted into the replay history.
        tms = res["metadata"].get("tool_messages") or []
        self.assertFalse(any(isinstance(m.get("content"), list) for m in tms))

    def test_no_injection_when_vision_unsupported(self):
        calls, _ = self._run(False)
        second = calls[1]
        img_msgs = [m for m in second
                    if m.get("role") == "user" and isinstance(m.get("content"), list)]
        self.assertFalse(img_msgs)


class VisionProbeSetup:  # noqa: D101 - marker namespace, not a test
    pass


if __name__ == "__main__":
    unittest.main()
