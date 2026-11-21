"""Mind map shows conversation nodes only: agent steps are never drawn, and
children stored under steps (legacy follow-ups) hang off the owning reply.

Runs the real buildPairTree/_liftMapKids from static/index.html under node.
"""
import json
import os
import shutil
import subprocess

import pytest

HTML = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
node_required = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _method(src: str, name: str) -> str:
    """Extract an object-literal method `name(...) { ... }` by brace matching."""
    start = src.index(f"    {name}(")
    i = src.index("{", start)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1].strip()
    raise ValueError(name)


def _run(tree: dict) -> dict:
    src = open(HTML, encoding="utf-8").read()
    obj = "{" + ",\n".join(_method(src, n) for n in ("_liftMapKids", "buildPairTree")) + "}"
    js = f"""
const o = {obj};
const strip = (n) => ({{id: n.id, replyId: n.replyId, owned: [...(n.ownedIds || [])].sort(),
  files: n.filesChanged, stop: n.stopReason, children: n.children.map(strip)}});
console.log(JSON.stringify(strip(o.buildPairTree({json.dumps(tree)}))));
"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def step(i, children=()):
    return {"id": i, "role": "step", "children": list(children)}


@node_required
def test_steps_are_not_cards_and_branches_hang_off_the_reply():
    # u1 → a1 → s1 → s2 → (legacy follow-up u2 → a2), plus a new-style fork
    # u3 → a3 directly under a1.
    tree = {"id": "u1", "role": "user", "children": [
        {"id": "a1", "role": "assistant", "agent": True, "files_changed": 3,
         "children": [
             step("s1", [step("s2", [
                 {"id": "u2", "role": "user", "children": [
                     {"id": "a2", "role": "assistant", "children": [step("s3")]}]}])]),
             {"id": "u3", "role": "user", "children": [
                 {"id": "a3", "role": "assistant", "stop_reason": "error", "children": []}]},
         ]}]}
    root = _run(tree)
    assert root["replyId"] == "a1" and root["files"] == 3
    assert root["owned"] == ["s1", "s2"]  # selecting a step centers its reply
    kids = sorted(root["children"], key=lambda c: c["id"])
    assert [k["id"] for k in kids] == ["u2", "u3"]  # 2 branches off turn 1
    assert all(k["children"] == [] for k in kids)  # no step cards anywhere
    assert kids[1]["stop"] == "error"
