"""Small read-only tool registry for chatable.

The registry keeps automatic tool use and native provider tool-calling on
the same internal contract.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from web_tools import WebToolResult, grep_content, web_fetch, web_search


ToolHandler = Callable[[dict[str, Any]], WebToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    read_only: bool = True
    result_max_chars: int = 20_000


@dataclass
class ToolResult:
    call_id: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    output: str
    error: str = ""
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_content(self, automatic: bool = False) -> str:
        label = "Automatic tool result" if automatic else "Tool result"
        return (
            f"{label}: {self.name}\n"
            f"Call ID: {self.call_id}\n"
            f"Arguments: {json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)}\n\n"
            f"{self.output}"
        )

    def node_metadata(self, automatic: bool = False) -> dict[str, Any]:
        return {
            "tool": self.name,
            "name": self.name,
            "call_id": self.call_id,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "auto": automatic,
            **self.metadata,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name:
            raise ValueError("tool name is required")
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as e:
            raise ValueError(f"unknown tool: {name}") from e

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> set[str]:
        return set(self._tools)

    def decision_prompt(self) -> str:
        """Build the auto-tool prompt from registered tool specs."""
        lines = [
            "You may use read-only tools before answering.",
            "",
            "Available tools:",
        ]
        for spec in self.list():
            schema = json.dumps(
                spec.input_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            lines.append(f"- {spec.name}: {spec.description} Arguments schema: {schema}")
        lines.extend([
            "",
            "Guidelines:",
            "- Use web_search/web_fetch when the user's question needs current information,",
            "  source lookup, paper content, or URLs.",
            "- Use web_search whenever the question hinges on a technical term, jargon, a",
            "  newly coined name, or a model/method/dataset abbreviation or acronym (e.g.",
            "  'RoPE', 'GRPO', 'Mamba', 'FlashAttention') that is not common knowledge or",
            "  that you cannot define with high confidence. Look it up first instead of",
            "  answering from memory — guessing a definition is worse than searching.",
            "- After fetching a paper/PDF, use grep to find specific sections: methods,",
            "  results, ablation studies, metrics, or any concept the user asks about.",
            "- Typical paper-reading workflow: web_fetch → grep(pattern=\"method\") →",
            "  grep(pattern=\"experiment\") → grep(pattern=\"result\").",
            "- Search and grep can be combined in one round. Do one round of fetch then",
            "  one round of grep for key terms, then answer.",
            "",
            "Do not use tools for pure reasoning or math derivation.",
            "",
            "Respond ONLY as JSON, with no markdown:",
            '{"tool_calls":[{"name":"web_search","arguments":{"query":"...","max_results":5}}]}',
            "",
            "If no tool is needed, respond exactly:",
            '{"tool_calls":[]}',
        ])
        return "\n".join(lines)

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible chat-completions tool specs."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in self.list()
        ]

    def validate_arguments(self, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        schema = spec.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        normalized = dict(arguments)
        for key in required:
            if key not in normalized or normalized[key] in ("", None):
                raise ValueError(f"missing required argument: {key}")
        for key, prop in properties.items():
            if key not in normalized or normalized[key] is None:
                continue
            typ = prop.get("type")
            if typ == "integer":
                try:
                    normalized[key] = int(normalized[key])
                except (TypeError, ValueError) as e:
                    raise ValueError(f"{key} must be an integer") from e
            elif typ == "boolean":
                value = normalized[key]
                if isinstance(value, bool):
                    continue
                if isinstance(value, str) and value.lower() in ("true", "1", "yes"):
                    normalized[key] = True
                elif isinstance(value, str) and value.lower() in ("false", "0", "no"):
                    normalized[key] = False
                else:
                    raise ValueError(f"{key} must be a boolean")
            elif typ == "string" and not isinstance(normalized[key], str):
                normalized[key] = str(normalized[key])
        return normalized

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.get(name)
        call_id = uuid4().hex[:12]
        start = time.monotonic()
        try:
            normalized = self.validate_arguments(spec, arguments)
            raw = spec.handler(normalized)
            output = raw.output
            if len(output) > spec.result_max_chars:
                output = output[:spec.result_max_chars] + (
                    f"\n\n[Tool output truncated at {spec.result_max_chars} chars.]"
                )
            return ToolResult(
                call_id=call_id,
                name=spec.name,
                arguments=normalized,
                ok=raw.ok,
                output=output,
                error="" if raw.ok else raw.output,
                duration_ms=int((time.monotonic() - start) * 1000),
                metadata=raw.metadata,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call_id,
                name=spec.name,
                arguments=arguments if isinstance(arguments, dict) else {},
                ok=False,
                output=f"{spec.name} error: {exc}",
                error=str(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )


def _web_search_handler(args: dict[str, Any]) -> WebToolResult:
    return web_search(
        str(args["query"]).strip(),
        max_results=int(args.get("max_results") or 8),
        language=str(args.get("language") or ""),
    )


def _web_fetch_handler(args: dict[str, Any]) -> WebToolResult:
    return web_fetch(
        str(args["url"]).strip(),
        max_length=int(args.get("max_length") or 8000),
        no_cache=bool(args.get("no_cache") or False),
    )


def _grep_handler(args: dict[str, Any]) -> WebToolResult:
    return grep_content(
        str(args["pattern"]).strip(),
        url=str(args.get("url") or "").strip(),
        max_lines=int(args.get("max_lines") or 30),
        ignore_case=bool(args.get("ignore_case", True)),
    )


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="web_search",
        description=(
            "Search the web for current sources or URLs. Use this to look up any "
            "technical term, jargon, newly coined name, or model/method/dataset "
            "abbreviation or acronym you are not fully certain about, instead of "
            "answering from memory."
        ),
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "1-20 results."},
                "language": {"type": "string", "description": "Region code, e.g. cn-zh for Chinese."},
            },
        },
        handler=_web_search_handler,
        result_max_chars=16_000,
    ))
    registry.register(ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a URL (HTML/PDF), local PDF file, or arXiv paper as readable "
            "text. The WHOLE document is indexed for grep even when the returned "
            "text is truncated for length — to read later sections of a long PDF, "
            "use grep instead of fetching the same URL again."
        ),
        input_schema={
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL or local file path."},
                "max_length": {"type": "integer", "description": "Chars of text to return, 500-500000 (default 8000). The full document is always searchable via grep regardless of this."},
                "no_cache": {"type": "boolean", "description": "Bypass in-memory cache."},
            },
        },
        handler=_web_fetch_handler,
        result_max_chars=500_000,
    ))
    registry.register(ToolSpec(
        name="grep",
        description="Search within previously fetched content for a regex pattern. Use after web_fetch to find specific sections.",
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "url": {"type": "string", "description": "Optional: restrict search to a specific fetched URL."},
                "max_lines": {"type": "integer", "description": "Max matching lines to show (default 30)."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default true)."},
            },
        },
        handler=_grep_handler,
        result_max_chars=16_000,
    ))
    return registry


TOOL_REGISTRY = default_registry()
