"""Node dataclass — the single unit of the conversation tree."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional
from uuid import uuid4
import time

Role = Literal["user", "assistant", "summary", "system_note"]


@dataclass
class Node:
    id: str
    role: Role
    content: str                      # assistant content grows during stream
    parent_id: Optional[str]          # None only for the synthetic ROOT node
    created_at: float                 # time.time() seconds
    usage: Optional[dict] = None      # assistant-only; set after stream ends
    pinned: bool = False              # always include in context preview/messages
    token_count: int = 0              # rough local estimate, not provider billing
    title: str = ""
    summary: str = ""
    metadata: Optional[dict] = None

    @staticmethod
    def new(role: Role, content: str, parent_id: Optional[str]) -> "Node":
        return Node(
            id=uuid4().hex[:12],
            role=role,
            content=content,
            parent_id=parent_id,
            created_at=time.time(),
            token_count=estimate_tokens(content),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bookmark:
    """A saved URL with fetched content and an auto-generated summary."""
    id: str
    url: str
    title: str = ""
    summary: str = ""
    content: str = ""            # full fetched text (possibly truncated)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Optional[dict] = None

    @staticmethod
    def new(url: str) -> "Bookmark":
        now = time.time()
        return Bookmark(
            id=uuid4().hex[:12],
            url=url,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Cheap context-budget estimate good enough for UI previews."""
    if not text:
        return 0
    return max(1, len(text) // 4)
