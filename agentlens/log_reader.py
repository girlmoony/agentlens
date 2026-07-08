"""Parse Claude Code local session logs (~/.claude/projects/**/*.jsonl).

Each JSONL line is one *content block* of an assistant message, not one full
message — a single API response with a thinking block, a text block, and two
tool_use blocks appears as four lines sharing the same message.id and an
identical (final) usage object. Lines for one message are contiguous, so we
accumulate blocks until message.id changes and count usage exactly once per
message id — summing per-line would inflate token/cost totals several-fold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class ToolCall:
    timestamp: Optional[datetime]
    name: str
    input: dict


@dataclass
class Turn:
    message_id: str
    timestamp: Optional[datetime]
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    # Split of cache_creation_input_tokens by TTL, since 5m and 1h writes are
    # priced differently. Logs that predate this breakdown report zeros here;
    # callers should treat that case as a full 5m write (see parse_session).
    cache_creation_5m_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0
    tool_calls: list = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    project_dir: str
    file_path: Path
    turns: list = field(default_factory=list)

    @property
    def start(self) -> Optional[datetime]:
        timestamps = [t.timestamp for t in self.turns if t.timestamp]
        return min(timestamps) if timestamps else None

    @property
    def end(self) -> Optional[datetime]:
        timestamps = [t.timestamp for t in self.turns if t.timestamp]
        return max(timestamps) if timestamps else None


def _parse_timestamp(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def find_session_files(
    projects_dir: Path = DEFAULT_PROJECTS_DIR, since: Optional[datetime] = None
) -> Iterator[Path]:
    """Yield every *.jsonl session log under projects_dir, optionally filtered by mtime."""
    if not projects_dir.exists():
        return
    for project_path in sorted(projects_dir.iterdir()):
        if not project_path.is_dir():
            continue
        for jsonl_file in sorted(project_path.glob("*.jsonl")):
            if since is not None:
                mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=timezone.utc)
                if mtime < since:
                    continue
            yield jsonl_file


def parse_session(file_path: Path) -> Session:
    """Parse one session JSONL file into a Session with one Turn per assistant message."""
    session = Session(session_id=file_path.stem, project_dir=file_path.parent.name, file_path=file_path)

    current_id: Optional[str] = None
    current_usage: dict = {}
    current_model = "unknown"
    current_timestamp: Optional[datetime] = None
    current_tool_calls: list = []

    def flush():
        if current_id is None:
            return
        cache_creation_total = current_usage.get("cache_creation_input_tokens", 0) or 0
        cache_creation = current_usage.get("cache_creation") or {}
        cache_5m = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0
        cache_1h = cache_creation.get("ephemeral_1h_input_tokens", 0) or 0
        if cache_5m == 0 and cache_1h == 0 and cache_creation_total > 0:
            # Older logs (or any turn missing the per-TTL breakdown) — assume
            # the default 5m TTL rather than dropping the cost entirely.
            cache_5m = cache_creation_total
        session.turns.append(
            Turn(
                message_id=current_id,
                timestamp=current_timestamp,
                model=current_model,
                input_tokens=current_usage.get("input_tokens", 0) or 0,
                output_tokens=current_usage.get("output_tokens", 0) or 0,
                cache_creation_input_tokens=cache_creation_total,
                cache_read_input_tokens=current_usage.get("cache_read_input_tokens", 0) or 0,
                cache_creation_5m_input_tokens=cache_5m,
                cache_creation_1h_input_tokens=cache_1h,
                tool_calls=list(current_tool_calls),
            )
        )

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated/corrupted trailing line

            if event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            msg_id = message.get("id")
            if not msg_id:
                continue

            if msg_id != current_id:
                flush()
                current_id = msg_id
                current_usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
                current_model = message.get("model", "unknown")
                current_timestamp = _parse_timestamp(event.get("timestamp", ""))
                current_tool_calls = []

            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        current_tool_calls.append(
                            ToolCall(
                                timestamp=_parse_timestamp(event.get("timestamp", "")),
                                name=block.get("name", "unknown"),
                                input=block.get("input") if isinstance(block.get("input"), dict) else {},
                            )
                        )

        flush()

    return session
