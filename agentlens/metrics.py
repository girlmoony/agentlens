"""Cost calculation and waste-pattern detection for AgentLens."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .log_reader import Session, Turn

PRICING_FILE = Path(__file__).parent / "pricing.yaml"


@dataclass
class Pricing:
    models: dict
    cache_multipliers: dict
    unknown_model_fallback: str
    introductory_pricing: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = PRICING_FILE) -> "Pricing":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            models=data["models"],
            cache_multipliers=data["cache_multipliers"],
            unknown_model_fallback=data["unknown_model_fallback"],
            introductory_pricing=data.get("introductory_pricing", {}),
        )

    def rate_for(self, model: str, as_of: Optional[date] = None) -> dict:
        intro = self.introductory_pricing.get(model)
        if intro is not None:
            if as_of is None:
                as_of = datetime.now(timezone.utc).date()
            valid_until = date.fromisoformat(intro["valid_until"])
            if as_of < valid_until:
                return {"input": intro["input"], "output": intro["output"]}
        return self.models.get(model, self.models[self.unknown_model_fallback])


def turn_cost(turn: Turn, pricing: Pricing) -> dict:
    """USD cost breakdown for one assistant turn."""
    as_of = turn.timestamp.date() if turn.timestamp else None
    rate = pricing.rate_for(turn.model, as_of)
    input_rate = rate["input"] / 1_000_000
    output_rate = rate["output"] / 1_000_000
    write_5m = pricing.cache_multipliers["write_5m"]
    write_1h = pricing.cache_multipliers["write_1h"]
    read_mult = pricing.cache_multipliers["read"]

    input_cost = turn.input_tokens * input_rate
    output_cost = turn.output_tokens * output_rate
    # Cache writes are priced by their actual TTL — a 1h write costs 2x base
    # input vs. 1.25x for a 5m write, so the two must be summed separately
    # rather than assuming every write is 5m.
    cache_write_cost = (
        turn.cache_creation_5m_input_tokens * input_rate * write_5m
        + turn.cache_creation_1h_input_tokens * input_rate * write_1h
    )
    cache_read_cost = turn.cache_read_input_tokens * input_rate * read_mult

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost,
    }


@dataclass
class SessionSummary:
    session_id: str
    project_dir: str
    start: Optional[object]
    end: Optional[object]
    turn_count: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_cost: float
    models_used: list
    findings: list = field(default_factory=list)


def summarize_session(session: Session, pricing: Pricing) -> SessionSummary:
    input_tokens = sum(t.input_tokens for t in session.turns)
    output_tokens = sum(t.output_tokens for t in session.turns)
    cache_creation = sum(t.cache_creation_input_tokens for t in session.turns)
    cache_read = sum(t.cache_read_input_tokens for t in session.turns)
    total_cost = sum(turn_cost(t, pricing)["total_cost"] for t in session.turns)
    tool_call_count = sum(len(t.tool_calls) for t in session.turns)
    models_used = sorted({t.model for t in session.turns})

    return SessionSummary(
        session_id=session.session_id,
        project_dir=session.project_dir,
        start=session.start,
        end=session.end,
        turn_count=len(session.turns),
        tool_call_count=tool_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        total_cost=total_cost,
        models_used=models_used,
        findings=detect_waste_patterns(session),
    )


# ---------------------------------------------------------------- Waste patterns ----

DUPLICATE_READ_TOOLS = {"Read"}
RAPID_CALL_GAP_SECONDS = 3.0
RAPID_CALL_MIN_STREAK = 3
CACHE_WRITE_TO_READ_RATIO_THRESHOLD = 1.5


def detect_waste_patterns(session: Session) -> list:
    """Rule-based detection of avoidable token/cost waste in one session."""
    findings = []
    findings.extend(_detect_duplicate_reads(session))
    findings.extend(_detect_low_cache_reuse(session))
    findings.extend(_detect_batchable_tool_calls(session))
    return findings


def _detect_duplicate_reads(session: Session) -> list:
    read_paths = Counter()
    for turn in session.turns:
        for call in turn.tool_calls:
            if call.name in DUPLICATE_READ_TOOLS:
                path = call.input.get("file_path")
                if path:
                    read_paths[path] += 1

    findings = []
    for path, count in read_paths.items():
        if count > 1:
            findings.append(
                {
                    "type": "duplicate_read",
                    "detail": f"{path} was read {count} times in this session",
                    "severity": "low" if count <= 2 else "medium",
                }
            )
    return findings


def _detect_low_cache_reuse(session: Session) -> list:
    cache_creation = sum(t.cache_creation_input_tokens for t in session.turns)
    cache_read = sum(t.cache_read_input_tokens for t in session.turns)
    if cache_creation == 0:
        return []
    ratio = cache_creation / max(cache_read, 1)
    if ratio >= CACHE_WRITE_TO_READ_RATIO_THRESHOLD:
        return [
            {
                "type": "low_cache_reuse",
                "detail": (
                    f"cache_creation_input_tokens ({cache_creation:,}) is {ratio:.1f}x "
                    f"cache_read_input_tokens ({cache_read:,}) — context is being rebuilt "
                    "more often than it's reused"
                ),
                "severity": "medium",
            }
        ]
    return []


def _detect_batchable_tool_calls(session: Session) -> list:
    """Flag runs of consecutive turns that each fire exactly one tool call in rapid
    succession — a signal those calls could have been batched into one turn."""
    findings = []
    streak = 0
    streak_start = None
    prev_timestamp = None

    def close_streak(end_index):
        nonlocal streak, streak_start
        if streak >= RAPID_CALL_MIN_STREAK:
            findings.append(
                {
                    "type": "batchable_tool_calls",
                    "detail": (
                        f"{streak} consecutive single-tool-call turns within "
                        f"{RAPID_CALL_GAP_SECONDS:.0f}s of each other (turns {streak_start}-{end_index}) "
                        "— independent calls like this can usually be batched into one turn"
                    ),
                    "severity": "low",
                }
            )
        streak = 0
        streak_start = None

    for i, turn in enumerate(session.turns):
        is_single_tool_turn = len(turn.tool_calls) == 1
        gap_ok = (
            prev_timestamp is not None
            and turn.timestamp is not None
            and (turn.timestamp - prev_timestamp).total_seconds() <= RAPID_CALL_GAP_SECONDS
        )
        if is_single_tool_turn and (streak == 0 or gap_ok):
            if streak == 0:
                streak_start = i
            streak += 1
        else:
            close_streak(i - 1)
            if is_single_tool_turn:
                streak = 1
                streak_start = i
        prev_timestamp = turn.timestamp if turn.timestamp else prev_timestamp

    close_streak(len(session.turns) - 1)
    return findings
