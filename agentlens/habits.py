"""User-habit-driven cost analysis for AgentLens.

Where metrics.py flags waste on the agent/log side (duplicate reads, low
cache reuse, batchable tool calls), this module looks at how the *user* is
driving the session — unbounded context growth, switching topics without a
/clear, mixing unrelated projects in one session — since those habits are
often the bigger cost lever and are invisible from the agent's own log.

Everything here is a heuristic over what's already in the JSONL log (token
usage per turn, tool-call inputs, system boundary events). There is no
ground truth for "the user changed topics" or "this session mixed two
projects", so thresholds are deliberately conservative and documented next
to their constants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .log_reader import Session, Turn

# Context-window warning bands, mirroring the 25/55/80/90% thresholds Claude
# Code's own context-usage indicator warns at.
CONTEXT_NOTICE_RATIO = 0.25
CONTEXT_WARNING_RATIO = 0.55
CONTEXT_HIGH_RATIO = 0.80
CONTEXT_CRITICAL_RATIO = 0.90

# Conservative default context window (tokens) used when we have no better
# figure for the model in play. Sessions rarely need per-model precision here
# since this is a "did you run hot for a while" signal, not a billing figure.
DEFAULT_CONTEXT_WINDOW = 200_000

# A session must spend at least this many consecutive turns at/above the
# "high" band before we call it a sustained budget overrun rather than a
# brief, self-correcting spike.
CONTEXT_BUDGET_EXCEEDED_TURNS = 5

# Two turns' referenced directories are considered "the same topic" when
# their Jaccard similarity is at least this high; below it, we call it a
# topic shift. 0.2 means "almost no directory overlap".
TOPIC_SHIFT_JACCARD_THRESHOLD = 0.2

# How many undisciplined topic shifts (no /clear or /compact in between)
# before we flag the session for not being split.
SESSION_NOT_SPLIT_MIN_SHIFTS = 2

# How many distinct "foreign" (unrelated to the session's own project)
# directory roots must show up before we call a session project-mixed.
MIXED_PROJECT_MIN_FOREIGN = 2

PATH_INPUT_KEYS = ("file_path", "path", "notebook_path")

_PENALTY_WEIGHTS = {
    "session_not_split": 15,
    "context_budget_exceeded": 20,
    "mixed_project_session": 15,
}
_SEVERITY_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.5}


@dataclass
class ContextPoint:
    turn_index: int
    timestamp: Optional[datetime]
    context_tokens: int
    ratio: float


@dataclass
class TopicZone:
    start_turn: int
    end_turn: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    label: str
    preceded_by_reset: bool  # True for the first zone, or if a /clear|/compact preceded it


@dataclass
class HabitMetrics:
    session_id: str
    duration_seconds: float
    turn_count: int
    context_timeline: list = field(default_factory=list)
    peak_context_tokens: int = 0
    peak_context_ratio: float = 0.0
    topic_zones: list = field(default_factory=list)
    topic_shift_count: int = 0
    undisciplined_shift_count: int = 0
    boundary_count: int = 0
    cache_hit_rate: Optional[float] = None
    foreign_project_keys: list = field(default_factory=list)
    project_mixing_score: float = 0.0
    findings: list = field(default_factory=list)
    habit_score: int = 100


def _normalize_segments(path: str) -> list:
    return [seg for seg in re.split(r"[\\/]+", path) if seg not in ("", ".", "..")]


def _dir_segments(path: str) -> list:
    segs = _normalize_segments(path)
    return segs[:-1] if len(segs) > 1 else segs


def _extract_paths(turn: Turn) -> list:
    paths = []
    for call in turn.tool_calls:
        for key in PATH_INPUT_KEYS:
            val = call.input.get(key)
            if isinstance(val, str) and val.strip():
                paths.append(val)
    return paths


def _turn_topic_dirs(turn: Turn) -> set:
    dirs = set()
    for p in _extract_paths(turn):
        segs = _dir_segments(p)
        if segs:
            dirs.add("/".join(segs))
    return dirs


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _zone_label(dirs: set) -> str:
    if not dirs:
        return "(no file activity)"
    ordered = sorted(dirs, key=len)
    if len(ordered) == 1:
        return ordered[0]
    return f"{ordered[0]} +{len(ordered) - 1} more"


def _context_timeline(session: Session, context_window: int) -> list:
    points = []
    for i, turn in enumerate(session.turns):
        tokens = turn.input_tokens + turn.cache_read_input_tokens + turn.cache_creation_input_tokens
        points.append(
            ContextPoint(
                turn_index=i,
                timestamp=turn.timestamp,
                context_tokens=tokens,
                ratio=tokens / context_window if context_window else 0.0,
            )
        )
    return points


def _detect_context_budget_finding(timeline: list) -> Optional[dict]:
    if not timeline:
        return None
    peak = max(timeline, key=lambda p: p.ratio)
    best_run = 0
    run = 0
    for p in timeline:
        if p.ratio >= CONTEXT_HIGH_RATIO:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0

    if peak.ratio >= CONTEXT_CRITICAL_RATIO:
        severity = "high"
    elif best_run >= CONTEXT_BUDGET_EXCEEDED_TURNS:
        severity = "medium"
    else:
        return None

    return {
        "type": "context_budget_exceeded",
        "detail": (
            f"context usage peaked at {peak.ratio * 100:.0f}% of the ~{DEFAULT_CONTEXT_WINDOW:,} token "
            f"budget (turn {peak.turn_index}) and stayed at/above {CONTEXT_HIGH_RATIO * 100:.0f}% for "
            f"{best_run} consecutive turn(s) — consider a /clear or /compact before this point"
        ),
        "severity": severity,
    }


def _reset_between(session: Session, ts_before: Optional[datetime], ts_after: Optional[datetime]) -> bool:
    if ts_before is None or ts_after is None:
        return False
    return any(b.timestamp is not None and ts_before <= b.timestamp <= ts_after for b in session.boundaries)


def _detect_topic_zones(session: Session) -> tuple:
    """Split the session into topic zones by directory-overlap between
    consecutive file-referencing turns, and record each shift's own
    "disciplined" flag (was there a /clear or /compact between the two
    turns that triggered it).

    Each TopicZone.preceded_by_reset describes *that zone's own* start: the
    first zone is trivially True (there's nothing before it to split from),
    and every later zone inherits the disciplined flag of the shift that
    created it.
    """
    zone_bounds = []  # (start_turn, end_turn, label_dirs)
    shifts = []  # [{"turn_index": int, "disciplined": bool}, ...], one per zone boundary after the first
    zone_start = 0
    zone_dirs_accum: set = set()
    prev_dirs: Optional[set] = None
    prev_index = None

    for i, turn in enumerate(session.turns):
        dirs = _turn_topic_dirs(turn)
        if not dirs:
            continue
        if prev_dirs is not None and _jaccard(prev_dirs, dirs) < TOPIC_SHIFT_JACCARD_THRESHOLD:
            reset = _reset_between(
                session,
                session.turns[prev_index].timestamp if prev_index is not None else None,
                turn.timestamp,
            )
            zone_bounds.append((zone_start, prev_index, zone_dirs_accum))
            shifts.append({"turn_index": i, "disciplined": reset})
            zone_start = i
            zone_dirs_accum = set()
        zone_dirs_accum |= dirs
        prev_dirs = dirs
        prev_index = i

    if session.turns:
        zone_bounds.append((zone_start, len(session.turns) - 1, zone_dirs_accum))

    zones = []
    for idx, (start, end, dirs_accum) in enumerate(zone_bounds):
        preceded_by_reset = True if idx == 0 else shifts[idx - 1]["disciplined"]
        zones.append(
            TopicZone(
                start_turn=start,
                end_turn=end,
                start_time=session.turns[start].timestamp,
                end_time=session.turns[end].timestamp,
                label=_zone_label(dirs_accum),
                preceded_by_reset=preceded_by_reset,
            )
        )

    return zones, shifts


def _detect_session_not_split_finding(shifts: list) -> Optional[dict]:
    undisciplined = [s for s in shifts if not s["disciplined"]]
    if len(undisciplined) < SESSION_NOT_SPLIT_MIN_SHIFTS:
        return None
    severity = "high" if len(undisciplined) >= 5 else ("medium" if len(undisciplined) >= 3 else "low")
    return {
        "type": "session_not_split",
        "detail": (
            f"{len(undisciplined)} topic shift(s) happened without a /clear, /compact, or new session "
            "in between — context kept accumulating across unrelated work instead of being reset"
        ),
        "severity": severity,
    }


def _project_tail_tokens(project_dir: str, depth: int = 2) -> list:
    tokens = [t.lower() for t in project_dir.split("-") if len(t) > 1]
    return tokens[-depth:] if tokens else []


def _detect_project_mixing(session: Session) -> tuple:
    """Returns (foreign_project_keys, mixing_score, finding_or_None).

    A referenced path counts as "foreign" when none of the session's own
    project-directory tail tokens (e.g. the repo folder name) appear among
    the path's own directory segments — i.e. the file plainly lives outside
    the project this session was started in.
    """
    home_tokens = set(_project_tail_tokens(session.project_dir))
    if not home_tokens:
        return [], 0.0, None

    foreign_keys = set()
    total_dirs = set()
    for turn in session.turns:
        for p in _extract_paths(turn):
            segs = [s.lower() for s in _dir_segments(p)]
            if not segs:
                continue
            key = "/".join(segs)
            total_dirs.add(key)
            if not (home_tokens & set(segs)):
                # Use the last couple of segments as a compact label for the foreign root.
                foreign_keys.add("/".join(segs[-2:]))

    mixing_score = len(foreign_keys) / len(total_dirs) if total_dirs else 0.0
    finding = None
    if len(foreign_keys) >= MIXED_PROJECT_MIN_FOREIGN:
        severity = "high" if len(foreign_keys) >= 4 else "medium"
        finding = {
            "type": "mixed_project_session",
            "detail": (
                f"{len(foreign_keys)} directories unrelated to this session's own project were touched "
                f"({', '.join(sorted(foreign_keys)[:5])}) — consider a separate session per project"
            ),
            "severity": severity,
        }
    return sorted(foreign_keys), mixing_score, finding


def _cache_hit_rate(session: Session) -> Optional[float]:
    creation = sum(t.cache_creation_input_tokens for t in session.turns)
    read = sum(t.cache_read_input_tokens for t in session.turns)
    denom = creation + read
    return (read / denom) if denom else None


def _habit_score(findings: list) -> int:
    score = 100.0
    for f in findings:
        weight = _PENALTY_WEIGHTS.get(f["type"])
        if weight is None:
            continue
        score -= weight * _SEVERITY_MULTIPLIER.get(f["severity"], 1.0)
    return max(0, min(100, round(score)))


def detect_habit_waste_patterns(session: Session) -> list:
    """Rule-based detection of cost waste caused by *how the user drives the
    session*, as opposed to metrics.detect_waste_patterns which looks at the
    agent/log side."""
    findings = []

    timeline = _context_timeline(session, DEFAULT_CONTEXT_WINDOW)
    budget_finding = _detect_context_budget_finding(timeline)
    if budget_finding:
        findings.append(budget_finding)

    _zones, shifts = _detect_topic_zones(session)
    split_finding = _detect_session_not_split_finding(shifts)
    if split_finding:
        findings.append(split_finding)

    _foreign_keys, _mixing_score, mixed_finding = _detect_project_mixing(session)
    if mixed_finding:
        findings.append(mixed_finding)

    return findings


def compute_habit_metrics(session: Session, context_window: int = DEFAULT_CONTEXT_WINDOW) -> HabitMetrics:
    """Compute the full set of usage-habit metrics and findings for one session."""
    duration = (session.end - session.start).total_seconds() if session.start and session.end else 0.0
    timeline = _context_timeline(session, context_window)
    peak = max(timeline, key=lambda p: p.ratio) if timeline else None

    zones, shifts = _detect_topic_zones(session)
    undisciplined = [s for s in shifts if not s["disciplined"]]
    foreign_keys, mixing_score, _mixed_finding = _detect_project_mixing(session)

    findings = detect_habit_waste_patterns(session)

    return HabitMetrics(
        session_id=session.session_id,
        duration_seconds=duration,
        turn_count=len(session.turns),
        context_timeline=timeline,
        peak_context_tokens=peak.context_tokens if peak else 0,
        peak_context_ratio=peak.ratio if peak else 0.0,
        topic_zones=zones,
        topic_shift_count=len(shifts),
        undisciplined_shift_count=len(undisciplined),
        boundary_count=len(session.boundaries),
        cache_hit_rate=_cache_hit_rate(session),
        foreign_project_keys=foreign_keys,
        project_mixing_score=mixing_score,
        findings=findings,
        habit_score=_habit_score(findings),
    )
