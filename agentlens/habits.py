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

Two pieces of external research inform (but do not mechanically dictate) the
context-growth side of this module:

- Chroma, "Context Rot: How Increasing Input Tokens Impacts LLM Performance"
  (Kelly Hong, Anton Troynikov, Jeff Huber; Chroma, July 2025).
  https://www.trychroma.com/research/context-rot
  Tested 18 frontier models (Claude Opus 4/Sonnet 4/Sonnet 3.7/Sonnet
  3.5/Haiku 3.5, GPT, Gemini, and Qwen3 families). The report explicitly
  does *not* give a universal token count or percentage-of-window
  threshold where degradation begins — it states performance grows
  "increasingly unreliable" in a model- and task-dependent, non-uniform
  way as input length grows, even on simple retrieval/replication tasks.
  Two findings we *do* build on directly: (a) a single distractor already
  measurably reduces accuracy vs. a distractor-free baseline, and adding
  more distractors compounds the loss further; (b) in their LongMemEval
  comparison, prompts trimmed to only the relevant content scored
  significantly higher than the full, unfiltered prompt — i.e. irrelevant
  content in context degrades accuracy independent of raw length.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts"
  (Stanford; arXiv 2023, TACL 2024). https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf
  Found a U-shaped position-sensitivity curve in multi-document QA: models
  use information at the very start or end of their context far more
  reliably than information placed in the middle, even when everything
  fits comfortably inside the stated context window. This module does not
  attempt to detect *where* in a session's context a fact lives (that
  isn't recoverable from the JSONL log), but it's the reason
  `long_context_quality_risk` below treats "the context is large" as a
  standalone risk signal rather than only a cost one.

Neither paper gives a number this module could cite as "the" degradation
threshold, so `QUALITY_RISK_CONTEXT_RATIO` below is a deliberately
conservative, explicitly-labeled heuristic — not a reproduction of a
published figure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from .log_reader import Session, Turn

# Context-window warning bands, mirroring the 25/55/80/90% thresholds Claude
# Code's own context-usage indicator warns at. These are cost/capacity bands
# (how close is this session to its token budget), not an accuracy-degradation
# curve — see the module docstring for why no such curve is encoded here.
CONTEXT_NOTICE_RATIO = 0.25
CONTEXT_WARNING_RATIO = 0.55
CONTEXT_HIGH_RATIO = 0.80
CONTEXT_CRITICAL_RATIO = 0.90

# Fallback context window (tokens) used only when no per-model resolver is
# supplied (e.g. a standalone call in tests). Real callers should pass
# metrics.Pricing.context_window_for so each turn is measured against the
# window of the model that actually produced it.
DEFAULT_CONTEXT_WINDOW = 200_000

# A session must spend at least this many consecutive turns at/above the
# "high" band before we call it a sustained budget overrun rather than a
# brief, self-correcting spike.
CONTEXT_BUDGET_EXCEEDED_TURNS = 5

# Neither Chroma's Context Rot report nor Liu et al.'s "Lost in the Middle"
# gives a universal token count or ratio at which quality degradation
# begins — both explicitly frame it as model- and task-dependent. This ratio
# is therefore *not* derived from either paper; it is a conservative,
# deliberately-labeled halfway point, picked because Chroma's finding that
# degradation can appear well before a window is full (i.e. before the
# cost-oriented CONTEXT_HIGH_RATIO/CONTEXT_CRITICAL_RATIO bands above) means
# waiting for those bands would miss the risk this finding is meant to
# surface. Treat `long_context_quality_risk` as a qualitative nudge, not a
# measured cutoff.
QUALITY_RISK_CONTEXT_RATIO = 0.5

# Which of the *other* findings we treat as the "distractor/irrelevant
# content" signal from Chroma's report — a session with one of these AND an
# elevated context ratio is exactly the "long context plus noise" combination
# their distractor experiments and LongMemEval comparison both flagged as
# worse than length alone.
QUALITY_RISK_NOISE_FINDING_TYPES = {"duplicate_read", "low_cache_reuse", "mixed_project_session"}

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
    # Lower weight than the cost-grounded findings above: this one rests on
    # a qualitative research finding rather than a measured cost figure.
    "long_context_quality_risk": 10,
}
_SEVERITY_MULTIPLIER = {"low": 0.6, "medium": 1.0, "high": 1.5}

ContextWindowResolver = Callable[[str], int]


def _default_context_window_resolver(_model: str) -> int:
    return DEFAULT_CONTEXT_WINDOW


@dataclass
class ContextPoint:
    turn_index: int
    timestamp: Optional[datetime]
    context_tokens: int
    context_window: int
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


def _context_timeline(session: Session, resolver: ContextWindowResolver) -> list:
    points = []
    for i, turn in enumerate(session.turns):
        tokens = turn.input_tokens + turn.cache_read_input_tokens + turn.cache_creation_input_tokens
        window = resolver(turn.model) or DEFAULT_CONTEXT_WINDOW
        points.append(
            ContextPoint(
                turn_index=i,
                timestamp=turn.timestamp,
                context_tokens=tokens,
                context_window=window,
                ratio=tokens / window,
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
            f"context usage peaked at {peak.ratio * 100:.0f}% of the ~{peak.context_window:,} token "
            f"budget (turn {peak.turn_index}) and stayed at/above {CONTEXT_HIGH_RATIO * 100:.0f}% for "
            f"{best_run} consecutive turn(s) — consider a /clear or /compact before this point"
        ),
        "severity": severity,
    }


def detect_long_context_quality_risk(timeline: list, other_finding_types: set) -> Optional[dict]:
    """Cross-cutting signal combining this session's context size with whether
    it also triggered a "redundant/irrelevant content" finding elsewhere
    (agent-side `duplicate_read`/`low_cache_reuse`, or this module's own
    `mixed_project_session`) — the specific combination Chroma's distractor
    experiments and LongMemEval comparison found compounds degradation beyond
    what context length alone causes. See the module docstring for citations
    and why `QUALITY_RISK_CONTEXT_RATIO` is a conservative heuristic, not a
    measured threshold."""
    if not timeline:
        return None
    peak = max(timeline, key=lambda p: p.ratio)
    if peak.ratio < QUALITY_RISK_CONTEXT_RATIO:
        return None
    noise_types = QUALITY_RISK_NOISE_FINDING_TYPES & other_finding_types
    if not noise_types:
        return None

    severity = "medium" if peak.ratio >= CONTEXT_HIGH_RATIO else "low"
    return {
        "type": "long_context_quality_risk",
        "detail": (
            f"context usage reached {peak.ratio * 100:.0f}% of the model's window (turn {peak.turn_index}) "
            f"in a session that also triggered {', '.join(sorted(noise_types))} — Chroma's \"Context Rot\" "
            "study (2025, trychroma.com/research/context-rot) found that distractor/irrelevant content "
            "compounds accuracy loss beyond what length alone causes. No published research gives a "
            "universal token threshold for when this starts, so treat this as a conservative, qualitative "
            "signal rather than a precise cutoff"
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
                f"({', '.join(sorted(foreign_keys)[:5])}) — consider a separate session per project. Beyond "
                "the extra tokens, Chroma's \"Context Rot\" study found irrelevant context measurably lowers "
                "response accuracy too, citing their focused-vs-full-prompt (LongMemEval) comparison"
            ),
            "severity": severity,
        }
    return sorted(foreign_keys), mixing_score, finding


def _cache_hit_rate(session: Session) -> Optional[float]:
    creation = sum(t.cache_creation_input_tokens for t in session.turns)
    read = sum(t.cache_read_input_tokens for t in session.turns)
    denom = creation + read
    return (read / denom) if denom else None


def habit_score(findings: list) -> int:
    """0-100 score derived from habit-related findings (see _PENALTY_WEIGHTS).
    Public so callers (e.g. metrics.summarize_session) can recompute it after
    merging in findings — like long_context_quality_risk — that depend on
    context beyond a single session's own habit findings."""
    score = 100.0
    for f in findings:
        weight = _PENALTY_WEIGHTS.get(f["type"])
        if weight is None:
            continue
        score -= weight * _SEVERITY_MULTIPLIER.get(f["severity"], 1.0)
    return max(0, min(100, round(score)))


def detect_habit_waste_patterns(session: Session, context_window_resolver: Optional[ContextWindowResolver] = None) -> list:
    """Rule-based detection of cost waste caused by *how the user drives the
    session*, as opposed to metrics.detect_waste_patterns which looks at the
    agent/log side. Does not include long_context_quality_risk, which needs
    visibility into agent-side findings too — see
    metrics.summarize_session / detect_long_context_quality_risk."""
    resolver = context_window_resolver or _default_context_window_resolver
    findings = []

    timeline = _context_timeline(session, resolver)
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


def compute_habit_metrics(
    session: Session, context_window_resolver: Optional[ContextWindowResolver] = None
) -> HabitMetrics:
    """Compute the full set of usage-habit metrics and findings for one
    session. `context_window_resolver` maps a model ID to its context window
    in tokens (see metrics.Pricing.context_window_for) — pass it whenever
    real per-model sizing matters; it defaults to a flat 200K fallback."""
    resolver = context_window_resolver or _default_context_window_resolver
    duration = (session.end - session.start).total_seconds() if session.start and session.end else 0.0
    timeline = _context_timeline(session, resolver)
    peak = max(timeline, key=lambda p: p.ratio) if timeline else None

    zones, shifts = _detect_topic_zones(session)
    undisciplined = [s for s in shifts if not s["disciplined"]]
    foreign_keys, mixing_score, mixed_finding = _detect_project_mixing(session)

    findings = []
    budget_finding = _detect_context_budget_finding(timeline)
    if budget_finding:
        findings.append(budget_finding)
    split_finding = _detect_session_not_split_finding(shifts)
    if split_finding:
        findings.append(split_finding)
    if mixed_finding:
        findings.append(mixed_finding)

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
        habit_score=habit_score(findings),
    )
