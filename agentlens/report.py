"""Self-contained static HTML report generation for AgentLens (no JS, no external assets)."""
from __future__ import annotations

import html
from collections import defaultdict
from datetime import date
from pathlib import Path

from . import habits
from .metrics import SessionSummary

CHART_HEIGHT = 160
CHART_BAR_WIDTH = 28
CHART_BAR_GAP = 14

# How many sessions to draw small-multiple charts for, in the context-growth
# and topic-timeline sections — showing every session would make the report
# unusably long, so each shows only the sessions most worth looking at.
CONTEXT_TIMELINE_MAX_SESSIONS = 5
TOPIC_TIMELINE_MAX_SESSIONS = 8

TOPIC_ZONE_COLORS = [
    "#6f5bd6", "#2a9d8f", "#e76f51", "#457b9d", "#e9c46a", "#8ac926", "#ef476f", "#118ab2",
]


def _daily_cost(summaries: list) -> dict:
    totals = defaultdict(float)
    for s in summaries:
        if s.start is None:
            continue
        day = s.start.date()
        totals[day] += s.total_cost
    return dict(sorted(totals.items()))


def _bar_chart_svg(daily: dict) -> str:
    if not daily:
        return "<p>No dated sessions to chart.</p>"
    max_cost = max(daily.values()) or 1.0
    width = max(len(daily) * (CHART_BAR_WIDTH + CHART_BAR_GAP) + CHART_BAR_GAP, 200)
    bars = []
    labels = []
    for i, (day, cost) in enumerate(daily.items()):
        bar_height = (cost / max_cost) * (CHART_HEIGHT - 30)
        x = CHART_BAR_GAP + i * (CHART_BAR_WIDTH + CHART_BAR_GAP)
        y = CHART_HEIGHT - 20 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{CHART_BAR_WIDTH}" height="{bar_height:.1f}" '
            f'rx="3" class="bar"><title>{day.isoformat()}: ${cost:.4f}</title></rect>'
        )
        labels.append(
            f'<text x="{x + CHART_BAR_WIDTH / 2}" y="{CHART_HEIGHT - 4}" '
            f'class="bar-label" text-anchor="middle">{day.strftime("%m/%d")}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {CHART_HEIGHT}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Daily cost chart">'
        + "".join(bars)
        + "".join(labels)
        + "</svg>"
    )


# ---------------------------------------------------------------- Usage-habit visuals ----


def _sessions_with_habits(summaries: list) -> list:
    return [s for s in summaries if getattr(s, "habit_metrics", None) is not None]


def _habit_score_summary_html(summaries: list) -> str:
    with_habits = _sessions_with_habits(summaries)
    if not with_habits:
        return "<p>No sessions to score.</p>"

    avg_score = sum(s.habit_metrics.habit_score for s in with_habits) / len(with_habits)
    counts = defaultdict(int)
    for s in with_habits:
        for f in s.habit_metrics.findings:
            counts[f["type"]] += 1

    labels = {
        "session_not_split": "session not split",
        "context_budget_exceeded": "context budget exceeded",
        "mixed_project_session": "mixed project session",
    }
    chips = "".join(
        f'<div class="stat"><div class="value">{counts.get(key, 0)}</div>'
        f'<div class="label">{html.escape(label)}</div></div>'
        for key, label in labels.items()
    )
    return (
        '<div class="stat-row">'
        f'<div class="stat"><div class="value">{avg_score:.0f}/100</div>'
        '<div class="label">avg. habit score</div></div>'
        f"{chips}"
        "</div>"
    )


def _context_timeline_svg(summaries: list) -> str:
    candidates = [
        s for s in _sessions_with_habits(summaries)
        if s.habit_metrics.context_timeline and s.habit_metrics.turn_count > 0
    ]
    if not candidates:
        return "<p>No sessions with context-usage data to chart.</p>"

    top = sorted(candidates, key=lambda s: s.habit_metrics.peak_context_ratio, reverse=True)
    top = top[:CONTEXT_TIMELINE_MAX_SESSIONS]

    width, height = 320, 150
    pad_left, pad_bottom = 34, 20
    plot_w = width - pad_left - 10
    plot_h = height - pad_bottom - 10

    def y_of(ratio: float) -> float:
        capped = min(ratio, 1.1)
        return 10 + plot_h * (1 - capped / 1.1)

    cards = []
    for s in top:
        hm = s.habit_metrics
        points = hm.context_timeline
        n = max(len(points) - 1, 1)
        polyline_pts = " ".join(
            f"{pad_left + plot_w * (p.turn_index / n):.1f},{y_of(p.ratio):.1f}" for p in points
        )
        danger_y = y_of(habits.CONTEXT_HIGH_RATIO)
        band_pieces = []
        for ratio, css_class in (
            (habits.CONTEXT_WARNING_RATIO, "budget-line-warning"),
            (habits.CONTEXT_HIGH_RATIO, "budget-line-high"),
            (habits.CONTEXT_CRITICAL_RATIO, "budget-line-critical"),
        ):
            ly = y_of(ratio)
            band_pieces.append(
                f'<line x1="{pad_left}" y1="{ly:.1f}" x2="{width - 10}" y2="{ly:.1f}" class="{css_class}"/>'
            )
        svg = (
            f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" aria-label="Context growth timeline for session {html.escape(s.session_id[:8])}">'
            f'<rect x="{pad_left}" y="10" width="{plot_w}" height="{danger_y - 10:.1f}" class="danger-zone"/>'
            + "".join(band_pieces)
            + f'<polyline points="{polyline_pts}" class="context-line"/>'
            f'<text x="4" y="14" class="axis-label">100%</text>'
            f'<text x="4" y="{10 + plot_h + 4:.0f}" class="axis-label">0%</text>'
            f'<text x="{pad_left}" y="{height - 4}" class="bar-label">turn 0</text>'
            f'<text x="{width - 40}" y="{height - 4}" class="bar-label">turn {len(points) - 1}</text>'
            "</svg>"
        )
        cards.append(
            f'<div class="chart-card"><div class="chart-card-title">'
            f'{html.escape(s.session_id[:8])} — peak {hm.peak_context_ratio * 100:.0f}%</div>{svg}</div>'
        )

    return '<div class="chart-grid">' + "".join(cards) + "</div>"


def _topic_timeline_svg(summaries: list) -> str:
    candidates = [
        s for s in _sessions_with_habits(summaries)
        if s.habit_metrics.turn_count > 0 and len(s.habit_metrics.topic_zones) > 1
    ]
    if not candidates:
        return "<p>No topic switches detected — nothing to split.</p>"

    top = sorted(
        candidates, key=lambda s: s.habit_metrics.undisciplined_shift_count, reverse=True
    )[:TOPIC_TIMELINE_MAX_SESSIONS]

    width = 640
    row_h = 34
    bar_h = 16
    label_w = 90
    plot_w = width - label_w - 10
    height = row_h * len(top) + 30

    rows = []
    for row_i, s in enumerate(top):
        hm = s.habit_metrics
        n_turns = hm.turn_count
        y = 10 + row_i * row_h
        segs = []
        markers = []
        for zone_i, zone in enumerate(hm.topic_zones):
            x = label_w + plot_w * (zone.start_turn / n_turns)
            w = plot_w * ((zone.end_turn - zone.start_turn + 1) / n_turns)
            color = TOPIC_ZONE_COLORS[zone_i % len(TOPIC_ZONE_COLORS)]
            segs.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="{bar_h}" fill="{color}" rx="2">'
                f"<title>{html.escape(zone.label)} (turns {zone.start_turn}-{zone.end_turn})</title></rect>"
            )
            if zone_i > 0:
                marker_cls = "clear-marker-ok" if zone.preceded_by_reset else "clear-marker-warn"
                marker_glyph = "✓" if zone.preceded_by_reset else "!"
                tooltip = (
                    "context was reset before this topic"
                    if zone.preceded_by_reset
                    else "no /clear or /compact here — consider splitting the session"
                )
                markers.append(
                    f'<text x="{x:.1f}" y="{y - 2}" class="{marker_cls}" text-anchor="middle">'
                    f"{marker_glyph}<title>{html.escape(tooltip)}</title></text>"
                )
        rows.append(
            f'<text x="0" y="{y + bar_h - 4}" class="bar-label">{html.escape(s.session_id[:8])}</text>'
            + "".join(segs)
            + "".join(markers)
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Topic timeline across sessions">'
        + "".join(rows)
        + "</svg>"
    )
    caption = (
        "<p class=\"chart-caption\">Each row is one session's turns, colored by topic zone. "
        '<span class="clear-marker-warn">!</span> marks a topic switch with no /clear or /compact '
        'before it; <span class="clear-marker-ok">✓</span> marks one that was properly reset.</p>'
    )
    return svg + caption


def _scatter_svg(points: list, x_label: str, y_label: str, aria_label: str) -> str:
    if not points:
        return "<p>Not enough data to chart.</p>"

    width, height = 420, 260
    pad_left, pad_bottom, pad_top, pad_right = 46, 30, 14, 14
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    def x_of(x):
        return pad_left + plot_w * (x - x_min) / (x_max - x_min)

    def y_of(y):
        return pad_top + plot_h * (1 - (y - y_min) / (y_max - y_min))

    dots = []
    for x, y, tooltip in points:
        dots.append(
            f'<circle cx="{x_of(x):.1f}" cy="{y_of(y):.1f}" r="4" class="scatter-dot">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )

    axes = (
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis-line"/>'
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
        f'y2="{height - pad_bottom}" class="axis-line"/>'
    )
    labels = (
        f'<text x="{pad_left}" y="{height - 6}" class="axis-label">{html.escape(x_label)}</text>'
        f'<text x="4" y="{pad_top + 8}" class="axis-label">{html.escape(y_label)}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{html.escape(aria_label)}">'
        + axes + labels + "".join(dots) + "</svg>"
    )


def _cache_hit_vs_length_svg(summaries: list) -> str:
    points = []
    for s in _sessions_with_habits(summaries):
        hm = s.habit_metrics
        if hm.cache_hit_rate is None or hm.turn_count == 0:
            continue
        points.append(
            (hm.turn_count, hm.cache_hit_rate * 100, f"{s.session_id[:8]}: {hm.turn_count} turns, "
             f"{hm.cache_hit_rate * 100:.0f}% cache hit rate")
        )
    return _scatter_svg(points, "turns per session", "cache hit rate (%)", "Cache hit rate vs session length")


def _cost_per_call_vs_peak_context_svg(summaries: list) -> str:
    points = []
    for s in _sessions_with_habits(summaries):
        hm = s.habit_metrics
        if s.turn_count == 0:
            continue
        cost_per_call = s.total_cost / s.turn_count
        points.append(
            (hm.peak_context_tokens, cost_per_call, f"{s.session_id[:8]}: peak "
             f"{hm.peak_context_tokens:,} tokens, ${cost_per_call:.4f}/call")
        )
    return _scatter_svg(
        points, "peak context size (tokens)", "cost per call ($)", "Cost per call vs peak context size"
    )


def _findings_html(summaries: list) -> str:
    rows = []
    for s in summaries:
        if not s.findings:
            continue
        for f in s.findings:
            rows.append(
                f"<tr><td>{html.escape(s.session_id[:8])}</td>"
                f'<td class="sev-{f["severity"]}">{html.escape(f["severity"])}</td>'
                f'<td>{html.escape(f["type"])}</td>'
                f'<td>{html.escape(f["detail"])}</td></tr>'
            )
    if not rows:
        return "<p>No waste patterns detected in this period.</p>"
    return (
        "<table><thead><tr><th>Session</th><th>Severity</th><th>Pattern</th><th>Detail</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _sessions_table_html(summaries: list) -> str:
    rows = []
    for s in sorted(summaries, key=lambda s: s.total_cost, reverse=True):
        started = s.start.strftime("%Y-%m-%d %H:%M") if s.start else "—"
        models = ", ".join(s.models_used) or "—"
        rows.append(
            f"<tr><td>{html.escape(s.session_id[:8])}</td>"
            f"<td>{html.escape(s.project_dir[:40])}</td>"
            f"<td>{started}</td>"
            f"<td>{s.turn_count}</td>"
            f"<td>{s.tool_call_count}</td>"
            f"<td>{s.input_tokens + s.output_tokens:,}</td>"
            f"<td>${s.total_cost:.4f}</td>"
            f"<td>{html.escape(models)}</td>"
            f"<td>{len(s.findings)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Session</th><th>Project</th><th>Started</th><th>Turns</th>"
        "<th>Tool calls</th><th>Tokens</th><th>Cost</th><th>Model(s)</th><th>Findings</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentLens report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --border: #ddd;
    --accent: #6f5bd6; --sev-low: #b58900; --sev-medium: #cb4b16; --sev-high: #dc322f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16161d; --fg: #eee; --muted: #999; --border: #333; --accent: #a996ff; }}
  }}
  body {{ background: var(--bg); color: var(--fg); font-family: -apple-system, Segoe UI, sans-serif;
          max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .subtitle {{ color: var(--muted); margin-top: 0; }}
  section {{ margin: 2.5rem 0; }}
  h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  .stat-row {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .stat {{ background: color-mix(in srgb, var(--accent) 10%, transparent); border-radius: 8px;
           padding: 0.8rem 1.2rem; min-width: 140px; }}
  .stat .value {{ font-size: 1.6rem; font-weight: 700; }}
  .stat .label {{ color: var(--muted); font-size: 0.8rem; }}
  .bar {{ fill: var(--accent); }}
  .bar-label {{ fill: var(--muted); font-size: 10px; }}
  .sev-low {{ color: var(--sev-low); }}
  .sev-medium {{ color: var(--sev-medium); font-weight: 600; }}
  .sev-high {{ color: var(--sev-high); font-weight: 700; }}
  .chart-grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .chart-card {{ border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 0.6rem; }}
  .chart-card-title {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 0.2rem; }}
  .chart-caption {{ color: var(--muted); font-size: 0.85rem; }}
  .axis-label {{ fill: var(--muted); font-size: 9px; }}
  .axis-line {{ stroke: var(--border); stroke-width: 1; }}
  .danger-zone {{ fill: var(--sev-high); opacity: 0.08; }}
  .context-line {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
  .budget-line-warning {{ stroke: var(--sev-low); stroke-width: 1; stroke-dasharray: 3 3; }}
  .budget-line-high {{ stroke: var(--sev-medium); stroke-width: 1; stroke-dasharray: 3 3; }}
  .budget-line-critical {{ stroke: var(--sev-high); stroke-width: 1; stroke-dasharray: 3 3; }}
  .scatter-dot {{ fill: var(--accent); opacity: 0.75; }}
  .clear-marker-ok {{ fill: #2a9d8f; font-size: 11px; font-weight: 700; }}
  .clear-marker-warn {{ fill: var(--sev-high); font-size: 11px; font-weight: 700; }}
  .scatter-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  footer {{ color: var(--muted); font-size: 0.8rem; margin-top: 3rem; }}
</style>
</head>
<body>
<h1>AgentLens</h1>
<p class="subtitle">Claude Code session cost &amp; efficiency report — generated {generated_at}</p>

<section>
  <h2>Summary</h2>
  <div class="stat-row">
    <div class="stat"><div class="value">{session_count}</div><div class="label">sessions</div></div>
    <div class="stat"><div class="value">${total_cost:.2f}</div><div class="label">total cost</div></div>
    <div class="stat"><div class="value">{total_tokens:,}</div><div class="label">total tokens</div></div>
    <div class="stat"><div class="value">{finding_count}</div><div class="label">waste findings</div></div>
  </div>
</section>

<section>
  <h2>Daily cost</h2>
  {chart}
</section>

<section>
  <h2>Sessions</h2>
  {sessions_table}
</section>

<section>
  <h2>Waste patterns</h2>
  {findings_table}
</section>

<section>
  <h2>Usage habits</h2>
  <p class="chart-caption">
    Cost driven by how sessions are used, rather than by the agent itself —
    unbounded context growth, topic switches without a reset, and unrelated
    projects mixed into one session.
  </p>
  {habit_summary}
</section>

<section>
  <h2>Topic timeline &amp; session-splitting</h2>
  {topic_timeline}
</section>

<section>
  <h2>Context growth timeline</h2>
  <p class="chart-caption">
    Top sessions by peak context usage. Dashed lines mark the
    55% / 80% / 90% warning bands; the shaded band is the 80%+ danger zone.
  </p>
  {context_timeline}
</section>

<section>
  <h2>Cache hit rate vs. session length</h2>
  <div class="scatter-row">{cache_vs_length}</div>
</section>

<section>
  <h2>Cost per call vs. peak context size</h2>
  <div class="scatter-row">{cost_vs_peak_context}</div>
</section>

<footer>AgentLens — local, self-contained report. No data leaves this machine.</footer>
</body>
</html>
"""


def generate_report(summaries: list, output_path: Path) -> dict:
    daily = _daily_cost(summaries)
    total_cost = sum(s.total_cost for s in summaries)
    total_tokens = sum(s.input_tokens + s.output_tokens for s in summaries)
    finding_count = sum(len(s.findings) for s in summaries)

    with_habits = _sessions_with_habits(summaries)
    avg_habit_score = (
        sum(s.habit_metrics.habit_score for s in with_habits) / len(with_habits) if with_habits else None
    )

    html_content = PAGE_TEMPLATE.format(
        generated_at=date.today().isoformat(),
        session_count=len(summaries),
        total_cost=total_cost,
        total_tokens=total_tokens,
        finding_count=finding_count,
        chart=_bar_chart_svg(daily),
        sessions_table=_sessions_table_html(summaries),
        findings_table=_findings_html(summaries),
        habit_summary=_habit_score_summary_html(summaries),
        topic_timeline=_topic_timeline_svg(summaries),
        context_timeline=_context_timeline_svg(summaries),
        cache_vs_length=_cache_hit_vs_length_svg(summaries),
        cost_vs_peak_context=_cost_per_call_vs_peak_context_svg(summaries),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    return {
        "output": str(output_path),
        "session_count": len(summaries),
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "finding_count": finding_count,
        "avg_habit_score": avg_habit_score,
    }
