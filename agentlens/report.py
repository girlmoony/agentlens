"""Self-contained static HTML report generation for AgentLens (no JS, no external assets)."""
from __future__ import annotations

import html
from collections import defaultdict
from datetime import date
from pathlib import Path

from .metrics import SessionSummary

CHART_HEIGHT = 160
CHART_BAR_WIDTH = 28
CHART_BAR_GAP = 14


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

<footer>AgentLens — local, self-contained report. No data leaves this machine.</footer>
</body>
</html>
"""


def generate_report(summaries: list, output_path: Path) -> dict:
    daily = _daily_cost(summaries)
    total_cost = sum(s.total_cost for s in summaries)
    total_tokens = sum(s.input_tokens + s.output_tokens for s in summaries)
    finding_count = sum(len(s.findings) for s in summaries)

    html_content = PAGE_TEMPLATE.format(
        generated_at=date.today().isoformat(),
        session_count=len(summaries),
        total_cost=total_cost,
        total_tokens=total_tokens,
        finding_count=finding_count,
        chart=_bar_chart_svg(daily),
        sessions_table=_sessions_table_html(summaries),
        findings_table=_findings_html(summaries),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    return {
        "output": str(output_path),
        "session_count": len(summaries),
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "finding_count": finding_count,
    }
