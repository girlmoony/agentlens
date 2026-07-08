"""AgentLens MCP server.

Exposes the same scan/report/habit-score logic as cli.py over the Model
Context Protocol (stdio transport), so MCP-aware clients — Claude Code,
Codex CLI, GitHub Copilot, etc. — can query session cost/efficiency data
directly instead of shelling out to the CLI.

Run:  python -m agentlens.mcp_server
      (or the `agentlens-mcp` console script installed via pyproject.toml)

This reuses log_reader.py / metrics.py / habits.py / report.py exactly as
cli.py does — no parsing or detection logic is duplicated here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .cli import DEFAULT_REPORT_PATH, _collect_summaries, _parse_since, summary_to_dict
from .log_reader import DEFAULT_PROJECTS_DIR
from .report import generate_report as _generate_html_report

mcp = FastMCP(
    "agentlens",
    instructions=(
        "Local cost & efficiency visibility for Claude Code sessions. "
        "Reads ~/.claude/projects/**/*.jsonl directly — no API keys, no data "
        "leaves this machine. Use scan_sessions for per-session token/cost/"
        "waste-pattern data, generate_report for a shareable HTML report, "
        "and get_habit_score for a quick read on whether recent sessions "
        "were driven efficiently."
    ),
)


def _resolve_projects_dir(projects_dir: Optional[str]) -> Path:
    return Path(projects_dir) if projects_dir else DEFAULT_PROJECTS_DIR


@mcp.tool()
def scan_sessions(since: Optional[str] = "7d", projects_dir: Optional[str] = None) -> dict:
    """Scan Claude Code session logs and return per-session token usage, USD
    cost, and waste-pattern findings (both agent-side patterns like
    duplicate_read, and usage-habit patterns like context_budget_exceeded).

    Args:
        since: only include sessions from the last N (h)ours/(d)ays/(w)eeks,
            e.g. "7d", "24h", "2w". Pass None or "" to scan all history.
        projects_dir: override the default ~/.claude/projects location.
    """
    since_dt = _parse_since(since) if since else None
    summaries = _collect_summaries(since_dt, _resolve_projects_dir(projects_dir))

    return {
        "session_count": len(summaries),
        "total_cost": sum(s.total_cost for s in summaries),
        "total_tokens": sum(s.input_tokens + s.output_tokens for s in summaries),
        "finding_count": sum(len(s.findings) for s in summaries),
        "sessions": [summary_to_dict(s) for s in summaries],
    }


@mcp.tool()
def generate_report(
    since: Optional[str] = "30d", output_path: Optional[str] = None, projects_dir: Optional[str] = None
) -> dict:
    """Generate a self-contained, offline HTML cost/efficiency report (charts,
    waste-pattern table, usage-habit visualizations) and return its file path.

    Args:
        since: only include sessions from the last N (h)ours/(d)ays/(w)eeks,
            e.g. "30d". Pass None or "" to include all history.
        output_path: where to write the HTML file (default:
            ./agentlens-report.html in the server's working directory).
        projects_dir: override the default ~/.claude/projects location.
    """
    since_dt = _parse_since(since) if since else None
    summaries = _collect_summaries(since_dt, _resolve_projects_dir(projects_dir))
    out_path = Path(output_path) if output_path else DEFAULT_REPORT_PATH
    return _generate_html_report(summaries, out_path)


@mcp.tool()
def get_habit_score(since: Optional[str] = "7d", projects_dir: Optional[str] = None) -> dict:
    """Return the usage-habit score (0-100, how efficiently recent sessions
    were driven — context growth, session splitting, project mixing) averaged
    across recent sessions, a breakdown of how many sessions hit each habit
    finding, and the lowest-scoring sessions worth investigating first.

    Args:
        since: only include sessions from the last N (h)ours/(d)ays/(w)eeks,
            e.g. "7d". Pass None or "" to include all history.
        projects_dir: override the default ~/.claude/projects location.
    """
    since_dt = _parse_since(since) if since else None
    summaries = _collect_summaries(since_dt, _resolve_projects_dir(projects_dir))
    with_habits = [s for s in summaries if s.habit_metrics is not None]

    finding_counts: dict = {}
    for s in with_habits:
        for f in s.habit_metrics.findings:
            finding_counts[f["type"]] = finding_counts.get(f["type"], 0) + 1

    worst = sorted(with_habits, key=lambda s: s.habit_metrics.habit_score)[:5]

    return {
        "session_count": len(summaries),
        "avg_habit_score": (
            sum(s.habit_metrics.habit_score for s in with_habits) / len(with_habits) if with_habits else None
        ),
        "finding_counts": finding_counts,
        "worst_sessions": [
            {
                "session_id": s.session_id,
                "started": s.start.isoformat() if s.start else None,
                "habit_score": s.habit_metrics.habit_score,
                "findings": [f["type"] for f in s.habit_metrics.findings],
            }
            for s in worst
        ],
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
