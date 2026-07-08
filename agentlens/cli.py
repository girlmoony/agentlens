"""AgentLens CLI — cost and efficiency visibility for local Claude Code sessions.

Run:  python -m agentlens.cli scan [--since 7d]
      python -m agentlens.cli report [--since 7d] [-o report.html]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .log_reader import DEFAULT_PROJECTS_DIR, find_session_files, parse_session
from .metrics import Pricing, summarize_session
from .report import generate_report

DEFAULT_REPORT_PATH = Path.cwd() / "agentlens-report.html"


def _parse_since(value: str) -> datetime:
    match = re.fullmatch(r"(\d+)([dhw])", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid --since value: {value!r} (expected e.g. 7d, 24h, 2w)")
    amount, unit = int(match.group(1)), match.group(2)
    delta = {"h": timedelta(hours=amount), "d": timedelta(days=amount), "w": timedelta(weeks=amount)}[unit]
    return datetime.now(timezone.utc) - delta


def _collect_summaries(since: Optional[datetime], projects_dir: Path):
    pricing = Pricing.load()
    summaries = []
    for file_path in find_session_files(projects_dir=projects_dir, since=since):
        session = parse_session(file_path)
        if not session.turns:
            continue
        if since is not None and session.end is not None and session.end < since:
            continue
        summaries.append(summarize_session(session, pricing))
    return summaries


def cmd_scan(args) -> None:
    since = _parse_since(args.since) if args.since else None
    summaries = _collect_summaries(since, Path(args.projects_dir))

    if not summaries:
        print("No sessions found.")
        return

    total_cost = sum(s.total_cost for s in summaries)
    total_tokens = sum(s.input_tokens + s.output_tokens for s in summaries)
    finding_count = sum(len(s.findings) for s in summaries)

    suffix = f" since {args.since}" if args.since else ""
    print(f"Scanned {len(summaries)} session(s){suffix}")
    print(f"  total cost:   ${total_cost:.4f}")
    print(f"  total tokens: {total_tokens:,}")
    print(f"  findings:     {finding_count}")
    print()

    for s in sorted(summaries, key=lambda s: s.total_cost, reverse=True)[:20]:
        started = s.start.strftime("%Y-%m-%d %H:%M") if s.start else "?"
        print(
            f"  {s.session_id[:8]}  {started}  turns={s.turn_count:<3} tools={s.tool_call_count:<3} "
            f"tokens={s.input_tokens + s.output_tokens:<8,} cost=${s.total_cost:.4f}  "
            f"findings={len(s.findings)}"
        )

    if args.json:
        payload = [
            {
                "session_id": s.session_id,
                "project_dir": s.project_dir,
                "started": s.start.isoformat() if s.start else None,
                "turn_count": s.turn_count,
                "tool_call_count": s.tool_call_count,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "total_cost": s.total_cost,
                "models_used": s.models_used,
                "findings": s.findings,
            }
            for s in summaries
        ]
        print(json.dumps(payload, indent=2))


def cmd_report(args) -> None:
    since = _parse_since(args.since) if args.since else None
    summaries = _collect_summaries(since, Path(args.projects_dir))
    output_path = Path(args.output) if args.output else DEFAULT_REPORT_PATH
    summary = generate_report(summaries, output_path)
    print(json.dumps(summary, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentlens", description="Local cost & efficiency visibility for Claude Code sessions."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--since", help="only include sessions from the last N (h)ours/(d)ays/(w)eeks, e.g. 7d")
    common.add_argument(
        "--projects-dir", default=str(DEFAULT_PROJECTS_DIR), help="override the Claude Code projects directory"
    )

    scan_parser = sub.add_parser("scan", parents=[common], help="print a cost/efficiency summary to the terminal")
    scan_parser.add_argument("--json", action="store_true", help="also print full results as JSON")
    scan_parser.set_defaults(func=cmd_scan)

    report_parser = sub.add_parser("report", parents=[common], help="generate a static HTML report")
    report_parser.add_argument("-o", "--output", help="output HTML path (default: ./agentlens-report.html)")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
