# AgentLens

Local cost & efficiency visibility for [Claude Code](https://claude.com/claude-code)
sessions. AgentLens reads your existing session logs under
`~/.claude/projects/**/*.jsonl`, computes per-session token usage and USD
cost, and flags recurring patterns that waste tokens — all offline, with no
extra instrumentation, no API keys, and no data ever leaving your machine.

## Background

AgentLens grew out of a trend-research project that was itself using Claude
Code heavily and wanted to know where its own session costs were going.
There was no lightweight, local way to answer "which of my sessions cost the
most, and why" without either trusting a third-party dashboard or reverse-
engineering the session log format — so this tool does that parsing and
cost accounting directly against the log format Claude Code already writes.

## Installation

```bash
pip install -r requirements.txt
```

No other setup is required — AgentLens only reads local files.

## Usage

```bash
# Terminal summary of the last 7 days
python -m agentlens.cli scan --since 7d

# Full JSON output (per-session token/cost breakdown)
python -m agentlens.cli scan --since 7d --json

# Self-contained HTML report (no JS, no external assets)
python -m agentlens.cli report --since 30d -o report.html
```

`--since` accepts `Nh` / `Nd` / `Nw` (hours/days/weeks). Omit it to scan all
available history. `--projects-dir` overrides the default
`~/.claude/projects` location if your logs live elsewhere.

## What it detects

Agent/log-side waste (`metrics.py`):

- **duplicate_read** — the same file read more than once in a session,
  a sign the model re-reads content it already had instead of relying on
  what's in context.
- **low_cache_reuse** — `cache_creation_input_tokens` far exceeds
  `cache_read_input_tokens`, meaning prompt-cache context is being rebuilt
  more often than it's actually reused.
- **batchable_tool_calls** — runs of consecutive single-tool-call turns
  fired in rapid succession (within a few seconds of each other) that could
  likely have been batched into one turn instead.

User-habit-side waste (`habits.py`) — cost driven by *how a session is
driven*, rather than by what the agent does within it:

- **context_budget_exceeded** — context usage (`input_tokens +
  cache_read_input_tokens + cache_creation_input_tokens` for a turn, as a
  fraction of a conservative 200K-token budget) peaked at 90%+, or stayed at
  80%+ for 5+ consecutive turns, without a `/clear` or `/compact` in
  between. Mirrors the 25/55/80/90% warning bands Claude Code's own
  context-usage indicator uses. The 200K figure is a flat, conservative
  default rather than the real context window of the model(s) used in the
  session, so on models with a larger window the reported ratio can exceed
  100% — read it as "how hot did this session run" rather than an exact
  percentage of the actual limit.
- **session_not_split** — the session changed topic (see below) at least
  twice without a `/clear`, `/compact`, or new session in between, so
  context kept accumulating across unrelated work instead of being reset.
  A topic change is detected heuristically: consecutive file-referencing
  turns whose directories have a Jaccard similarity below 0.2 are treated as
  a topic shift.
- **mixed_project_session** — two or more directories unrelated to the
  session's own project (i.e. that don't share any of the project folder's
  name tokens) were touched in the same session — a sign unrelated work got
  mixed into one context instead of using a session per project.

### Usage-habit score

Each session gets a 0–100 **habit score**, starting at 100 and losing points
for each habit finding above (15 for `session_not_split`, 20 for
`context_budget_exceeded`, 15 for `mixed_project_session`), scaled by
severity (0.6x low, 1x medium, 1.5x high) and floored at 0. It's a rough,
at-a-glance signal for "was this session driven efficiently," not a
precise cost figure — the findings and their `detail` text are the
authoritative explanation. `scan` and `report` both show the average habit
score across the scanned sessions, and the HTML report shows it per-session
plus a breakdown of how many sessions hit each habit finding.

The HTML report also charts, per flagged session: a **context growth
timeline** (context-usage ratio per turn, with the 55/80/90% bands and the
80%+ danger zone shaded) and a **topic timeline** — a Gantt-style view of
each session's turns colored by topic zone, with a marker at every topic
shift showing whether it was followed by a `/clear`/`/compact` (✓) or not
(!, the "you probably wanted to start a new session here" signal). Two
scatter plots round it out: cache hit rate vs. session length, and cost per
API call (`total_cost / turn_count`) vs. peak context size.

## Pricing logic

`pricing.yaml` holds USD-per-million-token rates per model plus the prompt-
cache write/read multipliers, matching Anthropic's published API pricing:

- **Cache write TTL**: Claude Code's session logs record cache writes split
  by TTL (`cache_creation.ephemeral_5m_input_tokens` /
  `ephemeral_1h_input_tokens`). AgentLens prices each bucket separately —
  1.25x base input rate for 5-minute writes, 2x for 1-hour writes — rather
  than assuming every write uses the same TTL. Cache reads are priced at
  0.1x. Older logs that predate this per-TTL breakdown fall back to treating
  the full write as 5-minute TTL.
- **Time-limited introductory pricing**: some models launch with a
  temporary discounted rate before settling into their standard price (for
  example, Claude Sonnet 5's introductory $2/$10 per MTok rate ahead of its
  standard $3/$15). These live in a separate `introductory_pricing` section
  in `pricing.yaml` with a `valid_until` date. Each turn is priced against
  its *own* timestamp, so historical turns keep the rate that was actually
  in effect when they ran, and turns dated on or after `valid_until`
  automatically use the standard rate in `models` — no code change needed
  when the introductory period ends.
- **Unknown models** fall back to a configurable default rate
  (`unknown_model_fallback`) rather than erroring, so a new or internal
  model name in the logs doesn't crash the scan.

Update `pricing.yaml` whenever Anthropic changes pricing — nothing in the
code needs to change for a rate update.

## Notes

- One Claude Code API response is logged as multiple JSONL lines (one per
  content block — thinking, text, tool use, etc.), sharing the same
  `message.id` and an identical `usage` object. `log_reader.parse_session`
  collapses these into a single `Turn` and counts usage once per message —
  summing per-line would inflate token/cost totals several-fold.
- The estimated cost is computed the same way Claude Code's own `/cost`
  (`/usage`) command does: locally, from token counts. It is not a
  replacement for the authoritative billing figures in the Claude Console.

## Development

```bash
pip install pytest
python -m pytest agentlens/tests/
```

## License

MIT — see [LICENSE](LICENSE).
