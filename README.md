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

Requires **Python 3.10+** — the `mcp` SDK used by the
[MCP server](#mcp-server) below requires 3.10 or newer (every published
`mcp` release does).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No other setup is required — AgentLens only reads local files. This also
installs the `mcp` SDK needed for the MCP server.

To get the `agentlens` / `agentlens-mcp` console scripts (used in the MCP
client config examples below), install the package itself instead:

```bash
pip install -e .
```

> **`error: externally-managed-environment`?** Some Linux distributions
> (Debian, Ubuntu, Fedora, etc.) and Homebrew's Python refuse a bare `pip
> install` against the system Python (PEP 668). The virtual environment
> above avoids this entirely — create and activate it first, then both
> `pip install` commands work normally. If you really need to install
> without a venv, `pip install --user -r requirements.txt` is the safer
> fallback; `pip install --break-system-packages ...` also works but can
> affect OS-managed Python packages.

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

## MCP Server

AgentLens also runs as an [MCP](https://modelcontextprotocol.io) server
(`agentlens/mcp_server.py`), so MCP-aware clients — Claude Code, Codex CLI,
GitHub Copilot Chat, etc. — can query session cost/efficiency data directly
instead of shelling out to the CLI. It reuses the exact same
`log_reader.py`/`metrics.py`/`habits.py`/`report.py` logic as the CLI; the
`scan`/`report` commands above are unaffected and keep working as before.

Built on the official [`mcp`](https://pypi.org/project/mcp/) Python SDK
(confirmed on PyPI; `mcp>=1.2.0` in `requirements.txt`/`pyproject.toml`),
using its `FastMCP` high-level API over the standard **stdio** transport.

### Tools

| Tool | Description |
|---|---|
| `scan_sessions(since="7d", projects_dir=None)` | Per-session token usage, USD cost, and waste-pattern findings (same data as `scan --json`). |
| `generate_report(since="30d", output_path=None, projects_dir=None)` | Generates the self-contained HTML report and returns its file path. |
| `get_habit_score(since="7d", projects_dir=None)` | Average usage-habit score, a breakdown of how many sessions hit each habit finding, and the lowest-scoring sessions worth investigating first. |

All three accept the same `since` format as the CLI (`Nh`/`Nd`/`Nw`, or
`None`/omitted for all history) and an optional `projects_dir` override.

### Running it directly

```bash
python -m agentlens.mcp_server
```

This starts the server on stdio and blocks — it's meant to be launched by an
MCP client, not run interactively. After installing the package
(`pip install -e .`, via `pyproject.toml`), the same server is also
available as the `agentlens-mcp` console script, which avoids depending on
the client's working directory.

### Claude Code

```bash
claude mcp add --transport stdio agentlens -- python -m agentlens.mcp_server
```

Or add directly to `.mcp.json` (project scope) / `~/.claude.json` (user
scope):

```json
{
  "mcpServers": {
    "agentlens": {
      "type": "stdio",
      "command": "agentlens-mcp"
    }
  }
}
```

(Use `"command": "python", "args": ["-m", "agentlens.mcp_server"]` instead
if you haven't installed the package and are running from the repo — Claude
Code needs a matching working directory in that case.)

### Codex CLI

Codex reads MCP servers from `[mcp_servers.<name>]` tables in
`~/.codex/config.toml` (or a project-scoped `.codex/config.toml` for
trusted projects):

```toml
[mcp_servers.agentlens]
command = "agentlens-mcp"
```

### GitHub Copilot (VS Code)

VS Code's Copilot Chat reads MCP servers from `.vscode/mcp.json` at the
workspace root — note the root key is `servers`, not `mcpServers` (VS Code's
own naming, different from Claude Code's):

```json
{
  "servers": {
    "agentlens": {
      "command": "agentlens-mcp"
    }
  }
}
```

Other MCP-compatible clients follow a similar shape (a `command` — and
`args` if not using the installed console script — for a stdio server);
consult the client's own docs for its exact config file location and key
name if it isn't one of the three above.

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
  fraction of *that turn's own model's* context window — see
  [Context window sizing](#context-window-sizing) below) peaked at 90%+, or
  stayed at 80%+ for 5+ consecutive turns, without a `/clear` or `/compact`
  in between. Mirrors the 25/55/80/90% warning bands Claude Code's own
  context-usage indicator uses. This is a cost/capacity signal (how close to
  the token budget did this session run), not an accuracy one — see
  `long_context_quality_risk` below for that angle.
- **session_not_split** — the session changed topic (see below) at least
  twice without a `/clear`, `/compact`, or new session in between, so
  context kept accumulating across unrelated work instead of being reset.
  A topic change is detected heuristically: consecutive file-referencing
  turns whose directories have a Jaccard similarity below 0.2 are treated as
  a topic shift.
- **mixed_project_session** — two or more directories unrelated to the
  session's own project (i.e. that don't share any of the project folder's
  name tokens) were touched in the same session — a sign unrelated work got
  mixed into one context instead of using a session per project. Beyond the
  token cost, this is also flagged as a possible quality risk — see
  [Context Rot research](#context-rot-research) below.
- **long_context_quality_risk** — context usage reached 50%+ of the model's
  window *and* the same session also triggered `duplicate_read`,
  `low_cache_reuse`, or `mixed_project_session`. This combination — a large
  context plus redundant/irrelevant content in it — is what external
  research (below) found compounds accuracy loss beyond what context length
  alone causes. There is no published token threshold for when this
  actually starts, so this finding is explicitly a conservative, qualitative
  nudge, not a measured cutoff — see [Context Rot research](#context-rot-research).

### Context window sizing

Each turn's context-usage ratio is computed against *that turn's own
model's* real context window, looked up from a new `context_windows` section
in `pricing.yaml` (populated the same way as the cost tables — see
[Pricing logic](#pricing-logic)). A session that switches models mid-way
(rare, but seen in real logs) is scored per-turn against the model that
actually produced that turn, rather than one flat number for the whole
session. Models not in the table fall back to a conservative
`unknown_model_context_window` (200K) rather than assuming the larger figure
most current models share.

### Context Rot research

`context_budget_exceeded` and `long_context_quality_risk` are informed by
two pieces of external research on context-length degradation, cited
directly in code comments and finding `detail` text so the claims stay
traceable:

- **Chroma, ["Context Rot: How Increasing Input Tokens Impacts LLM
  Performance"](https://www.trychroma.com/research/context-rot)** (Kelly
  Hong, Anton Troynikov, Jeff Huber; July 2025). Tested 18 frontier models.
  **Explicitly does not give a universal token count or percentage-of-window
  threshold** where degradation begins — it reports that reliability
  declines in a non-uniform, model- and task-dependent way as input grows,
  even on simple retrieval tasks. Two findings this project *does* encode
  directly: (1) a single distractor already measurably reduces accuracy
  vs. a distractor-free baseline, and additional distractors compound the
  loss further; (2) in their LongMemEval comparison, prompts trimmed to only
  the relevant content scored significantly higher than the full,
  unfiltered prompt — irrelevant context degrades accuracy independent of
  raw length.
- **Liu et al., ["Lost in the Middle: How Language Models Use Long
  Contexts"](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf)**
  (Stanford; arXiv 2023, TACL 2024). Found a U-shaped position-sensitivity
  curve — models use information at the very start or end of context far
  more reliably than information in the middle, even well inside the stated
  context window. AgentLens can't recover *where* in a session's context a
  given fact lives from the JSONL log, so this isn't mechanically detected;
  it's the reason `long_context_quality_risk` treats "the context is large"
  as a standalone risk rather than only a cost one.

**Because neither paper gives a reproducible numeric threshold, AgentLens
does not pretend to have one.** `QUALITY_RISK_CONTEXT_RATIO` (50% of the
model's window) in `agentlens/habits.py` is explicitly documented in code as
a conservative, made-up-for-this-project heuristic — not a published
figure — and `long_context_quality_risk`'s `detail` text repeats that
caveat every time it fires. Treat both context-based habit findings as
directional signals to investigate, not precise measurements.

### Usage-habit score

Each session gets a 0–100 **habit score**, starting at 100 and losing points
for each habit finding above (15 for `session_not_split`, 20 for
`context_budget_exceeded`, 15 for `mixed_project_session`, 10 for
`long_context_quality_risk` — weighted lower since it rests on a qualitative
research finding rather than a measured cost figure), scaled by severity
(0.6x low, 1x medium, 1.5x high) and floored at 0. It's a single composite
score rather than separate cost/quality axes, deliberately: the weighting
already reflects "the cost-grounded findings count more than the
research-informed one" without needing two numbers to track. It's a rough,
at-a-glance signal for "was this session driven efficiently," not a
precise cost figure — the findings and their `detail` text are the
authoritative explanation. `scan` and `report` both show the average habit
score across the scanned sessions, and the HTML report shows it per-session
plus a breakdown of how many sessions hit each habit finding.
`scan --json` additionally includes `habit_score`, `peak_context_ratio`,
`cache_hit_rate`, `topic_shift_count`, and `undisciplined_shift_count` per
session.

### Usage-habit visualizations

The HTML report charts, per flagged session: a **context growth
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
- **Context windows** (`context_windows` section, plus
  `unknown_model_context_window` as the fallback) hold each model's real
  context window in tokens. These aren't used for cost at all — they only
  feed the usage-habit context-ratio calculation in `habits.py`; see
  [Context window sizing](#context-window-sizing) above.

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
