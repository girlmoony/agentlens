import unittest
from datetime import datetime, timezone

from agentlens.log_reader import Session, Turn, ToolCall
from agentlens.metrics import Pricing, turn_cost, detect_waste_patterns

PRICING = Pricing(
    models={
        "claude-sonnet-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    },
    cache_multipliers={"write_5m": 1.25, "write_1h": 2.0, "read": 0.1},
    unknown_model_fallback="claude-sonnet-5",
)


def _ts(sec: int) -> datetime:
    return datetime(2026, 7, 8, 10, 0, sec, tzinfo=timezone.utc)


class TestTurnCost(unittest.TestCase):
    def test_basic_input_output_cost(self):
        turn = Turn(
            message_id="m1", timestamp=_ts(0), model="claude-sonnet-5",
            input_tokens=1_000_000, output_tokens=1_000_000,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        cost = turn_cost(turn, PRICING)
        self.assertAlmostEqual(cost["input_cost"], 3.00)
        self.assertAlmostEqual(cost["output_cost"], 15.00)
        self.assertAlmostEqual(cost["total_cost"], 18.00)

    def test_cache_write_and_read_multipliers(self):
        turn = Turn(
            message_id="m1", timestamp=_ts(0), model="claude-opus-4-8",
            input_tokens=0, output_tokens=0,
            cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000,
            cache_creation_5m_input_tokens=1_000_000,
        )
        cost = turn_cost(turn, PRICING)
        # input rate 5.00/1M; write = 5.00 * 1.25 = 6.25; read = 5.00 * 0.1 = 0.5
        self.assertAlmostEqual(cost["cache_write_cost"], 6.25)
        self.assertAlmostEqual(cost["cache_read_cost"], 0.5)

    def test_cache_writes_priced_separately_by_ttl(self):
        """A 1h cache write costs 2x base input vs. 1.25x for a 5m write — mixing them
        into one bucket and assuming 5m would under-price the 1h portion."""
        turn = Turn(
            message_id="m1", timestamp=_ts(0), model="claude-opus-4-8",
            input_tokens=0, output_tokens=0,
            cache_creation_input_tokens=2_000_000, cache_read_input_tokens=0,
            cache_creation_5m_input_tokens=1_000_000,
            cache_creation_1h_input_tokens=1_000_000,
        )
        cost = turn_cost(turn, PRICING)
        # 5m: 5.00 * 1.25 = 6.25; 1h: 5.00 * 2.0 = 10.00
        self.assertAlmostEqual(cost["cache_write_cost"], 16.25)

    def test_sonnet_5_introductory_pricing_applies_before_valid_until(self):
        pricing = Pricing(
            models={"claude-sonnet-5": {"input": 3.00, "output": 15.00}},
            cache_multipliers={"write_5m": 1.25, "write_1h": 2.0, "read": 0.1},
            unknown_model_fallback="claude-sonnet-5",
            introductory_pricing={"claude-sonnet-5": {"valid_until": "2026-09-01", "input": 2.00, "output": 10.00}},
        )
        turn = Turn(
            message_id="m1", timestamp=_ts(0), model="claude-sonnet-5",
            input_tokens=1_000_000, output_tokens=1_000_000,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        cost = turn_cost(turn, pricing)
        self.assertAlmostEqual(cost["input_cost"], 2.00)
        self.assertAlmostEqual(cost["output_cost"], 10.00)

    def test_sonnet_5_reverts_to_standard_pricing_after_valid_until(self):
        pricing = Pricing(
            models={"claude-sonnet-5": {"input": 3.00, "output": 15.00}},
            cache_multipliers={"write_5m": 1.25, "write_1h": 2.0, "read": 0.1},
            unknown_model_fallback="claude-sonnet-5",
            introductory_pricing={"claude-sonnet-5": {"valid_until": "2026-09-01", "input": 2.00, "output": 10.00}},
        )
        turn = Turn(
            message_id="m1", timestamp=datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc), model="claude-sonnet-5",
            input_tokens=1_000_000, output_tokens=1_000_000,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        cost = turn_cost(turn, pricing)
        self.assertAlmostEqual(cost["input_cost"], 3.00)
        self.assertAlmostEqual(cost["output_cost"], 15.00)

    def test_unknown_model_falls_back(self):
        turn = Turn(
            message_id="m1", timestamp=_ts(0), model="some-future-model",
            input_tokens=1_000_000, output_tokens=0,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        cost = turn_cost(turn, PRICING)
        self.assertAlmostEqual(cost["input_cost"], 3.00)  # fallback == claude-sonnet-5 rate


def _session_with_turns(turns) -> Session:
    return Session(session_id="s1", project_dir="proj", file_path="s1.jsonl", turns=turns)


class TestWastePatterns(unittest.TestCase):
    def test_duplicate_read_detected(self):
        calls = [ToolCall(timestamp=_ts(i), name="Read", input={"file_path": "/a.py"}) for i in range(3)]
        turn = Turn("m1", _ts(0), "claude-sonnet-5", 0, 0, 0, 0, tool_calls=calls)
        findings = detect_waste_patterns(_session_with_turns([turn]))
        types = [f["type"] for f in findings]
        self.assertIn("duplicate_read", types)

    def test_single_read_not_flagged(self):
        calls = [ToolCall(timestamp=_ts(0), name="Read", input={"file_path": "/a.py"})]
        turn = Turn("m1", _ts(0), "claude-sonnet-5", 0, 0, 0, 0, tool_calls=calls)
        findings = detect_waste_patterns(_session_with_turns([turn]))
        self.assertEqual([f for f in findings if f["type"] == "duplicate_read"], [])

    def test_low_cache_reuse_flagged(self):
        turn = Turn("m1", _ts(0), "claude-sonnet-5", 0, 0,
                     cache_creation_input_tokens=10_000, cache_read_input_tokens=1_000)
        findings = detect_waste_patterns(_session_with_turns([turn]))
        types = [f["type"] for f in findings]
        self.assertIn("low_cache_reuse", types)

    def test_healthy_cache_reuse_not_flagged(self):
        turn = Turn("m1", _ts(0), "claude-sonnet-5", 0, 0,
                     cache_creation_input_tokens=1_000, cache_read_input_tokens=50_000)
        findings = detect_waste_patterns(_session_with_turns([turn]))
        types = [f["type"] for f in findings]
        self.assertNotIn("low_cache_reuse", types)

    def test_rapid_single_tool_calls_flagged(self):
        turns = [
            Turn(f"m{i}", _ts(i), "claude-sonnet-5", 0, 0, 0, 0,
                 tool_calls=[ToolCall(timestamp=_ts(i), name="Bash", input={})])
            for i in range(4)
        ]
        findings = detect_waste_patterns(_session_with_turns(turns))
        types = [f["type"] for f in findings]
        self.assertIn("batchable_tool_calls", types)

    def test_spaced_out_single_tool_calls_not_flagged(self):
        gaps = [_ts(0), _ts(30), _ts(0).replace(minute=5), _ts(0).replace(minute=10)]
        turns = [
            Turn(f"m{i}", ts, "claude-sonnet-5", 0, 0, 0, 0,
                 tool_calls=[ToolCall(timestamp=ts, name="Bash", input={})])
            for i, ts in enumerate(gaps)
        ]
        findings = detect_waste_patterns(_session_with_turns(turns))
        types = [f["type"] for f in findings]
        self.assertNotIn("batchable_tool_calls", types)


if __name__ == "__main__":
    unittest.main()
