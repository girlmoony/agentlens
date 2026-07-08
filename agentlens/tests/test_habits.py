import unittest
from datetime import datetime, timezone

from agentlens.log_reader import Boundary, Session, ToolCall, Turn
from agentlens import habits


def _ts(sec: int, micro: int = 0) -> datetime:
    return datetime(2026, 7, 8, 10, 0, sec, micro, tzinfo=timezone.utc)


def _read_turn(msg_id: str, sec: int, file_path: str, input_tokens: int = 100) -> Turn:
    return Turn(
        msg_id, _ts(sec), "claude-sonnet-5", input_tokens, 100, 0, 0,
        tool_calls=[ToolCall(timestamp=_ts(sec), name="Read", input={"file_path": file_path})],
    )


def _session(turns, project_dir="proj", boundaries=None) -> Session:
    return Session(
        session_id="s1", project_dir=project_dir, file_path="s1.jsonl",
        turns=turns, boundaries=boundaries or [],
    )


class TestContextBudget(unittest.TestCase):
    def test_sustained_high_context_flagged(self):
        turns = [Turn(f"m{i}", _ts(i), "claude-sonnet-5", 170_000, 100, 0, 0) for i in range(6)]
        findings = habits.detect_habit_waste_patterns(_session(turns))
        types = [f["type"] for f in findings]
        self.assertIn("context_budget_exceeded", types)

    def test_brief_spike_not_flagged(self):
        turns = [Turn("m0", _ts(0), "claude-sonnet-5", 170_000, 100, 0, 0)] + [
            Turn(f"m{i}", _ts(i), "claude-sonnet-5", 1_000, 100, 0, 0) for i in range(1, 6)
        ]
        findings = habits.detect_habit_waste_patterns(_session(turns))
        types = [f["type"] for f in findings]
        self.assertNotIn("context_budget_exceeded", types)

    def test_low_context_not_flagged(self):
        turns = [Turn(f"m{i}", _ts(i), "claude-sonnet-5", 1_000, 100, 0, 0) for i in range(6)]
        findings = habits.detect_habit_waste_patterns(_session(turns))
        types = [f["type"] for f in findings]
        self.assertNotIn("context_budget_exceeded", types)


class TestSessionNotSplit(unittest.TestCase):
    def _three_topic_turns(self):
        return (
            [_read_turn(f"a{i}", i, "/proj/src/auth/login.py") for i in range(3)]
            + [_read_turn(f"b{i}", i + 3, "/proj/docs/readme.md") for i in range(3)]
            + [_read_turn(f"c{i}", i + 6, "/proj/infra/terraform/main.tf") for i in range(3)]
        )

    def test_repeated_topic_shifts_without_reset_flagged(self):
        findings = habits.detect_habit_waste_patterns(_session(self._three_topic_turns()))
        types = [f["type"] for f in findings]
        self.assertIn("session_not_split", types)

    def test_topic_shift_with_clear_between_not_flagged(self):
        turns = self._three_topic_turns()
        boundaries = [
            Boundary(timestamp=_ts(2, 500_000), kind="clear"),
            Boundary(timestamp=_ts(5, 500_000), kind="clear"),
        ]
        findings = habits.detect_habit_waste_patterns(_session(turns, boundaries=boundaries))
        types = [f["type"] for f in findings]
        self.assertNotIn("session_not_split", types)

    def test_single_topic_session_not_flagged(self):
        turns = [_read_turn(f"a{i}", i, "/proj/src/auth/login.py") for i in range(5)]
        findings = habits.detect_habit_waste_patterns(_session(turns))
        types = [f["type"] for f in findings]
        self.assertNotIn("session_not_split", types)


class TestMixedProjectSession(unittest.TestCase):
    def test_multiple_unrelated_projects_flagged(self):
        turns = [
            _read_turn("m0", 0, "/home/user/mcp/agentlens/src/x.py"),
            _read_turn("m1", 1, "/home/user/other-repo-a/foo.py"),
            _read_turn("m2", 2, "/home/user/other-repo-b/bar.py"),
        ]
        findings = habits.detect_habit_waste_patterns(
            _session(turns, project_dir="home-user-mcp-agentlens")
        )
        types = [f["type"] for f in findings]
        self.assertIn("mixed_project_session", types)

    def test_single_project_not_flagged(self):
        turns = [
            _read_turn("m0", 0, "/home/user/mcp/agentlens/src/x.py"),
            _read_turn("m1", 1, "/home/user/mcp/agentlens/tests/y.py"),
            _read_turn("m2", 2, "/home/user/mcp/agentlens/docs/z.md"),
        ]
        findings = habits.detect_habit_waste_patterns(
            _session(turns, project_dir="home-user-mcp-agentlens")
        )
        types = [f["type"] for f in findings]
        self.assertNotIn("mixed_project_session", types)

    def test_single_foreign_reference_not_flagged(self):
        """One incidental foreign-directory read (e.g. a shared config file)
        shouldn't be enough to call the session mixed."""
        turns = [
            _read_turn("m0", 0, "/home/user/mcp/agentlens/src/x.py"),
            _read_turn("m1", 1, "/home/user/mcp/agentlens/tests/y.py"),
            _read_turn("m2", 2, "/home/user/shared-config/settings.json"),
        ]
        findings = habits.detect_habit_waste_patterns(
            _session(turns, project_dir="home-user-mcp-agentlens")
        )
        types = [f["type"] for f in findings]
        self.assertNotIn("mixed_project_session", types)


class TestHabitMetrics(unittest.TestCase):
    def test_habit_score_100_when_no_findings(self):
        turns = [_read_turn(f"a{i}", i, "/proj/src/auth/login.py") for i in range(3)]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertEqual(hm.habit_score, 100)
        self.assertEqual(hm.findings, [])

    def test_habit_score_decreases_with_findings(self):
        turns = [Turn(f"m{i}", _ts(i), "claude-sonnet-5", 170_000, 100, 0, 0) for i in range(6)]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertLess(hm.habit_score, 100)

    def test_cache_hit_rate_computed(self):
        turns = [
            Turn("m0", _ts(0), "claude-sonnet-5", 0, 0,
                 cache_creation_input_tokens=1_000, cache_read_input_tokens=9_000)
        ]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertAlmostEqual(hm.cache_hit_rate, 0.9)

    def test_cache_hit_rate_none_when_no_cache_activity(self):
        turns = [Turn("m0", _ts(0), "claude-sonnet-5", 100, 100, 0, 0)]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertIsNone(hm.cache_hit_rate)

    def test_context_timeline_has_one_point_per_turn(self):
        turns = [Turn(f"m{i}", _ts(i), "claude-sonnet-5", 100, 100, 0, 0) for i in range(4)]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertEqual(len(hm.context_timeline), 4)

    def test_topic_zones_cover_all_turns_contiguously(self):
        turns = [_read_turn(f"a{i}", i, "/proj/src/auth/login.py") for i in range(3)] + [
            _read_turn(f"b{i}", i + 3, "/proj/docs/readme.md") for i in range(3)
        ]
        hm = habits.compute_habit_metrics(_session(turns))
        self.assertEqual(hm.topic_zones[0].start_turn, 0)
        self.assertEqual(hm.topic_zones[-1].end_turn, len(turns) - 1)
        self.assertTrue(hm.topic_zones[0].preceded_by_reset)


if __name__ == "__main__":
    unittest.main()
