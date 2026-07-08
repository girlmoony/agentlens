import json
import tempfile
import unittest
from pathlib import Path

from agentlens import mcp_server


def _write_jsonl(lines: list, path: Path) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _assistant_event(msg_id, ts, model, usage, content):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {"id": msg_id, "model": model, "role": "assistant", "usage": usage, "content": content},
    }


class TestToolRegistration(unittest.TestCase):
    def test_expected_tools_are_registered(self):
        tools = mcp_server.mcp._tool_manager.list_tools()
        names = {t.name for t in tools}
        self.assertEqual(names, {"scan_sessions", "generate_report", "get_habit_score"})

    def test_tool_schemas_have_since_and_projects_dir_params(self):
        tools = {t.name: t for t in mcp_server.mcp._tool_manager.list_tools()}
        for name in ("scan_sessions", "generate_report", "get_habit_score"):
            props = tools[name].parameters["properties"]
            self.assertIn("since", props)
            self.assertIn("projects_dir", props)


class TestMcpToolsAgainstFakeProject(unittest.TestCase):
    """The @mcp.tool() decorator returns the original function unchanged, so
    these tools can be called directly like any other function — no MCP
    protocol machinery needed for unit tests."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.projects_dir = Path(self.tmpdir.name)

        project_dir = self.projects_dir / "fake-project"
        project_dir.mkdir()
        usage = {
            "input_tokens": 100, "output_tokens": 50,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
        lines = [
            _assistant_event(
                "msg_1", "2026-07-08T10:00:00Z", "claude-sonnet-5", usage,
                [{"type": "text", "text": "hi"}],
            )
        ]
        _write_jsonl(lines, project_dir / "session.jsonl")

    def test_scan_sessions_returns_expected_shape(self):
        result = mcp_server.scan_sessions(since=None, projects_dir=str(self.projects_dir))
        self.assertEqual(result["session_count"], 1)
        self.assertEqual(len(result["sessions"]), 1)
        session = result["sessions"][0]
        self.assertEqual(session["input_tokens"], 100)
        self.assertEqual(session["output_tokens"], 50)
        self.assertIn("habit_score", session)
        self.assertIn("findings", session)

    def test_scan_sessions_empty_projects_dir(self):
        empty_dir = Path(self.tmpdir.name) / "empty"
        empty_dir.mkdir()
        result = mcp_server.scan_sessions(since=None, projects_dir=str(empty_dir))
        self.assertEqual(result["session_count"], 0)
        self.assertEqual(result["sessions"], [])
        self.assertEqual(result["total_cost"], 0)

    def test_generate_report_writes_html_file_and_returns_path(self):
        output_path = Path(self.tmpdir.name) / "report.html"
        result = mcp_server.generate_report(
            since=None, output_path=str(output_path), projects_dir=str(self.projects_dir)
        )
        self.assertEqual(result["output"], str(output_path))
        self.assertTrue(output_path.exists())
        self.assertIn("AgentLens", output_path.read_text(encoding="utf-8"))

    def test_get_habit_score_returns_expected_shape(self):
        result = mcp_server.get_habit_score(since=None, projects_dir=str(self.projects_dir))
        self.assertEqual(result["session_count"], 1)
        self.assertIsNotNone(result["avg_habit_score"])
        self.assertIsInstance(result["finding_counts"], dict)
        self.assertEqual(len(result["worst_sessions"]), 1)
        self.assertIn("habit_score", result["worst_sessions"][0])

    def test_get_habit_score_empty_projects_dir(self):
        empty_dir = Path(self.tmpdir.name) / "empty"
        empty_dir.mkdir()
        result = mcp_server.get_habit_score(since=None, projects_dir=str(empty_dir))
        self.assertEqual(result["session_count"], 0)
        self.assertIsNone(result["avg_habit_score"])
        self.assertEqual(result["worst_sessions"], [])


if __name__ == "__main__":
    unittest.main()
