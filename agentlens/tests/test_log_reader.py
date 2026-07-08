import json
import tempfile
import unittest
from pathlib import Path

from agentlens.log_reader import parse_session


def _write_jsonl(lines: list, path: Path) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _assistant_event(msg_id, ts, model, usage, content):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {"id": msg_id, "model": model, "role": "assistant", "usage": usage, "content": content},
    }


class TestParseSession(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name="session.jsonl"):
        return Path(self.tmpdir.name) / name

    def test_dedupes_repeated_blocks_of_same_message(self):
        """One API response streamed as 3 JSONL lines (thinking, text, tool_use) must
        collapse into exactly one Turn — summing per-line would triple-count tokens."""
        usage = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        lines = [
            _assistant_event("msg_1", "2026-07-08T10:00:00Z", "claude-sonnet-5", usage, [{"type": "thinking", "thinking": "..."}]),
            _assistant_event("msg_1", "2026-07-08T10:00:01Z", "claude-sonnet-5", usage, [{"type": "text", "text": "hi"}]),
            _assistant_event(
                "msg_1", "2026-07-08T10:00:02Z", "claude-sonnet-5", usage,
                [{"type": "tool_use", "name": "Read", "input": {"file_path": "/a.py"}}],
            ),
        ]
        path = self._path()
        _write_jsonl(lines, path)

        session = parse_session(path)

        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session.turns[0].input_tokens, 100)
        self.assertEqual(session.turns[0].output_tokens, 50)
        self.assertEqual(len(session.turns[0].tool_calls), 1)
        self.assertEqual(session.turns[0].tool_calls[0].name, "Read")

    def test_two_distinct_messages_produce_two_turns(self):
        usage = {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        lines = [
            _assistant_event("msg_1", "2026-07-08T10:00:00Z", "claude-opus-4-8", usage, [{"type": "text", "text": "a"}]),
            _assistant_event("msg_2", "2026-07-08T10:00:05Z", "claude-opus-4-8", usage, [{"type": "text", "text": "b"}]),
        ]
        path = self._path()
        _write_jsonl(lines, path)

        session = parse_session(path)
        self.assertEqual(len(session.turns), 2)

    def test_skips_non_assistant_and_malformed_lines(self):
        path = self._path()
        raw_lines = [
            json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
            "{not valid json",
            json.dumps(
                _assistant_event(
                    "msg_1", "2026-07-08T10:00:00Z", "claude-haiku-4-5",
                    {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                    [{"type": "text", "text": "ok"}],
                )
            ),
            "",
        ]
        path.write_text("\n".join(raw_lines), encoding="utf-8")

        session = parse_session(path)
        self.assertEqual(len(session.turns), 1)

    def test_splits_cache_writes_by_ttl(self):
        usage = {
            "input_tokens": 2, "output_tokens": 10,
            "cache_creation_input_tokens": 500, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 200},
        }
        lines = [_assistant_event("msg_1", "2026-07-08T10:00:00Z", "claude-sonnet-5", usage, [{"type": "text", "text": "hi"}])]
        path = self._path()
        _write_jsonl(lines, path)

        session = parse_session(path)
        turn = session.turns[0]
        self.assertEqual(turn.cache_creation_input_tokens, 500)
        self.assertEqual(turn.cache_creation_5m_input_tokens, 300)
        self.assertEqual(turn.cache_creation_1h_input_tokens, 200)

    def test_cache_write_without_ttl_breakdown_assumed_5m(self):
        """Logs predating the ephemeral_*_input_tokens breakdown only have the aggregate
        cache_creation_input_tokens — that must still be priced, as a 5m write."""
        usage = {"input_tokens": 2, "output_tokens": 10, "cache_creation_input_tokens": 400, "cache_read_input_tokens": 0}
        lines = [_assistant_event("msg_1", "2026-07-08T10:00:00Z", "claude-sonnet-5", usage, [{"type": "text", "text": "hi"}])]
        path = self._path()
        _write_jsonl(lines, path)

        session = parse_session(path)
        turn = session.turns[0]
        self.assertEqual(turn.cache_creation_5m_input_tokens, 400)
        self.assertEqual(turn.cache_creation_1h_input_tokens, 0)

    def test_ignores_assistant_message_without_usage(self):
        path = self._path()
        line = {
            "type": "assistant",
            "timestamp": "2026-07-08T10:00:00Z",
            "message": {"id": "msg_1", "model": "claude-opus-4-8", "role": "assistant", "content": [{"type": "text", "text": "x"}]},
        }
        _write_jsonl([line], path)

        session = parse_session(path)
        # No usage dict present at all -> current_usage defaults to {} -> counted as a
        # zero-token turn (not silently dropped), since it still has a message id.
        self.assertEqual(len(session.turns), 1)
        self.assertEqual(session.turns[0].input_tokens, 0)


if __name__ == "__main__":
    unittest.main()
