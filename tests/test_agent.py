"""Tests for agent components: knowledge base, tools, LLM client."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from avo.agent.knowledge_base import KnowledgeBase
from avo.agent.llm_client import OllamaCloudClient, create_llm_client
from avo.agent.tools import ToolExecutor
from avo.config import LLMConfig, LLMProvider
from avo.core.types import Score


class TestKnowledgeBase:
    def test_from_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase.from_directory(tmpdir)
            assert len(kb.documents) == 0

    def test_load_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc1.md").write_text("# Hello\nThis is about sorting algorithms.")
            (Path(tmpdir) / "doc2.txt").write_text("Optimization techniques for CUDA kernels.")
            (Path(tmpdir) / "ignored.png").write_bytes(b"\x89PNG")

            kb = KnowledgeBase.from_directory(tmpdir)
            assert len(kb.documents) == 2

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "sorting.md").write_text("Quicksort partitions the array recursively.")
            (Path(tmpdir) / "cuda.md").write_text("CUDA threads execute in warps of 32.")

            kb = KnowledgeBase.from_directory(tmpdir)
            results = kb.search("quicksort partition")
            assert len(results) > 0
            assert results[0].name == "sorting.md"

    def test_get_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("Test content")
            kb = KnowledgeBase.from_directory(tmpdir)
            doc = kb.get_document("test.md")
            assert doc is not None
            assert doc.content == "Test content"

    def test_catalog(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.md").write_text("Content")
            kb = KnowledgeBase.from_directory(tmpdir)
            catalog = kb.catalog()
            assert "doc.md" in catalog

    def test_nonexistent_directory(self):
        kb = KnowledgeBase.from_directory("/nonexistent/path")
        assert len(kb.documents) == 0


class TestToolExecutor:
    def _make_dummy_scorer(self):
        from avo.core.scoring import ScoringFunction

        class DummyScorer(ScoringFunction):
            def evaluate(self, source_code, source_file="solution.py"):
                return Score(values={"test": 1.0}, passes_correctness=True, correctness_message="OK")

            def get_configurations(self):
                return ["test"]

        return DummyScorer()

    def test_read_write_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scorer = self._make_dummy_scorer()
            kb = KnowledgeBase()
            executor = ToolExecutor(workspace, "solution.py", scorer, kb)

            result = executor.execute("write_file", {"path": "test.txt", "content": "hello"})
            assert "Written" in result

            result = executor.execute("read_file", {"path": "test.txt"})
            assert result == "hello"

    def test_read_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scorer = self._make_dummy_scorer()
            kb = KnowledgeBase()
            executor = ToolExecutor(workspace, "solution.py", scorer, kb)

            result = executor.execute("read_file", {"path": "nope.txt"})
            assert "not found" in result.lower()

    def test_run_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scorer = self._make_dummy_scorer()
            kb = KnowledgeBase()
            executor = ToolExecutor(workspace, "solution.py", scorer, kb)

            result = executor.execute("run_command", {"command": "echo hello"})
            assert "hello" in result

    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scorer = self._make_dummy_scorer()
            kb = KnowledgeBase()
            executor = ToolExecutor(workspace, "solution.py", scorer, kb)

            result = executor.execute("nonexistent_tool", {})
            assert "unknown" in result.lower()


class TestLLMClientFactory:
    def test_create_ollama_client(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        client = create_llm_client(config)
        assert client is not None
        assert client.config.model == "llama3"
        client.close()

    def test_create_ollama_cloud_client(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA_CLOUD, model="llama3", api_key="test-key")
        client = create_llm_client(config)
        assert client is not None
        client.close()

    def test_create_openai_client(self):
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o")
        client = create_llm_client(config)
        assert client is not None
        client.close()

    def test_create_anthropic_client(self):
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-sonnet-20240229")
        client = create_llm_client(config)
        assert client is not None
        client.close()


def _mock_ollama_response(content="Hello!", tool_calls=None, **extra):
    """Build a mock httpx.Response that looks like Ollama's /api/chat."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body = {
        "model": "test-model",
        "created_at": "2026-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 20,
        **extra,
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


class TestOllamaCloudClient:
    """Tests for OllamaCloudClient: endpoint, payload format, message
    conversion, response parsing, and multi-turn tool conversations."""

    def _make_client(self, **overrides):
        defaults = dict(provider=LLMProvider.OLLAMA_CLOUD, model="test-model", api_key="test-key")
        defaults.update(overrides)
        return OllamaCloudClient(LLMConfig(**defaults))

    def test_uses_api_chat_endpoint(self):
        """Must hit /api/chat, not the OpenAI-compatible /chat/completions."""
        client = self._make_client()
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
            url = mock_post.call_args[1].get("url") or mock_post.call_args[0][0]
            assert url.endswith("/api/chat")
            assert "/chat/completions" not in url

    def test_default_base_url(self):
        client = self._make_client()
        assert client._base_url == "https://ollama.com"

    def test_custom_base_url(self):
        client = self._make_client(base_url="http://my-server:11434")
        assert client._base_url == "http://my-server:11434"

    def test_bearer_auth_header(self):
        client = self._make_client(api_key="my-secret")
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer my-secret"

    @patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=False)
    def test_api_key_from_env(self):
        """Uses OLLAMA_API_KEY (not the old OLLAMA_CLOUD_API_KEY)."""
        client = OllamaCloudClient(LLMConfig(provider=LLMProvider.OLLAMA_CLOUD, model="m"))
        assert client._api_key == "env-key"

    @patch.dict("os.environ", {}, clear=False)
    def test_no_auth_header_without_key(self):
        config = LLMConfig(provider=LLMProvider.OLLAMA_CLOUD, model="m", api_key="")
        # Remove env var if present
        import os
        os.environ.pop("OLLAMA_API_KEY", None)
        client = OllamaCloudClient(config)
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
            headers = mock_post.call_args[1]["headers"]
            assert "Authorization" not in headers

    def test_payload_ollama_native_format(self):
        """Payload must use Ollama fields: stream, options.temperature,
        options.num_predict — not OpenAI's top-level temperature/max_tokens."""
        client = self._make_client(temperature=0.5, max_tokens=1024)
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
            payload = mock_post.call_args[1]["json"]
            assert payload["stream"] is False
            assert payload["options"]["temperature"] == 0.5
            assert payload["options"]["num_predict"] == 1024
            assert "temperature" not in payload
            assert "max_tokens" not in payload

    def test_tools_passed_through(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
        client = self._make_client()
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}], tools=tools)
            payload = mock_post.call_args[1]["json"]
            assert payload["tools"] == tools

    def test_no_tools_key_when_none(self):
        client = self._make_client()
        with patch.object(client._http, "post", return_value=_mock_ollama_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
            payload = mock_post.call_args[1]["json"]
            assert "tools" not in payload

    def test_parse_text_response(self):
        client = self._make_client()
        with patch.object(client._http, "post", return_value=_mock_ollama_response("The answer is 42.")):
            result = client.chat([{"role": "user", "content": "hi"}])
            assert result["content"] == "The answer is 42."
            assert result["tool_calls"] == []
            assert result["finish_reason"] == "stop"
            assert result["usage"]["prompt_tokens"] == 10
            assert result["usage"]["completion_tokens"] == 20
            assert result["usage"]["total_tokens"] == 30

    def test_parse_tool_call_response(self):
        """Ollama returns tool_call arguments as a dict; we must normalise
        them to JSON strings for the internal format."""
        ollama_tool_calls = [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ]
        client = self._make_client()
        with patch.object(client._http, "post", return_value=_mock_ollama_response("", ollama_tool_calls)):
            result = client.chat([{"role": "user", "content": "weather?"}])
            tc = result["tool_calls"][0]
            assert tc["function"]["name"] == "get_weather"
            assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

    # -- _to_ollama_message conversion tests (the exact bug that caused 400) --

    def test_convert_simple_user_message(self):
        msg = {"role": "user", "content": "hello"}
        assert OllamaCloudClient._to_ollama_message(msg) == {"role": "user", "content": "hello"}

    def test_convert_system_message(self):
        msg = {"role": "system", "content": "You are helpful."}
        assert OllamaCloudClient._to_ollama_message(msg) == {"role": "system", "content": "You are helpful."}

    def test_convert_assistant_with_tool_calls_strips_id_and_type(self):
        """OpenAI-style tool_calls have id/type fields; Ollama must not see them."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
                }
            ],
        }
        converted = OllamaCloudClient._to_ollama_message(msg)
        assert converted["role"] == "assistant"
        tc = converted["tool_calls"][0]
        assert "id" not in tc
        assert "type" not in tc
        assert tc["function"]["name"] == "read_file"
        assert tc["function"]["arguments"] == {"path": "x.py"}

    def test_convert_tool_call_arguments_from_json_string_to_dict(self):
        """The internal format stores arguments as JSON strings; Ollama
        expects them as dicts. This mismatch caused the 400 Bad Request."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "f", "arguments": '{"a": 1, "b": "two"}'}}
            ],
        }
        converted = OllamaCloudClient._to_ollama_message(msg)
        args = converted["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"a": 1, "b": "two"}

    def test_convert_tool_call_arguments_already_dict(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "f", "arguments": {"key": "val"}}}
            ],
        }
        converted = OllamaCloudClient._to_ollama_message(msg)
        assert converted["tool_calls"][0]["function"]["arguments"] == {"key": "val"}

    def test_convert_tool_call_malformed_arguments_fallback(self):
        msg = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "f", "arguments": "not valid json{"}}
            ],
        }
        converted = OllamaCloudClient._to_ollama_message(msg)
        assert converted["tool_calls"][0]["function"]["arguments"] == {}

    def test_convert_tool_result_strips_tool_call_id(self):
        """Ollama tool messages only have role + content; tool_call_id is
        an OpenAI-ism that triggers 400 if sent."""
        msg = {"role": "tool", "tool_call_id": "call_abc123", "content": "file contents"}
        converted = OllamaCloudClient._to_ollama_message(msg)
        assert converted == {"role": "tool", "content": "file contents"}
        assert "tool_call_id" not in converted

    def test_multi_turn_tool_conversation_payload(self):
        """Reproduce the exact scenario that caused the 400: a first call
        returns tool_calls, tool results are appended, and the full
        conversation is sent for the second call. The second payload must
        have Ollama-native messages (no id/type/tool_call_id, arguments
        as dicts)."""
        client = self._make_client()

        # Simulate first-call response with tool calls
        first_response = _mock_ollama_response(
            content="",
            tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "sol.py"}}}],
        )
        second_response = _mock_ollama_response(content="Done!")

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return first_response if call_count == 1 else second_response

        with patch.object(client._http, "post", side_effect=mock_post) as mock_post_fn:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Optimize the code."},
            ]

            r1 = client.chat(messages, tools=[{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}])

            # Build conversation as variation_operator does (OpenAI-style)
            messages.append({"role": "assistant", "content": "", "tool_calls": r1["tool_calls"]})
            messages.append({"role": "tool", "tool_call_id": r1["tool_calls"][0].get("id", ""), "content": "def sort(arr): ..."})

            r2 = client.chat(messages)

            # Verify the second call's payload has Ollama-native messages
            second_payload = mock_post_fn.call_args_list[1][1]["json"]
            sent_messages = second_payload["messages"]

            # Assistant message: no id/type, arguments as dict
            assistant_msg = sent_messages[2]
            assert assistant_msg["role"] == "assistant"
            tc = assistant_msg["tool_calls"][0]
            assert "id" not in tc
            assert "type" not in tc
            assert isinstance(tc["function"]["arguments"], dict)

            # Tool result message: no tool_call_id
            tool_msg = sent_messages[3]
            assert tool_msg["role"] == "tool"
            assert "tool_call_id" not in tool_msg
            assert tool_msg["content"] == "def sort(arr): ..."
