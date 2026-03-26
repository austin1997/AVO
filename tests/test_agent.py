"""Tests for agent components: knowledge base, tools, LLM client."""

import tempfile
from pathlib import Path

import pytest

from avo.agent.knowledge_base import KnowledgeBase
from avo.agent.llm_client import create_llm_client
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
