"""Tests for avo.config."""

import tempfile
from pathlib import Path

import pytest

from avo.config import EvolutionConfig, LLMConfig, LLMProvider, SupervisorConfig


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == LLMProvider.OPENAI
        assert cfg.model == "gpt-4o"

    def test_effective_base_url_default(self):
        cfg = LLMConfig(provider=LLMProvider.OLLAMA)
        assert cfg.effective_base_url() == "http://localhost:11434/v1"

    def test_effective_base_url_custom(self):
        cfg = LLMConfig(provider=LLMProvider.OPENAI, base_url="http://custom:8080/v1")
        assert cfg.effective_base_url() == "http://custom:8080/v1"

    def test_ollama_cloud_url(self):
        cfg = LLMConfig(provider=LLMProvider.OLLAMA_CLOUD)
        assert "ollamacloud" in cfg.effective_base_url()


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.max_versions == 100
        assert cfg.llm.provider == LLMProvider.OPENAI

    def test_yaml_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            cfg = EvolutionConfig(
                project_name="test-project",
                max_versions=50,
                llm=LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
            )
            cfg.to_yaml(path)

            cfg2 = EvolutionConfig.from_yaml(path)
            assert cfg2.project_name == "test-project"
            assert cfg2.max_versions == 50
            assert cfg2.llm.provider == LLMProvider.OLLAMA
            assert cfg2.llm.model == "llama3"

    def test_workspace_path(self):
        cfg = EvolutionConfig(workspace_dir="/tmp/test_workspace")
        assert cfg.workspace_path() == Path("/tmp/test_workspace")


class TestSupervisorConfig:
    def test_defaults(self):
        cfg = SupervisorConfig()
        assert cfg.enabled is True
        assert cfg.max_failed_attempts == 10
