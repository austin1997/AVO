"""Pydantic-based configuration for the AVO framework."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OLLAMA_CLOUD = "ollama_cloud"
    CUSTOM = "custom"


class LLMConfig(BaseModel):
    """Configuration for the LLM backend."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 16384
    timeout: float = 120.0

    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        defaults = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com",
            LLMProvider.OLLAMA: "http://localhost:11434/v1",
            LLMProvider.OLLAMA_CLOUD: "https://api.ollamacloud.com/v1",
            LLMProvider.CUSTOM: "http://localhost:8000/v1",
        }
        return defaults[self.provider]


class SupervisorConfig(BaseModel):
    """Configuration for the self-supervision mechanism."""

    enabled: bool = True
    max_failed_attempts: int = 10
    stall_window: int = 5
    cycle_detection_window: int = 8
    redirect_with_llm: bool = True


class EvolutionConfig(BaseModel):
    """Top-level configuration for an AVO evolution run."""

    project_name: str = "avo-evolution"
    workspace_dir: str = "./workspace"
    solution_file: str = "solution.py"
    seed_file: str = ""

    max_versions: int = 100
    max_wall_time_hours: float = 168.0  # 7 days default, matching paper
    max_agent_steps_per_variation: int = 50

    llm: LLMConfig = Field(default_factory=LLMConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)

    knowledge_base_dir: str = ""
    scoring_module: str = ""
    scoring_class: str = "Scorer"

    git_persist: bool = True
    log_dir: str = "./logs"
    log_level: str = "INFO"
    verbose: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvolutionConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).resolve()

    def knowledge_base_path(self) -> Path | None:
        if not self.knowledge_base_dir:
            return None
        return Path(self.knowledge_base_dir).resolve()

    def log_path(self) -> Path:
        return Path(self.log_dir).resolve()
