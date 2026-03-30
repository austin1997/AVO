"""Multi-provider LLM client for AVO.

Supports OpenAI, Anthropic, Ollama (local), Ollama Cloud, and any
OpenAI-compatible endpoint. The agent's conversation history is sent
as the message list, serving as the persistent memory described in
Section 4.1 of the paper.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from avo.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._http = httpx.Client(timeout=config.timeout)

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Returns the full API response as a dict with at minimum:
        - "content": the assistant's text reply
        - "tool_calls": list of tool call dicts (if any)
        - "usage": token usage dict
        """

    def close(self) -> None:
        self._http.close()


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (also used by Ollama variants)."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.effective_base_url().rstrip("/")
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("LLM request to %s model=%s messages=%d", url, self.config.model, len(messages))

        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice.get("message", {})

        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason", ""),
            "raw": data,
        }


class AnthropicClient(BaseLLMClient):
    """Client for the Anthropic Messages API."""

    API_VERSION = "2023-06-01"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.effective_base_url().rstrip("/")
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
        }

        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                chat_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": chat_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if tools:
            payload["tools"] = self._convert_tools_to_anthropic(tools)

        logger.debug("Anthropic request model=%s messages=%d", self.config.model, len(chat_messages))

        resp = self._http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                })

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "usage": data.get("usage", {}),
            "finish_reason": data.get("stop_reason", ""),
            "raw": data,
        }

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool defs to Anthropic format."""
        result = []
        for tool in tools:
            func = tool.get("function", tool)
            result.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return result


class OllamaClient(OpenAIClient):
    """Client for local Ollama server.

    Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint,
    so this is a thin wrapper that sets the default base URL and does
    not require an API key.
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url:
            config = config.model_copy(update={"base_url": "http://localhost:11434/v1"})
        super().__init__(config)
        self._api_key = ""


class OllamaCloudClient(BaseLLMClient):
    """Client for Ollama Cloud API (https://ollama.com).

    Uses Ollama's native /api/chat endpoint with Bearer token auth.
    See https://docs.ollama.com/cloud for the API specification.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = config.effective_base_url().rstrip("/")
        self._api_key = config.api_key or os.environ.get("OLLAMA_API_KEY", "")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        ollama_messages = [self._to_ollama_message(m) for m in messages]

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        logger.debug("Ollama Cloud request to %s model=%s messages=%d", url, self.config.model, len(messages))

        resp = self._http.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            body = resp.text
            logger.error("Ollama Cloud error %d: %s", resp.status_code, body)
            resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                },
            })

        usage = {}
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

        return {
            "content": message.get("content", ""),
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": data.get("done_reason", ""),
            "raw": data,
        }

    @staticmethod
    def _to_ollama_message(msg: dict[str, Any]) -> dict[str, Any]:
        """Convert an internal (OpenAI-style) message to Ollama native format.

        Ollama's /api/chat expects tool_call arguments as objects (not JSON
        strings), and tool-result messages use only role + content (no
        tool_call_id).
        """
        result: dict[str, Any] = {"role": msg["role"]}

        if "content" in msg:
            result["content"] = msg["content"]

        if "tool_calls" in msg:
            ollama_tcs = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", tc)
                raw_args = func.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        raw_args = {}
                ollama_tcs.append({"function": {"name": func.get("name", ""), "arguments": raw_args}})
            result["tool_calls"] = ollama_tcs

        return result


def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Factory: create the appropriate LLM client based on provider config."""
    clients = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OLLAMA: OllamaClient,
        LLMProvider.OLLAMA_CLOUD: OllamaCloudClient,
        LLMProvider.CUSTOM: OpenAIClient,
    }
    client_cls = clients[config.provider]
    logger.info("Creating LLM client: provider=%s model=%s", config.provider.value, config.model)
    return client_cls(config)
