"""Agent tool definitions for AVO.

These tools give the AVO agent the ability to interact with its environment:
editing files, running shell commands, reading knowledge base docs, and
invoking the scoring function -- matching the capabilities described in
Section 3.2 and 4.1 of the paper.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from avo.agent.knowledge_base import KnowledgeBase
from avo.core.scoring import ScoringFunction
from avo.core.types import Score

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "The full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr. Use for compilation, testing, profiling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120).", "default": 120},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_solution",
            "description": "Run the scoring function on the current solution file. Returns correctness and performance scores.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the domain knowledge base for relevant documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query terms."},
                    "max_results": {"type": "integer", "description": "Max documents to return.", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge_doc",
            "description": "Read a specific document from the knowledge base by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Document name or path."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_docs",
            "description": "List all available documents in the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_lineage",
            "description": "View the full evolutionary lineage with all committed versions and their scores.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_version",
            "description": "Read the source code of a specific committed version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer", "description": "Version number to read."},
                },
                "required": ["version"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Submit the current solution for commitment. The solution will be evaluated; if it passes correctness and improves performance, it will be committed as a new version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Brief description of what was changed."},
                },
                "required": ["description"],
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against the workspace and environment."""

    def __init__(
        self,
        workspace_dir: Path,
        solution_file: str,
        scoring_fn: ScoringFunction,
        knowledge_base: KnowledgeBase,
        lineage_summary_fn: Any = None,
        read_version_fn: Any = None,
    ) -> None:
        self._workspace = workspace_dir
        self._solution_file = solution_file
        self._scoring_fn = scoring_fn
        self._kb = knowledge_base
        self._lineage_summary_fn = lineage_summary_fn
        self._read_version_fn = read_version_fn
        self._last_score: Score | None = None

    @property
    def last_score(self) -> Score | None:
        return self._last_score

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call and return the result as a string."""
        handlers = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_command": self._run_command,
            "evaluate_solution": self._evaluate_solution,
            "search_knowledge_base": self._search_kb,
            "read_knowledge_doc": self._read_kb_doc,
            "list_knowledge_docs": self._list_kb_docs,
            "view_lineage": self._view_lineage,
            "read_version": self._read_version,
            "submit_solution": self._submit_solution,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            return handler(**arguments)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return f"Error executing {tool_name}: {e}"

    def _read_file(self, path: str) -> str:
        full = self._workspace / path
        if not full.exists():
            return f"File not found: {path}"
        return full.read_text(errors="replace")

    def _write_file(self, path: str, content: str) -> str:
        full = self._workspace / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Written {len(content)} bytes to {path}"

    def _run_command(self, command: str, timeout: int = 120) -> str:
        logger.info("Executing command: %s", command[:200])
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._workspace),
            )
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            output += f"Exit code: {result.returncode}"
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"

    def _evaluate_solution(self) -> str:
        sol_path = self._workspace / self._solution_file
        if not sol_path.exists():
            return f"Solution file not found: {self._solution_file}"
        source_code = sol_path.read_text()
        score = self._scoring_fn.evaluate(source_code, self._solution_file)
        self._last_score = score
        return json.dumps(score.to_dict(), indent=2)

    def _search_kb(self, query: str, max_results: int = 3) -> str:
        docs = self._kb.search(query, max_results)
        if not docs:
            return "No matching documents found."
        parts = []
        for doc in docs:
            parts.append(f"--- {doc.path} ---\n{doc.content[:2000]}")
        return "\n\n".join(parts)

    def _read_kb_doc(self, name: str) -> str:
        doc = self._kb.get_document(name)
        if doc is None:
            return f"Document not found: {name}"
        return f"--- {doc.path} ---\n{doc.content}"

    def _list_kb_docs(self) -> str:
        docs = self._kb.list_documents()
        if not docs:
            return "Knowledge base is empty."
        return "Available documents:\n" + "\n".join(f"  - {d}" for d in docs)

    def _view_lineage(self) -> str:
        if self._lineage_summary_fn:
            return self._lineage_summary_fn()
        return "Lineage not available."

    def _read_version(self, version: int) -> str:
        if self._read_version_fn:
            code = self._read_version_fn(version)
            if code:
                return code
        return f"Version {version} not found."

    def _submit_solution(self, description: str = "") -> str:
        sol_path = self._workspace / self._solution_file
        if not sol_path.exists():
            return f"Solution file not found: {self._solution_file}"
        source_code = sol_path.read_text()
        score = self._scoring_fn.evaluate(source_code, self._solution_file)
        self._last_score = score

        result = {
            "description": description,
            "score": score.to_dict(),
            "passes_correctness": score.passes_correctness,
            "geomean": score.geomean,
        }
        return json.dumps(result, indent=2)
