"""AVO Agent / Variation Operator.

Implements Equation 4 from the paper:
    Vary(P_t) = Agent(P_t, K, f)

The agent is a self-directed coding agent that subsumes sampling, generation,
and evaluation into a single autonomous loop. Each variation step is a
multi-turn agent conversation with tool use (Section 3.2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from avo.agent.knowledge_base import KnowledgeBase
from avo.agent.llm_client import BaseLLMClient, create_llm_client
from avo.agent.prompts import SYSTEM_PROMPT, VARIATION_STEP_PROMPT, SUPERVISOR_DIRECTION_PROMPT
from avo.agent.tools import TOOL_DEFINITIONS, ToolExecutor
from avo.config import EvolutionConfig
from avo.core.population import Population
from avo.core.scoring import ScoringFunction
from avo.core.types import Score, Solution

logger = logging.getLogger(__name__)


class AVOAgent:
    """The Agentic Variation Operator.

    Replaces the fixed mutation/crossover pipeline of classical evolutionary
    search with an autonomous coding agent that can consult the current lineage,
    a domain-specific knowledge base, and execution feedback to propose, repair,
    critique, and verify implementation edits.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        population: Population,
        scoring_fn: ScoringFunction,
        knowledge_base: KnowledgeBase,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self._config = config
        self._population = population
        self._scoring_fn = scoring_fn
        self._kb = knowledge_base
        self._llm = llm_client or create_llm_client(config.llm)
        self._workspace = config.workspace_path()
        self._step_count = 0

        self._tool_executor = ToolExecutor(
            workspace_dir=self._workspace,
            solution_file=config.solution_file,
            scoring_fn=scoring_fn,
            knowledge_base=knowledge_base,
            lineage_summary_fn=population.summary,
            read_version_fn=self._read_committed_version,
        )

    def _read_committed_version(self, version: int) -> str | None:
        entry = self._population.get_entry(version)
        if entry:
            return entry.solution.source_code
        return None

    def variation_step(self, redirect_direction: str | None = None) -> Solution | None:
        """Execute a single variation step (Section 3.2).

        This is the core agent loop: the agent autonomously plans, implements,
        tests, and debugs until it produces an improved solution or exhausts
        its step budget.

        Returns the committed Solution on success, None if no improvement found.
        """
        self._step_count += 1
        logger.info("=== Variation step %d ===", self._step_count)

        messages = self._build_initial_messages(redirect_direction)

        committed = None
        for step in range(self._config.max_agent_steps_per_variation):
            logger.debug("Agent turn %d/%d", step + 1, self._config.max_agent_steps_per_variation)

            try:
                response = self._llm.chat(messages, tools=TOOL_DEFINITIONS)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                messages.append({"role": "user", "content": f"LLM error: {e}. Please try again."})
                continue

            assistant_content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if assistant_content:
                assistant_msg["content"] = assistant_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                logger.debug("Agent finished without tool calls")
                break

            for tc in tool_calls:
                func = tc.get("function", tc)
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                logger.info("Tool call: %s(%s)", tool_name, _truncate(str(args), 200))
                result = self._tool_executor.execute(tool_name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": _truncate(result, 8000),
                })

                if tool_name == "submit_solution":
                    committed = self._handle_submission(result)
                    if committed:
                        logger.info("Variation step %d: committed v%d", self._step_count, committed.version)
                        return committed

        logger.info("Variation step %d: no improvement committed", self._step_count)
        return None

    def _build_initial_messages(self, redirect_direction: str | None = None) -> list[dict[str, Any]]:
        """Construct the system + user messages for a variation step."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if redirect_direction:
            best = self._population.best_entry
            user_msg = SUPERVISOR_DIRECTION_PROMPT.format(
                direction=redirect_direction,
                best_version=best.solution.version if best else 0,
                best_geomean=best.score.geomean if best else 0.0,
            )
        else:
            best = self._population.best_entry
            user_msg = VARIATION_STEP_PROMPT.format(
                lineage_summary=self._population.summary(),
                scoring_context=self._scoring_fn.get_scoring_context(),
                kb_catalog=self._kb.catalog(),
                step_number=self._step_count,
                best_version=best.solution.version if best else 0,
                best_geomean=best.score.geomean if best else 0.0,
                solution_file=self._config.solution_file,
            )

        messages.append({"role": "user", "content": user_msg})
        return messages

    def _handle_submission(self, result_json: str) -> Solution | None:
        """Process a submit_solution result and try to commit."""
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            return None

        score = Score.from_dict(result.get("score", {}))
        if not score.passes_correctness:
            return None

        sol_path = self._workspace / self._config.solution_file
        if not sol_path.exists():
            return None
        source_code = sol_path.read_text()

        metadata = {"description": result.get("description", "")}
        return self._population.try_commit(source_code, score, metadata)

    @property
    def step_count(self) -> int:
        return self._step_count

    def close(self) -> None:
        self._llm.close()


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n... (truncated, {len(s)} total chars)"
