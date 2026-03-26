# AVO: Agentic Variation Operators for Autonomous Evolutionary Search

A Python implementation of the AVO framework from [arXiv:2603.24517](https://arxiv.org/abs/2603.24517).

AVO replaces the fixed mutation and crossover heuristics of classical evolutionary search with an autonomous LLM-powered coding agent. The agent iteratively plans, implements, tests, and debugs code optimizations, using a domain-specific knowledge base, the full lineage of prior solutions, and execution feedback to guide its search.

## Architecture

```
Vary(P_t) = Agent(P_t, K, f)
```

Where:
- **P_t** is the lineage of all committed solutions and their scores
- **K** is a domain-specific knowledge base (docs, reference code)
- **f** is the scoring function (correctness + performance)

The agent autonomously decides what to consult, what to edit, and when to evaluate. Each committed version is persisted as a git commit with score metadata.

### Key Components

| Component | Description |
|---|---|
| `avo/core/types.py` | Solution, Score, Lineage data types |
| `avo/core/population.py` | Single-lineage population management |
| `avo/core/scoring.py` | Abstract scoring function interface |
| `avo/core/evolution.py` | Main continuous evolution loop |
| `avo/agent/variation_operator.py` | The AVO agent (core variation operator) |
| `avo/agent/llm_client.py` | Multi-provider LLM client |
| `avo/agent/tools.py` | Agent tools (file edit, shell, eval) |
| `avo/agent/knowledge_base.py` | Knowledge base loader/retriever |
| `avo/supervisor/self_supervision.py` | Stagnation detection and redirection |
| `avo/persistence/git_backend.py` | Git-based lineage persistence |

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For GPU-based examples (attention kernel):

```bash
pip install -e ".[gpu]"
```

## Quick Start

### 1. Create a config file

```bash
avo init -o my_config.yaml
```

### 2. Set up your optimization target

You need three things:
- A **seed program** (the starting point for evolution)
- A **scoring function** (implements `avo.core.scoring.ScoringFunction`)
- A **knowledge base** directory (optional, domain-specific docs)

### 3. Configure your LLM provider

Edit the config YAML to set your LLM provider:

```yaml
llm:
  provider: openai          # openai, anthropic, ollama, ollama_cloud, custom
  model: gpt-4o
  api_key: ""               # or set OPENAI_API_KEY env var
  temperature: 0.7
  max_tokens: 16384
```

For local Ollama:

```yaml
llm:
  provider: ollama
  model: llama3
  base_url: "http://localhost:11434/v1"
```

For Ollama Cloud:

```yaml
llm:
  provider: ollama_cloud
  model: llama3
  api_key: "your-api-key"   # or set OLLAMA_CLOUD_API_KEY env var
```

### 4. Run the evolution

```bash
avo run my_config.yaml
```

Or use the Python API:

```python
from avo.config import EvolutionConfig
from avo.core.evolution import EvolutionRunner

config = EvolutionConfig.from_yaml("my_config.yaml")
runner = EvolutionRunner(config)
runner.run()
print(runner.get_results())
```

## Examples

### Sorting Optimization (CPU-only)

Evolves a naive bubble sort into a faster sorting algorithm:

```bash
python examples/sorting_optimization/run.py
```

### Attention Kernel Optimization (GPU required)

Optimizes attention forward pass, matching the paper's benchmark setup:

```bash
python examples/attention_kernel/run.py
```

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `project_name` | `avo-evolution` | Name for this evolution run |
| `workspace_dir` | `./workspace` | Working directory for the agent |
| `solution_file` | `solution.py` | Filename for the evolving solution |
| `seed_file` | (required) | Path to the initial seed program |
| `max_versions` | `100` | Maximum committed versions |
| `max_wall_time_hours` | `168.0` | Maximum wall-clock time (7 days) |
| `max_agent_steps_per_variation` | `50` | Max LLM turns per variation step |
| `llm.provider` | `openai` | LLM provider |
| `llm.model` | `gpt-4o` | Model name |
| `supervisor.enabled` | `true` | Enable self-supervision |
| `supervisor.max_failed_attempts` | `10` | Failures before redirect |
| `knowledge_base_dir` | (optional) | Path to knowledge base docs |
| `scoring_module` | (required) | Path to scoring function module |
| `scoring_class` | `Scorer` | Class name in scoring module |
| `git_persist` | `true` | Persist versions as git commits |

## How It Works

The evolution loop:

1. **Initialize**: Load seed program as v0
2. **Variation Step**: The AVO agent autonomously:
   - Reviews the lineage and current best solution
   - Consults the knowledge base for optimization ideas
   - Edits the solution file
   - Runs the scoring function to check correctness and performance
   - Iterates (edit-evaluate-diagnose) until an improvement is found
   - Submits the improved solution for commitment
3. **Commit**: If the solution passes correctness and improves the geomean score, it becomes the new best version and is persisted as a git commit
4. **Self-Supervision**: If the agent stalls (too many failed attempts), the supervisor analyzes the trajectory and redirects exploration toward fresh optimization directions
5. **Repeat** until max versions or wall time is reached

## Running Tests

```bash
pytest tests/ -v
```

## References

- [AVO Paper (arXiv:2603.24517)](https://arxiv.org/abs/2603.24517) - Chen et al., NVIDIA, 2026
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) - Baseline attention kernel
- [AlphaEvolve](https://arxiv.org/abs/2506.13131) - Prior LLM-augmented evolutionary search

## License

MIT
