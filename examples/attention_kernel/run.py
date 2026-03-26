#!/usr/bin/env python3
"""Run the attention kernel optimization AVO example.

Usage:
    python -m examples.attention_kernel.run
    python examples/attention_kernel/run.py
    python examples/attention_kernel/run.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avo.config import EvolutionConfig
from avo.core.evolution import EvolutionRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="AVO Attention Kernel Optimization")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = EvolutionConfig.from_yaml(args.config)
    runner = EvolutionRunner(config)
    runner.run()

    results = runner.get_results()
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
