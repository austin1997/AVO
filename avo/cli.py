"""CLI entry point for AVO."""

from __future__ import annotations

import argparse
import json
import sys

from avo.config import EvolutionConfig
from avo.core.evolution import EvolutionRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="avo",
        description="AVO: Agentic Variation Operators for Autonomous Evolutionary Search",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an evolution from a YAML config")
    run_parser.add_argument("config", help="Path to the YAML config file")

    init_parser = subparsers.add_parser("init", help="Create a default config file")
    init_parser.add_argument("-o", "--output", default="avo_config.yaml", help="Output path")

    args = parser.parse_args()

    if args.command == "run":
        config = EvolutionConfig.from_yaml(args.config)
        runner = EvolutionRunner(config)
        runner.run()
        results = runner.get_results()
        print(json.dumps(results, indent=2))

    elif args.command == "init":
        config = EvolutionConfig()
        config.to_yaml(args.output)
        print(f"Default config written to {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
