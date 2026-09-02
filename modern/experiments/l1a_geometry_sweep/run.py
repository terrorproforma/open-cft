"""Command-line entry point for the reproducible L1a geometry sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--parity-count", type=int, default=6)
    arguments = parser.parse_args()
    result = run_experiment(
        arguments.output,
        count=arguments.count,
        seed=arguments.seed,
        parity_count=arguments.parity_count,
    )
    summary = result["dataset"]["summary"]
    print(
        json.dumps(
            {
                "classification": result["dataset"]["classification"],
                "evaluated_count": summary["evaluated_count"],
                "failed_count": summary["failed_count"],
                "feasible_count": summary["feasible_count"],
                "nondominated_count": summary["nondominated_count"],
                "output": str(arguments.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
