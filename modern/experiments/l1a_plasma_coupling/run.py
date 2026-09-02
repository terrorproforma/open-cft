from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    arguments = parser.parse_args()
    result = run_experiment(arguments.output)
    summary = result["dataset"]["summary"]
    print(
        f"accepted={summary['accepted_count']} failed={summary['failed_count']} "
        f"compatible={summary['compatible_count']} plasma={summary['plasma_solve_count']}"
    )


if __name__ == "__main__":
    main()
