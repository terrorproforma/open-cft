"""Run the declared 128-case four-cell topology search."""

from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    output = Path(__file__).resolve().parent / "results"
    result = run_experiment(output)
    print(result["dataset"]["summary"])


if __name__ == "__main__":
    main()
