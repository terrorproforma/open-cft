"""Execute the preregistered held-out validation exactly once."""

from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    result = run_experiment(Path(__file__).resolve().parent / "results")
    print(result["dataset"]["summary"])


if __name__ == "__main__":
    main()

