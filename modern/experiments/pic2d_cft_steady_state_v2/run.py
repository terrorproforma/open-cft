"""Steady-state run v2: model v1.3 (quasi-steady 0-D neutral inventory) on the divergent-exit channel.

Thin wrapper over the shared runner in ``experiments.pic2d_cft_steady_state_v1.run``
with this experiment's protocol and results directory.  From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_steady_state_v2.run run        # start / resume
    python -m experiments.pic2d_cft_steady_state_v2.run status
    python -m experiments.pic2d_cft_steady_state_v2.run finalize

Development/screening: not preregistered, no validated physics claim.
"""

from __future__ import annotations

from pathlib import Path
import sys

from experiments.pic2d_cft_steady_state_v1 import run as runner

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"


def main(argv: list[str] | None = None) -> int:
    return runner.main(argv, protocol_path=PROTOCOL_PATH, results=RESULTS)


if __name__ == "__main__":
    sys.exit(main())
