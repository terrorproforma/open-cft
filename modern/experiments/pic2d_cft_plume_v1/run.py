"""Plume development run v1: model v2.0 (channel + 12 x 12 mm plume box, cathode emission region,
two-zone neutrals, momentum / thrust ledgers, plume-boundary gate) on top of v1.4.

Thin wrapper over the shared runner in ``experiments.pic2d_cft_steady_state_v1.run``
with this experiment's protocol and results directory.  From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_plume_v1.run run        # start / resume
    python -m experiments.pic2d_cft_plume_v1.run status
    python -m experiments.pic2d_cft_plume_v1.run finalize

Development/screening: not preregistered, no validated physics claim, not a performance
prediction (thrust, Isp, efficiency, divergence and IEDF are development numbers).
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
