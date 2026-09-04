"""Plume development run v2.1: model v2.0.2 physics on an axially extended plume box (channel + 12 x 24 mm plume,
far plane at z = 48 mm) with the static B field from the domain-padding-1.5 P2 solution
(``spec/pic2d/p2-field-plume-extension-v2.json``).  PREPARED, NOT LAUNCHED (2026-09-04).

Thin wrapper over the shared runner in ``experiments.pic2d_cft_steady_state_v1.run`` with this
experiment's protocol and results directory.  From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_plume_v2_1.run run        # start / resume (fresh start: no v2.0.x checkpoint is resumable)
    python -m experiments.pic2d_cft_plume_v2_1.run status
    python -m experiments.pic2d_cft_plume_v2_1.run finalize

Development/screening: not preregistered, no validated physics claim, not a performance
prediction (thrust, Isp, efficiency, divergence and IEDF are development numbers).  Do not
launch while another PIC run holds the GPU (attempt 8, PID 51256, until ~20:00 AEST 2026-09-04)
or while another host factorisation is running.
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
