"""Check an exact-order replacement for PIC2D's quadratic Coulomb rank stage.

This is a review experiment, not a production backend or a GPU speed benchmark.
Run from modern/: PYTHONPATH=src python -m tools.coulomb_ordering_probe
The existing Warp rank kernel and Warp 1.14's radix sort run on CPU. Synthetic
cells include dead slots, spare capacity, empty cells, and concentrated loads.
"""

from __future__ import annotations

import json

import numpy as np
import warp as wp

from cft_revival.pic2d.warp_coulomb import coulomb_rank_kernel


def check_case(name: str, cells: np.ndarray, n_cells: int, seed: int) -> dict:
    """Compare the full live permutation before the existing pairing shuffle."""
    capacity = len(cells)
    live = np.flatnonzero(cells >= 0).astype(np.int32)
    counts = np.bincount(cells[live], minlength=n_cells).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    expected = live[np.lexsort((live, cells[live]))]
    # Emulate arbitrary atomic arrival order in each cell's provisional segment.
    rng = np.random.default_rng(seed)
    provisional = np.zeros(capacity, dtype=np.int32)
    for c in range(n_cells):
        provisional[starts[c]:starts[c + 1]] = rng.permutation(live[cells[live] == c])

    old_slots = wp.zeros(capacity, dtype=wp.int32, device="cpu")
    old_cells = wp.zeros(capacity, dtype=wp.int32, device="cpu")
    wp.launch(
        coulomb_rank_kernel, dim=capacity,
        inputs=[wp.array(cells, dtype=wp.int32, device="cpu"),
                wp.array(starts, dtype=wp.int32, device="cpu"),
                wp.array(provisional, dtype=wp.int32, device="cpu"), old_slots, old_cells],
        device="cpu",
    )

    # Positive int64 composite keys sort by cell, then ORIGINAL slot. Inactive
    # slots sort after all live slots. Particle arrays and RNG keys never move.
    key_host = np.full(2 * capacity, np.iinfo(np.int64).max, dtype=np.int64)
    key_host[live] = cells[live].astype(np.int64) * (1 << 32) + live.astype(np.int64)
    value_host = np.zeros(2 * capacity, dtype=np.int32)
    value_host[:capacity] = np.arange(capacity, dtype=np.int32)
    keys = wp.array(key_host, dtype=wp.int64, device="cpu")
    values = wp.array(value_host, dtype=wp.int32, device="cpu")
    wp.utils.radix_sort_pairs(keys, values, capacity)
    actual = values.numpy()[:len(live)]
    np.testing.assert_array_equal(old_slots.numpy()[:len(live)], expected)
    np.testing.assert_array_equal(old_cells.numpy()[:len(live)], cells[expected])
    np.testing.assert_array_equal(actual, expected)

    return {
        "case": name, "capacity": capacity, "live": len(live),
        "max_cell_occupancy": int(counts.max(initial=0)),
        "old_rank_comparisons": sum(int(k) ** 2 for k in counts),
        "exact_live_permutation": True,
        "radix_key_value_storage_bytes": int(2 * capacity * (8 + 4)),
    }


def main() -> None:
    wp.init()
    rng = np.random.default_rng(20260905)
    cases = [
        ("all_inactive", np.full(32, -1, dtype=np.int32), 16),
        ("one_live_with_holes", np.array([-1, 3, -1, -1], dtype=np.int32), 16),
    ]
    for occupancy in (4, 32, 256):
        live_cells = np.repeat(np.arange(16, dtype=np.int32), occupancy)
        cells = np.concatenate((live_cells, np.full(len(live_cells) // 2, -1, dtype=np.int32)))
        rng.shuffle(cells)
        cases.append((f"balanced_k{occupancy}", cells, 16))
    hotspot = rng.integers(0, 64, size=8192, dtype=np.int32)
    hotspot[:2048] = 0
    hotspot[4096:6144] = -1
    cases.append(("dense_hotspot", hotspot, 64))
    results = [check_case(name, cells, n_cells, 7) for name, cells, n_cells in cases]
    print(json.dumps({
        "scope": "CPU ordering proof only; no CUDA performance or graph qualification",
        "warp_version": wp.__version__, "numpy_version": np.__version__,
        "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
