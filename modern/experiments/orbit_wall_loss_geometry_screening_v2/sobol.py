"""Scrambled Sobol' sequences (dependency-free: the system interpreter has no scipy).

Base sequence: Joe & Kuo direction numbers (``new-joe-kuo-6.21201``) in Gray-code order,
32 binary digits. Scrambling: Matousek's linear matrix scrambling (a random lower-
triangular binary matrix with unit diagonal per dimension) followed by a random digital
shift, both drawn from ``numpy.random.default_rng`` seeded by the caller's 64-bit seed.
Both operations are GF(2)-linear in the digits, so they are applied once to the direction
numbers and the shift is XOR-ed into every point; the scrambled sequence keeps the
``(t, m, s)``-net property of every ``2**m``-point prefix (Owen 1998; Matousek 1998), which
is what makes the stage-2 top-up (indices 16..63 of the same sequence) an extension of the
stage-1 prefix (indices 0..15) instead of an independent sample.

Only the properties the campaign relies on are implemented: the sequence is deterministic
in ``(dimension, seed)``, any prefix of length ``2**m`` is balanced over the dyadic
intervals of each 1-D projection, and the first two dimensions form a ``(0, m, 2)``-net.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import numpy as np

BITS = 32
_SCALE = float(2**BITS)

# (s, a, m_1..m_s) for dimensions 2..8 from Joe & Kuo's ``new-joe-kuo-6.21201``.
# Dimension 1 is the van der Corput sequence (all direction numbers 1).
_JOE_KUO: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 1, 3, 3)),
    (4, 4, (1, 3, 5, 13)),
    (5, 2, (1, 1, 5, 5, 17)),
)
MAX_DIMENSION = 1 + len(_JOE_KUO)


def direction_numbers(dimension_index: int) -> list[int]:
    """Unscrambled direction numbers ``v_1..v_32`` (as 32-bit integers) of one dimension (0-based)."""

    if not 0 <= dimension_index < MAX_DIMENSION:
        raise ValueError(f"dimension index {dimension_index} outside 0..{MAX_DIMENSION - 1}")
    if dimension_index == 0:
        return [1 << (BITS - i) for i in range(1, BITS + 1)]
    s, a, m = _JOE_KUO[dimension_index - 1]
    v = [0] * (BITS + 1)
    for i in range(1, s + 1):
        v[i] = m[i - 1] << (BITS - i)
    for i in range(s + 1, BITS + 1):
        v[i] = v[i - s] ^ (v[i - s] >> s)
        for k in range(1, s):
            if (a >> (s - 1 - k)) & 1:
                v[i] ^= v[i - k]
    return v[1:]


def seed_from_bytes(material: bytes) -> int:
    """64-bit seed from arbitrary bytes (first 8 bytes of SHA-256)."""

    return int.from_bytes(sha256(material).digest()[:8], "big")


def _scramble_rows(rng: np.random.Generator) -> list[int]:
    """Row masks of a random lower-triangular 32x32 binary matrix with unit diagonal.

    Digit ``i`` (0 = most significant) of the output is the parity of ``mask_i & x``;
    ``mask_i`` covers digits ``k < i`` at random and digit ``i`` always.
    """

    random_rows = rng.integers(0, 2**BITS, size=BITS, dtype=np.uint64)
    rows = []
    for i in range(BITS):
        high = ((1 << i) - 1) << (BITS - i)
        rows.append((int(random_rows[i]) & high) | (1 << (BITS - 1 - i)))
    return rows


def _apply_rows(rows: Sequence[int], x: int) -> int:
    y = 0
    for i, mask in enumerate(rows):
        if (mask & x).bit_count() & 1:
            y |= 1 << (BITS - 1 - i)
    return y


def scrambled_sobol(dimension: int, count: int, seed: int, *, start: int = 0) -> np.ndarray:
    """Points ``start .. start+count-1`` of the scrambled sequence, shape ``(count, dimension)`` in ``[0, 1)``.

    The scramble is a deterministic function of ``(dimension, seed)``; ``start`` selects a
    window of the same sequence, so ``scrambled_sobol(d, 64, seed)[16:]`` equals
    ``scrambled_sobol(d, 48, seed, start=16)``.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError(f"dimension must lie in 1..{MAX_DIMENSION}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if start + count > 2**BITS - 1:
        raise ValueError("sequence window exceeds the 32-bit period")
    rng = np.random.default_rng(seed)
    scrambled_directions: list[list[int]] = []
    shifts: list[int] = []
    for j in range(dimension):
        rows = _scramble_rows(rng)
        scrambled_directions.append([_apply_rows(rows, v) for v in direction_numbers(j)])
        shifts.append(int(rng.integers(0, 2**BITS, dtype=np.uint64)))
    output = np.empty((count, dimension), dtype=np.float64)
    if count == 0:
        return output
    # Gray-code state at index ``start``: X(n) = XOR of v_k over the set bits of gray(n).
    gray = start ^ (start >> 1)
    state = []
    for j in range(dimension):
        x = 0
        bit = 0
        g = gray
        while g:
            if g & 1:
                x ^= scrambled_directions[j][bit]
            g >>= 1
            bit += 1
        state.append(x)
    for row in range(count):
        n = start + row
        for j in range(dimension):
            output[row, j] = (state[j] ^ shifts[j]) / _SCALE
        # advance: X(n+1) = X(n) ^ v_c where c = index of the lowest zero bit of n.
        c = 0
        while (n >> c) & 1:
            c += 1
        for j in range(dimension):
            state[j] ^= scrambled_directions[j][c]
    return output


def dyadic_balance(points: np.ndarray, resolution_bits: int) -> bool:
    """True iff every dyadic interval of width ``2**-resolution_bits`` in every 1-D projection holds ``n / 2**resolution_bits`` points."""

    n, dimension = points.shape
    cells = 2**resolution_bits
    if n % cells:
        raise ValueError("point count must be a multiple of the interval count")
    expected = n // cells
    for j in range(dimension):
        counts = np.bincount(np.floor(points[:, j] * cells).astype(int), minlength=cells)
        if len(counts) != cells or np.any(counts != expected):
            return False
    return True
