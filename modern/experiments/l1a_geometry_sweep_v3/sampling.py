"""Scrambled Sobol space-filling design in pure Python (no SciPy dependency).

Why this module exists: the accepted sampler ``cft_revival.optimization.sampling`` is a
shifted Halton sequence (its docstring says so explicitly) and the campaign v3 protocol
asks for a scrambled Sobol design. SciPy is not installed in the project interpreter, so
the sequence is implemented here from its definition:

* direction numbers: Joe and Kuo, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM J. Sci. Comput. 30(5), 2635-2654 (2008), DOI 10.1137/070709359 - the
  ``new-joe-kuo-6.21201`` table (first 16 dimensions embedded verbatim below; the file was
  downloaded from the authors' page on 2026-09-03 and the rows were copied byte for byte);
* Gray-code generation (Antonov and Saleev 1979) of the unscrambled digital net;
* scrambling: linear matrix scramble (random lower-triangular GF(2) matrices with a unit
  diagonal applied to the generating matrices) followed by a random digital shift, i.e. the
  Matousek (1998) "LMS + shift" scheme that ``scipy.stats.qmc.Sobol(scramble=True)`` uses.
  Both preserve the (t, m, s)-net property, so the first 2**m points remain balanced in
  every dyadic elementary interval (this is what the tests check).

The random bits of the scramble are derived from SHA-256 of ``"<seed>:<dimension>:<row>"``
rather than from any library RNG, so the design is a pure function of the seed on every
platform and Python version.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from cft_revival.optimization import Design, Variable

BITS = 32
_SCALE = float(1 << BITS)

# d: (s, a, m_1..m_s) from new-joe-kuo-6.21201 (dimension 1 is the van der Corput sequence).
JOE_KUO_DIRECTION_TABLE: dict[int, tuple[int, int, tuple[int, ...]]] = {
    2: (1, 0, (1,)),
    3: (2, 1, (1, 3)),
    4: (3, 1, (1, 3, 1)),
    5: (3, 2, (1, 1, 1)),
    6: (4, 1, (1, 1, 3, 3)),
    7: (4, 4, (1, 3, 5, 13)),
    8: (5, 2, (1, 1, 5, 5, 17)),
    9: (5, 4, (1, 1, 5, 5, 5)),
    10: (5, 7, (1, 1, 7, 11, 19)),
    11: (5, 11, (1, 1, 5, 1, 1)),
    12: (5, 13, (1, 1, 1, 3, 11)),
    13: (5, 14, (1, 3, 5, 5, 31)),
    14: (6, 1, (1, 3, 3, 9, 7, 49)),
    15: (6, 13, (1, 1, 1, 15, 21, 21)),
    16: (6, 16, (1, 3, 1, 13, 27, 49)),
}
MAX_DIMENSIONS = max(JOE_KUO_DIRECTION_TABLE)


def direction_numbers(dimension: int, bits: int = BITS) -> list[int]:
    """Direction numbers V[1..bits] of one Sobol dimension (1-based), scaled to 2**bits."""

    if dimension < 1 or dimension > MAX_DIMENSIONS:
        raise ValueError(f"Sobol dimension {dimension} outside the embedded Joe-Kuo table")
    if dimension == 1:
        return [1 << (bits - i) for i in range(1, bits + 1)]
    s, a, m = JOE_KUO_DIRECTION_TABLE[dimension]
    v = [0] * (bits + 1)
    for i in range(1, s + 1):
        v[i] = m[i - 1] << (bits - i)
    for i in range(s + 1, bits + 1):
        value = v[i - s] ^ (v[i - s] >> s)
        for k in range(1, s):
            if (a >> (s - 1 - k)) & 1:
                value ^= v[i - k]
        v[i] = value
    return v[1:]


def _bit_stream(seed: int, dimension: int, row: int) -> int:
    """256 deterministic bits for (seed, dimension, row)."""

    digest = hashlib.sha256(f"{seed}:{dimension}:{row}".encode("ascii")).digest()
    return int.from_bytes(digest, "big")


def _parity(value: int) -> int:
    return bin(value).count("1") & 1


def scrambled_direction_numbers(dimension: int, seed: int, bits: int = BITS) -> tuple[list[int], int]:
    """Linear-matrix-scrambled direction numbers and the digital shift of one dimension.

    Row ``j`` (0 = most significant digit) of the lower-triangular matrix has a unit
    diagonal and random entries in columns ``k < j``; the shift is a random ``bits``-bit word.
    """

    base = direction_numbers(dimension, bits)
    rows: list[int] = []
    for j in range(bits):
        random_bits = _bit_stream(seed, dimension, j)
        mask = 1 << (bits - 1 - j)  # unit diagonal
        for k in range(j):
            if (random_bits >> k) & 1:
                mask |= 1 << (bits - 1 - k)
        rows.append(mask)
    scrambled = []
    for column in base:
        value = 0
        for j, mask in enumerate(rows):
            if _parity(mask & column):
                value |= 1 << (bits - 1 - j)
        scrambled.append(value)
    shift = _bit_stream(seed, dimension, bits) & ((1 << bits) - 1)
    return scrambled, shift


def sobol_points(count: int, dimensions: int, *, seed: int, scramble: bool = True, bits: int = BITS) -> tuple[tuple[float, ...], ...]:
    """The first ``count`` points of the (scrambled) Sobol sequence in [0, 1)**dimensions."""

    if count < 0 or dimensions < 1 or dimensions > MAX_DIMENSIONS:
        raise ValueError("invalid Sobol shape")
    columns: list[list[int]] = []
    shifts: list[int] = []
    for dimension in range(1, dimensions + 1):
        if scramble:
            scrambled, shift = scrambled_direction_numbers(dimension, seed, bits)
        else:
            scrambled, shift = direction_numbers(dimension, bits), 0
        columns.append(scrambled)
        shifts.append(shift)
    state = [0] * dimensions
    points: list[tuple[float, ...]] = []
    for index in range(count):
        if index > 0:
            # Gray code: flip direction number c where c is the rightmost zero bit of index-1.
            previous = index - 1
            c = 0
            while (previous >> c) & 1:
                c += 1
            if c >= bits:
                raise ValueError("Sobol index exceeds the bit depth")
            for dimension in range(dimensions):
                state[dimension] ^= columns[dimension][c]
        points.append(tuple((state[d] ^ shifts[d]) / _SCALE for d in range(dimensions)))
    return tuple(points)


def sobol_designs(variables: Sequence[Variable], count: int, *, seed: int, scramble: bool = True) -> tuple[Design, ...]:
    """Exactly ``count`` unique scaled designs from the scrambled Sobol sequence.

    The provenance string records the sequence index so a design can be regenerated from
    (seed, index) alone. Duplicate design ids cannot occur for distinct Sobol indices below
    2**32, but the check is kept fail-closed.
    """

    variables_tuple = tuple(variables)
    points = sobol_points(count, len(variables_tuple), seed=seed, scramble=scramble)
    designs: list[Design] = []
    seen: set[str] = set()
    for index, point in enumerate(points):
        values = tuple(
            variable.lower + coordinate * (variable.upper - variable.lower)
            for coordinate, variable in zip(point, variables_tuple, strict=True)
        )
        design = Design(values, variables_tuple, provenance=f"scrambled-sobol:seed={seed}:index={index}")
        if design.design_id in seen:
            raise RuntimeError("duplicate Sobol design identity")
        seen.add(design.design_id)
        designs.append(design)
    return tuple(designs)
