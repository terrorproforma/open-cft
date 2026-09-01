"""Small linear-algebra boundary with an optional NumPy fast path."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

try:  # NumPy is an acceleration, not a runtime requirement.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by the explicit fallback tests.
    _np = None


def numpy_available() -> bool:
    return _np is not None


def cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if _np is not None:
        try:
            return _np.linalg.cholesky(_np.asarray(matrix, dtype=float)).tolist()
        except _np.linalg.LinAlgError as error:
            raise ArithmeticError("matrix is not positive definite") from error
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = matrix[row][column] - sum(
                lower[row][index] * lower[column][index] for index in range(column)
            )
            if row == column:
                if remainder <= 0.0:
                    raise ArithmeticError("matrix is not positive definite")
                lower[row][column] = sqrt(remainder)
            else:
                lower[row][column] = remainder / lower[column][column]
    return lower


def solve_cholesky(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    size = len(lower)
    forward = [0.0] * size
    for row in range(size):
        forward[row] = (
            rhs[row]
            - sum(lower[row][column] * forward[column] for column in range(row))
        ) / lower[row][row]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        result[row] = (
            forward[row]
            - sum(lower[column][row] * result[column] for column in range(row + 1, size))
        ) / lower[row][row]
    return result


def solve_lower(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    result = [0.0] * len(lower)
    for row in range(len(lower)):
        result[row] = (
            rhs[row]
            - sum(lower[row][column] * result[column] for column in range(row))
        ) / lower[row][row]
    return result


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def snapshot_pod(
    centered_fields: Sequence[Sequence[float]], rank: int
) -> tuple[list[list[float]], list[float]]:
    """Return orthonormal spatial modes and singular values.

    NumPy uses a thin SVD. The fallback uses deterministic power iteration on
    ``A A^T`` followed by Gram-Schmidt deflation.
    """
    if _np is not None:
        matrix = _np.asarray(centered_fields, dtype=float)
        _, singular_values, right = _np.linalg.svd(matrix, full_matrices=False)
        return right[:rank].tolist(), singular_values[:rank].tolist()

    width = len(centered_fields[0])
    covariance = [
        [
            sum(field[row] * field[column] for field in centered_fields)
            for column in range(width)
        ]
        for row in range(width)
    ]
    modes: list[list[float]] = []
    values: list[float] = []
    for component in range(rank):
        vector = [1.0 + ((index + component) % 7) / 7.0 for index in range(width)]
        for _ in range(128):
            candidate = [
                sum(covariance[row][column] * vector[column] for column in range(width))
                for row in range(width)
            ]
            for mode in modes:
                projection = dot(candidate, mode)
                candidate = [
                    value - projection * basis
                    for value, basis in zip(candidate, mode, strict=True)
                ]
            norm = sqrt(max(dot(candidate, candidate), 0.0))
            if norm <= 1e-15:
                break
            updated = [value / norm for value in candidate]
            if sqrt(sum((a - b) ** 2 for a, b in zip(updated, vector, strict=True))) < 1e-12:
                vector = updated
                break
            vector = updated
        eigenvalue = max(
            dot(
                vector,
                [
                    sum(covariance[row][column] * vector[column] for column in range(width))
                    for row in range(width)
                ],
            ),
            0.0,
        )
        if eigenvalue <= 1e-24:
            break
        modes.append(vector)
        values.append(sqrt(eigenvalue))
    return modes, values
