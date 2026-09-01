"""Optional NVIDIA Warp backend for verified batch kernels.

Warp is imported only in this module. The rest of the package remains usable
without Warp, CUDA, NumPy, or a native compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .kernels import validate_cusp_fields
from .models import ValidationError

try:
    import warp as wp
except ImportError:  # pragma: no cover - exercised in environments without Warp.
    wp = None  # type: ignore[assignment]


if wp is not None:

    @wp.kernel
    def _cusp_arrival_probability_kernel(
        low_field_t: wp.array(dtype=wp.float64),
        high_field_t: wp.array(dtype=wp.float64),
        output: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        ratio = low_field_t[index] / high_field_t[index]
        zero = wp.float64(0.0)
        half = wp.float64(0.5)
        one = wp.float64(1.0)
        if ratio == zero:
            output[index] = zero
        else:
            output[index] = half * ratio / (one + wp.sqrt(one - ratio))


@dataclass(frozen=True)
class WarpBatchResult:
    probabilities: tuple[float, ...]
    device: str


def warp_available() -> bool:
    return wp is not None


def available_warp_devices() -> tuple[str, ...]:
    if wp is None:
        return ()
    wp.init()
    return tuple(str(device) for device in wp.get_devices())


def warp_device_available(device: str) -> bool:
    if wp is None:
        return False
    try:
        _resolve_device(device)
    except (RuntimeError, ValueError):
        return False
    return True


def _resolve_device(device: str):
    if wp is None:
        raise RuntimeError(
            "NVIDIA Warp is unavailable; install the optional 'gpu' dependency"
        )
    wp.init()
    requested = device.strip().lower()
    if requested == "auto":
        return wp.get_preferred_device()
    if requested == "cuda":
        requested = "cuda:0"
    if requested != "cpu" and not requested.startswith("cuda:"):
        raise ValueError("Warp device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
    try:
        return wp.get_device(requested)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(f"Warp device {requested!r} is unavailable") from error


def cusp_arrival_probabilities_warp(
    low_field_t: Sequence[float],
    high_field_t: Sequence[float],
    *,
    device: str = "auto",
) -> WarpBatchResult:
    """Evaluate a validated batch on a Warp CPU or CUDA device."""

    try:
        batch_size = len(low_field_t)
        high_batch_size = len(high_field_t)
    except TypeError as error:
        raise ValidationError("cusp field batches must be one-dimensional sequences") from error
    if batch_size != high_batch_size:
        raise ValidationError("low/high field batches must have equal length")
    if batch_size == 0:
        raise ValidationError("cusp field batch cannot be empty")

    for values in (low_field_t, high_field_t):
        shape = getattr(values, "shape", None)
        if shape is not None and len(shape) != 1:
            raise ValidationError("cusp field batches must be one-dimensional")
    for low, high in zip(low_field_t, high_field_t, strict=True):
        validate_cusp_fields(float(low), float(high))

    resolved = _resolve_device(device)
    low_array = wp.array(low_field_t, dtype=wp.float64, device=resolved)
    high_array = wp.array(high_field_t, dtype=wp.float64, device=resolved)
    output = wp.empty(batch_size, dtype=wp.float64, device=resolved)
    wp.launch(
        kernel=_cusp_arrival_probability_kernel,
        dim=batch_size,
        inputs=[low_array, high_array, output],
        device=resolved,
    )
    wp.synchronize_device(resolved)
    return WarpBatchResult(
        probabilities=tuple(float(value) for value in output.numpy()),
        device=str(resolved),
    )
