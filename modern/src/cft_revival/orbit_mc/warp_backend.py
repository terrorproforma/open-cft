"""Optional Warp CPU/CUDA relativistic Boris kernel used for backend parity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AxisymmetricField, ElectronLaunch, OrbitConfig, OrbitResult, OrbitValidationError

_PUSH_KERNEL = None


@dataclass(frozen=True, slots=True)
class WarpStatus:
    import_available: bool
    cpu_available: bool
    cuda_available: bool
    version: str | None
    reason: str


def warp_status() -> WarpStatus:
    try:
        import warp as wp
        wp.init()
        cuda = bool(wp.is_cuda_available())
        version = getattr(wp, "__version__", None)
        return WarpStatus(True, True, cuda, version, "Warp initialized")
    except Exception as error:
        return WarpStatus(False, False, False, None, f"{type(error).__name__}: {error}")


def warp_boris_push_batch(
    velocities_m_per_s: np.ndarray,
    electric_v_per_m: np.ndarray,
    magnetic_t: np.ndarray,
    dt_s: float,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Run one float64 uniform-field push; map interpolation remains host-verified."""

    try:
        import warp as wp
    except ImportError as error:
        raise OrbitValidationError("Warp is not installed") from error
    velocity = np.asarray(velocities_m_per_s, dtype=np.float64)
    electric = np.asarray(electric_v_per_m, dtype=np.float64)
    magnetic = np.asarray(magnetic_t, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise OrbitValidationError("velocities must have shape (N,3)")
    if electric.shape == (3,):
        electric = np.broadcast_to(electric, velocity.shape).copy()
    if magnetic.shape == (3,):
        magnetic = np.broadcast_to(magnetic, velocity.shape).copy()
    if electric.shape != velocity.shape or magnetic.shape != velocity.shape:
        raise OrbitValidationError("field arrays must have shape (3,) or (N,3)")
    if not np.isfinite(velocity).all() or not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
        raise OrbitValidationError("Warp inputs must be finite")

    global _PUSH_KERNEL
    if _PUSH_KERNEL is None:
        @wp.kernel
        def push(
            velocity_in: wp.array(dtype=wp.vec3d),
            electric_in: wp.array(dtype=wp.vec3d),
            magnetic_in: wp.array(dtype=wp.vec3d),
            dt: wp.float64,
            velocity_out: wp.array(dtype=wp.vec3d),
        ):
            index = wp.tid()
            c = wp.float64(299792458.0)
            qm = wp.float64(-175882000837.79984)
            v = velocity_in[index]
            gamma = wp.sqrt(wp.float64(1.0) / (wp.float64(1.0) - wp.dot(v, v)/(c*c)))
            u_minus = gamma*v + wp.float64(0.5)*qm*dt*electric_in[index]
            gamma_minus = wp.sqrt(wp.float64(1.0) + wp.dot(u_minus, u_minus)/(c*c))
            t = wp.float64(0.5)*qm*dt*magnetic_in[index]/gamma_minus
            s = wp.float64(2.0)*t/(wp.float64(1.0) + wp.dot(t, t))
            u_prime = u_minus + wp.cross(u_minus, t)
            u_plus = u_minus + wp.cross(u_prime, s)
            u_new = u_plus + wp.float64(0.5)*qm*dt*electric_in[index]
            gamma_new = wp.sqrt(wp.float64(1.0) + wp.dot(u_new, u_new)/(c*c))
            velocity_out[index] = u_new/gamma_new

        _PUSH_KERNEL = push

    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        raise OrbitValidationError("Warp CUDA device is unavailable")
    vin = wp.from_numpy(velocity, dtype=wp.vec3d, device=device)
    ein = wp.from_numpy(electric, dtype=wp.vec3d, device=device)
    bin_ = wp.from_numpy(magnetic, dtype=wp.vec3d, device=device)
    output = wp.empty(len(velocity), dtype=wp.vec3d, device=device)
    wp.launch(
        _PUSH_KERNEL, dim=len(velocity),
        inputs=[vin, ein, bin_, float(dt_s)], outputs=[output], device=device,
    )
    wp.synchronize_device(device)
    return np.asarray(output.numpy(), dtype=np.float64)


def integrate_orbit_warp(
    launch: ElectronLaunch,
    field: AxisymmetricField,
    config: OrbitConfig,
    *,
    device: str = "cpu",
) -> OrbitResult:
    """Execute the complete host event loop with each relativistic push on Warp."""

    from .integrator import integrate_orbit

    def push(
        velocity: np.ndarray, electric: np.ndarray, magnetic: np.ndarray, dt: float
    ) -> np.ndarray:
        return warp_boris_push_batch(
            velocity[np.newaxis, :], electric, magnetic, dt, device=device
        )[0]

    return integrate_orbit(
        launch, field, config, backend=f"warp-{device}-relativistic-boris",
        velocity_pusher=push,
    )
