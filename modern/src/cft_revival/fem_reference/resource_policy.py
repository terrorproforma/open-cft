"""Fail-closed resource policy for third-level adaptive studies."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os

from .models import FEMValidationError

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ThirdLevelResourcePolicy:
    maximum_p2_dofs: int = 1_500_000
    minimum_free_ram_bytes: int = 8 * GIB
    one_design_at_a_time: bool = True
    safety_factor: float = 1.75
    fixed_python_reserve_bytes: int = 256 * 1024**2
    maximum_triangles: int = 1_500_000
    maximum_robin_edges: int = 1_500_000
    heavy_mesh_p2_dofs: int = 100_000
    heavy_serialized_bytes: int = 64 * 1024**2


class ResourceBlockedError(FEMValidationError):
    """Typed fail-closed result for a resource-gated numerical phase."""

    status = "NOT_EVALUATED"


def estimate_peak_allocation_bytes(
    *,
    p2_dofs: int,
    triangles: int,
    robin_edges: int = 0,
    serialized_bytes: int = 0,
    policy: ThirdLevelResourcePolicy = ThirdLevelResourcePolicy(),
) -> dict[str, int | float]:
    """Conservative calibrated peak for topology COO, CSR, and solver work."""

    for name, value in (
        ("p2_dofs", p2_dofs),
        ("triangles", triangles),
        ("robin_edges", robin_edges),
        ("serialized_bytes", serialized_bytes),
    ):
        minimum = 0 if name in {"robin_edges", "serialized_bytes"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise FEMValidationError(f"{name} must be an integer >= {minimum}")
    if p2_dofs > policy.maximum_p2_dofs:
        raise ResourceBlockedError(
            f"NOT_EVALUATED: {p2_dofs} P2 DOFs exceeds "
            f"{policy.maximum_p2_dofs} cap"
        )
    if (
        triangles > policy.maximum_triangles
        or robin_edges > policy.maximum_robin_edges
    ):
        raise ResourceBlockedError("NOT_EVALUATED: projected topology exceeds policy")
    contributions = 36 * triangles + 9 * robin_edges
    # Peak topology phase: key array, sort workspace, unique mask/keys, and
    # conservative final CSR indices/data. Numerical/solver phase includes
    # CSR, RHS/load, PCG vectors, and Python/NumPy allocator slack.
    coo_keys = 8 * contributions
    sort_workspace = 8 * contributions
    unique_masks_and_keys = 9 * contributions
    csr_upper_bound = 16 * contributions + 8 * (p2_dofs + 1)
    vectors_and_mesh_temporaries = 20 * 8 * p2_dofs + 192 * triangles
    serialization_buffers = 6 * serialized_bytes
    variable = (
        coo_keys
        + sort_workspace
        + unique_masks_and_keys
        + csr_upper_bound
        + vectors_and_mesh_temporaries
        + serialization_buffers
    )
    reserve = min(
        policy.fixed_python_reserve_bytes,
        max(1024**2, variable // 4),
    )
    modeled = variable + reserve
    required = int(modeled * policy.safety_factor)
    return {
        "coo_keys_bytes": coo_keys,
        "sort_workspace_bytes": sort_workspace,
        "unique_masks_and_keys_bytes": unique_masks_and_keys,
        "csr_upper_bound_bytes": csr_upper_bound,
        "vectors_and_mesh_temporaries_bytes": vectors_and_mesh_temporaries,
        "serialization_parse_buffers_bytes": serialization_buffers,
        "fixed_python_reserve_bytes": reserve,
        "safety_factor": policy.safety_factor,
        "modeled_bytes": modeled,
        "required_free_ram_bytes": required,
        "calibration_probe_tracemalloc_bytes": 2_914_493,
        "calibration_probe_retained_rss_delta_bytes": 675_840,
        "calibration_probe_rss_and_allocator_margin": (
            "covered_by_1.75x_plus_scaled_reserve_up_to_256MiB"
        ),
    }


def available_ram_bytes() -> int:
    """Return currently available physical RAM using only the standard library."""

    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            )

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise FEMValidationError("unable to query available physical RAM")
        return int(status.available_physical)
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(available_pages * page_size)


def current_process_rss_bytes() -> int:
    """Return current resident working-set bytes without optional packages."""

    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        process = get_current_process()
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_memory_info.restype = ctypes.c_int
        if not get_memory_info(
            process, ctypes.byref(counters), counters.cb
        ):
            raise FEMValidationError("unable to query process resident memory")
        return int(counters.working_set_size)
    with open("/proc/self/statm", encoding="ascii") as source:
        resident_pages = int(source.read().split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def preflight_third_level(
    selected_designs: int,
    *,
    policy: ThirdLevelResourcePolicy = ThirdLevelResourcePolicy(),
    available_bytes: int | None = None,
) -> dict[str, int | bool]:
    if (
        isinstance(selected_designs, bool)
        or not isinstance(selected_designs, int)
        or selected_designs <= 0
    ):
        raise FEMValidationError("selected design count must be a positive integer")
    if policy.one_design_at_a_time and selected_designs != 1:
        raise FEMValidationError("third-level execution requires exactly one design")
    free = available_ram_bytes() if available_bytes is None else available_bytes
    if isinstance(free, bool) or not isinstance(free, int) or free < 0:
        raise FEMValidationError("available RAM must be a non-negative integer")
    passed = free >= policy.minimum_free_ram_bytes
    if not passed:
        raise ResourceBlockedError(
            "NOT_EVALUATED: third-level execution requires at least "
            f"{policy.minimum_free_ram_bytes / GIB:.0f} GiB free RAM; "
            f"only {free / GIB:.2f} GiB is available"
        )
    return {
        "passed": True,
        "available_ram_bytes": free,
        "minimum_free_ram_bytes": policy.minimum_free_ram_bytes,
        "maximum_p2_dofs": policy.maximum_p2_dofs,
        "one_design_at_a_time": policy.one_design_at_a_time,
    }


def preflight_level_allocation(
    *,
    p2_dofs: int,
    triangles: int,
    robin_edges: int = 0,
    third_level: bool,
    policy: ThirdLevelResourcePolicy = ThirdLevelResourcePolicy(),
    available_bytes: int | None = None,
    serialized_bytes: int = 0,
    phase: str = "level",
) -> dict[str, int | float | bool | str]:
    estimate = estimate_peak_allocation_bytes(
        p2_dofs=p2_dofs,
        triangles=triangles,
        robin_edges=robin_edges,
        serialized_bytes=serialized_bytes,
        policy=policy,
    )
    available = available_ram_bytes() if available_bytes is None else available_bytes
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise FEMValidationError("available RAM must be a non-negative integer")
    required = int(estimate["required_free_ram_bytes"])
    if third_level:
        required = max(required, policy.minimum_free_ram_bytes)
    if available < required:
        raise ResourceBlockedError(
            f"NOT_EVALUATED: {phase} allocation preflight failed: requires "
            f"{required} free bytes, found {available}"
        )
    return {
        **estimate,
        "passed": True,
        "available_ram_bytes": available,
        "effective_required_free_ram_bytes": required,
        "third_level": third_level,
        "check_timing": "immediately_before_level_and_assembly",
        "phase": phase,
    }


def guard_allocation(
    phase: str,
    *,
    p2_dofs: int,
    triangles: int,
    robin_edges: int = 0,
    serialized_bytes: int = 0,
    third_level: bool = False,
    policy: ThirdLevelResourcePolicy = ThirdLevelResourcePolicy(),
    available_bytes: int | None = None,
) -> dict[str, int | float | bool | str]:
    if not isinstance(phase, str) or not phase:
        raise FEMValidationError("allocation phase must be non-empty")
    if (
        p2_dofs < policy.heavy_mesh_p2_dofs
        and serialized_bytes < policy.heavy_serialized_bytes
    ):
        report = preflight_level_allocation(
            p2_dofs=p2_dofs,
            triangles=triangles,
            robin_edges=robin_edges,
            serialized_bytes=serialized_bytes,
            third_level=False,
            policy=policy,
            available_bytes=1 << 62,
            phase=phase,
        )
        report["memory_check"] = "not_heavy_below_p2_and_serialized_thresholds"
        return report
    return preflight_level_allocation(
        p2_dofs=p2_dofs,
        triangles=triangles,
        robin_edges=robin_edges,
        serialized_bytes=serialized_bytes,
        third_level=third_level,
        policy=policy,
        available_bytes=available_bytes,
        phase=phase,
    )
