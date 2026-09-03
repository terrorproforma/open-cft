"""The screened design catalogue: binding, per-cell wall-loss counts and the frozen posterior sample.

The catalogue is the immutable screening artifact ``geometry-wall-loss-dataset.json`` of
``orbit_wall_loss_geometry_screening_v1`` (classification
``SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS``; 96 accepted sweep-v2 designs, four
axial launch cells of 128 launches each at the accepted-2N timestep).  No surrogate
stands between the dataset and the optimiser: each design's per-cell wall-hit
probabilities enter the L0 model DIRECTLY as uncertain inputs whose binomial
uncertainty is the Jeffreys Beta posterior of the recorded counts.

Everything here is deterministic pure Python (``math``, no scipy) so the frozen sample
replays without the ML runtime; the regularised incomplete beta function and its
inverse are implemented locally and cross-checked against scipy in the tests.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import canonical_bytes, strict_json_file

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent

SOURCE_CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
REPORTED_CASE = "accepted-2N"
CELLS: tuple[str, ...] = ("gs1-cell-1", "gs1-cell-2", "gs1-cell-3", "gs1-cell-4")
CELL_TRIALS = 128
POOLED_TRIALS = 512
CATALOGUE_SIZE = 96
JEFFREYS_PRIOR = 0.5

# Frozen binding of the screening dataset (bytes, Git blob, manifest entry, ancestry).
DATASET_BINDING: dict[str, Any] = {
    "experiment": "modern/experiments/orbit_wall_loss_geometry_screening_v1",
    "classification": SOURCE_CLASSIFICATION,
    "dataset_path": "modern/experiments/orbit_wall_loss_geometry_screening_v1/results/artifacts/geometry-wall-loss-dataset.json",
    "dataset_file_sha256": "9104e3bf0694f7c1c88d1c7bc377b43b24cbc42a94bd75eed2cfa6dc21f5114a",
    "dataset_bytes": 2659656,
    "dataset_git_blob": "858de21adf1b3ea40237d5e7f0531a2013eb58e2",
    "manifest_path": "modern/experiments/orbit_wall_loss_geometry_screening_v1/results/manifest.json",
    "manifest_file_sha256": "39bd52133fefd3adae45e9593e7312b8e9027322ca3142d59912bbc13e2e027a",
    "manifest_git_blob": "ff3b3cd8a4cb59047b6856031faaadb941f11877",
    "manifest_state": "accepted_result",
    "screening_preregistration_commit": "c86bfca37fdf285f4f2a53a01c2f32f14516d868",
    "screening_result_commit": "ab7c28977963822b2ad6eac451d2bafef5185e6c",
    "screening_merge_commit": "22e2156b5f66f5e85bc4d2238da2ce6a2936bd46",
    "design_count": CATALOGUE_SIZE,
    "reported_case": REPORTED_CASE,
    "cells": list(CELLS),
    "cell_trials": CELL_TRIALS,
    "pooled_trials": POOLED_TRIALS,
}


class CatalogueBindingError(RuntimeError):
    """The dataset on disk is not the recorded screening artifact."""


# --------------------------------------------------------------------------
# Binding (hashes + Git blobs + manifest entry + ancestry)
# --------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPOSITORY, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _blob_of(commit: str, relative: str) -> str | None:
    try:
        return _git("rev-parse", f"{commit}:{relative}")
    except subprocess.CalledProcessError:
        return None


def _is_ancestor(commit: str) -> bool | None:
    try:
        completed = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"), cwd=REPOSITORY, capture_output=True
        )
    except OSError:
        return None
    return completed.returncode == 0


def binding_report(spec: Mapping[str, Any] = DATASET_BINDING, *, use_git: bool = True) -> dict[str, Any]:
    """Byte hashes, Git blob identities, manifest entry and ancestry of the dataset."""

    dataset_path = REPOSITORY / spec["dataset_path"]
    manifest_path = REPOSITORY / spec["manifest_path"]
    dataset_bytes = dataset_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json_file(manifest_path)
    entry = next(
        (item for item in manifest["artifacts"] if item.get("path") == "artifacts/geometry-wall-loss-dataset.json"),
        None,
    )
    checks = {
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest() == spec["dataset_file_sha256"],
        "dataset_bytes": len(dataset_bytes) == spec["dataset_bytes"],
        "dataset_lf": b"\r" not in dataset_bytes,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest() == spec["manifest_file_sha256"],
        "manifest_state": manifest.get("state") == spec["manifest_state"],
        "manifest_entry_matches": (
            entry is not None
            and entry.get("byte_sha256") == spec["dataset_file_sha256"]
            and entry.get("bytes") == spec["dataset_bytes"]
        ),
    }
    git: dict[str, Any] = {"used": use_git}
    if use_git:
        dataset_blob = _blob_of(spec["screening_result_commit"], spec["dataset_path"])
        manifest_blob = _blob_of(spec["screening_result_commit"], spec["manifest_path"])
        working_dataset_blob = _git("hash-object", str(dataset_path))
        working_manifest_blob = _git("hash-object", str(manifest_path))
        git.update(
            {
                "dataset_blob_at_result_commit": dataset_blob,
                "manifest_blob_at_result_commit": manifest_blob,
                "dataset_blob_working_tree": working_dataset_blob,
                "manifest_blob_working_tree": working_manifest_blob,
                "result_commit_is_ancestor_of_head": _is_ancestor(spec["screening_result_commit"]),
                "merge_commit_is_ancestor_of_head": _is_ancestor(spec["screening_merge_commit"]),
            }
        )
        checks["dataset_git_blob"] = dataset_blob == spec["dataset_git_blob"] == working_dataset_blob
        checks["manifest_git_blob"] = manifest_blob == spec["manifest_git_blob"] == working_manifest_blob
        checks["result_commit_is_ancestor"] = git["result_commit_is_ancestor_of_head"] is True
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_bytes": len(dataset_bytes),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_artifact_count": manifest.get("artifact_count"),
        "manifest_state": manifest.get("state"),
        "git": git,
        "screening_result_commit": spec["screening_result_commit"],
        "screening_preregistration_commit": spec["screening_preregistration_commit"],
        "screening_merge_commit": spec["screening_merge_commit"],
    }


def require_binding(spec: Mapping[str, Any] = DATASET_BINDING, *, use_git: bool = True) -> dict[str, Any]:
    report = binding_report(spec, use_git=use_git)
    if not report["passed"]:
        failed = sorted(name for name, ok in report["checks"].items() if not ok)
        raise CatalogueBindingError("catalogue binding failed: " + ", ".join(failed))
    return report


# --------------------------------------------------------------------------
# Catalogue rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueDesign:
    """One screened design: identity, sealed geometry and the accepted-2N wall-hit counts."""

    index: int  # catalogue index == sweep_index == position in the dataset list
    case_id: str
    design_id: str
    geometry_sha256: str
    cell_wall_hits: tuple[int, int, int, int]  # successes per cell (128 launches each)
    cell_trials: tuple[int, int, int, int]
    pooled_wall_hits: int  # successes over the 512 launches of the case
    pooled_trials: int
    reflected: int
    geometry: Mapping[str, Any]
    design_values: Mapping[str, float]
    converged: bool

    @property
    def cell_point_estimates(self) -> tuple[float, ...]:
        return tuple(s / n for s, n in zip(self.cell_wall_hits, self.cell_trials, strict=True))

    @property
    def pooled_point_estimate(self) -> float:
        return self.pooled_wall_hits / self.pooled_trials

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "case_id": self.case_id,
            "design_id": self.design_id,
            "geometry_sha256": self.geometry_sha256,
            "cell_wall_hits": list(self.cell_wall_hits),
            "cell_trials": list(self.cell_trials),
            "pooled_wall_hits": self.pooled_wall_hits,
            "pooled_trials": self.pooled_trials,
            "reflected": self.reflected,
            "cell_point_estimates": list(self.cell_point_estimates),
            "pooled_point_estimate": self.pooled_point_estimate,
            "geometry": dict(self.geometry),
            "design_values": dict(self.design_values),
            "converged": self.converged,
        }


_GEOMETRY_KEYS = (
    "chamber_length_m",
    "wall_radius_m",
    "exit_start_m",
    "exit_length_m",
    "exit_outer_radius_m",
    "has_divergent_exit",
    "stage_count",
    "stage_pitch_m",
    "stage_centers_m",
    "magnet_inner_radius_m",
    "magnet_outer_radius_m",
    "magnet_axial_thickness_m",
    "dielectric_thickness_m",
    "injector_length_m",
    "first_polarity",
)


def load_catalogue(spec: Mapping[str, Any] = DATASET_BINDING) -> tuple[CatalogueDesign, ...]:
    """Load the 96 designs in dataset order; every stored probability must equal its count ratio."""

    dataset = strict_json_file(REPOSITORY / spec["dataset_path"])
    if dataset["classification"] != SOURCE_CLASSIFICATION:
        raise CatalogueBindingError("dataset classification is not the screening label")
    if dataset["design_count"] != spec["design_count"] or len(dataset["designs"]) != spec["design_count"]:
        raise CatalogueBindingError("dataset design count differs from the declared catalogue size")
    if dataset.get("evidentiary") is not True or dataset.get("plan_kind") != "evidentiary":
        raise CatalogueBindingError("dataset is not the evidentiary screening record")
    designs: list[CatalogueDesign] = []
    for position, record in enumerate(dataset["designs"]):
        if record["classification"] != SOURCE_CLASSIFICATION:
            raise CatalogueBindingError(f"{record['case_id']}: row classification is not the screening label")
        if int(record["sweep_index"]) != position:
            raise CatalogueBindingError(f"{record['case_id']}: sweep_index {record['sweep_index']} != position {position}")
        if not record["convergence"]["converged"] or not record["convergence"]["sealed"]:
            raise CatalogueBindingError(f"{record['case_id']}: accepted-2N not converged/sealed")
        case = record["cases"][spec["reported_case"]]
        if not case["sealed"] or int(case["trial_count"]) != spec["pooled_trials"]:
            raise CatalogueBindingError(f"{record['case_id']}: reported case not sealed at {spec['pooled_trials']} launches")
        cells = record["per_cell"][spec["reported_case"]]
        hits = []
        trials = []
        for cell in spec["cells"]:
            item = cells[cell]["wall_hit"]
            successes, n = int(item["successes"]), int(item["trials"])
            if n != spec["cell_trials"]:
                raise CatalogueBindingError(f"{record['case_id']}/{cell}: trials {n} != {spec['cell_trials']}")
            if float(item["probability"]) != successes / n:
                raise CatalogueBindingError(f"{record['case_id']}/{cell}: stored probability is not the count ratio")
            if int(cells[cell]["counts"]["wall_hit"]) != successes:
                raise CatalogueBindingError(f"{record['case_id']}/{cell}: counts disagree with the Wilson block")
            hits.append(successes)
            trials.append(n)
        pooled_hits = int(case["termination_counts"]["wall_hit"])
        if sum(hits) != pooled_hits:
            raise CatalogueBindingError(f"{record['case_id']}: cell wall counts do not sum to the pooled count")
        if float(case["wall_hit"]["probability"]) != pooled_hits / int(case["trial_count"]):
            raise CatalogueBindingError(f"{record['case_id']}: pooled probability is not the count ratio")
        geometry = record["geometry"]
        designs.append(
            CatalogueDesign(
                index=position,
                case_id=str(record["case_id"]),
                design_id=str(record["design_id"]),
                geometry_sha256=str(record["identities"]["geometry_sha256"]),
                cell_wall_hits=tuple(hits),  # type: ignore[arg-type]
                cell_trials=tuple(trials),  # type: ignore[arg-type]
                pooled_wall_hits=pooled_hits,
                pooled_trials=int(case["trial_count"]),
                reflected=int(case["termination_counts"]["reflected"]),
                geometry={key: geometry[key] for key in _GEOMETRY_KEYS},
                design_values={key: float(value) for key, value in record["design_values"].items()},
                converged=bool(record["convergence"]["converged"]),
            )
        )
    if len({d.case_id for d in designs}) != len(designs) or len({d.design_id for d in designs}) != len(designs):
        raise CatalogueBindingError("case or design ids are not unique")
    return tuple(designs)


def catalogue_sha256(designs: Sequence[CatalogueDesign]) -> str:
    """Identity of the catalogue as consumed (indices, ids, counts, geometry)."""

    return hashlib.sha256(canonical_bytes([d.to_record() for d in designs])).hexdigest()


# --------------------------------------------------------------------------
# Regularised incomplete beta and its inverse (pure Python, deterministic)
# --------------------------------------------------------------------------

_CF_MAX_ITERATIONS = 400
_CF_EPSILON = 1e-16
_CF_TINY = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz continued fraction for the incomplete beta (Numerical Recipes ``betacf``)."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPSILON:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) for a, b > 0 and x in [0, 1]."""

    if not (a > 0.0 and b > 0.0):
        raise ValueError("beta parameters must be positive")
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_log_pdf(a: float, b: float, x: float) -> float:
    return (a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - log_beta(a, b)


@lru_cache(maxsize=None)
def beta_quantile(u: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b): bisection to a bracket, safeguarded Newton to full precision.

    Deterministic (fixed schedule), monotone in ``u`` on the bracket and clamped to the open
    interval (0, 1) so the result is always an admissible probability.
    """

    if not (0.0 < u < 1.0):
        raise ValueError("quantile level must lie in (0, 1)")
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < u:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    for _ in range(60):
        f = regularized_incomplete_beta(a, b, x) - u
        if f < 0.0:
            lo = max(lo, x)
        else:
            hi = min(hi, x)
        if hi - lo <= 4.0 * math.ulp(x):
            break
        try:
            pdf = math.exp(beta_log_pdf(a, b, x))
        except (ValueError, OverflowError):
            pdf = 0.0
        step = f / pdf if pdf > 0.0 and math.isfinite(pdf) else 0.0
        candidate = x - step
        if not (lo < candidate < hi) or step == 0.0:
            candidate = 0.5 * (lo + hi)
        if candidate == x:
            break
        x = candidate
    x = min(max(x, math.ulp(0.0)), math.nextafter(1.0, 0.0))
    return x


# --------------------------------------------------------------------------
# Frozen QMC sample of the uncertain inputs
# --------------------------------------------------------------------------

QMC_BASES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17)
QMC_SEED = 20260903
QMC_SAMPLE_SIZE = 64

# Uncertain inputs shared by every design (the accepted L0 sweep ranges, as in v1).
SHARED_UNCERTAIN_INPUTS: tuple[tuple[str, float, float, str], ...] = (
    ("ionized_number_fraction", 0.65, 0.98, "1"),
    ("xe_double_plus_fraction_of_ions", 0.0, 0.15, "1"),
    ("axial_momentum_fraction_of_ion_momentum", 0.75, 0.98, "1"),
)
CUSP_NAMES: tuple[str, ...] = (
    "wall_loss_probability_cell_1",
    "wall_loss_probability_cell_2",
    "wall_loss_probability_cell_3",
    "wall_loss_probability_cell_4",
)
POOLED_NAME = "wall_loss_probability_pooled"
UNCERTAIN_NAMES: tuple[str, ...] = CUSP_NAMES + tuple(item[0] for item in SHARED_UNCERTAIN_INPUTS)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def unit_qmc_rows(count: int = QMC_SAMPLE_SIZE, seed: int = QMC_SEED) -> tuple[tuple[float, ...], ...]:
    """Deterministic prime-base radical-inverse rows (offset ``17 + seed * 104729``); identical to v1."""

    if count < 1 or seed < 0:
        raise ValueError("QMC sample requires count >= 1 and seed >= 0")
    start = 17 + seed * 104_729
    return tuple(
        tuple(_radical_inverse(start + row, base) for base in QMC_BASES) for row in range(1, count + 1)
    )


def posterior_parameters(successes: int, trials: int, *, width_scale: float = 1.0) -> tuple[float, float]:
    """Jeffreys Beta posterior (s + 1/2, n - s + 1/2); ``width_scale`` rescales the counts.

    ``width_scale`` w multiplies both counts (s -> w s, n - s -> w (n - s)) so the posterior
    mean stays (approximately) fixed while its standard deviation scales like 1/sqrt(w); w = 1
    is the campaign posterior; the sensitivity analysis uses w in {1/4, 1, 4} and the point
    estimate (no width).
    """

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not (width_scale > 0.0 and math.isfinite(width_scale)):
        raise ValueError("width scale must be positive and finite")
    return successes * width_scale + JEFFREYS_PRIOR, (trials - successes) * width_scale + JEFFREYS_PRIOR


def posterior_mean(successes: int, trials: int) -> float:
    """Jeffreys posterior mean (s + 1/2) / (n + 1), unrounded (v1 audit F22)."""

    a, b = posterior_parameters(successes, trials)
    return a / (a + b)


def design_theta_rows(
    design: CatalogueDesign,
    *,
    rows: Sequence[Sequence[float]] | None = None,
    width_scale: float | None = 1.0,
) -> tuple[dict[str, float], ...]:
    """The frozen 64 uncertain-input rows of one design.

    Cells: ``p_k = BetaQuantile(u_k; s_k + 1/2, n_k - s_k + 1/2)`` from the design's own
    accepted-2N counts (``width_scale`` None = point estimate, the posterior mean, no width);
    the pooled probability (used only by the sensitivity closure CL-2) takes the first
    radical-inverse coordinate through the pooled-count posterior; the three shared inputs
    are uniform on the accepted L0 sweep ranges as in v1.
    """

    unit = unit_qmc_rows() if rows is None else rows
    out = []
    for row in unit:
        theta: dict[str, float] = {}
        for k, name in enumerate(CUSP_NAMES):
            s, n = design.cell_wall_hits[k], design.cell_trials[k]
            if width_scale is None:
                theta[name] = posterior_mean(s, n)
            else:
                a, b = posterior_parameters(s, n, width_scale=width_scale)
                theta[name] = beta_quantile(row[k], a, b)
        if width_scale is None:
            theta[POOLED_NAME] = posterior_mean(design.pooled_wall_hits, design.pooled_trials)
        else:
            a, b = posterior_parameters(design.pooled_wall_hits, design.pooled_trials, width_scale=width_scale)
            theta[POOLED_NAME] = beta_quantile(row[0], a, b)
        for (name, lower, upper, _units), coordinate in zip(
            SHARED_UNCERTAIN_INPUTS, row[len(CUSP_NAMES) :], strict=True
        ):
            theta[name] = lower + coordinate * (upper - lower)
        out.append(theta)
    return tuple(out)


def design_nominal_theta(design: CatalogueDesign) -> dict[str, float]:
    """Nominal point: Jeffreys posterior means of the counts, midpoints of the shared priors."""

    theta = {
        name: posterior_mean(design.cell_wall_hits[k], design.cell_trials[k]) for k, name in enumerate(CUSP_NAMES)
    }
    theta[POOLED_NAME] = posterior_mean(design.pooled_wall_hits, design.pooled_trials)
    for name, lower, upper, _units in SHARED_UNCERTAIN_INPUTS:
        theta[name] = 0.5 * (lower + upper)
    return theta


def catalogue_sample(
    designs: Sequence[CatalogueDesign], *, width_scale: float | None = 1.0
) -> tuple[tuple[dict[str, float], ...], ...]:
    """Per-design frozen sample (index-aligned with ``designs``)."""

    rows = unit_qmc_rows()
    return tuple(design_theta_rows(design, rows=rows, width_scale=width_scale) for design in designs)


def catalogue_sample_sha256(sample: Sequence[Sequence[Mapping[str, float]]]) -> str:
    return hashlib.sha256(
        canonical_bytes([[dict(theta) for theta in design_rows] for design_rows in sample])
    ).hexdigest()


def unit_rows_sha256(rows: Sequence[Sequence[float]] | None = None) -> str:
    unit = unit_qmc_rows() if rows is None else rows
    return hashlib.sha256(canonical_bytes([list(row) for row in unit])).hexdigest()
