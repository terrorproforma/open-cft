"""Read-only post-hoc audit of the MDO L0 campaign v1 bundle (``results/``).

Adversarial stance: every number the campaign reports is re-derived here from
the immutable bundle, from Git, or from an INDEPENDENT re-implementation of the
evaluation chain (closure CL-1 + L0 conservation relations), the CVaR
aggregation, Pareto dominance, the hypervolume (WFG exclusive-hypervolume
recursion, a different algorithm from the campaign's slicing implementation),
the Wilson interval and the stdlib Latin-hypercube designs.  The package's own
``model.evaluate_design`` is then replayed separately so the reader can see
which comparisons are bit-exact and which are "within tolerance".

Nothing is written under ``results/``; the script refuses ``--json`` targets
inside it.  Optional stages that need the pinned ML runtime (pymoo for the
NSGA-III replay, torch/botorch for the qLogNEHVI replay) are skipped with a
reason when the libraries are absent, never failed.

Usage (from ``modern/``)::

    python -m experiments.mdo_l0_campaign_v1.audit_replay [--table] [--json PATH]
        [--dense] [--nsga3] [--bo SEED [--bo SEED ...]]

Exit code 0 iff ``report["passed"]`` (integrity + replay checks; disclosures
never change the exit code).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Callable, Mapping, Sequence

EXPERIMENT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENT / "results"
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent
RESULTS_REL = "modern/experiments/mdo_l0_campaign_v1/results"
EXPERIMENT_REL = "modern/experiments/mdo_l0_campaign_v1"

# ---- immutable bindings (asserted, never edited) ------------------------------
PREREGISTRATION_COMMIT = "4898d0fd3decddc5f308072e724d1936660c00e9"
PREREGISTRATION_SUBJECT = "preregister MDO L0 campaign v1"
RESULT_COMMIT = "c553124b7393890d8ee9c6fc022e536c8a1fd35e"
DASHBOARD_COMMIT = "e642f38cd613e3d687c32777080d8aefae93c7b3"
PAPER_COMMIT = "ba6875f604746e8fbeaf2aee2bdf06b8f06bdc04"
SHAKEDOWN_GIT_HEAD = "a1a53300cdcfcb59d9b82b75697737fe772390c4"
RESULTS_TREE = "89e6e69f861fa201f8ad91ca9635577eba44a683"
MANIFEST_SHA256 = "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381"
SOURCE_SHA256 = "da21671f9661f183f3f980044e000a1fdfd0d3495782ed3b0e1fbf5763a9682e"
PROTOCOL_SEMANTIC_SHA256 = "09755b85393d3b3248941ce52f8c21edb832ce30c6f31e5c6919079c41d496ba"
SHAKEDOWN_FILE_SHA256 = "8b5a829302e7aa800d2c60ca1146d86195a71594482c1699a2698e79d76d5c1e"
SAMPLE_SHA256 = "6e574ff122894e0facf951cdf89069c1b4625d6082a33b7026ff4d8a776db33e"
FROZEN_BLOBS = {
    "protocol.json": "06eef451c8c5f3a3b161f893fc9116787caf2c4c",
    "authorities.json": "6d2ba9a87327ba66e626e7ae98031ed9c5392953",
    "shakedown.json": "bec04e5b7df0c2210942bc58de68a542e43751a1",
    "model.py": "b300e0a68e8d06edbc62f857f37e8f232f8c6253",
    "optimizers.py": "431116046125da84b705937f7df83892a9abb122",
    "experiment.py": "bf5037e553ed4d8af5d8dc9d05b1bad94854ce66",
    "run.py": "8c2d241dad2375fa0d82212c08ebb4bb4ab05376",
}
CLASSIFICATION = (
    "l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_"
    "not_thruster_performance"
)
CLOSURE_ID = "CL-1-multiplicative-cusp-survival"
INFEASIBLE_CODE = "beam_current_exceeds_anode_current"
EXPECTED_NON_RESULTS_FILES_IN_RESULT_COMMIT = (
    "modern/spec/optimization/mdo-l0-campaign-v1.json",
    "modern/tests/experiments/mdo_l0_campaign_v1/test_mdo_v1_results.py",
)
# packages the campaign imports but the frozen hash scope never bound (disclosure F10)
NON_SCOPED_DEPENDENCY_PATHS = (
    "modern/src/cft_revival/experiment_runtime",
    "modern/src/cft_revival/models.py",
    "modern/src/cft_revival/kernels.py",
    "modern/src/cft_revival/kernels",
)

# ---- independent evaluation chain (no cft_revival import) ---------------------
ELEMENTARY_CHARGE_C = 1.602176634e-19
XENON_ATOM_MASS_KG = 2.180171556711138e-25
STANDARD_GRAVITY_M_PER_S2 = 9.80665
CATHODE_INPUT_POWER_W = 15.0
OBJECTIVE_NAMES = (
    "axial_thrust_n",
    "specific_impulse_s",
    "thruster_electrical_to_beam_efficiency",
    "anode_input_power_w",
)
MAXIMIZE = (True, True, True, False)
COMPARISON_SCALE = (0.06, 3000.0, 1.0, 1300.0)
REFERENCE = (0.0, 0.0, 0.0, 1300.0)
DESIGN_VARIABLES = (
    ("discharge_voltage_v", 150.0, 500.0),
    ("anode_current_a", 0.1, 2.5),
    ("propellant_mass_flow_kg_per_s", 2.0e-7, 2.0e-6),
)
THETA_NAMES = (
    "cusp_probability_cell_1",
    "cusp_probability_cell_2",
    "cusp_probability_cell_3",
    "cusp_probability_cell_4",
    "ionized_number_fraction",
    "xe_double_plus_fraction_of_ions",
    "axial_momentum_fraction_of_ion_momentum",
)
THETA_BOUNDS = ((0.0, 0.45),) * 4 + ((0.65, 0.98), (0.0, 0.15), (0.75, 0.98))
QMC_BASES = (2, 3, 5, 7, 11, 13, 17)
QMC_SEED = 20260903
QMC_COUNT = 64
CVAR_TAIL = 16
SEEDS = (101, 202, 303)
STRATEGIES = ("qlognehvi", "nsga3", "lhs")
INITIAL_DESIGN = 16
EVALUATIONS_PER_RUN = 96
# Tolerance declared by THIS audit for the independent re-implementation
# (different operation order than the package's exponent-separated arithmetic).
INDEPENDENT_RELATIVE_TOLERANCE = 1e-12
Z_95 = 1.959963984540054


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class Bundle:
    """Read-only view of the results bundle."""

    def __init__(self, root: Path = RESULTS_ROOT) -> None:
        self.root = root

    def bytes(self, relative: str) -> bytes:
        return (self.root / relative).read_bytes()

    def load(self, relative: str) -> Any:
        return json.loads(self.bytes(relative).decode("utf-8"))

    def run(self, strategy: str, seed: int) -> dict[str, Any]:
        return self.load(f"artifacts/runs/{strategy}-{seed}.json")

    def all_records(self) -> list[dict[str, Any]]:
        """Assessment order: for seed in seeds: for strategy in strategies."""

        records: list[dict[str, Any]] = []
        for seed in SEEDS:
            for strategy in STRATEGIES:
                for record in self.run(strategy, seed)["records"]:
                    copy = dict(record)
                    copy["run"] = f"{strategy}:{seed}"
                    records.append(copy)
        return records

    def event_files(self, directory: str) -> list[dict[str, Any]]:
        names = sorted(
            path.name
            for path in (self.root / directory).iterdir()
            if path.name.endswith(".json") and not path.name.endswith(".sha256.json")
        )
        return [self.load(f"{directory}/{name}") for name in names]


# =============================================================================
# 1. bundle integrity
# =============================================================================


def classify_entry(entry: Mapping[str, Any], data: bytes) -> str:
    if sha256_hex(data) == entry["byte_sha256"] and len(data) == entry["bytes"]:
        return "byte_exact"
    crlf = data.replace(b"\n", b"\r\n")
    if b"\r" not in data and sha256_hex(crlf) == entry["byte_sha256"] and len(crlf) == entry["bytes"]:
        return "eol_only"
    return "mismatch"


def bundle_integrity(bundle: Bundle) -> dict[str, Any]:
    manifest_bytes = bundle.bytes("manifest.json")
    manifest = json.loads(manifest_bytes)
    files = [entry for entry in manifest["artifacts"] if entry["type"] == "file"]
    directories = [entry for entry in manifest["artifacts"] if entry["type"] == "directory"]
    counts = {"byte_exact": 0, "eol_only": 0, "mismatch": 0}
    not_exact: list[dict[str, Any]] = []
    contracts: dict[str, int] = {}
    sidecar_pairs_ok = True
    sidecar_pair_count = 0
    blob_artifacts: list[str] = []
    carriage_returns: list[str] = []
    for entry in files:
        data = bundle.bytes(entry["path"])
        status = classify_entry(entry, data)
        counts[status] += 1
        contracts[entry["contract"]] = contracts.get(entry["contract"], 0) + 1
        if status != "byte_exact":
            not_exact.append({"path": entry["path"], "status": status})
        if b"\r" in data:
            carriage_returns.append(entry["path"])
        if entry["contract"] == "hash-sidecar":
            sidecar_pair_count += 1
            sidecar = json.loads(bundle.bytes(entry["sidecar"]))
            pair_ok = (
                sidecar["artifact"] == entry["path"]
                and sidecar["byte_sha256"] == entry["byte_sha256"] == sha256_hex(data)
                and sidecar["bytes"] == entry["bytes"] == len(data)
                and sidecar["semantic_sha256"] == entry.get("semantic_sha256")
            )
            if sidecar["semantic_sha256"] is None:
                # write_blob artifacts (the frozen shakedown record) carry no semantic hash
                blob_artifacts.append(entry["path"])
            else:
                pair_ok = pair_ok and sidecar["semantic_sha256"] == sha256_hex(data)
                pair_ok = pair_ok and entry.get("canonical_json_sha256") == sha256_hex(data)
            sidecar_pairs_ok = sidecar_pairs_ok and pair_ok
    on_disk = {path.relative_to(bundle.root).as_posix() for path in bundle.root.rglob("*") if path.is_file()}
    listed = {entry["path"] for entry in files} | {"manifest.json"}
    lock_bytes = bundle.bytes("execution-lock.json")
    terminal_bytes = bundle.bytes("terminal.json")
    lock = json.loads(lock_bytes)
    terminal = json.loads(terminal_bytes)
    transitions = bundle.event_files("transitions")
    accesses = bundle.event_files("access")
    counters = bundle.event_files("counters")
    transition_names = [item["transition"] for item in transitions]
    times = {item["transition"]: _utc(item["recorded_at_utc"]["value"]) for item in transitions}
    lock_time = _utc(lock["acquired_at_utc"]["value"])
    # access records precede the operation they announce
    phase_start = {"prebundle": "prebundle-started", "development": "development-started", "assessment": "assessment-started"}
    access_before_operation = all(item["recorded_before_operation"] is True for item in accesses)
    for item in accesses:
        if item["kind"] == "phase":
            access_before_operation = access_before_operation and (
                _utc(item["recorded_at_utc"]["value"]) < times[phase_start[item["operation"]]]
            )
    access_times = [_utc(item["recorded_at_utc"]["value"]) for item in accesses]
    counter_times = [_utc(item["recorded_at_utc"]["value"]) for item in counters]
    # counter k+1 ("before-<op>") is written before access k
    counter_before_access = len(counters) == len(accesses) + 1 and all(
        counter_times[index + 1] < access_times[index] for index in range(len(accesses))
    )
    # solver access spacing must accommodate the recorded run wall clocks
    run_access = {item["operation"]: _utc(item["recorded_at_utc"]["value"]) for item in accesses if item["kind"] == "solver"}
    metrics = bundle.load("artifacts/metrics.json")
    spacing_ok = True
    spacing: dict[str, dict[str, float]] = {}
    ordered_runs = [f"run-{strategy}-{seed}" for seed in SEEDS for strategy in STRATEGIES]
    for index, operation in enumerate(ordered_runs):
        strategy, seed = operation.split("-")[1], operation.split("-")[2]
        wall = metrics["timing"][f"{strategy}:{seed}"]["wall_clock_seconds"]
        following = run_access[ordered_runs[index + 1]] if index + 1 < len(ordered_runs) else times["assessment-accepted"]
        gap = (following - run_access[operation]).total_seconds()
        spacing[operation] = {"gap_to_next_s": gap, "recorded_wall_s": wall}
        spacing_ok = spacing_ok and gap >= wall
    from_lock_to_terminal = (times["terminal"] - lock_time).total_seconds()
    # Git-common lock (only available on the producing clone's common dir)
    git_lock: dict[str, Any] = {"available": False}
    try:
        common = _git("rev-parse", "--git-common-dir")
        common_path = Path(common) if Path(common).is_absolute() else (REPOSITORY / common).resolve()
        lock_path = common_path / "mdo-l0-campaign-v1.execution.lock"
        if lock_path.is_file():
            content = lock_path.read_bytes()
            created = datetime.fromtimestamp(os.stat(lock_path).st_ctime, tz=lock_time.tzinfo)
            git_lock = {
                "available": True,
                "path": lock_path.as_posix(),
                "content_is_preregistration_commit": content == (PREREGISTRATION_COMMIT + "\n").encode("ascii"),
                "bytes": len(content),
                "created_utc": created.isoformat(),
                "created_before_runtime_lock": created <= lock_time,
                "seconds_before_runtime_lock": (lock_time - created).total_seconds(),
            }
    except Exception as error:  # pragma: no cover - git not available
        git_lock = {"available": False, "error": f"{type(error).__name__}: {error}"}
    passed = (
        counts["mismatch"] == 0
        and counts["eol_only"] == 0
        and not carriage_returns
        and sidecar_pairs_ok
        and on_disk == listed
        and manifest["artifact_count"] == len(manifest["artifacts"])
        and manifest["lock_byte_sha256"] == sha256_hex(lock_bytes)
        and manifest["terminal_byte_sha256"] == sha256_hex(terminal_bytes)
        and manifest["state"] == terminal["state"] == "accepted_result"
        and [item["sequence"] for item in transitions] == list(range(1, len(transitions) + 1))
        and transition_names[0] == "lock-acquired"
        and transition_names[-1] == "terminal"
        and [item["sequence"] for item in accesses] == list(range(1, len(accesses) + 1))
        and access_before_operation
        and counter_before_access
        and spacing_ok
        and lock["attempt"] == 1
        and lock["commit"] == PREREGISTRATION_COMMIT
        and terminal["counts"]["attempt_count"] == 1
        and (not git_lock["available"] or (git_lock["content_is_preregistration_commit"] and git_lock["created_before_runtime_lock"]))
    )
    return {
        "manifest_sha256": sha256_hex(manifest_bytes),
        "manifest_sha256_expected": sha256_hex(manifest_bytes) == MANIFEST_SHA256,
        "manifest_state": manifest["state"],
        "artifact_count": manifest["artifact_count"],
        "file_entries": len(files),
        "directory_entries": len(directories),
        "counts": counts,
        "not_byte_exact": not_exact,
        "carriage_return_files": carriage_returns,
        "contracts": contracts,
        "sidecar_pairs": sidecar_pair_count,
        "sidecar_pairs_consistent": sidecar_pairs_ok,
        "blob_artifacts_without_semantic_hash": blob_artifacts,
        "on_disk_not_in_manifest": sorted(on_disk - listed),
        "in_manifest_not_on_disk": sorted(listed - on_disk),
        "lock_byte_sha256_ok": manifest["lock_byte_sha256"] == sha256_hex(lock_bytes),
        "terminal_byte_sha256_ok": manifest["terminal_byte_sha256"] == sha256_hex(terminal_bytes),
        "lock": {key: lock[key] for key in ("attempt", "commit", "command", "device", "host", "clean_worktree_attested", "immutable")},
        "lock_acquired_utc": lock["acquired_at_utc"]["value"],
        "terminal_counts": terminal["counts"],
        "terminal_payload": terminal["payload"],
        "transitions": [(item["transition"], item["recorded_at_utc"]["value"]) for item in transitions],
        "transition_sequence_contiguous": [item["sequence"] for item in transitions] == list(range(1, len(transitions) + 1)),
        "lock_to_terminal_s": from_lock_to_terminal,
        "development_s": (times["development-accepted"] - times["development-started"]).total_seconds(),
        "assessment_s": (times["assessment-accepted"] - times["assessment-started"]).total_seconds(),
        "access_records": len(accesses),
        "access_records_before_operation": access_before_operation,
        "counter_records": len(counters),
        "counter_before_access": counter_before_access,
        "run_access_spacing": spacing,
        "run_access_spacing_accommodates_wall_clocks": spacing_ok,
        "git_common_lock": git_lock,
        "passed": passed,
    }


# =============================================================================
# 2. preregistration integrity (Git + hash chain)
# =============================================================================


def _git(*arguments: str, binary: bool = False) -> Any:
    completed = subprocess.run(["git", *arguments], cwd=REPOSITORY, check=True, capture_output=True)
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


def _git_ok(*arguments: str) -> bool:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, capture_output=True).returncode == 0


def source_hash_from_git(protocol_value: Mapping[str, Any], commit: str) -> tuple[str, list[dict[str, Any]]]:
    """Recompute the code-contract hash from Git blobs, independent of the working tree."""

    digest = hashlib.sha256()
    entries = []
    for pattern in protocol_value["code_contract"]["source_hash_scope"]:
        directory, _, name = pattern.rpartition("/")
        listing = _git("ls-tree", "--name-only", f"{commit}:{directory}").splitlines()
        for relative in sorted(f"{directory}/{item}" for item in listing if fnmatch.fnmatch(item, name)):
            data = _git("cat-file", "blob", f"{commit}:{relative}", binary=True)
            if b"\r" in data:
                raise ValueError(f"carriage return in hashed blob {relative}@{commit}")
            file_sha = sha256_hex(data)
            entries.append({"path": relative, "sha256": file_sha, "bytes": len(data)})
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha.encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest(), entries


def source_hash_from_tree(protocol_value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for pattern in protocol_value["code_contract"]["source_hash_scope"]:
        relative = pattern[len("modern/"):]
        directory, _, name = relative.rpartition("/")
        for path in sorted((MODERN / directory).glob(name)):
            data = path.read_bytes()
            if b"\r" in data:
                raise ValueError(f"carriage return in hashed file {path}")
            rel = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_hex(data).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def preregistration_integrity(bundle: Bundle) -> dict[str, Any]:
    protocol_bytes = (EXPERIMENT / "protocol.json").read_bytes()
    protocol_value = json.loads(protocol_bytes)
    authorities = json.loads((EXPERIMENT / "authorities.json").read_bytes())
    shakedown_bytes = (EXPERIMENT / "shakedown.json").read_bytes()
    shakedown = json.loads(shakedown_bytes)
    sealed_protocol = bundle.load("artifacts/protocol.json")
    sealed_authorities = bundle.load("artifacts/authorities.json")
    sealed_shakedown_bytes = bundle.bytes("artifacts/shakedown.json")
    contract = bundle.load("artifacts/code-contract.json")
    # protocol semantic hash: canonical JSON (sorted keys, compact) of the payload
    def semantic(value: Any) -> str:
        return sha256_hex(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))

    report: dict[str, Any] = {
        "protocol_semantic_sha256": semantic(protocol_value),
        "protocol_semantic_matches_authorities": semantic(protocol_value) == authorities["protocol_semantic_sha256"] == PROTOCOL_SEMANTIC_SHA256,
        "protocol_semantic_matches_shakedown_record": shakedown["protocol_semantic_sha256"] == semantic(protocol_value),
        "sealed_protocol_payload_equals_frozen_file": sealed_protocol == protocol_value,
        "sealed_protocol_is_canonical_json": bundle.bytes("artifacts/protocol.json") == json.dumps(protocol_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        "sealed_authorities_equal_frozen_file": sealed_authorities == authorities,
        "sealed_shakedown_bytes_equal_frozen_file": sealed_shakedown_bytes == shakedown_bytes,
        "shakedown_file_sha256": sha256_hex(shakedown_bytes),
        "shakedown_file_sha256_matches_authorities": sha256_hex(shakedown_bytes) == authorities["shakedown_file_sha256"] == SHAKEDOWN_FILE_SHA256,
        "shakedown_evidentiary": shakedown["evidentiary"],
        "shakedown_outcomes_enter_estimand": shakedown["outcomes_enter_estimand"],
        "shakedown_passed": shakedown["passed"],
        "shakedown_seeds": shakedown["shakedown_plan"]["seeds"],
        "shakedown_seed_overlap": shakedown["disjointness"]["seed_overlap"],
        "shakedown_initial_design_overlap": shakedown["disjointness"]["initial_design_overlap_count"],
        "shakedown_disjointness_proven": shakedown["disjointness"]["proven"],
        "shakedown_source_sha256_equals_authorities": shakedown["source_sha256"] == authorities["source_sha256"],
        "shakedown_git_head": shakedown["git"]["head"],
        "shakedown_git_dirty_entries": shakedown["git"]["dirty_entries"],
        "shakedown_generated_at_utc": shakedown["generated_at_utc"],
        "shakedown_result_root_in_temp": "shakedown" in shakedown["runtime"]["result_root"] and RESULTS_REL not in shakedown["runtime"]["result_root"].replace("\\", "/"),
        "authorities_source_sha256": authorities["source_sha256"],
        "code_contract_artifact_matches_authorities": contract["source_sha256"] == authorities["source_sha256"] == SOURCE_SHA256 and contract["matches"] is True and contract["source_files"] == authorities["source_files"],
        "package_versions_declared_equal_observed": contract["declared_package_versions"] == contract["observed_package_versions"] == protocol_value["code_contract"]["package_versions"],
        "working_tree_source_sha256_equals_authorities": None,
        "git": {"available": False},
    }
    try:
        report["working_tree_source_sha256_equals_authorities"] = source_hash_from_tree(protocol_value) == authorities["source_sha256"]
    except ValueError as error:
        report["working_tree_source_sha256_equals_authorities"] = f"error: {error}"
    # independent recomputation of the seeded shakedown disjointness
    evidentiary_rows = {tuple(row) for seed in SEEDS for row in lhs_rows(INITIAL_DESIGN, Random(seed))}
    shakedown_rows = [tuple(row) for seed in shakedown["shakedown_plan"]["seeds"] for row in lhs_rows(shakedown["shakedown_plan"]["initial_design"], Random(seed))]
    report["shakedown_initial_design_overlap_recomputed"] = sum(1 for row in shakedown_rows if row in evidentiary_rows)
    report["shakedown_seed_namespace_rule_recomputed"] = all(seed < 1000 for seed in SEEDS) and all(seed >= 900_000 for seed in shakedown["shakedown_plan"]["seeds"])
    try:
        git: dict[str, Any] = {"available": True}
        git["prereg_subject"] = _git("show", "-s", "--format=%s", PREREGISTRATION_COMMIT)
        git["prereg_subject_ok"] = git["prereg_subject"] == PREREGISTRATION_SUBJECT
        git["prereg_author_date"] = _git("show", "-s", "--format=%aI", PREREGISTRATION_COMMIT)
        git["prereg_commit_date"] = _git("show", "-s", "--format=%cI", PREREGISTRATION_COMMIT)
        changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", PREREGISTRATION_COMMIT).splitlines()
        git["prereg_changed_files"] = changed
        git["prereg_experiment_path_isolated"] = bool(changed) and all(item.startswith(EXPERIMENT_REL + "/") for item in changed)
        git["prereg_contains_no_results"] = not any("/results/" in item for item in changed)
        git["prereg_pushed_to_authorized_branch"] = _git_ok("merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "origin/exp/mdo-l0-campaign-v1")
        git["prereg_ancestor_of_result"] = _git_ok("merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, RESULT_COMMIT)
        git["result_parent_is_prereg"] = _git("show", "-s", "--format=%P", RESULT_COMMIT) == PREREGISTRATION_COMMIT
        result_changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", RESULT_COMMIT).splitlines()
        git["result_commit_files"] = len(result_changed)
        git["result_commit_files_outside_results"] = sorted(item for item in result_changed if not item.startswith(RESULTS_REL + "/"))
        git["result_commit_outside_results_as_expected"] = tuple(git["result_commit_files_outside_results"]) == EXPECTED_NON_RESULTS_FILES_IN_RESULT_COMMIT
        git["results_tree_at_result_commit"] = _git("rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}")
        git["results_tree_at_head"] = _git("rev-parse", f"HEAD:{RESULTS_REL}")
        git["results_tree_unchanged"] = git["results_tree_at_result_commit"] == git["results_tree_at_head"] == RESULTS_TREE
        git["results_untouched_by_later_commits"] = _git("log", "--oneline", f"{RESULT_COMMIT}..HEAD", "--", RESULTS_REL) == ""
        git["results_worktree_clean"] = _git("status", "--porcelain", "--", RESULTS_REL) == ""
        git["results_worktree_lf"] = "w/crlf" not in _git("ls-files", "--eol", "--", RESULTS_REL)
        blobs = {}
        for name, expected in FROZEN_BLOBS.items():
            at_prereg = _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{EXPERIMENT_REL}/{name}")
            at_head = _git("rev-parse", f"HEAD:{EXPERIMENT_REL}/{name}")
            blobs[name] = {"prereg": at_prereg, "head": at_head, "expected": expected, "ok": at_prereg == at_head == expected}
        git["frozen_blobs"] = blobs
        git["frozen_blobs_unchanged"] = all(item["ok"] for item in blobs.values())
        scope_paths = [
            "modern/src/cft_revival/optimization", "modern/src/cft_revival/active_learning",
            "modern/src/cft_revival/surrogates", "modern/src/cft_revival/physics",
            "modern/spec/optimization/campaign-v1.json",
            f"{EXPERIMENT_REL}/model.py", f"{EXPERIMENT_REL}/optimizers.py", f"{EXPERIMENT_REL}/experiment.py",
        ]
        git["hashed_sources_untouched_since_prereg"] = _git("log", "--oneline", f"{PREREGISTRATION_COMMIT}..HEAD", "--", *scope_paths) == ""
        git["frozen_files_untouched_since_prereg"] = _git("log", "--oneline", f"{PREREGISTRATION_COMMIT}..HEAD", "--", f"{EXPERIMENT_REL}/protocol.json", f"{EXPERIMENT_REL}/authorities.json", f"{EXPERIMENT_REL}/shakedown.json") == ""
        # Imported-but-not-hash-bound packages (disclosure F10). Whether they moved between the
        # preregistration and the RESULT commit is a frozen fact about the evidence; whether they
        # have moved in the CHECKOUT since then is a live-tree fact that legitimately changes
        # (experiment_runtime did at bb756418) and is therefore RECORDED, never asserted.
        git["non_scoped_dependencies_unchanged_prereg_to_result"] = _git("diff", "--name-only", PREREGISTRATION_COMMIT, RESULT_COMMIT, "--", *NON_SCOPED_DEPENDENCY_PATHS) == ""
        git["non_scoped_dependencies_changed_since_prereg"] = _git("diff", "--name-only", PREREGISTRATION_COMMIT, "HEAD", "--", *NON_SCOPED_DEPENDENCY_PATHS).splitlines()
        git["non_scoped_dependencies_unchanged_since_prereg"] = git["non_scoped_dependencies_changed_since_prereg"] == []
        source_hashes = {}
        for commit in (PREREGISTRATION_COMMIT, RESULT_COMMIT, DASHBOARD_COMMIT, PAPER_COMMIT):
            digest, entries = source_hash_from_git(protocol_value, commit)
            source_hashes[commit[:8]] = {"source_sha256": digest, "equals_authorities": digest == authorities["source_sha256"], "entries_equal_authorities": entries == authorities["source_files"], "files": len(entries)}
        git["source_hash_from_blobs"] = source_hashes
        git["source_hash_from_blobs_all_equal"] = all(item["equals_authorities"] and item["entries_equal_authorities"] for item in source_hashes.values())
        # shakedown head: experiment files were untracked (see dirty_entries); package files must be identical
        package_files = [item for item in authorities["source_files"] if not item["path"].startswith(EXPERIMENT_REL)]
        identical = 0
        shakedown_head_available = _git_ok("cat-file", "-e", f"{SHAKEDOWN_GIT_HEAD}^{{commit}}")
        if shakedown_head_available:
            for item in package_files:
                data = _git("cat-file", "blob", f"{SHAKEDOWN_GIT_HEAD}:{item['path']}", binary=True)
                identical += int(sha256_hex(data) == item["sha256"])
        git["shakedown_head_available"] = shakedown_head_available
        git["shakedown_head_package_files_identical"] = f"{identical}/{len(package_files)}" if shakedown_head_available else "unavailable"
        git["shakedown_head_is_rebased_tests_commit"] = (
            shakedown_head_available
            and _git("show", "-s", "--format=%s%n%aI", SHAKEDOWN_GIT_HEAD) == _git("show", "-s", "--format=%s%n%aI", f"{PREREGISTRATION_COMMIT}^")
            and _git("diff", "--stat", SHAKEDOWN_GIT_HEAD, f"{PREREGISTRATION_COMMIT}^", "--", "modern/src", "modern/spec/optimization", EXPERIMENT_REL) == ""
        )
        report["git"] = git
    except Exception as error:  # pragma: no cover - git absent or shallow clone
        report["git"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
    # hash-scope coverage vs. modules actually imported by the campaign
    report["import_scope"] = import_scope_analysis()
    git_report = report["git"]
    passed = bool(
        report["protocol_semantic_matches_authorities"]
        and report["protocol_semantic_matches_shakedown_record"]
        and report["sealed_protocol_payload_equals_frozen_file"]
        and report["sealed_authorities_equal_frozen_file"]
        and report["sealed_shakedown_bytes_equal_frozen_file"]
        and report["shakedown_file_sha256_matches_authorities"]
        and report["shakedown_evidentiary"] is False
        and report["shakedown_outcomes_enter_estimand"] is False
        and report["shakedown_passed"] is True
        and report["shakedown_seed_overlap"] == []
        and report["shakedown_initial_design_overlap"] == 0 == report["shakedown_initial_design_overlap_recomputed"]
        and report["shakedown_seed_namespace_rule_recomputed"]
        and report["shakedown_source_sha256_equals_authorities"]
        and report["code_contract_artifact_matches_authorities"]
        and report["package_versions_declared_equal_observed"]
        and report["working_tree_source_sha256_equals_authorities"] is True
        and (
            not git_report["available"]
            or (
                git_report["prereg_subject_ok"]
                and git_report["prereg_experiment_path_isolated"]
                and git_report["prereg_contains_no_results"]
                and git_report["prereg_pushed_to_authorized_branch"]
                and git_report["result_parent_is_prereg"]
                and git_report["result_commit_outside_results_as_expected"]
                and git_report["results_tree_unchanged"]
                and git_report["results_untouched_by_later_commits"]
                and git_report["frozen_blobs_unchanged"]
                and git_report["hashed_sources_untouched_since_prereg"]
                and git_report["frozen_files_untouched_since_prereg"]
                and git_report["source_hash_from_blobs_all_equal"]
            )
        )
    )
    report["passed"] = passed
    return report


def import_scope_analysis() -> dict[str, Any]:
    """Which cft_revival packages the campaign imports versus the hash-bound scope."""

    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(MODERN / 'src')!r}); sys.path.insert(0, {str(MODERN)!r})\n"
        "before = set(sys.modules)\n"
        "from experiments.mdo_l0_campaign_v1 import model, optimizers, experiment, run\n"
        "print(json.dumps(sorted(m for m in set(sys.modules) - before if m.startswith('cft_revival'))))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).stdout
        modules = json.loads(out.strip().splitlines()[-1])
    except Exception as error:  # pragma: no cover
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    scoped = {"cft_revival.optimization", "cft_revival.active_learning", "cft_revival.surrogates", "cft_revival.physics"}
    loaded = sorted({".".join(item.split(".")[:2]) for item in modules if item.count(".") >= 1})
    return {
        "available": True,
        "imported_packages": loaded,
        "hash_scoped_packages_never_imported": sorted(scoped - set(loaded)),
        "imported_packages_outside_hash_scope": sorted(set(loaded) - scoped),
        "modules": modules,
    }


# =============================================================================
# 3. independent re-implementation of the evaluation chain
# =============================================================================


def radical_inverse(index: int, base: int) -> float:
    result, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def qmc_sample(count: int = QMC_COUNT, seed: int = QMC_SEED, cusp_upper: float = 0.45) -> list[dict[str, float]]:
    start = 17 + seed * 104_729
    sample = []
    for row in range(1, count + 1):
        theta = {}
        for index, (name, base) in enumerate(zip(THETA_NAMES, QMC_BASES)):
            lower, upper = THETA_BOUNDS[index]
            if index < 4:
                upper = cusp_upper
            unit = radical_inverse(start + row, base)
            theta[name] = lower + unit * (upper - lower) if upper > lower else lower
        sample.append(theta)
    return sample


def nominal_theta(cusp_upper: float = 0.45) -> dict[str, float]:
    theta = {}
    for index, name in enumerate(THETA_NAMES):
        lower, upper = THETA_BOUNDS[index]
        if index < 4:
            upper = cusp_upper
        theta[name] = 0.5 * (lower + upper)
    return theta


def cl1_survival(theta: Mapping[str, float]) -> float:
    """CL-1 exactly as written in protocol.json: S(p) = prod_k (1 - p_k)."""

    survival = 1.0
    for name in THETA_NAMES[:4]:
        survival *= 1.0 - theta[name]
    return survival


def charge_fractions(theta: Mapping[str, float]) -> tuple[float, float, float]:
    ionized = theta["ionized_number_fraction"] * cl1_survival(theta)
    neutral = 1.0 - ionized
    double = ionized * theta["xe_double_plus_fraction_of_ions"]
    plus = 1.0 - neutral - double
    return neutral, plus, double


def beam_current_a(mass_flow: float, theta: Mapping[str, float]) -> float:
    _neutral, plus, double = charge_fractions(theta)
    return mass_flow / XENON_ATOM_MASS_KG * ELEMENTARY_CHARGE_C * (plus + 2.0 * double)


def l0_objectives(values: Sequence[float], theta: Mapping[str, float]) -> tuple[float, float, float, float]:
    """Plain-arithmetic L0 conservation relations (thrust, Isp, efficiency, anode power)."""

    voltage, anode_current, mass_flow = values
    _neutral, plus, double = charge_fractions(theta)
    gamma = theta["axial_momentum_fraction_of_ion_momentum"]
    plus_speed = math.sqrt(2.0 * ELEMENTARY_CHARGE_C * voltage / XENON_ATOM_MASS_KG)
    double_speed = math.sqrt(2.0) * plus_speed
    momentum_speed = plus * plus_speed + double * double_speed
    thrust = gamma * mass_flow * momentum_speed
    specific_impulse = gamma * momentum_speed / STANDARD_GRAVITY_M_PER_S2
    beam_current = mass_flow / XENON_ATOM_MASS_KG * ELEMENTARY_CHARGE_C * (plus + 2.0 * double)
    anode_power = voltage * anode_current
    efficiency = voltage * beam_current / (anode_power + CATHODE_INPUT_POWER_W)
    return thrust, specific_impulse, efficiency, anode_power


def cvar(column: Sequence[float], maximize: bool, tail: int = CVAR_TAIL) -> float:
    ordered = sorted(column)
    worst = ordered[:tail] if maximize else ordered[-tail:]
    return math.fsum(worst) / tail


def normalize(objectives: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        ((value - reference) / scale) if maximize else ((reference - value) / scale)
        for value, reference, scale, maximize in zip(objectives, REFERENCE, COMPARISON_SCALE, MAXIMIZE)
    )


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right)) and any(a > b for a, b in zip(left, right))


def nondominated(points: Sequence[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Pairwise O(n^2) dominance filter; exact duplicates collapse to one point."""

    unique = list(dict.fromkeys(points))
    return [p for index, p in enumerate(unique) if not any(dominates(q, p) for j, q in enumerate(unique) if j != index)]


def _product(point: Sequence[float]) -> float:
    result = 1.0
    for coordinate in point:
        result *= coordinate
    return result


def wfg_hypervolume(points: Sequence[tuple[float, ...]]) -> float:
    """WFG exclusive-hypervolume recursion against the origin (all-maximise frame)."""

    front = [p for p in nondominated(points) if all(c > 0.0 for c in p)]
    front.sort(key=lambda p: p[-1], reverse=True)
    return _wfg(front)


def _wfg(front: list[tuple[float, ...]]) -> float:
    if not front:
        return 0.0
    if len(front) == 1:
        return _product(front[0])
    total = 0.0
    for index, point in enumerate(front):
        limited = nondominated([tuple(min(a, b) for a, b in zip(point, other)) for other in front[index + 1:]])
        total += _product(point) - _wfg(limited)
    return total


def lhs_rows(count: int, rng: Random, dimensions: int = 3) -> list[tuple[float, ...]]:
    columns = []
    for _dimension in range(dimensions):
        strata = list(range(count))
        rng.shuffle(strata)
        columns.append([(stratum + rng.random()) / count for stratum in strata])
    return [tuple(columns[d][i] for d in range(dimensions)) for i in range(count)]


def denormalize(unit: Sequence[float]) -> list[float]:
    return [lower + float(coordinate) * (upper - lower) for coordinate, (_name, lower, upper) in zip(unit, DESIGN_VARIABLES)]


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return centre - half, centre + half


def _front_ids(rows: Mapping[str, tuple[float, ...]]) -> set[str]:
    """Design ids on the front; an exactly duplicated point keeps its first id only."""

    front = set(nondominated(list(rows.values())))
    ids: set[str] = set()
    seen: set[tuple[float, ...]] = set()
    for key, point in rows.items():
        if point in front and point not in seen:
            ids.add(key)
            seen.add(point)
    return ids


def independent_replay(bundle: Bundle) -> dict[str, Any]:
    started = time.perf_counter()
    sealed_sample = bundle.load("artifacts/uncertain-sample.json")
    sample = qmc_sample()
    nominal = nominal_theta()
    survivals = [cl1_survival(theta) for theta in sample]
    report: dict[str, Any] = {
        "sample_bit_exact": sample == sealed_sample["sample"],
        "nominal_bit_exact": nominal == sealed_sample["nominal"],
        "sample_size": len(sample),
        "survival_min": min(survivals),
        "survival_max": max(survivals),
        "survival_mean": math.fsum(survivals) / len(survivals),
    }
    runs = {f"{strategy}:{seed}": bundle.run(strategy, seed) for seed in SEEDS for strategy in STRATEGIES}
    # (a) shared initial design and (b) full LHS runs from the stdlib stream
    initial_ok: dict[str, bool] = {}
    lhs_ok: dict[str, bool] = {}
    for seed in SEEDS:
        rng = Random(seed)
        first = [denormalize(row) for row in lhs_rows(INITIAL_DESIGN, rng)]
        rest = [denormalize(row) for row in lhs_rows(EVALUATIONS_PER_RUN - INITIAL_DESIGN, rng)]
        for strategy in STRATEGIES:
            records = runs[f"{strategy}:{seed}"]["records"]
            initial_ok[f"{strategy}:{seed}"] = [r["design"]["values"] for r in records[:INITIAL_DESIGN]] == first and all(r["batch"] == 0 for r in records[:INITIAL_DESIGN])
        lhs_ok[str(seed)] = [r["design"]["values"] for r in runs[f"lhs:{seed}"]["records"]] == first + rest
    report["shared_initial_design_bit_exact"] = initial_ok
    report["lhs_designs_bit_exact"] = lhs_ok
    # every recorded evaluation through the independent chain
    worst = {"robust_objectives": 0.0, "robust_statistics": 0.0, "nominal_objectives": 0.0, "margins": 0.0}
    counts = {"success": 0, "infeasible": 0}
    duplicates: dict[str, int] = {}
    consistency_failures: list[dict[str, Any]] = []
    out_of_bounds: list[dict[str, Any]] = []
    non_finite: list[dict[str, Any]] = []
    for key, run in runs.items():
        seen: set[str] = set()
        duplicates[key] = 0
        for record in run["records"]:
            values = record["design"]["values"]
            if record["design"]["design_id"] in seen:
                duplicates[key] += 1
            seen.add(record["design"]["design_id"])
            for (name, lower, upper), value in zip(DESIGN_VARIABLES, values):
                if not (lower <= value <= upper) or not math.isfinite(value):
                    out_of_bounds.append({"run": key, "index": record["index"], "variable": name, "value": value})
            robust_margin = values[1] - max(beam_current_a(values[2], theta) for theta in sample)
            nominal_margin = values[1] - beam_current_a(values[2], nominal)
            worst["margins"] = max(worst["margins"], _relative(robust_margin, record["constraints"]["robust_beam_current_margin_a"]), _relative(nominal_margin, record["constraints"]["nominal_beam_current_margin_a"]))
            counts[record["status"]] += 1
            if record["status"] == "infeasible":
                ok = (
                    robust_margin < 0.0
                    and record["constraints"]["robust_beam_current_margin_a"] < 0.0
                    and record["failure_code"] == INFEASIBLE_CODE
                    and record["robust_objectives"] is None
                    and record["robust_statistics"] is None
                    and record["sample_result_sha256"] is None
                )
            else:
                ok = (
                    robust_margin >= 0.0
                    and record["constraints"]["robust_beam_current_margin_a"] >= 0.0
                    and record["failure_code"] is None
                    and record["robust_objectives"] is not None
                )
                if ok:
                    rows = [l0_objectives(values, theta) for theta in sample]
                    for j, name in enumerate(OBJECTIVE_NAMES):
                        column = [row[j] for row in rows]
                        recorded = record["robust_objectives"][name]
                        if not math.isfinite(recorded):
                            non_finite.append({"run": key, "index": record["index"], "objective": name})
                        worst["robust_objectives"] = max(worst["robust_objectives"], _relative(cvar(column, MAXIMIZE[j]), recorded))
                        stats = record["robust_statistics"][name]
                        worst["robust_statistics"] = max(
                            worst["robust_statistics"],
                            _relative(stats["cvar"], recorded),
                            _relative(stats["mean"], math.fsum(column) / len(column)),
                            _relative(stats["minimum"], min(column)),
                            _relative(stats["maximum"], max(column)),
                        )
            if record["nominal_objectives"] is None:
                ok = ok and nominal_margin < 0.0
            else:
                ok = ok and nominal_margin >= 0.0
                nominal_row = l0_objectives(values, nominal)
                for j, name in enumerate(OBJECTIVE_NAMES):
                    worst["nominal_objectives"] = max(worst["nominal_objectives"], _relative(nominal_row[j], record["nominal_objectives"][name]))
            if not ok:
                consistency_failures.append({"run": key, "index": record["index"]})
    report["records"] = sum(counts.values())
    report["status_counts"] = counts
    report["duplicate_evaluations_per_run"] = duplicates
    report["worst_relative_difference"] = worst
    report["within_tolerance"] = all(value <= INDEPENDENT_RELATIVE_TOLERANCE for value in worst.values())
    report["relative_tolerance"] = INDEPENDENT_RELATIVE_TOLERANCE
    report["fail_closed_consistency_failures"] = consistency_failures
    report["out_of_bounds"] = out_of_bounds
    report["non_finite"] = non_finite
    # Pareto sets, hypervolumes, paired tests, seed variance
    per_run: dict[str, Any] = {}
    final_hv: dict[str, float] = {}
    curve_worst = 0.0
    for key, run in runs.items():
        records = run["records"]
        successful = [(r["index"], r["design"]["design_id"], normalize([r["robust_objectives"][n] for n in OBJECTIVE_NAMES])) for r in records if r["status"] == "success"]
        rows = {}
        for index, design_id, point in successful:
            rows.setdefault(design_id, (index, point))
        front_ids = _front_ids({design_id: point for design_id, (_index, point) in rows.items()})
        pareto_indices = sorted(rows[design_id][0] for design_id in front_ids)
        hypervolume = wfg_hypervolume([point for _i, _d, point in successful])
        final_hv[key] = hypervolume
        recorded = run["summary"]
        curve = run["hypervolume_curve"]
        for n in range(8, EVALUATIONS_PER_RUN + 1, 8):
            prefix = [normalize([r["robust_objectives"][nm] for nm in OBJECTIVE_NAMES]) for r in records[:n] if r["status"] == "success"]
            value = wfg_hypervolume(prefix)
            curve_worst = max(curve_worst, _relative(value, curve[n - 1]["hypervolume"]) if curve[n - 1]["hypervolume"] else abs(value))
        per_run[key] = {
            "pareto_indices_equal": pareto_indices == recorded["pareto_record_indices"],
            "pareto_set_size": len(pareto_indices),
            "hypervolume_wfg": hypervolume,
            "hypervolume_recorded": recorded["final_hypervolume"],
            "hypervolume_relative_difference": _relative(hypervolume, recorded["final_hypervolume"]),
            "hypervolume_bit_exact": hypervolume == recorded["final_hypervolume"],
            "infeasible_evaluations": recorded["infeasible_evaluations"],
            "curve_monotone": all(b["hypervolume"] >= a["hypervolume"] for a, b in zip(curve, curve[1:])),
            "evaluations": recorded["evaluations"],
        }
    report["per_run"] = per_run
    report["curve_spot_check_worst_relative"] = curve_worst
    report["paired"] = {}
    for right in ("lhs", "nsga3"):
        pairs = [{"seed": seed, "qlognehvi": final_hv[f"qlognehvi:{seed}"], right: final_hv[f"{right}:{seed}"], "bo_wins": final_hv[f"qlognehvi:{seed}"] > final_hv[f"{right}:{seed}"]} for seed in SEEDS]
        wins = sum(item["bo_wins"] for item in pairs)
        report["paired"][f"bo_beats_{right}"] = {"wins": wins, "seeds": len(SEEDS), "pairs": pairs, "passed_at_2_of_3": wins >= 2, "one_sided_sign_test_p": 0.5 ** len(SEEDS) if wins == len(SEEDS) else None}
    report["seed_variance"] = {
        strategy: {
            "mean": statistics.fmean([final_hv[f"{strategy}:{seed}"] for seed in SEEDS]),
            "sample_std": statistics.stdev([final_hv[f"{strategy}:{seed}"] for seed in SEEDS]),
            "minimum": min(final_hv[f"{strategy}:{seed}"] for seed in SEEDS),
            "maximum": max(final_hv[f"{strategy}:{seed}"] for seed in SEEDS),
        }
        for strategy in STRATEGIES
    }
    # pooled fronts and Jaccard
    unique: dict[str, dict[str, Any]] = {}
    for record in bundle.all_records():
        unique.setdefault(record["design"]["design_id"], record)
    robust_rows = {k: normalize([r["robust_objectives"][n] for n in OBJECTIVE_NAMES]) for k, r in unique.items() if r["status"] == "success"}
    nominal_rows = {k: normalize([r["nominal_objectives"][n] for n in OBJECTIVE_NAMES]) for k, r in unique.items() if r["nominal_objectives"] is not None}
    robust_front = _front_ids(robust_rows)
    nominal_front = _front_ids(nominal_rows)
    pooled = bundle.load("artifacts/pooled-fronts.json")
    report["pooled"] = {
        "unique_designs": len(unique),
        "robust_candidates": len(robust_rows),
        "nominal_candidates": len(nominal_rows),
        "robust_front_size": len(robust_front),
        "nominal_front_size": len(nominal_front),
        "shared_designs": len(robust_front & nominal_front),
        "jaccard": len(robust_front & nominal_front) / len(robust_front | nominal_front),
        "robust_front_ids_equal_recorded": robust_front == set(pooled["robust"]["design_ids"]),
        "nominal_front_ids_equal_recorded": nominal_front == set(pooled["nominal"]["design_ids"]),
        "jaccard_equal_recorded": len(robust_front & nominal_front) / len(robust_front | nominal_front) == pooled["jaccard_robust_nominal"],
        "nominal_front_members_robust_feasible": sum(1 for k in nominal_front if unique[k]["constraints"]["robust_beam_current_margin_a"] >= 0.0),
        "robust_hypervolume_wfg": wfg_hypervolume(list(robust_rows.values())),
        "robust_hypervolume_recorded": pooled["robust"]["hypervolume"],
        "nominal_hypervolume_wfg": wfg_hypervolume(list(nominal_rows.values())),
        "nominal_hypervolume_recorded": pooled["nominal"]["hypervolume"],
    }
    report["pooled"]["robust_hypervolume_relative_difference"] = _relative(report["pooled"]["robust_hypervolume_wfg"], pooled["robust"]["hypervolume"])
    report["pooled"]["nominal_hypervolume_relative_difference"] = _relative(report["pooled"]["nominal_hypervolume_wfg"], pooled["nominal"]["hypervolume"])
    # dense reference from the sealed columnar records
    dense = bundle.load("artifacts/dense-reference.json")
    columns = dense["records"]
    dense_robust = [normalize(o) for o, status in zip(columns["robust_objectives"], columns["status"]) if status == "success"]
    dense_nominal = [normalize(o) for o in columns["nominal_objectives"] if o is not None]
    dense_bounds = all(lower <= v[i] <= upper for v in columns["values"] for i, (_n, lower, upper) in enumerate(DESIGN_VARIABLES))
    dense_front = nondominated(dense_robust)
    dense_hv = wfg_hypervolume(dense_front)
    dense_nominal_front = nondominated(dense_nominal)
    dense_nominal_hv = wfg_hypervolume(dense_nominal_front)
    # spot-replay 256 dense designs through the independent chain
    dense_worst = 0.0
    dense_status_ok = True
    stride = max(1, columns["count"] // 256)
    for index in range(0, columns["count"], stride):
        values = columns["values"][index]
        margin = values[1] - max(beam_current_a(values[2], theta) for theta in sample)
        dense_worst = max(dense_worst, _relative(margin, columns["robust_beam_current_margin_a"][index]))
        if (margin < 0.0) != (columns["status"][index] == "infeasible"):
            dense_status_ok = False
        if columns["status"][index] == "success":
            rows = [l0_objectives(values, theta) for theta in sample]
            for j in range(4):
                dense_worst = max(dense_worst, _relative(cvar([row[j] for row in rows], MAXIMIZE[j]), columns["robust_objectives"][index][j]))
    report["dense"] = {
        "count": columns["count"],
        "feasible": sum(1 for status in columns["status"] if status == "success"),
        "infeasible": sum(1 for status in columns["status"] if status != "success"),
        "all_designs_within_bounds": dense_bounds,
        "robust_front_size": len(dense_front),
        "robust_front_size_recorded": dense["fronts"]["robust"]["front_size"],
        "robust_hypervolume_wfg": dense_hv,
        "robust_hypervolume_recorded": dense["fronts"]["robust"]["hypervolume"],
        "robust_hypervolume_relative_difference": _relative(dense_hv, dense["fronts"]["robust"]["hypervolume"]),
        "nominal_front_size": len(dense_nominal_front),
        "nominal_front_size_recorded": dense["fronts"]["nominal"]["front_size"],
        "nominal_hypervolume_wfg": dense_nominal_hv,
        "nominal_hypervolume_recorded": dense["fronts"]["nominal"]["hypervolume"],
        "nominal_hypervolume_relative_difference": _relative(dense_nominal_hv, dense["fronts"]["nominal"]["hypervolume"]),
        "spot_replay_count": len(range(0, columns["count"], stride)),
        "spot_replay_worst_relative": dense_worst,
        "spot_replay_status_consistent": dense_status_ok,
        "attained_fraction_bo_mean": statistics.fmean(final_hv[f"qlognehvi:{seed}"] for seed in SEEDS) / dense_hv,
    }
    report["seconds"] = time.perf_counter() - started
    metrics = bundle.load("artifacts/metrics.json")
    gates = bundle.load("artifacts/gates.json")
    report["passed"] = bool(
        report["sample_bit_exact"]
        and report["nominal_bit_exact"]
        and all(initial_ok.values())
        and all(lhs_ok.values())
        and counts == {"success": 734, "infeasible": 130}
        and report["within_tolerance"]
        and not consistency_failures
        and not out_of_bounds
        and not non_finite
        and all(item["pareto_indices_equal"] for item in per_run.values())
        and all(item["hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE for item in per_run.values())
        and all(item["curve_monotone"] for item in per_run.values())
        and curve_worst <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["paired"]["bo_beats_lhs"]["wins"] == 3 == report["paired"]["bo_beats_nsga3"]["wins"]
        and gates["reported_not_binding"]["bo_beats_random"]["wins"] == 3
        and gates["reported_not_binding"]["bo_beats_nsga3"]["wins"] == 3
        and report["pooled"]["robust_front_ids_equal_recorded"]
        and report["pooled"]["nominal_front_ids_equal_recorded"]
        and report["pooled"]["jaccard_equal_recorded"]
        and report["pooled"]["robust_hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["pooled"]["nominal_hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["dense"]["all_designs_within_bounds"]
        and report["dense"]["robust_front_size"] == report["dense"]["robust_front_size_recorded"]
        and report["dense"]["robust_hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["dense"]["nominal_hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["dense"]["spot_replay_worst_relative"] <= INDEPENDENT_RELATIVE_TOLERANCE
        and report["dense"]["spot_replay_status_consistent"]
        and metrics["dense_reference"]["count"] == 8192
    )
    return report


# =============================================================================
# 4. sensitivity tables (priors + scenarios) re-derived independently
# =============================================================================


def sensitivity_replay(bundle: Bundle) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_value = json.loads((EXPERIMENT / "protocol.json").read_bytes())
    recorded = bundle.load("artifacts/sensitivity.json")
    pooled = bundle.load("artifacts/pooled-fronts.json")
    unique: dict[str, dict[str, Any]] = {}
    for record in bundle.all_records():
        unique.setdefault(record["design"]["design_id"], record)
    campaign_rows = {k: normalize([r["robust_objectives"][n] for n in OBJECTIVE_NAMES]) for k, r in unique.items() if r["status"] == "success"}
    campaign_front = _front_ids(campaign_rows)
    priors = []
    priors_ok = True
    for upper, rec in zip(protocol_value["uncertain_inputs"]["sensitivity_priors"]["cusp_upper_bounds"], recorded["priors"]):
        sample = qmc_sample(cusp_upper=float(upper))
        rows: dict[str, tuple[float, ...]] = {}
        infeasible = 0
        for key, record in unique.items():
            values = record["design"]["values"]
            if values[1] - max(beam_current_a(values[2], theta) for theta in sample) < 0.0:
                infeasible += 1
                continue
            l0_rows = [l0_objectives(values, theta) for theta in sample]
            rows[key] = normalize([cvar([row[j] for row in l0_rows], MAXIMIZE[j]) for j in range(4)])
        front = _front_ids(rows)
        common = set(rows) & set(campaign_rows)
        alternative_common = _front_ids({k: rows[k] for k in common})
        campaign_common = _front_ids({k: campaign_rows[k] for k in common})
        hypervolume = wfg_hypervolume(list(rows.values()))
        survivals = [cl1_survival(theta) for theta in sample]
        row = {
            "cusp_upper": float(upper),
            "feasible": len(rows),
            "infeasible": infeasible,
            "front_size": len(front),
            "common_feasible_designs": len(common),
            "identical_on_common_feasible_set": alternative_common == campaign_common,
            "common_front_symmetric_difference": len(alternative_common ^ campaign_common),
            "jaccard_with_campaign_front": len(front & campaign_front) / len(front | campaign_front),
            "hypervolume_wfg": hypervolume,
            "hypervolume_recorded": rec["hypervolume"],
            "survival_min": min(survivals),
            "survival_max": max(survivals),
            "survival_mean": math.fsum(survivals) / len(survivals),
            "front_ids_equal_recorded": front == set(rec["front_design_ids"]),
        }
        row["matches_recorded"] = bool(
            row["feasible"] == rec["feasible"]
            and row["infeasible"] == rec["infeasible"]
            and row["front_size"] == rec["front_size"]
            and row["common_feasible_designs"] == rec["common_feasible_designs"]
            and row["identical_on_common_feasible_set"] == rec["identical_on_common_feasible_set"]
            and row["jaccard_with_campaign_front"] == rec["jaccard_with_campaign_front"]
            and _relative(hypervolume, rec["hypervolume"]) <= INDEPENDENT_RELATIVE_TOLERANCE
            and _relative(row["survival_max"], rec["survival_max"]) <= INDEPENDENT_RELATIVE_TOLERANCE
            and row["front_ids_equal_recorded"]
        )
        priors_ok = priors_ok and row["matches_recorded"]
        priors.append(row)
    scenarios = []
    scenarios_ok = True
    base = nominal_theta()
    pareto_ids = pooled["robust"]["design_ids"]
    for scenario, rec in zip(protocol_value["uncertain_inputs"]["sensitivity_scenarios"], recorded["scenarios"]):
        theta = dict(base)
        for name, probability in zip(THETA_NAMES[:4], scenario["cusp_probabilities"]):
            theta[name] = float(probability)
        survival = cl1_survival(theta)
        rows = []
        infeasible = 0
        for key in pareto_ids:
            values = unique[key]["design"]["values"]
            if values[1] - beam_current_a(values[2], theta) < 0.0:
                infeasible += 1
                continue
            rows.append(l0_objectives(values, theta))
        hypervolume = wfg_hypervolume([normalize(row) for row in rows]) if rows else 0.0
        row = {
            "id": scenario["id"],
            "cusp_probabilities": list(scenario["cusp_probabilities"]),
            "survival": survival,
            "survival_recorded": rec["survival"],
            "pareto_designs_evaluated": len(rows),
            "pareto_designs_infeasible": infeasible,
            "thrust_max": max(r[0] for r in rows) if rows else None,
            "thrust_max_recorded": rec["objective_ranges"]["axial_thrust_n"]["maximum"] if rec["objective_ranges"]["axial_thrust_n"] else None,
            "efficiency_max": max(r[2] for r in rows) if rows else None,
            "hypervolume_wfg": hypervolume,
            "hypervolume_recorded": rec["hypervolume"],
        }
        row["matches_recorded"] = bool(
            _relative(survival, rec["survival"]) <= INDEPENDENT_RELATIVE_TOLERANCE
            and len(rows) == rec["pareto_designs_evaluated"]
            and infeasible == rec["pareto_designs_infeasible"]
            and (row["thrust_max"] is None or _relative(row["thrust_max"], row["thrust_max_recorded"]) <= INDEPENDENT_RELATIVE_TOLERANCE)
            and (rec["hypervolume"] == 0.0 or _relative(hypervolume, rec["hypervolume"]) <= INDEPENDENT_RELATIVE_TOLERANCE)
        )
        scenarios_ok = scenarios_ok and row["matches_recorded"]
        scenarios.append(row)
    # the Jeffreys rule as written versus the frozen scenario numbers
    v4 = protocol_value["authority"]["wall_loss_v4"]["per_cell_wall_hit"]
    jeffreys_rule = [round((cell["successes"] + 0.5) / (cell["trials"] + 1), 4) for cell in v4.values()]
    frozen_jeffreys = next(item["cusp_probabilities"] for item in protocol_value["uncertain_inputs"]["sensitivity_scenarios"] if item["id"] == "v4_per_cell_jeffreys")
    jeffreys_theta = dict(base)
    rule_theta = dict(base)
    for index, name in enumerate(THETA_NAMES[:4]):
        jeffreys_theta[name] = frozen_jeffreys[index]
        rule_theta[name] = (list(v4.values())[index]["successes"] + 0.5) / (list(v4.values())[index]["trials"] + 1)
    report = {
        "priors": priors,
        "priors_match_recorded": priors_ok,
        "scenarios": scenarios,
        "scenarios_match_recorded": scenarios_ok,
        "design_set_invariance_on_common_set_all_priors": all(item["identical_on_common_feasible_set"] for item in priors),
        "jeffreys_rule_rounded_4dp": jeffreys_rule,
        "jeffreys_frozen_in_protocol": frozen_jeffreys,
        "jeffreys_rule_equals_frozen": jeffreys_rule == frozen_jeffreys,
        "jeffreys_survival_frozen": cl1_survival(jeffreys_theta),
        "jeffreys_survival_unrounded_rule": cl1_survival(rule_theta),
        "seconds": time.perf_counter() - started,
    }
    report["passed"] = bool(priors_ok and scenarios_ok and report["design_set_invariance_on_common_set_all_priors"])
    return report


# =============================================================================
# 5. statistics sanity (Wilson, calibration, gate semantics)
# =============================================================================


def statistics_sanity(bundle: Bundle) -> dict[str, Any]:
    protocol_value = json.loads((EXPERIMENT / "protocol.json").read_bytes())
    v4 = protocol_value["authority"]["wall_loss_v4"]
    export = v4["coupling_export"]
    pooled_wall_hit = v4["pooled_wall_hit"]
    wilson_330 = wilson_interval(330, export["trial_count"])
    survival_v4 = 1.0 - pooled_wall_hit["successes"] / pooled_wall_hit["trials"]
    implied = (1.0 - 0.5 * THETA_BOUNDS[0][1]) ** 4
    gates = bundle.load("artifacts/gates.json")
    binding = gates["binding"]
    return {
        "wilson_330_of_512": list(wilson_330),
        "wilson_matches_protocol_authority": wilson_330 == tuple(export["confidence_interval_95"]) and export["probability"] == 330 / 512,
        "v4_pooled_survival": survival_v4,
        "prior_implied_mean_survival": implied,
        "calibration_gap": implied - survival_v4,
        "calibration_gap_below_0_005": abs(implied - survival_v4) < 0.005,
        "uniform_split_rule": 1.0 - survival_v4 ** 0.25,
        "uniform_split_frozen": next(item["cusp_probabilities"][0] for item in protocol_value["uncertain_inputs"]["sensitivity_scenarios"] if item["id"] == "v4_pooled_uniform_split"),
        "uniform_split_rounds_to_frozen": round(1.0 - survival_v4 ** 0.25, 4) == next(item["cusp_probabilities"][0] for item in protocol_value["uncertain_inputs"]["sensitivity_scenarios"] if item["id"] == "v4_pooled_uniform_split"),
        "binding_gates": {name: item["passed"] for name, item in binding.items()},
        "binding_gate_count": len(binding),
        "all_binding_passed": gates["all_binding_passed"],
        "replay_bit_exact_replayed": binding["replay_bit_exact"]["replayed"],
        "replay_bit_exact_mismatches": binding["replay_bit_exact"]["mismatches"],
        "reported_required_wins": gates["reported_not_binding"]["bo_beats_random"]["required_wins"],
        "reported_seeds": gates["reported_not_binding"]["bo_beats_random"]["seeds"],
        "null_probability_of_passing_reported_gate": sum(math.comb(3, k) for k in (2, 3)) / 8,
        "one_sided_sign_test_p_3_of_3": 0.125,
        "binding_gate_semantics": {
            "replay_bit_exact": "recording integrity (same code, same machine)",
            "l0_domain": "recording integrity (fail-closed rule honoured)",
            "hypervolume_monotone": "self-consistency; implied by set growth for a correct hypervolume",
            "budget_exact": "recording integrity",
            "shared_initial_design": "recording integrity (fairness by construction)",
            "sample_hash": "frozen-constant integrity",
            "pareto_replay": "recording integrity",
            "code_contract": "already enforced at prebundle; cannot fail once prebundle passed",
        },
        "passed": bool(wilson_330 == tuple(export["confidence_interval_95"]) and abs(implied - survival_v4) < 0.005 and gates["all_binding_passed"]),
    }


def metadata_labels(bundle: Bundle) -> dict[str, Any]:
    """Descriptive strings in the run artifacts versus what the protocol and code did."""

    protocol_value = json.loads((EXPERIMENT / "protocol.json").read_bytes())
    acquisition_labels = {str(seed): bundle.run("qlognehvi", seed)["optimizer"]["acquisition"] for seed in SEEDS}
    nsga_info = {str(seed): bundle.run("nsga3", seed)["optimizer"] for seed in SEEDS}
    generations_in_provenance = {
        str(seed): sorted({int(r["provenance"].split("generation=")[1].split(":")[0]) for r in bundle.run("nsga3", seed)["records"]})
        for seed in SEEDS
    }
    label_says_sequential = all("sequential greedy" in label for label in acquisition_labels.values())
    return {
        "protocol_candidate_optimizer": protocol_value["optimizers"]["qlognehvi"]["candidate_optimizer"],
        "recorded_acquisition_label": acquisition_labels["101"],
        "acquisition_label_says_sequential_while_protocol_declares_joint": label_says_sequential and "joint q" in protocol_value["optimizers"]["qlognehvi"]["candidate_optimizer"],
        "nsga3_generations_completed_reported_by_pymoo": {seed: info["generations_completed"] for seed, info in nsga_info.items()},
        "nsga3_generations_declared": protocol_value["budget"]["nsga3_generations"],
        "nsga3_generation_indices_in_provenance": generations_in_provenance,
        "nsga3_pymoo_reported_evaluations": {seed: info["pymoo_reported_evaluations"] for seed, info in nsga_info.items()},
        "qlognehvi_iterations": {str(seed): bundle.run("qlognehvi", seed)["optimizer"]["iterations"] for seed in SEEDS},
        "qlognehvi_device": {str(seed): bundle.run("qlognehvi", seed)["optimizer"]["device"] for seed in SEEDS},
        "passed": all(info["pymoo_reported_evaluations"] == EVALUATIONS_PER_RUN for info in nsga_info.values())
        and all(gens == list(range(6)) for gens in generations_in_provenance.values())
        and all(bundle.run("qlognehvi", seed)["optimizer"]["iterations"] == 20 for seed in SEEDS)
        and all(bundle.run("qlognehvi", seed)["optimizer"]["device"] == "cpu" for seed in SEEDS),
    }


# =============================================================================
# 6. claim boundary
# =============================================================================


def claim_boundary(bundle: Bundle) -> dict[str, Any]:
    protocol_value = json.loads((EXPERIMENT / "protocol.json").read_bytes())
    result = bundle.load("artifacts/campaign-result.json")
    sealed_protocol = bundle.load("artifacts/protocol.json")
    spec_index = json.loads((MODERN / "spec/optimization/mdo-l0-campaign-v1.json").read_bytes())
    campaign_v1 = json.loads((MODERN / "spec/optimization/campaign-v1.json").read_bytes())
    paper_dir = REPOSITORY / "paper"
    texts = {
        "protocol.json": (EXPERIMENT / "protocol.json").read_text(encoding="utf-8"),
        "README.md": (EXPERIMENT / "README.md").read_text(encoding="utf-8"),
        "results/artifacts/campaign-result.json": bundle.bytes("artifacts/campaign-result.json").decode("utf-8"),
        "spec/optimization/mdo-l0-campaign-v1.json": (MODERN / "spec/optimization/mdo-l0-campaign-v1.json").read_text(encoding="utf-8"),
        "visualization/mdo-l0-campaign-v1.html": (MODERN / "visualization/mdo-l0-campaign-v1.html").read_text(encoding="utf-8"),
        "paper/evidence/claims.json": (paper_dir / "evidence/claims.json").read_text(encoding="utf-8"),
        "paper/evidence/result-gates.json": (paper_dir / "evidence/result-gates.json").read_text(encoding="utf-8"),
        "paper/evidence/manifests/mdo-l0-v1.json": (paper_dir / "evidence/manifests/mdo-l0-v1.json").read_text(encoding="utf-8"),
    }
    classification_present = {name: CLASSIFICATION in text for name, text in texts.items()}
    # The spec index and the experiment README carry the boundary sentence rather
    # than the classification identifier; that is recorded, not hidden.
    boundary_sentence_present = {name: "no thruster-performance claim" in text for name, text in texts.items()}
    classification_or_boundary = {
        name: classification_present[name] or (name in ("spec/optimization/mdo-l0-campaign-v1.json", "README.md") and boundary_sentence_present[name])
        for name in texts
    }
    geometry_present = {name: ("geometry" in text.lower()) for name, text in texts.items()}
    geometry_present["results/artifacts/protocol.json"] = "why_geometry_variables_are_excluded" in sealed_protocol["claim_boundary"] and len(sealed_protocol["excluded_legacy_variables"]) == 5
    claims = json.loads(texts["paper/evidence/claims.json"])["claims"]
    mdo_claims = [item for item in claims if item.get("classification") == CLASSIFICATION]
    clm030 = next(item for item in claims if item["id"] == "CLM-030")
    gates_doc = json.loads(texts["paper/evidence/result-gates.json"])
    gate = next(item for item in gates_doc["gates"] if item["id"] == "GATE-MDO-L0-V1") if isinstance(gates_doc, dict) and "gates" in gates_doc else None
    if gate is None and isinstance(gates_doc, dict):
        for value in gates_doc.values():
            if isinstance(value, list):
                gate = next((item for item in value if isinstance(item, dict) and item.get("id") == "GATE-MDO-L0-V1"), None)
                if gate:
                    break
    non_claims = list(clm030.get("non_claims", []))
    return {
        "classification": CLASSIFICATION,
        "classification_present": classification_present,
        "boundary_sentence_present": boundary_sentence_present,
        "classification_or_boundary_present": classification_or_boundary,
        "documents_without_classification_identifier": sorted(name for name, present in classification_present.items() if not present),
        "protocol_classification": protocol_value["classification"] == CLASSIFICATION,
        "sealed_protocol_classification": sealed_protocol["classification"] == CLASSIFICATION,
        "campaign_result_classification": result["classification"] == CLASSIFICATION,
        "campaign_result_claim_boundary_equals_protocol": result["claim_boundary"] == protocol_value["claim_boundary"]["statement"],
        "campaign_result_closure": result["closure"] == CLOSURE_ID == protocol_value["closures"]["CL-1"]["id"],
        "forbidden_readings": protocol_value["claim_boundary"]["forbidden_readings"],
        "geometry_exclusion_in_protocol": "why_geometry_variables_are_excluded" in protocol_value["claim_boundary"] and len(protocol_value["excluded_legacy_variables"]) == 5,
        "geometry_mentioned": geometry_present,
        "paper_claims_with_classification": [item["id"] for item in mdo_claims],
        "clm030_non_claims": non_claims,
        "clm030_non_claims_cover_performance_and_geometry": any("thruster-performance" in item for item in non_claims) and any("geometry" in item for item in non_claims),
        "clm030_binds_result_and_prereg_commits": clm030["bindings"]["results_commit"] == RESULT_COMMIT and clm030["bindings"]["preregistration_commit"] == PREREGISTRATION_COMMIT and clm030["bindings"]["dashboard_commit"] == DASHBOARD_COMMIT,
        "gate_found": gate is not None,
        "gate_opens_level": gate.get("opens_level", "?") if gate else "?",
        "gate_kind": gate.get("kind", "?") if gate else "?",
        "spec_index_results_pointer_ok": spec_index["results"]["manifest_sha256"] == MANIFEST_SHA256 and spec_index["results"]["preregistration_commit"] == PREREGISTRATION_COMMIT and spec_index["results"]["terminal_state"] == "accepted_result",
        "campaign_v1_benchmark_results_null": campaign_v1["benchmark"]["results"] is None,
        "passed": bool(
            all(classification_or_boundary.values())
            and protocol_value["classification"] == CLASSIFICATION
            and sealed_protocol["classification"] == CLASSIFICATION
            and geometry_present["results/artifacts/protocol.json"]
            and result["classification"] == CLASSIFICATION
            and result["claim_boundary"] == protocol_value["claim_boundary"]["statement"]
            and result["closure"] == CLOSURE_ID
            and len(protocol_value["excluded_legacy_variables"]) == 5
            and geometry_present["protocol.json"]
            and geometry_present["README.md"]
            and geometry_present["paper/evidence/claims.json"]
            and any("geometry" in item for item in non_claims)
            and any("thruster-performance" in item for item in non_claims)
            and clm030["bindings"]["results_commit"] == RESULT_COMMIT
            and gate is not None
            and (gate.get("opens_level", "x") is None)
            and spec_index["results"]["manifest_sha256"] == MANIFEST_SHA256
            and campaign_v1["benchmark"]["results"] is None
        ),
    }


# =============================================================================
# 7. package replays (bit-exact, optional ML runtime)
# =============================================================================


def package_replay(bundle: Bundle, *, dense: bool = False, nsga3: bool = False, bo_seeds: Sequence[int] = (), progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from cft_revival.experiment_runtime.canonical import canonical_bytes

    from . import experiment as ex
    from . import model as m
    from . import optimizers as opt

    def say(text: str) -> None:
        if progress is not None:
            progress(text)

    report: dict[str, Any] = {}
    started = time.perf_counter()
    records = bundle.all_records()
    replay = ex.replay_records(records)
    design_ids_ok = all(m.make_design(r["design"]["values"]).design_id == r["design"]["design_id"] for r in records)
    report["records_864"] = {"replayed": replay["replayed"], "mismatches": replay["mismatches"], "bit_exact": replay["passed"] and design_ids_ok, "design_ids_recompute": design_ids_ok, "seconds": time.perf_counter() - started}
    pooled = bundle.load("artifacts/pooled-fronts.json")
    report["pooled_fronts_bit_exact"] = canonical_bytes(ex.pooled_fronts(records)) == canonical_bytes(pooled)
    per_strategy = bundle.load("artifacts/per-strategy-fronts.json")
    report["per_strategy_fronts_bit_exact"] = all(
        canonical_bytes(ex.pooled_fronts([r for r in records if r["run"].startswith(strategy + ":")])) == canonical_bytes(per_strategy[strategy])
        for strategy in STRATEGIES
    )
    tick = time.perf_counter()
    sensitivity = ex.cusp_sensitivity(records, ex.protocol(), pooled["robust"]["design_ids"])
    report["sensitivity_bit_exact"] = {"bit_exact": canonical_bytes(sensitivity) == canonical_bytes(bundle.load("artifacts/sensitivity.json")), "seconds": time.perf_counter() - tick}
    report["sample_sha256"] = m.sample_sha256(m.uncertain_sample())
    report["sample_sha256_matches_protocol"] = report["sample_sha256"] == SAMPLE_SHA256
    keys = ("design", "constraints", "status", "failure_code", "robust_objectives", "robust_statistics", "nominal_objectives", "sample_result_sha256")

    def compare(replayed: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        mismatches = []
        for a, b in zip(replayed, recorded, strict=True):
            for key in keys:
                if canonical_bytes(a[key]) != canonical_bytes(b[key]):
                    mismatches.append({"index": a["index"], "key": key})
                    break
            if a["batch"] != b["batch"] or a["provenance"] != b["provenance"]:
                mismatches.append({"index": a["index"], "key": "batch/provenance"})
        return mismatches

    def ledger(strategy: str, seed: int) -> Any:
        return opt.RunLedger(strategy=strategy, seed=seed, budget=EVALUATIONS_PER_RUN, sample=m.uncertain_sample(), nominal=m.nominal_theta(), tail_fraction=m.CVAR_TAIL_FRACTION)

    if dense:
        say("dense reference (8192 designs)")
        tick = time.perf_counter()
        reference = ex.dense_reference(8192, QMC_SEED)
        recorded = bundle.load("artifacts/dense-reference.json")
        compact = ex.compact_records(reference["records"])
        report["dense"] = {
            "columns_bit_exact": {key: compact[key] == recorded["records"][key] for key in recorded["records"] if key != "count"},
            "fronts_bit_exact": canonical_bytes(reference["fronts"]) == canonical_bytes(recorded["fronts"]),
            "separability_bit_exact": canonical_bytes(ex.separability_check(reference["records"])) == canonical_bytes(bundle.load("artifacts/dense-reference-summary.json")["separability"]),
            "feasible": reference["feasible"],
            "infeasible": reference["infeasible"],
            "seconds": time.perf_counter() - tick,
        }
        report["dense"]["bit_exact"] = all(report["dense"]["columns_bit_exact"].values()) and report["dense"]["fronts_bit_exact"] and report["dense"]["separability_bit_exact"]
    else:
        report["dense"] = {"skipped": "pass --dense to replay the 8192-point reference through the package"}
    if nsga3:
        try:
            import pymoo  # noqa: F401
        except ImportError as error:
            report["nsga3"] = {"skipped": f"pymoo unavailable: {error}"}
        else:
            runs = {}
            for seed in SEEDS:
                say(f"NSGA-III seed {seed}")
                recorded = bundle.run("nsga3", seed)
                led = ledger("nsga3", seed)
                tick = time.perf_counter()
                info = opt.run_nsga3(led, initial_count=INITIAL_DESIGN, population_size=INITIAL_DESIGN, generations=EVALUATIONS_PER_RUN // INITIAL_DESIGN, reference_direction_seed=1)
                summary = led.summary()
                runs[str(seed)] = {
                    "wall_seconds": time.perf_counter() - tick,
                    "record_mismatches": compare(led.records, recorded["records"]),
                    "curve_bit_exact": [c["hypervolume"] for c in led.hypervolume_curve] == [c["hypervolume"] for c in recorded["hypervolume_curve"]],
                    "final_hypervolume_bit_exact": summary["final_hypervolume"] == recorded["summary"]["final_hypervolume"],
                    "pareto_indices_equal": summary["pareto_record_indices"] == recorded["summary"]["pareto_record_indices"],
                    "pymoo_reported_evaluations": info["pymoo_reported_evaluations"],
                    "generations_completed_reported_by_pymoo": info["generations_completed"],
                }
                runs[str(seed)]["bit_exact"] = not runs[str(seed)]["record_mismatches"] and runs[str(seed)]["curve_bit_exact"] and runs[str(seed)]["final_hypervolume_bit_exact"] and runs[str(seed)]["pareto_indices_equal"]
            report["nsga3"] = {"runs": runs, "bit_exact": all(item["bit_exact"] for item in runs.values())}
    else:
        report["nsga3"] = {"skipped": "pass --nsga3 to replay NSGA-III through pymoo"}
    if bo_seeds:
        try:
            import torch  # noqa: F401
            from cft_revival.optimization.botorch_adapter import load_api

            load_api()
        except Exception as error:
            report["qlognehvi"] = {"skipped": f"torch/botorch unavailable: {type(error).__name__}: {error}"}
        else:
            runs = {}
            for seed in bo_seeds:
                say(f"qLogNEHVI seed {seed} (cpu float64, ~9 min)")
                recorded = bundle.run("qlognehvi", seed)
                led = ledger("qlognehvi", seed)
                tick = time.perf_counter()
                info = opt.run_qlognehvi(
                    led,
                    initial_count=INITIAL_DESIGN,
                    batch_size=4,
                    device="cpu",
                    num_restarts=4,
                    raw_samples=128,
                    mc_samples=32,
                    fit_noise_floor=1e-6,
                    sequential=False,
                    maxiter=100,
                    progress=lambda entry: say(f"  it {entry['iteration']} n={entry['evaluations']} fit {entry['fit_seconds']:.1f}s acq {entry['acquisition_seconds']:.1f}s hv {entry['hypervolume']:.6f}"),
                )
                wall = time.perf_counter() - tick
                summary = led.summary()
                mismatches = compare(led.records, recorded["records"])
                first_divergent = next((i for i, (a, b) in enumerate(zip(led.records, recorded["records"])) if a["design"]["values"] != b["design"]["values"]), None)
                recorded_acq = [e["acquisition_value"] for e in recorded["optimizer"]["iteration_log"]]
                replay_acq = [e["acquisition_value"] for e in info["iteration_log"]]
                runs[str(seed)] = {
                    "wall_seconds": wall,
                    "recorded_wall_seconds": recorded["summary"]["wall_clock_seconds"],
                    "record_mismatches": len(mismatches),
                    "first_divergent_record_index": first_divergent,
                    "curve_bit_exact": [c["hypervolume"] for c in led.hypervolume_curve] == [c["hypervolume"] for c in recorded["hypervolume_curve"]],
                    "final_hypervolume_replay": summary["final_hypervolume"],
                    "final_hypervolume_recorded": recorded["summary"]["final_hypervolume"],
                    "final_hypervolume_bit_exact": summary["final_hypervolume"] == recorded["summary"]["final_hypervolume"],
                    "pareto_indices_equal": summary["pareto_record_indices"] == recorded["summary"]["pareto_record_indices"],
                    "acquisition_values_bit_exact": recorded_acq == replay_acq,
                    "acquisition_values_max_abs_diff": max(abs(a - b) for a, b in zip(recorded_acq, replay_acq)),
                    "iterations": info["iterations"],
                    "torch_threads": torch.get_num_threads(),
                }
                runs[str(seed)]["bit_exact"] = not mismatches and runs[str(seed)]["curve_bit_exact"] and runs[str(seed)]["final_hypervolume_bit_exact"]
            report["qlognehvi"] = {"runs": runs, "bit_exact": all(item["bit_exact"] for item in runs.values())}
    else:
        report["qlognehvi"] = {"skipped": "pass --bo SEED to replay qLogNEHVI through BoTorch (cpu)"}
    report["seconds"] = time.perf_counter() - started
    report["passed"] = bool(
        report["records_864"]["bit_exact"]
        and report["pooled_fronts_bit_exact"]
        and report["per_strategy_fronts_bit_exact"]
        and report["sensitivity_bit_exact"]["bit_exact"]
        and report["sample_sha256_matches_protocol"]
        and report["dense"].get("bit_exact", True)
        and report["nsga3"].get("bit_exact", True)
        and report["qlognehvi"].get("bit_exact", True)
    )
    return report


# =============================================================================
# findings table + driver
# =============================================================================


def findings(report: Mapping[str, Any]) -> list[dict[str, str]]:
    b, p, r, s, t, c, k = (report[key] for key in ("bundle", "preregistration", "independent", "sensitivity", "statistics", "claims", "package"))
    git = p["git"]
    rows: list[dict[str, str]] = []

    def add(fid: str, check: str, observed: str, verdict: str) -> None:
        rows.append({"id": fid, "check": check, "observed": observed, "verdict": verdict})

    add("F1", "manifest entries byte-exact on this LF checkout", f"{b['counts']['byte_exact']} byte-exact, {b['counts']['eol_only']} EOL-only, {b['counts']['mismatch']} mismatch of {b['file_entries']} files (+{b['directory_entries']} dirs); {b['sidecar_pairs']} sidecar pairs consistent={b['sidecar_pairs_consistent']}; CR bytes in {len(b['carriage_return_files'])} files", "PASS" if b["counts"]["mismatch"] == 0 and b["counts"]["eol_only"] == 0 else "FAIL")
    add("F2", "lock / terminal / transition-log hashes and ordering", f"lock={b['lock_byte_sha256_ok']}, terminal={b['terminal_byte_sha256_ok']}, transitions 1..{len(b['transitions'])} contiguous={b['transition_sequence_contiguous']}, {b['access_records']} access records before their operations={b['access_records_before_operation']}, counters precede access={b['counter_before_access']}, run spacing >= wall clocks={b['run_access_spacing_accommodates_wall_clocks']}", "PASS" if b["passed"] else "FAIL")
    lock = b["git_common_lock"]
    add("F3", "Git-common execution lock", (f"content == prereg commit: {lock['content_is_preregistration_commit']}, created {lock['seconds_before_runtime_lock']:.3f} s before the runtime lock" if lock["available"] else "not on this clone"), "PASS" if (not lock["available"] or (lock["content_is_preregistration_commit"] and lock["created_before_runtime_lock"])) else "FAIL")
    add("F4", "preregistration commit isolation / push / no results", (f"isolated={git['prereg_experiment_path_isolated']}, no results={git['prereg_contains_no_results']}, on origin/exp branch={git['prereg_pushed_to_authorized_branch']}, result parent is prereg={git['result_parent_is_prereg']}" if git["available"] else "git unavailable"), "PASS" if (not git["available"] or (git["prereg_experiment_path_isolated"] and git["prereg_contains_no_results"] and git["prereg_pushed_to_authorized_branch"])) else "FAIL")
    add("F5", "protocol / authorities / shakedown bindings", f"protocol semantic == authorities == shakedown record: {p['protocol_semantic_matches_authorities'] and p['protocol_semantic_matches_shakedown_record']}; shakedown file sha == authorities: {p['shakedown_file_sha256_matches_authorities']}; sealed copies equal frozen files: {p['sealed_protocol_payload_equals_frozen_file'] and p['sealed_authorities_equal_frozen_file'] and p['sealed_shakedown_bytes_equal_frozen_file']}", "PASS" if p["protocol_semantic_matches_authorities"] and p["shakedown_file_sha256_matches_authorities"] else "FAIL")
    add("F6", "shakedown non-evidentiary and disjoint", f"evidentiary={p['shakedown_evidentiary']}, outcomes_enter_estimand={p['shakedown_outcomes_enter_estimand']}, seeds {p['shakedown_seeds']} vs {list(SEEDS)}, initial-design overlap recomputed={p['shakedown_initial_design_overlap_recomputed']}, temp result root={p['shakedown_result_root_in_temp']}", "PASS" if p["shakedown_evidentiary"] is False and p["shakedown_initial_design_overlap_recomputed"] == 0 else "FAIL")
    add("F7", "code hash from Git blobs at prereg/result/dashboard/paper commits", (", ".join(f"{k}={v['equals_authorities']}" for k, v in git["source_hash_from_blobs"].items()) + f"; working tree={p['working_tree_source_sha256_equals_authorities']}; shakedown head package files {git['shakedown_head_package_files_identical']}") if git["available"] else "git unavailable", "PASS" if (not git["available"] or git["source_hash_from_blobs_all_equal"]) and p["working_tree_source_sha256_equals_authorities"] is True else "FAIL")
    add("F8", "no hashed-source or frozen-file change since prereg; results tree immutable", (f"hashed sources untouched={git['hashed_sources_untouched_since_prereg']}, frozen files untouched={git['frozen_files_untouched_since_prereg']}, results tree {git['results_tree_at_head'][:12]} unchanged={git['results_tree_unchanged']}, non-scoped deps unchanged={git['non_scoped_dependencies_unchanged_since_prereg']}") if git["available"] else "git unavailable", "PASS" if (not git["available"] or (git["hashed_sources_untouched_since_prereg"] and git["frozen_files_untouched_since_prereg"] and git["results_tree_unchanged"])) else "FAIL")
    add("F9", "files outside results/ in the result commit", (", ".join(git["result_commit_files_outside_results"]) or "none") if git["available"] else "git unavailable", "DISCLOSURE" if (git["available"] and git["result_commit_files_outside_results"]) else "PASS")
    scope = p["import_scope"]
    add("F10", "hash scope vs modules actually imported", f"never imported but hash-bound: {scope.get('hash_scoped_packages_never_imported')}; imported but not hash-bound: {scope.get('imported_packages_outside_hash_scope')}" if scope.get("available") else "unavailable", "DISCLOSURE")
    add("F11", "frozen QMC sample and nominal point (independent radical inverse)", f"sample bit-exact={r['sample_bit_exact']}, nominal bit-exact={r['nominal_bit_exact']}, survival {r['survival_min']:.4f}..{r['survival_max']:.4f} (mean {r['survival_mean']:.4f})", "PASS" if r["sample_bit_exact"] and r["nominal_bit_exact"] else "FAIL")
    add("F12", "shared 16-point initial design per seed (stdlib LHS)", f"{sum(r['shared_initial_design_bit_exact'].values())}/9 runs bit-exact", "PASS" if all(r["shared_initial_design_bit_exact"].values()) else "FAIL")
    add("F13", "LHS baseline: all 96 designs per seed", f"{sum(r['lhs_designs_bit_exact'].values())}/3 seeds bit-exact", "PASS" if all(r["lhs_designs_bit_exact"].values()) else "FAIL")
    add("F14", "864 records through an independent CL-1 + L0 + CVaR implementation", f"worst relative difference {max(r['worst_relative_difference'].values()):.1e} (tolerance {r['relative_tolerance']:.0e}); status counts {r['status_counts']}; fail-closed inconsistencies {len(r['fail_closed_consistency_failures'])}; out of bounds {len(r['out_of_bounds'])}; non-finite {len(r['non_finite'])}", "PASS" if r["within_tolerance"] and not r["fail_closed_consistency_failures"] and not r["out_of_bounds"] and not r["non_finite"] and r["status_counts"]["infeasible"] == 130 else "FAIL")
    add("F15", "864 records through the package (model.evaluate_design)", f"{k['records_864']['replayed']} replayed, {len(k['records_864']['mismatches'])} mismatches, design ids recompute={k['records_864']['design_ids_recompute']}", "PASS" if k["records_864"]["bit_exact"] else "FAIL")
    hv_bits = sum(item["hypervolume_bit_exact"] for item in r["per_run"].values())
    add("F16", "Pareto sets (pairwise dominance) and final hypervolumes (WFG)", f"Pareto indices equal 9/9={all(i['pareto_indices_equal'] for i in r['per_run'].values())}; HV worst relative {max(i['hypervolume_relative_difference'] for i in r['per_run'].values()):.1e} ({hv_bits}/9 bit-exact); curve spot checks worst {r['curve_spot_check_worst_relative']:.1e}", "PASS" if all(i["pareto_indices_equal"] for i in r["per_run"].values()) and max(i["hypervolume_relative_difference"] for i in r["per_run"].values()) <= INDEPENDENT_RELATIVE_TOLERANCE else "FAIL")
    add("F17", "BO beats random / NSGA-III (paired by seed, independent HV)", f"{r['paired']['bo_beats_lhs']['wins']}/3 and {r['paired']['bo_beats_nsga3']['wins']}/3; one-sided sign-test p = {t['one_sided_sign_test_p_3_of_3']}; the predeclared >=2/3 rule passes with probability {t['null_probability_of_passing_reported_gate']} under a no-difference null", "PASS" if r["paired"]["bo_beats_lhs"]["wins"] == 3 == r["paired"]["bo_beats_nsga3"]["wins"] else "FAIL")
    add("F18", "pooled robust vs nominal fronts", f"robust {r['pooled']['robust_front_size']} vs nominal {r['pooled']['nominal_front_size']}, shared {r['pooled']['shared_designs']}, Jaccard {r['pooled']['jaccard']:.6f} (recorded equal={r['pooled']['jaccard_equal_recorded']}); HV rel diff {r['pooled']['robust_hypervolume_relative_difference']:.1e} / {r['pooled']['nominal_hypervolume_relative_difference']:.1e}", "PASS" if r["pooled"]["robust_front_ids_equal_recorded"] and r["pooled"]["nominal_front_ids_equal_recorded"] else "FAIL")
    add("F19", "dense 8192-point reference (independent fronts/HV from the sealed columns)", f"feasible {r['dense']['feasible']}, infeasible {r['dense']['infeasible']}, all within bounds={r['dense']['all_designs_within_bounds']}; robust front {r['dense']['robust_front_size']} (recorded {r['dense']['robust_front_size_recorded']}), HV rel diff {r['dense']['robust_hypervolume_relative_difference']:.1e}; nominal front {r['dense']['nominal_front_size']} (recorded {r['dense']['nominal_front_size_recorded']}), HV rel diff {r['dense']['nominal_hypervolume_relative_difference']:.1e}; {r['dense']['spot_replay_count']} designs spot-replayed independently (worst {r['dense']['spot_replay_worst_relative']:.1e}); BO mean attains {r['dense']['attained_fraction_bo_mean']:.4f} of the reference", "PASS" if r["dense"]["robust_front_size"] == r["dense"]["robust_front_size_recorded"] and r["dense"]["robust_hypervolume_relative_difference"] <= INDEPENDENT_RELATIVE_TOLERANCE else "FAIL")
    add("F20", "prior-sensitivity table (independent re-evaluation of 758 designs x 4 priors)", "; ".join(f"a={row['cusp_upper']}: feasible {row['feasible']}, front {row['front_size']}, common-set identical {row['identical_on_common_feasible_set']}, Jaccard {row['jaccard_with_campaign_front']:.3f}" for row in s["priors"]) + f"; all match recorded={s['priors_match_recorded']}", "PASS" if s["priors_match_recorded"] else "FAIL")
    jeff = next(row for row in s["scenarios"] if row["id"] == "v4_per_cell_jeffreys")
    nwl = next(row for row in s["scenarios"] if row["id"] == "no_wall_loss")
    add("F21", "scenario table (114 robust-Pareto designs x 5 scenarios)", f"Jeffreys S = {jeff['survival']:.3e}, thrust max {jeff['thrust_max']:.3e} N; no_wall_loss {nwl['pareto_designs_evaluated']} evaluated / {nwl['pareto_designs_infeasible']} infeasible; all match recorded={s['scenarios_match_recorded']}", "PASS" if s["scenarios_match_recorded"] else "FAIL")
    add("F22", "Jeffreys rule vs frozen scenario numbers", f"rule rounded to 4 dp {s['jeffreys_rule_rounded_4dp']} vs frozen {s['jeffreys_frozen_in_protocol']}; S(frozen) {s['jeffreys_survival_frozen']:.4e} vs S(rule) {s['jeffreys_survival_unrounded_rule']:.4e}", "PASS" if s["jeffreys_rule_equals_frozen"] else "DISCLOSURE")
    add("F23", "Wilson interval / prior calibration", f"Wilson(330/512) equals protocol authority={t['wilson_matches_protocol_authority']}; implied survival {t['prior_implied_mean_survival']:.4f} vs v4 {t['v4_pooled_survival']:.4f} (gap {t['calibration_gap']:.4f} < 0.005: {t['calibration_gap_below_0_005']})", "PASS" if t["passed"] else "FAIL")
    add(
        "F24",
        "package replays (this run; the audit document records the pinned-runtime run)",
        "dense: " + (f"bit-exact={k['dense']['bit_exact']} in {k['dense']['seconds']:.0f} s" if "bit_exact" in k["dense"] else "skipped")
        + "; NSGA-III: " + (f"bit-exact {sum(i['bit_exact'] for i in k['nsga3']['runs'].values())}/3 seeds" if "runs" in k["nsga3"] else "skipped")
        + "; qLogNEHVI: " + (", ".join(f"seed {seed}: bit-exact={i['bit_exact']}, {i['wall_seconds']:.0f} s (recorded {i['recorded_wall_seconds']:.0f} s)" for seed, i in k["qlognehvi"]["runs"].items()) if "runs" in k["qlognehvi"] else "skipped"),
        "PASS" if k["dense"].get("bit_exact", True) and k["nsga3"].get("bit_exact", True) and k["qlognehvi"].get("bit_exact", True) else "FAIL",
    )
    add("F25", "claim boundary consistency", f"classification identifier in {sum(c['classification_present'].values())}/{len(c['classification_present'])} documents (boundary sentence only in {c['documents_without_classification_identifier']}); campaign-result claim boundary == protocol: {c['campaign_result_claim_boundary_equals_protocol']}; geometry exclusion in protocol/sealed protocol/paper: {c['geometry_exclusion_in_protocol'] and c['geometry_mentioned']['results/artifacts/protocol.json'] and c['geometry_mentioned']['paper/evidence/claims.json']}; CLM-030 non-claims cover performance+geometry: {c['clm030_non_claims_cover_performance_and_geometry']}; gate {c['gate_kind']} opens_level={c['gate_opens_level']}; campaign-v1 benchmark null={c['campaign_v1_benchmark_results_null']}", "PASS" if c["passed"] else "FAIL")
    add("F26", "binding gates are recording-integrity gates", f"{t['binding_gate_count']}/8 passed; none is an outcome gate (see semantics); acceptance = pipeline integrity, efficacy statements are reported-not-binding", "DISCLOSURE")
    add("F27", "NSGA-III duplicate evaluations (eliminate_duplicates=false)", ", ".join(f"{key}: {value}" for key, value in r["duplicate_evaluations_per_run"].items() if value), "DISCLOSURE" if any(r["duplicate_evaluations_per_run"].values()) else "PASS")
    lab = report["labels"]
    add("F28", "descriptive labels in run artifacts", f"qLogNEHVI optimizer.acquisition says '{lab['recorded_acquisition_label'].split(', ')[-1]}' while protocol/code use joint q (optimize_acqf sequential=False); pymoo reports generations_completed {set(lab['nsga3_generations_completed_reported_by_pymoo'].values())} for {lab['nsga3_generations_declared']} declared generations (provenance generations 0..5, 96 evaluations); iterations {set(lab['qlognehvi_iterations'].values())}, device {set(lab['qlognehvi_device'].values())}", "DISCLOSURE" if lab["acquisition_label_says_sequential_while_protocol_declares_joint"] else ("PASS" if lab["passed"] else "FAIL"))
    return rows


def audit(*, dense: bool = False, nsga3: bool = False, bo_seeds: Sequence[int] = (), progress: Callable[[str], None] | None = None, results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    bundle = Bundle(results_root)
    report: dict[str, Any] = {
        "schema_version": "cft-revival.mdo-l0-campaign-v1.posthoc-audit/1.0.0",
        "results_root": results_root.as_posix(),
        "read_only": True,
        "bundle": bundle_integrity(bundle),
        "preregistration": preregistration_integrity(bundle),
        "independent": independent_replay(bundle),
        "sensitivity": sensitivity_replay(bundle),
        "statistics": statistics_sanity(bundle),
        "claims": claim_boundary(bundle),
        "labels": metadata_labels(bundle),
        "package": package_replay(bundle, dense=dense, nsga3=nsga3, bo_seeds=bo_seeds, progress=progress),
    }
    report["findings"] = findings(report)
    report["passed"] = all(report[key]["passed"] for key in ("bundle", "preregistration", "independent", "sensitivity", "statistics", "claims", "labels", "package"))
    report["disclosures"] = [row["id"] for row in report["findings"] if row["verdict"] == "DISCLOSURE"]
    report["failures"] = [row["id"] for row in report["findings"] if row["verdict"] == "FAIL"]
    return report


def format_table(report: Mapping[str, Any]) -> str:
    lines = ["| id | check | observed | verdict |", "| --- | --- | --- | --- |"]
    for row in report["findings"]:
        lines.append(f"| {row['id']} | {row['check']} | {row['observed']} | {row['verdict']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here (never inside results/)")
    parser.add_argument("--table", action="store_true", help="print the Markdown findings table only")
    parser.add_argument("--dense", action="store_true", help="replay the 8192-point dense reference through the package")
    parser.add_argument("--nsga3", action="store_true", help="replay NSGA-III through pymoo (skipped if unavailable)")
    parser.add_argument("--bo", type=int, action="append", default=[], metavar="SEED", help="replay qLogNEHVI for SEED through BoTorch on cpu (skipped if unavailable)")
    arguments = parser.parse_args(argv)
    if arguments.json is not None:
        target = arguments.json.resolve()
        if RESULTS_ROOT.resolve() in target.parents or target == RESULTS_ROOT.resolve():
            parser.error("--json must not point inside results/ (immutable evidence)")
    report = audit(dense=arguments.dense, nsga3=arguments.nsga3, bo_seeds=tuple(arguments.bo), progress=lambda text: print(text, file=sys.stderr, flush=True))
    if arguments.table:
        sys.stdout.write(format_table(report))
    else:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if arguments.json is not None:
        arguments.json.write_bytes(json.dumps(report, indent=2, sort_keys=True, default=str).encode("utf-8"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
