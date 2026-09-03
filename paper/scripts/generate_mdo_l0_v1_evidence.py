"""Generate hash-bound paper evidence for the MDO L0 campaign v1.

Reads the sealed results bundle of ``modern/experiments/mdo_l0_campaign_v1``
(verified byte-for-byte against ``results/manifest.json``; no end-of-line
tolerance is needed or granted), binds it to the committed results revision,
cross-checks the committed results dashboard against the same bundle, and
writes:

* ``paper/evidence/mdo-l0-v1.json`` — every macro value with the artifact
  path, JSON pointer, formatter and artifact SHA-256 it was read from, or the
  derivation and inputs of a derived macro;
* ``paper/generated/mdo-l0-v1.tex`` — ``\\newcommand`` macros and three
  generated tables (each wrapped in ``\\ArtifactClaim``) for the admitted
  results subsection ``paper/sections/mdo-l0-v1.tex``;
* ``paper/generated/mdo-l0-v1.provenance.json`` — generator/input/output
  hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The campaign is optimiser evidence about the L0
conservation model under the declared closure CL-1; it is not thruster
performance, and every number below is conditional on that closure and on the
declared priors.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
from typing import Any, Callable

EXPERIMENT = Path("modern/experiments/mdo_l0_campaign_v1")
RESULTS = EXPERIMENT / "results"
EVIDENCE_PATH = Path("paper/evidence/mdo-l0-v1.json")
OUTPUT_PATH = Path("paper/generated/mdo-l0-v1.tex")
SIDECAR_PATH = Path("paper/generated/mdo-l0-v1.provenance.json")
SECTION_PATH = Path("paper/sections/mdo-l0-v1.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_mdo_l0_campaign_v1_dashboard.py")
DASHBOARD_HTML = Path("modern/visualization/mdo-l0-campaign-v1.html")

RESULTS_COMMIT_SHA = "c553124b7393890d8ee9c6fc022e536c8a1fd35e"
PREREGISTRATION_COMMIT_SHA = "4898d0fd3decddc5f308072e724d1936660c00e9"
DASHBOARD_COMMIT_SHA = "e642f38cd613e3d687c32777080d8aefae93c7b3"

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "MDO-L0-V1-20260903-864-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/mdo-l0-v1.json")
GATE_ID = "GATE-MDO-L0-V1"
ARTIFACT_ID = "TAB-MDO-L0-V1"
ARTIFACT_CLAIM_ID = "CLM-031"
PROSE_CLAIM_IDS = ("CLM-029", "CLM-030", "CLM-032", "CLM-033", "CLM-034", "CLM-035")
SECTION_BINDING = "\\input{sections/mdo-l0-v1.tex}"
GENERATED_BINDING = "\\input{generated/mdo-l0-v1.tex}"
SECTION_HEADING = (
    "Robust optimiser comparison on the operating-point model under declared cusp-probability uncertainty"
)
TABLE_MACROS = ("MdoHvTable", "MdoRobustNominalTable", "MdoScenarioTable")

EXPERIMENT_ID = "mdo-l0-campaign-v1"
CLASSIFICATION = (
    "l0_model_robust_multiobjective_optimisation_under_declared_input_uncertainty_not_thruster_performance"
)
CLOSURE_ID = "CL-1-multiplicative-cusp-survival"
STRATEGIES = ("qlognehvi", "nsga3", "lhs")
STRATEGY_TOKENS = {"qlognehvi": "Bo", "nsga3": "Nsga", "lhs": "Lhs"}
STRATEGY_LABELS = {"qlognehvi": "BoTorch qLogNEHVI", "nsga3": "pymoo NSGA-III", "lhs": "Latin hypercube"}
SEED_TOKENS = ("A", "B", "C")
OBJECTIVES = (
    "axial_thrust_n",
    "specific_impulse_s",
    "thruster_electrical_to_beam_efficiency",
    "anode_input_power_w",
)
SCENARIO_TOKENS = {
    "no_wall_loss": "NoWallLoss",
    "wide_prior_mean": "PriorMean",
    "v4_pooled_uniform_split": "PooledSplit",
    "wide_prior_upper": "PriorUpper",
    "v4_per_cell_jeffreys": "Jeffreys",
}
PRIOR_TOKENS = ("Zero", "Low", "Campaign", "High")
RUN_ARTIFACTS = tuple(f"artifacts/runs/{s}-{seed}.json" for seed in (101, 202, 303) for s in STRATEGIES)


# --------------------------------------------------------------------------- #
# Formatting (shared with the tests through this module)
# --------------------------------------------------------------------------- #
def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _sci(value: float, digits: int) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def _sig(value: float, digits: int) -> str:
    """Significant-digit rendering: plain decimal in [1e-3, 1e5), scientific outside."""

    value = float(value)
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return _sci(value, digits - 1)
    decimals = digits - 1 - math.floor(math.log10(magnitude))
    if decimals <= 0:
        return f"{round(value, decimals):.0f}"
    return f"{value:.{decimals}f}"


FORMATTERS: dict[str, Callable[[Any], str]] = {
    "int": lambda v: f"{int(v):d}",
    "int_comma": lambda v: f"{int(v):,d}".replace(",", "{,}"),
    "fixed0": lambda v: f"{float(v):.0f}",
    "fixed1": lambda v: f"{float(v):.1f}",
    "fixed2": lambda v: f"{float(v):.2f}",
    "fixed3": lambda v: f"{float(v):.3f}",
    "fixed4": lambda v: f"{float(v):.4f}",
    "fixed6": lambda v: f"{float(v):.6f}",
    "pct1": lambda v: f"{100.0 * float(v):.1f}\\%",
    "min1": lambda v: f"{float(v) / 60.0:.1f}",
    "sci1": lambda v: _sci(float(v), 1),
    "sci2": lambda v: _sci(float(v), 2),
    "sig3": lambda v: _sig(float(v), 3),
    "g": lambda v: f"{float(v):g}",
    "text": lambda v: _tex_escape(str(v)),
    "ident": lambda v: _tex_escape(str(v)).replace("\\_", "\\_\\allowbreak{}").replace("-", "-\\allowbreak{}"),
    "bool": lambda v: "true" if v is True else "false" if v is False else _tex_escape(str(v)),
    "list_g": lambda v: ", ".join(f"{float(x):g}" for x in v),
    "list_int": lambda v: ", ".join(f"{int(x):d}" for x in v),
    "list_text": lambda v: ", ".join(_tex_escape(str(x)) for x in v),
    "sha_short": lambda v: _tex_escape(str(v)[:12]),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


# --------------------------------------------------------------------------- #
# Strict JSON, hashing, pointers
# --------------------------------------------------------------------------- #
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json_bytes(raw: bytes, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate key {key!r}")
            result[key] = value
        return result

    def reject(constant: str) -> None:
        raise ValueError(f"{label}: nonfinite constant {constant!r}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid UTF-8 JSON") from exc


def resolve_pointer(value: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer."""

    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    current = value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"pointer {pointer!r}: missing key {token!r}")
            current = current[token]
        else:
            raise KeyError(f"pointer {pointer!r}: cannot descend into scalar at {token!r}")
    return current


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _lf(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    """The sealed results bundle, verified file by file against its own manifest."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.root = repo / RESULTS
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, "results manifest")
        if self.manifest.get("state") != "accepted_result":
            raise ValueError("results manifest state is not accepted_result")
        if self.manifest.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("results manifest experiment identity differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        entries = self.manifest["artifacts"]
        if len(entries) != self.manifest["artifact_count"]:
            raise ValueError("results manifest artifact count differs")
        for entry in entries:
            if entry["type"] != "file":
                continue
            relative = entry["path"]
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"manifest path escapes the bundle: {relative}")
            raw = (self.root / pure).read_bytes()
            actual = sha256_bytes(raw)
            if actual != entry["byte_sha256"] or len(raw) != entry["bytes"]:
                raise ValueError(f"bundle file SHA-256 or size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
        if self.hashes["terminal.json"] != self.manifest["terminal_byte_sha256"]:
            raise ValueError("terminal.json hash differs from the manifest binding")
        if self.hashes["execution-lock.json"] != self.manifest["lock_byte_sha256"]:
            raise ValueError("execution-lock.json hash differs from the manifest binding")
        # Every artifact except the lock (bound by lock_byte_sha256 above) carries a
        # sidecar whose byte hash must agree with the manifest.
        for relative in list(self.hashes):
            if relative.endswith(".sha256.json") or relative == "execution-lock.json":
                continue
            sidecar_rel = f"{relative}.sha256.json"
            if sidecar_rel not in self.hashes:
                raise ValueError(f"artifact without manifest-bound sidecar: {relative}")
            sidecar = load_json_bytes((self.root / sidecar_rel).read_bytes(), sidecar_rel)
            if sidecar["artifact"] != relative or sidecar["byte_sha256"] != self.hashes[relative]:
                raise ValueError(f"sidecar disagrees with the manifest: {sidecar_rel}")
            if sidecar["bytes"] != self.sizes[relative]:
                raise ValueError(f"sidecar size disagrees with the manifest: {sidecar_rel}")
        self.used: dict[str, dict[str, Any]] = {}

    def load(self, relative: str) -> Any:
        if relative not in self.hashes:
            raise ValueError(f"{relative} is not manifest-bound")
        raw = (self.root / relative).read_bytes()
        self.used[relative] = {"sha256": self.hashes[relative], "bytes": self.sizes[relative]}
        return load_json_bytes(raw, relative)

    def bind_committed(self) -> dict[str, Any]:
        """Prove the working-tree bundle equals the committed results revision."""

        head = _git(self.repo, "rev-parse", "HEAD")
        for commit, label in (
            (RESULTS_COMMIT_SHA, "results"),
            (PREREGISTRATION_COMMIT_SHA, "preregistration"),
            (DASHBOARD_COMMIT_SHA, "dashboard"),
        ):
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, head],
                cwd=self.repo, check=False, capture_output=True,
            ).returncode == 0
            if not ancestor:
                raise ValueError(f"{label} commit is not an ancestor of HEAD")
        manifest_rel = (RESULTS / "manifest.json").as_posix()
        committed_blob = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{manifest_rel}")
        working_blob = _git(self.repo, "hash-object", "--", manifest_rel)
        if committed_blob != working_blob:
            raise ValueError("working-tree results manifest differs from the committed blob")
        results_tree = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{RESULTS.as_posix()}")
        subject = _git(self.repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
        return {
            "results_commit": RESULTS_COMMIT_SHA,
            "results_commit_subject": subject,
            "results_tree": results_tree,
            "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
            "dashboard_commit": DASHBOARD_COMMIT_SHA,
            "manifest_git_blob": committed_blob,
            "manifest_path": manifest_rel,
        }


def dashboard_payload(html: bytes) -> dict[str, Any]:
    """Extract the JSON payload embedded by the committed dashboard generator."""

    text = html.decode("utf-8")
    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>', text, re.DOTALL
    )
    if match is None:
        raise ValueError("dashboard HTML carries no embedded payload")
    return load_json_bytes(match.group(1).replace("<\\/", "</").encode("utf-8"), "dashboard payload")


def cross_check_dashboard(repo: Path, bundle: Bundle, campaign: dict[str, Any], metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same bundle; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    for constant, value in (
        ("EXPECTED_MANIFEST_SHA256", bundle.manifest_sha256),
        ("RESULTS_COMMIT_SHA", RESULTS_COMMIT_SHA),
        ("PREREGISTRATION_COMMIT_SHA", PREREGISTRATION_COMMIT_SHA),
    ):
        if f'"{value}"' not in generator_text or constant not in generator_text:
            raise ValueError(f"dashboard generator does not pin {constant} to the bundle identity")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_sha256"] != bundle.manifest_sha256:
        raise ValueError("dashboard payload names a different results manifest")
    if identity["results_commit"] != RESULTS_COMMIT_SHA or identity["preregistration_commit"] != PREREGISTRATION_COMMIT_SHA:
        raise ValueError("dashboard payload names different revisions")
    if identity["terminal_state"] != bundle.manifest["state"] or identity["verified_files"] != len(bundle.hashes):
        raise ValueError("dashboard payload identity differs from the bundle")
    if payload["campaign_result"] != campaign:
        raise ValueError("dashboard campaign_result differs from the sealed artifact")
    if payload["seed_variance"] != metrics["seed_variance"]:
        raise ValueError("dashboard seed variance differs from the sealed metrics")
    if payload["gates"]["robust_vs_nominal"] != gates["reported_not_binding"]["robust_vs_nominal"]:
        raise ValueError("dashboard robust-vs-nominal block differs from the sealed gates")
    for key in ("bo_beats_random", "bo_beats_nsga3"):
        if payload["gates"][key] != gates["reported_not_binding"][key]:
            raise ValueError(f"dashboard {key} differs from the sealed gates")
    if payload["gates"]["design_set_invariance"]["per_prior"] != gates["reported_not_binding"]["design_set_invariance"]["per_prior"]:
        raise ValueError("dashboard design-set invariance differs from the sealed gates")
    for key, run in payload["runs"].items():
        if run["final_hypervolume"] != metrics["runs"][key]["final_hypervolume"]:
            raise ValueError(f"dashboard final hypervolume differs for {key}")
    return {
        "generator_path": DASHBOARD_GENERATOR.as_posix(),
        "generator_sha256_lf": sha256_bytes(_lf(generator_raw)),
        "html_path": DASHBOARD_HTML.as_posix(),
        "html_sha256_lf": sha256_bytes(_lf(html_raw)),
        "html_schema": payload["schema"],
        "payload_manifest_sha256": identity["manifest_sha256"],
        "rule": (
            "the committed dashboard pins the bundle's manifest SHA-256 and revisions and embeds its own "
            "extraction of the campaign result; the generator requires that extraction to equal the sealed "
            "artifacts before writing any macro"
        ),
    }


# --------------------------------------------------------------------------- #
# Macro construction
# --------------------------------------------------------------------------- #
class Macros:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()
        self.docs: dict[str, Any] = {}

    def doc(self, relative: str) -> Any:
        if relative not in self.docs:
            self.docs[relative] = self.bundle.load(relative)
        return self.docs[relative]

    def add(self, name: str, artifact: str, pointer: str, fmt: str, description: str) -> str:
        if name in self.names or not name.isalpha():
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        raw = resolve_pointer(self.doc(artifact), pointer)
        value = format_value(fmt, raw)
        self.items.append(
            {
                "name": name,
                "value": value,
                "raw": raw,
                "format": fmt,
                "derived": False,
                "source": {"artifact": artifact, "pointer": pointer},
                "description": description,
            }
        )
        self.names.add(name)
        return value

    def add_derived(
        self, name: str, raw: Any, fmt: str, description: str, derivation: str,
        inputs: list[dict[str, str]],
    ) -> str:
        if name in self.names or not name.isalpha():
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        value = format_value(fmt, raw)
        self.items.append(
            {
                "name": name,
                "value": value,
                "raw": raw,
                "format": fmt,
                "derived": True,
                "derivation": derivation,
                "inputs": inputs,
                "description": description,
            }
        )
        self.names.add(name)
        return value


def _seconds(start: str, end: str) -> float:
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (b - a).total_seconds()


PROBE_PATTERN = re.compile(
    r"closed only for p = \(0,0,0,0\) \((?P<closed>\d+)/(?P<total>\d+) cases\); every nonzero p ended at "
    r"iteration_limit with residual floors (?P<rmin>[0-9.e+-]+) \.\. (?P<rmax>[0-9.e+-]+); ~(?P<sec>\d+) s per multistart solve"
)


def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo)
    binding = bundle.bind_committed()
    m = Macros(bundle)
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    protocol = m.doc("artifacts/protocol.json")
    metrics = m.doc("artifacts/metrics.json")
    dense = m.doc("artifacts/dense-reference-summary.json")
    sensitivity = m.doc("artifacts/sensitivity.json")
    pooled = m.doc("artifacts/pooled-fronts.json")
    per_strategy = m.doc("artifacts/per-strategy-fronts.json")
    plan = m.doc("artifacts/campaign-plan.json")
    contract = m.doc("artifacts/code-contract.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    sample = m.doc("artifacts/uncertain-sample.json")
    consistency = m.doc("artifacts/protocol-consistency.json")
    probes = m.doc("artifacts/device-probes.json")
    runtime = m.doc("artifacts/runtime.json")
    curves = m.doc("artifacts/hypervolume-curves.json")
    pareto_sets = m.doc("artifacts/pareto-sets.json")
    runs = {rel: m.doc(rel) for rel in RUN_ARTIFACTS}
    dashboard = cross_check_dashboard(repo, bundle, campaign, metrics, gates)

    # Internal consistency of the sealed bundle (fail closed on any disagreement).
    if terminal["state"] != bundle.manifest["state"] or terminal["payload"]["all_binding_gates_passed"] is not True:
        raise ValueError("terminal record disagrees with the manifest or records a failed gate")
    if campaign["classification"] != CLASSIFICATION or campaign["closure"] != CLOSURE_ID:
        raise ValueError("campaign classification or closure differs from the admitted identity")
    if campaign["all_binding_gates_passed"] is not True or gates["all_binding_passed"] is not True:
        raise ValueError("binding gates are not all passed")
    if any(item["passed"] is not True for item in gates["binding"].values()):
        raise ValueError("a binding gate records a failure")
    if terminal["payload"]["binding_gate_results"] != {k: v["passed"] for k, v in gates["binding"].items()}:
        raise ValueError("terminal gate results differ from gates.json")
    if campaign["hypervolume_table"] != metrics["hypervolume_table"] or campaign["seed_variance"] != metrics["seed_variance"]:
        raise ValueError("campaign-result and metrics disagree")
    if campaign["robust_vs_nominal"] != gates["reported_not_binding"]["robust_vs_nominal"]:
        raise ValueError("campaign-result and gates disagree on robust versus nominal")
    if lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["attempt"] != 1 or lock["immutable"] is not True:
        raise ValueError("execution lock does not record the single preregistered attempt")
    if protocol["classification"] != CLASSIFICATION or protocol["closures"]["CL-1"]["id"] != CLOSURE_ID:
        raise ValueError("protocol classification or closure differs from the campaign")
    if any(v is not True for v in consistency.values()):
        raise ValueError("protocol-consistency records a failed check")
    if plan != authorities["evidentiary_plan"] or plan["kind"] != "evidentiary":
        raise ValueError("campaign plan differs from the preregistered authorities")
    if contract["matches"] is not True or contract["source_sha256"] != authorities["source_sha256"]:
        raise ValueError("code contract does not match the preregistered authorities")
    if sample["sha256"] != protocol["uncertain_inputs"]["sample"]["sha256"] or sample["sha256"] != authorities["sample_sha256"]:
        raise ValueError("frozen sample hash differs between artifacts")
    if len(sample["sample"]) != protocol["uncertain_inputs"]["sample"]["count"]:
        raise ValueError("frozen sample row count differs from the protocol")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["disjointness"]["proven"] is not True:
        raise ValueError("shakedown disjointness is not proven")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]:
        raise ValueError("shakedown artifact differs from the bound authority")
    # The frozen preregistration files (pretty-printed) must carry the payload the bundle sealed.
    for frozen, sealed in (
        ("protocol.json", protocol),
        ("authorities.json", authorities),
        ("shakedown.json", shakedown),
    ):
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != sealed:
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    if list(plan["strategies"]) != list(STRATEGIES) or len(plan["seeds"]) != len(SEED_TOKENS):
        raise ValueError("plan strategies or seeds differ from the admitted layout")
    if len(metrics["runs"]) != len(plan["run_ids"]) or campaign["runs"] != len(plan["run_ids"]):
        raise ValueError("run count differs between plan, metrics and campaign result")
    if probes["cuda"]["device"] != "cuda:0" or probes["cpu"]["device"] != "cpu":
        raise ValueError("device probes do not record the cpu and cuda:0 devices")
    if protocol["optimizers"]["qlognehvi"]["device"] != "cpu":
        raise ValueError("protocol does not declare the cpu device for BO")
    if len(protocol["design_variables"]) != len(protocol["design_variables"]) or protocol["design_variables"][0]["name"] != "discharge_voltage_v":
        raise ValueError("design variables are not in the declared order")

    # Identity and lifecycle.
    m.add("MdoClassification", "artifacts/campaign-result.json", "/classification", "ident", "campaign classification string")
    m.add("MdoTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("MdoClosureId", "artifacts/campaign-result.json", "/closure", "ident", "declared closure identifier")
    m.add("MdoFidelity", "artifacts/protocol.json", "/authority/l0_model/fidelity", "ident", "declared model fidelity label")
    m.add("MdoExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add("MdoAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("MdoRuns", "artifacts/campaign-result.json", "/runs", "int", "optimiser runs (strategies times seeds)")
    m.add("MdoTotalEvaluations", "artifacts/campaign-result.json", "/total_evaluations", "int", "L0 design evaluations in the campaign")
    m.add("MdoInfeasibleEvaluations", "artifacts/campaign-result.json", "/infeasible_evaluations", "int", "constraint-violating design evaluations")
    m.add("MdoEvaluationsPerRun", "artifacts/campaign-plan.json", "/evaluations_per_run", "int", "evaluation budget per run")
    m.add("MdoInitialDesign", "artifacts/campaign-plan.json", "/initial_design", "int", "shared initial design size per seed")
    m.add("MdoSeeds", "artifacts/campaign-plan.json", "/seeds", "list_int", "evidentiary seeds")
    m.add_derived("MdoSeedCount", len(plan["seeds"]), "int", "number of seeds", "len(plan.seeds)", [{"artifact": "artifacts/campaign-plan.json", "pointer": "/seeds"}])
    m.add_derived("MdoStrategyCount", len(plan["strategies"]), "int", "number of optimisers", "len(plan.strategies)", [{"artifact": "artifacts/campaign-plan.json", "pointer": "/strategies"}])
    m.add("MdoBoBatch", "artifacts/campaign-plan.json", "/qlognehvi_batch_size", "int", "BO batch size")
    m.add("MdoBoIterations", "artifacts/campaign-plan.json", "/qlognehvi_iterations", "int", "BO iterations")
    m.add("MdoNsgaPopulation", "artifacts/campaign-plan.json", "/nsga3_population_size", "int", "NSGA-III population size")
    m.add("MdoNsgaGenerations", "artifacts/campaign-plan.json", "/nsga3_generations", "int", "NSGA-III generations")
    m.add("MdoBoMcSamples", "artifacts/protocol.json", "/optimizers/qlognehvi/mc_samples", "int", "BO Monte Carlo samples")
    m.add("MdoBoDevice", "artifacts/protocol.json", "/optimizers/qlognehvi/device", "ident", "BO device")
    m.add("MdoBoDtype", "artifacts/protocol.json", "/optimizers/qlognehvi/dtype", "ident", "BO floating-point type")
    m.add("MdoBoLibrary", "artifacts/protocol.json", "/optimizers/qlognehvi/library", "text", "BO library")
    m.add("MdoNsgaLibrary", "artifacts/protocol.json", "/optimizers/nsga3/library", "text", "NSGA-III library")
    m.add("MdoBotorchVersion", "artifacts/code-contract.json", "/observed_package_versions/botorch", "text", "BoTorch version")
    m.add("MdoPymooVersion", "artifacts/code-contract.json", "/observed_package_versions/pymoo", "text", "pymoo version")
    m.add("MdoTorchVersion", "artifacts/code-contract.json", "/observed_package_versions/torch", "text", "torch version")
    m.add("MdoCodeContractMatches", "artifacts/code-contract.json", "/matches", "bool", "code contract matches the preregistered authorities")
    m.add("MdoSourceSha", "artifacts/code-contract.json", "/source_sha256", "sha_short", "hashed source prefix")
    m.add("MdoProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("MdoSampleSha", "artifacts/uncertain-sample.json", "/sha256", "sha_short", "frozen sample hash prefix")
    m.add("MdoPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add("MdoLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add_derived("MdoResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("MdoDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "git commit that added the results dashboard cross-checked by this generator", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("MdoManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("MdoVerifiedFiles", len(bundle.hashes), "int", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("MdoArtifactCount", bundle.manifest["artifact_count"], "int", "manifest entries (files and directories)", "manifest.artifact_count", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("MdoToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [{"artifact": "manifest.json", "pointer": "/artifacts"}])

    # Gates.
    m.add_derived("MdoGateCount", len(gates["binding"]), "int", "binding gates", "len(gates.binding)", [{"artifact": "artifacts/gates.json", "pointer": "/binding"}])
    m.add_derived("MdoGatesPassed", sum(1 for v in gates["binding"].values() if v["passed"] is True), "int", "binding gates passed", "count(gates.binding[*].passed == true)", [{"artifact": "artifacts/gates.json", "pointer": "/binding"}])
    m.add("MdoReplayed", "artifacts/gates.json", "/binding/replay_bit_exact/replayed", "int", "evaluations replayed bit-exactly")
    m.add_derived("MdoReplayMismatches", len(gates["binding"]["replay_bit_exact"]["mismatches"]), "int", "replay mismatches", "len(gates.binding.replay_bit_exact.mismatches)", [{"artifact": "artifacts/gates.json", "pointer": "/binding/replay_bit_exact/mismatches"}])
    if gates["binding"]["replay_bit_exact"]["replayed"] != campaign["total_evaluations"]:
        raise ValueError("replayed evaluation count differs from the total")
    m.add("MdoDenseReplayed", "artifacts/dense-reference-summary.json", "/replay/replayed", "int", "dense-reference designs replayed")
    m.add("MdoDenseReplayPassed", "artifacts/dense-reference-summary.json", "/replay/passed", "bool", "dense-reference replay passed")
    m.add("MdoSeparabilityPassed", "artifacts/dense-reference-summary.json", "/separability/passed", "bool", "separability check passed")
    spreads = [v["relative_spread"] for v in dense["separability"]["ratios"].values()]
    m.add_derived("MdoSeparabilitySpreadMax", max(spreads), "sci1", "largest relative spread of the robust-to-nominal objective ratio", "max over objectives of separability.ratios[*].relative_spread", [{"artifact": "artifacts/dense-reference-summary.json", "pointer": "/separability/ratios"}])
    m.add("MdoSeparabilityTolerance", "artifacts/dense-reference-summary.json", "/separability/tolerance_relative_spread", "sci1", "separability tolerance")

    # Design variables, uncertain inputs, closure and objectives.
    dv = protocol["design_variables"]
    if [d["name"] for d in dv] != ["discharge_voltage_v", "anode_current_a", "propellant_mass_flow_kg_per_s"]:
        raise ValueError("design variables differ from the declared three")
    m.add_derived("MdoDesignVariableCount", len(dv), "int", "design variables", "len(protocol.design_variables)", [{"artifact": "artifacts/protocol.json", "pointer": "/design_variables"}])
    m.add("MdoUaLower", "artifacts/protocol.json", "/design_variables/0/lower", "g", "discharge voltage lower bound (V)")
    m.add("MdoUaUpper", "artifacts/protocol.json", "/design_variables/0/upper", "g", "discharge voltage upper bound (V)")
    m.add("MdoIaLower", "artifacts/protocol.json", "/design_variables/1/lower", "g", "anode current lower bound (A)")
    m.add("MdoIaUpper", "artifacts/protocol.json", "/design_variables/1/upper", "g", "anode current upper bound (A)")
    m.add("MdoMdotLower", "artifacts/protocol.json", "/design_variables/2/lower", "sci1", "mass flow lower bound (kg/s)")
    m.add("MdoMdotUpper", "artifacts/protocol.json", "/design_variables/2/upper", "sci1", "mass flow upper bound (kg/s)")
    m.add_derived("MdoExcludedVariableCount", len(protocol["excluded_legacy_variables"]), "int", "excluded legacy geometry variables", "len(protocol.excluded_legacy_variables)", [{"artifact": "artifacts/protocol.json", "pointer": "/excluded_legacy_variables"}])
    inputs = protocol["uncertain_inputs"]["inputs"]
    cusp_inputs = [u for u in inputs if u["name"].startswith("cusp_probability_cell_")]
    if len(cusp_inputs) != 4 or any(u["lower"] != 0.0 or u["upper"] != cusp_inputs[0]["upper"] for u in cusp_inputs):
        raise ValueError("cusp priors are not four identical uniform intervals from zero")
    m.add_derived("MdoUncertainInputCount", len(inputs), "int", "uncertain inputs", "len(protocol.uncertain_inputs.inputs)", [{"artifact": "artifacts/protocol.json", "pointer": "/uncertain_inputs/inputs"}])
    m.add_derived("MdoCellCount", len(cusp_inputs), "int", "cusp cells with an uncertain probability", "count of uncertain inputs named cusp_probability_cell_*", [{"artifact": "artifacts/protocol.json", "pointer": "/uncertain_inputs/inputs"}])
    m.add("MdoCuspLower", "artifacts/protocol.json", "/uncertain_inputs/inputs/0/lower", "g", "cusp probability prior lower bound")
    m.add("MdoCuspUpper", "artifacts/protocol.json", "/uncertain_inputs/inputs/0/upper", "g", "cusp probability prior upper bound")
    m.add("MdoEtaLower", "artifacts/protocol.json", "/uncertain_inputs/inputs/4/lower", "g", "ionised fraction lower bound")
    m.add("MdoEtaUpper", "artifacts/protocol.json", "/uncertain_inputs/inputs/4/upper", "g", "ionised fraction upper bound")
    m.add("MdoZetaLower", "artifacts/protocol.json", "/uncertain_inputs/inputs/5/lower", "g", "doubly charged share lower bound")
    m.add("MdoZetaUpper", "artifacts/protocol.json", "/uncertain_inputs/inputs/5/upper", "g", "doubly charged share upper bound")
    m.add("MdoGammaLower", "artifacts/protocol.json", "/uncertain_inputs/inputs/6/lower", "g", "divergence factor lower bound")
    m.add("MdoGammaUpper", "artifacts/protocol.json", "/uncertain_inputs/inputs/6/upper", "g", "divergence factor upper bound")
    if [u["name"] for u in inputs[4:]] != ["ionized_number_fraction", "xe_double_plus_fraction_of_ions", "axial_momentum_fraction_of_ion_momentum"]:
        raise ValueError("non-cusp uncertain inputs are not in the declared order")
    m.add("MdoSampleCount", "artifacts/protocol.json", "/uncertain_inputs/sample/count", "int", "frozen QMC sample rows")
    m.add("MdoSampleBases", "artifacts/protocol.json", "/uncertain_inputs/sample/bases", "list_int", "Halton bases")
    m.add("MdoSampleFrozen", "artifacts/protocol.json", "/uncertain_inputs/sample/frozen", "bool", "sample frozen flag")
    m.add("MdoTailCount", "artifacts/protocol.json", "/robust_formulation/tail_count", "int", "CVaR tail count")
    m.add("MdoTailFraction", "artifacts/protocol.json", "/robust_formulation/tail_fraction", "g", "CVaR tail fraction")
    m.add("MdoRiskMeasure", "artifacts/protocol.json", "/robust_formulation/risk_measure", "text", "risk measure")
    if protocol["robust_formulation"]["tail_count"] != round(protocol["robust_formulation"]["tail_fraction"] * protocol["uncertain_inputs"]["sample"]["count"]):
        raise ValueError("CVaR tail count is not tail_fraction times the sample count")
    m.add("MdoNominalCusp", "artifacts/uncertain-sample.json", "/nominal/cusp_probability_cell_1", "g", "nominal cusp probability")
    m.add("MdoCathodePowerW", "artifacts/protocol.json", "/closures/fixed/cathode_input_power_w", "g", "fixed cathode input power (W)")
    m.add("MdoPpuEfficiency", "artifacts/protocol.json", "/closures/fixed/ppu_efficiency_fraction", "g", "fixed PPU efficiency")
    m.add("MdoClosureStatus", "artifacts/protocol.json", "/closures/CL-1/status", "text", "closure status")
    upper = cusp_inputs[0]["upper"]
    implied = (1.0 - upper / 2.0) ** len(cusp_inputs)
    m.add_derived("MdoImpliedPriorSurvival", implied, "fixed3", "mean CL-1 survival implied by the cusp prior", "(1 - upper/2) ** cell_count for independent uniform [0, upper] priors", [{"artifact": "artifacts/protocol.json", "pointer": "/uncertain_inputs/inputs/0/upper"}])
    v4 = protocol["authority"]["wall_loss_v4"]["pooled_wall_hit"]
    v4_survival = 1.0 - v4["successes"] / v4["trials"]
    m.add_derived("MdoVFourSurvival", v4_survival, "fixed3", "pooled electron survival of the wall-loss campaign as recorded in the protocol authority", "1 - successes / trials", [{"artifact": "artifacts/protocol.json", "pointer": "/authority/wall_loss_v4/pooled_wall_hit"}])
    m.add_derived("MdoSurvivalCalibrationGap", abs(implied - v4_survival), "fixed4", "gap between the implied prior survival and the recorded pooled survival", "abs(MdoImpliedPriorSurvival - MdoVFourSurvival)", [{"artifact": "artifacts/protocol.json", "pointer": "/uncertain_inputs/inputs/0/upper"}, {"artifact": "artifacts/protocol.json", "pointer": "/authority/wall_loss_v4/pooled_wall_hit"}])
    m.add("MdoVFourWallHits", "artifacts/protocol.json", "/authority/wall_loss_v4/pooled_wall_hit/successes", "int", "pooled wall hits recorded in the protocol authority")
    m.add("MdoVFourTrials", "artifacts/protocol.json", "/authority/wall_loss_v4/pooled_wall_hit/trials", "int", "pooled trials recorded in the protocol authority")
    m.add("MdoVFourReflections", "artifacts/protocol.json", "/authority/wall_loss_v4/reflections", "int", "reflections recorded in the protocol authority")
    m.add("MdoVFourResultCommit", "artifacts/protocol.json", "/authority/wall_loss_v4/result_commit", "text", "wall-loss result commit prefix recorded in the protocol authority")
    objectives = protocol["objectives"]
    if [o["name"] for o in objectives] != list(OBJECTIVES):
        raise ValueError("objectives differ from the declared four")
    m.add_derived("MdoObjectiveCount", len(objectives), "int", "objectives", "len(protocol.objectives)", [{"artifact": "artifacts/protocol.json", "pointer": "/objectives"}])
    m.add("MdoRefAnodePowerW", "artifacts/protocol.json", "/reference_point/anode_input_power_w", "g", "hypervolume reference point, anode power (W)")
    m.add("MdoScaleThrust", "artifacts/protocol.json", "/objectives/0/comparison_scale", "g", "thrust comparison scale (N)")
    m.add("MdoScaleIsp", "artifacts/protocol.json", "/objectives/1/comparison_scale", "g", "specific impulse comparison scale (s)")
    m.add("MdoScalePower", "artifacts/protocol.json", "/objectives/3/comparison_scale", "g", "anode power comparison scale (W)")
    m.add("MdoConstraintName", "artifacts/protocol.json", "/constraints/0/name", "ident", "robust constraint name")
    m.add("MdoConstraintThreshold", "artifacts/protocol.json", "/constraints/0/threshold", "g", "robust constraint threshold (A)")
    m.add("MdoDenseCount", "artifacts/dense-reference-summary.json", "/count", "int_comma", "dense reference design count")
    m.add("MdoDenseFeasible", "artifacts/dense-reference-summary.json", "/feasible", "int_comma", "dense reference feasible designs")
    m.add("MdoDenseInfeasible", "artifacts/dense-reference-summary.json", "/infeasible", "int_comma", "dense reference infeasible designs")
    m.add("MdoDenseRobustHv", "artifacts/dense-reference-summary.json", "/robust_hypervolume", "fixed6", "dense reference robust hypervolume")
    m.add("MdoDenseNominalHv", "artifacts/dense-reference-summary.json", "/nominal_hypervolume", "fixed4", "dense reference nominal hypervolume")
    m.add("MdoDenseRobustFront", "artifacts/dense-reference-summary.json", "/robust_front_size", "int", "dense reference robust front size")
    m.add("MdoDenseNominalFront", "artifacts/dense-reference-summary.json", "/nominal_front_size", "int", "dense reference nominal front size")
    m.add("MdoDenseSeed", "artifacts/protocol.json", "/dense_reference/seed", "int", "dense reference seed")
    if dense["count"] != metrics["dense_reference"]["count"] or dense["robust_hypervolume"] != metrics["dense_reference"]["robust_hypervolume"]:
        raise ValueError("dense reference summary differs from metrics")
    if dense["feasible"] + dense["infeasible"] != dense["count"]:
        raise ValueError("dense reference feasible and infeasible counts do not sum to the count")
    m.add_derived("MdoDenseToBudgetRatio", dense["count"] / plan["evaluations_per_run"], "fixed0", "dense reference designs per optimiser-run evaluation budget", "dense_reference.count / plan.evaluations_per_run", [{"artifact": "artifacts/dense-reference-summary.json", "pointer": "/count"}, {"artifact": "artifacts/campaign-plan.json", "pointer": "/evaluations_per_run"}])

    # Per-run estimands and the hypervolume table.
    seeds = [int(s) for s in plan["seeds"]]
    hv_rows: list[str] = []
    failed_runs = 0
    infeasible_total = 0
    bo_acq: list[float] = []
    bo_fit: list[float] = []
    per_strategy_hv: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    per_strategy_pareto: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    per_strategy_infeasible: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    per_strategy_wall: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    per_strategy_attained: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        for seed, seed_token in zip(seeds, SEED_TOKENS, strict=True):
            key = f"{strategy}:{seed}"
            summary = metrics["runs"][key]
            table = metrics["hypervolume_table"][key]
            run = runs[f"artifacts/runs/{strategy}-{seed}.json"]
            curve = curves[key]
            if summary["evaluations"] != plan["evaluations_per_run"] or summary["budget"] != plan["evaluations_per_run"]:
                failed_runs += 1
            if summary["final_hypervolume"] != table["final_hypervolume"] or run["summary"] != summary:
                raise ValueError(f"metrics disagree for {key}")
            if curve[-1]["hypervolume"] != summary["final_hypervolume"] or len(curve) != summary["evaluations"]:
                raise ValueError(f"hypervolume curve disagrees with the summary for {key}")
            if any(b["hypervolume"] < a["hypervolume"] for a, b in zip(curve, curve[1:], strict=False)):
                raise ValueError(f"hypervolume curve is not monotone for {key}")
            if summary["feasible_evaluations"] + summary["infeasible_evaluations"] != summary["evaluations"]:
                raise ValueError(f"feasible and infeasible counts do not sum to the evaluations for {key}")
            if len(run["records"]) != summary["evaluations"]:
                raise ValueError(f"record count differs from the evaluations for {key}")
            if pareto_sets[key]["size"] != summary["pareto_set_size"] or pareto_sets[key]["replay_passed"] is not True or pareto_sets[key]["nondominated_recomputed"] is not True:
                raise ValueError(f"pareto set disagrees with the summary for {key}")
            if abs(table["attained_fraction_of_dense_reference"] - summary["final_hypervolume"] / dense["robust_hypervolume"]) > 1e-12:
                raise ValueError(f"attained fraction does not reproduce for {key}")
            infeasible_total += summary["infeasible_evaluations"]
            per_strategy_hv[strategy].append(summary["final_hypervolume"])
            per_strategy_pareto[strategy].append(summary["pareto_set_size"])
            per_strategy_infeasible[strategy].append(summary["infeasible_evaluations"])
            per_strategy_wall[strategy].append(summary["wall_clock_seconds"])
            per_strategy_attained[strategy].append(table["attained_fraction_of_dense_reference"])
            base = f"/runs/{key}"
            m.add(f"MdoHv{token}{seed_token}", "artifacts/metrics.json", f"{base}/final_hypervolume", "fixed6", f"final robust hypervolume, {key}")
            m.add(f"MdoPareto{token}{seed_token}", "artifacts/metrics.json", f"{base}/pareto_set_size", "int", f"final Pareto-set size, {key}")
            m.add(f"MdoInfeasible{token}{seed_token}", "artifacts/metrics.json", f"{base}/infeasible_evaluations", "int", f"infeasible evaluations, {key}")
            m.add(f"MdoAttained{token}{seed_token}", "artifacts/metrics.json", f"/hypervolume_table/{key}/attained_fraction_of_dense_reference", "fixed3", f"fraction of the dense-reference robust hypervolume, {key}")
            timing = metrics["timing"][key]
            if strategy == "qlognehvi":
                bo_acq.append(timing["bo_acquisition_seconds"])
                bo_fit.append(timing["bo_fit_seconds"])
                if len(run["optimizer"]["iteration_log"]) != plan["qlognehvi_iterations"]:
                    raise ValueError(f"BO iteration log length differs from the plan for {key}")
            elif run["optimizer"].get("iteration_log"):
                raise ValueError(f"non-BO run carries a BO iteration log: {key}")
            hv_rows.append(
                f"{STRATEGY_LABELS[strategy]} & {seed} & {format_value('fixed6', summary['final_hypervolume'])} & "
                f"{format_value('fixed3', table['attained_fraction_of_dense_reference'])} & {summary['pareto_set_size']} & "
                f"{summary['infeasible_evaluations']} & {format_value('fixed1', summary['wall_clock_seconds'])}\\\\"
            )
    if infeasible_total != campaign["infeasible_evaluations"]:
        raise ValueError("per-run infeasible counts do not sum to the campaign total")
    if sum(len(v) for v in per_strategy_hv.values()) * plan["evaluations_per_run"] != campaign["total_evaluations"]:
        raise ValueError("runs times budget differs from the total evaluations")
    m.add_derived("MdoFailedRuns", failed_runs, "int", "runs that did not record exactly the budget", "count of runs with evaluations != plan.evaluations_per_run", [{"artifact": "artifacts/metrics.json", "pointer": "/runs"}])
    run_inputs = [{"artifact": "artifacts/metrics.json", "pointer": f"/runs/{s}:{seed}/final_hypervolume"} for s in STRATEGIES for seed in seeds]
    summary_rows: list[str] = []
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        hv = per_strategy_hv[strategy]
        recorded = metrics["seed_variance"][strategy]
        if abs(statistics.mean(hv) - recorded["mean"]) > 1e-15 or abs(statistics.stdev(hv) - recorded["sample_std"]) > 1e-15:
            raise ValueError(f"seed variance does not reproduce for {strategy}")
        if min(hv) != recorded["minimum"] or max(hv) != recorded["maximum"]:
            raise ValueError(f"seed extrema do not reproduce for {strategy}")
        m.add(f"MdoHv{token}Mean", "artifacts/metrics.json", f"/seed_variance/{strategy}/mean", "fixed6", f"mean final hypervolume, {strategy}")
        m.add(f"MdoHv{token}Std", "artifacts/metrics.json", f"/seed_variance/{strategy}/sample_std", "sci1", f"sample standard deviation of the final hypervolume, {strategy}")
        m.add(f"MdoHv{token}Min", "artifacts/metrics.json", f"/seed_variance/{strategy}/minimum", "fixed6", f"minimum final hypervolume, {strategy}")
        m.add(f"MdoHv{token}Max", "artifacts/metrics.json", f"/seed_variance/{strategy}/maximum", "fixed6", f"maximum final hypervolume, {strategy}")
        strategy_inputs = [{"artifact": "artifacts/metrics.json", "pointer": f"/runs/{strategy}:{seed}"} for seed in seeds]
        m.add_derived(f"MdoAttained{token}Mean", statistics.mean(per_strategy_attained[strategy]), "fixed3", f"mean fraction of the dense-reference robust hypervolume, {strategy}", "mean over seeds of hypervolume_table[*].attained_fraction_of_dense_reference", [{"artifact": "artifacts/metrics.json", "pointer": f"/hypervolume_table/{strategy}:{seed}/attained_fraction_of_dense_reference"} for seed in seeds])
        m.add_derived(f"MdoAttained{token}MeanTwo", statistics.mean(per_strategy_attained[strategy]), "fixed2", f"mean attained fraction to two decimals, {strategy}", "mean over seeds of hypervolume_table[*].attained_fraction_of_dense_reference", [{"artifact": "artifacts/metrics.json", "pointer": f"/hypervolume_table/{strategy}:{seed}/attained_fraction_of_dense_reference"} for seed in seeds])
        m.add_derived(f"MdoAttained{token}Min", min(per_strategy_attained[strategy]), "fixed3", f"minimum attained fraction, {strategy}", "min over seeds", strategy_inputs)
        m.add_derived(f"MdoAttained{token}Max", max(per_strategy_attained[strategy]), "fixed3", f"maximum attained fraction, {strategy}", "max over seeds", strategy_inputs)
        m.add_derived(f"MdoPareto{token}Min", min(per_strategy_pareto[strategy]), "int", f"minimum Pareto-set size, {strategy}", "min over seeds of runs[*].pareto_set_size", strategy_inputs)
        m.add_derived(f"MdoPareto{token}Max", max(per_strategy_pareto[strategy]), "int", f"maximum Pareto-set size, {strategy}", "max over seeds of runs[*].pareto_set_size", strategy_inputs)
        m.add_derived(f"MdoInfeasible{token}Min", min(per_strategy_infeasible[strategy]), "int", f"minimum infeasible evaluations, {strategy}", "min over seeds of runs[*].infeasible_evaluations", strategy_inputs)
        m.add_derived(f"MdoInfeasible{token}Max", max(per_strategy_infeasible[strategy]), "int", f"maximum infeasible evaluations, {strategy}", "max over seeds of runs[*].infeasible_evaluations", strategy_inputs)
        m.add_derived(f"MdoWall{token}Max", max(per_strategy_wall[strategy]), "fixed1", f"maximum run wall time (s), {strategy}", "max over seeds of runs[*].wall_clock_seconds", strategy_inputs)
        m.add_derived(f"MdoWall{token}Min", min(per_strategy_wall[strategy]), "fixed1", f"minimum run wall time (s), {strategy}", "min over seeds of runs[*].wall_clock_seconds", strategy_inputs)
        summary_rows.append(
            f"{STRATEGY_LABELS[strategy]} & mean $\\pm$ s & ${format_value('fixed6', recorded['mean'])} \\pm {format_value('sci1', recorded['sample_std'])[1:-1]}$ & "
            f"{format_value('fixed3', statistics.mean(per_strategy_attained[strategy]))} & "
            f"{format_value('fixed1', statistics.mean(per_strategy_pareto[strategy]))} & "
            f"{format_value('fixed1', statistics.mean(per_strategy_infeasible[strategy]))} & "
            f"{format_value('fixed1', statistics.mean(per_strategy_wall[strategy]))}\\\\"
        )
    m.add_derived("MdoBoAcqSecondsMin", min(bo_acq), "fixed0", "minimum BO acquisition seconds per seed", "min over BO seeds of timing[*].bo_acquisition_seconds", [{"artifact": "artifacts/metrics.json", "pointer": f"/timing/qlognehvi:{seed}/bo_acquisition_seconds"} for seed in seeds])
    m.add_derived("MdoBoAcqSecondsMax", max(bo_acq), "fixed0", "maximum BO acquisition seconds per seed", "max over BO seeds of timing[*].bo_acquisition_seconds", [{"artifact": "artifacts/metrics.json", "pointer": f"/timing/qlognehvi:{seed}/bo_acquisition_seconds"} for seed in seeds])
    m.add_derived("MdoBoFitSecondsMax", max(bo_fit), "fixed1", "maximum BO fit seconds per seed", "max over BO seeds of timing[*].bo_fit_seconds", [{"artifact": "artifacts/metrics.json", "pointer": f"/timing/qlognehvi:{seed}/bo_fit_seconds"} for seed in seeds])
    m.add("MdoAssessmentSeconds", "artifacts/campaign-result.json", "/assessment_seconds", "fixed0", "assessment stage wall time (s)")
    m.add("MdoDenseEvaluationSeconds", "artifacts/dense-reference-summary.json", "/evaluation_seconds", "fixed1", "dense reference evaluation time (s)")
    first = m.doc("transitions/0001-lock-acquired.json")
    last = m.doc("transitions/0009-terminal.json")
    m.add_derived(
        "MdoLifecycleWallMin", _seconds(first["recorded_at_utc"]["value"], last["recorded_at_utc"]["value"]), "min1",
        "lock-acquired to terminal wall time (min)", "(terminal.recorded_at_utc - lock_acquired.recorded_at_utc) / 60",
        [{"artifact": "transitions/0001-lock-acquired.json", "pointer": "/recorded_at_utc/value"}, {"artifact": "transitions/0009-terminal.json", "pointer": "/recorded_at_utc/value"}],
    )
    m.add("MdoCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "host CPU count")
    m.add("MdoCudaDevice", "artifacts/device-probes.json", "/cuda/device_name", "text", "CUDA device probed and not used")

    # Predeclared comparisons (reported, not binding).
    reported = gates["reported_not_binding"]
    for key, token in (("bo_beats_random", "Random"), ("bo_beats_nsga3", "Nsga")):
        block = reported[key]
        if block["wins"] != sum(1 for p in block["pairs"] if p["left_wins"] is True) or block["seeds"] != len(block["pairs"]):
            raise ValueError(f"{key} wins do not reproduce from the pairs")
        right = block["right"]
        for pair in block["pairs"]:
            seed = pair["seed"]
            if pair["qlognehvi"] != metrics["runs"][f"qlognehvi:{seed}"]["final_hypervolume"] or pair[right] != metrics["runs"][f"{right}:{seed}"]["final_hypervolume"]:
                raise ValueError(f"{key} pair values differ from the run summaries")
            if pair["left_wins"] is not (pair["qlognehvi"] > pair[right]):
                raise ValueError(f"{key} left_wins flag is inconsistent")
        if block["passed"] is not (block["wins"] >= block["required_wins"]):
            raise ValueError(f"{key} pass flag is inconsistent")
        m.add(f"MdoBoBeats{token}Wins", "artifacts/gates.json", f"/reported_not_binding/{key}/wins", "int", f"{key}: seeds won by qLogNEHVI")
        m.add(f"MdoBoBeats{token}Seeds", "artifacts/gates.json", f"/reported_not_binding/{key}/seeds", "int", f"{key}: paired seeds")
        m.add(f"MdoBoBeats{token}Required", "artifacts/gates.json", f"/reported_not_binding/{key}/required_wins", "int", f"{key}: predeclared required wins")
        m.add(f"MdoBoBeats{token}Passed", "artifacts/gates.json", f"/reported_not_binding/{key}/passed", "bool", f"{key}: predeclared comparison outcome")
    if campaign["bo_beats_random"] is not reported["bo_beats_random"]["passed"] or campaign["bo_beats_nsga3"] is not reported["bo_beats_nsga3"]["passed"]:
        raise ValueError("campaign-result comparison flags differ from gates.json")

    # Robust versus nominal (pooled designs).
    rvn = reported["robust_vs_nominal"]
    if (
        rvn["robust_front_size"] != pooled["robust"]["front_size"]
        or rvn["nominal_front_size"] != pooled["nominal"]["front_size"]
        or rvn["shared_designs"] != len(pooled["shared_design_ids"])
        or rvn["jaccard"] != pooled["jaccard_robust_nominal"]
        or rvn["robust_hypervolume"] != pooled["robust"]["hypervolume"]
        or rvn["nominal_hypervolume"] != pooled["nominal"]["hypervolume"]
        or rvn["nominal_front_members_robust_feasible"] != pooled["nominal"]["robust_feasible_members"]
    ):
        raise ValueError("robust-versus-nominal summary differs from the pooled fronts")
    shared = set(pooled["robust"]["design_ids"]) & set(pooled["nominal"]["design_ids"])
    if shared != set(pooled["shared_design_ids"]) or len(pooled["robust"]["design_ids"]) != pooled["robust"]["front_size"] or len(pooled["nominal"]["design_ids"]) != pooled["nominal"]["front_size"]:
        raise ValueError("shared design ids do not reproduce from the fronts")
    union = pooled["robust"]["front_size"] + pooled["nominal"]["front_size"] - len(shared)
    if abs(len(shared) / union - pooled["jaccard_robust_nominal"]) > 1e-15:
        raise ValueError("Jaccard index does not reproduce")
    if len(pooled["robust"]["designs"]) != pooled["robust"]["front_size"] or len(pooled["nominal"]["designs"]) != pooled["nominal"]["front_size"]:
        raise ValueError("front design lists differ from the front sizes")
    for front in ("robust", "nominal"):
        key = "robust_objectives" if front == "robust" else "nominal_objectives"
        for name in OBJECTIVES:
            values = [d[key][name] for d in pooled[front]["designs"]]
            rng = pooled[front]["objective_ranges"][name]
            if min(values) != rng["minimum"] or max(values) != rng["maximum"]:
                raise ValueError(f"{front} objective range does not reproduce for {name}")
    if any(d["constraints"]["robust_beam_current_margin_a"] < 0 for d in pooled["robust"]["designs"]):
        raise ValueError("a robust-front design violates the robust margin")
    if any(d["constraints"]["nominal_beam_current_margin_a"] < 0 for d in pooled["nominal"]["designs"]):
        raise ValueError("a nominal-front design violates the nominal margin")
    m.add("MdoUniqueDesigns", "artifacts/pooled-fronts.json", "/unique_designs", "int", "unique evaluated designs pooled over all runs")
    m.add("MdoRobustCandidates", "artifacts/pooled-fronts.json", "/robust/candidates", "int", "robust-feasible pooled designs")
    m.add("MdoNominalCandidates", "artifacts/pooled-fronts.json", "/nominal/candidates", "int", "nominally feasible pooled designs")
    m.add("MdoRobustFront", "artifacts/pooled-fronts.json", "/robust/front_size", "int", "pooled robust front size")
    m.add("MdoNominalFront", "artifacts/pooled-fronts.json", "/nominal/front_size", "int", "pooled nominal front size")
    m.add("MdoPooledRobustHv", "artifacts/pooled-fronts.json", "/robust/hypervolume", "fixed6", "pooled robust hypervolume")
    m.add("MdoPooledNominalHv", "artifacts/pooled-fronts.json", "/nominal/hypervolume", "fixed4", "pooled nominal hypervolume")
    m.add_derived("MdoSharedDesigns", len(pooled["shared_design_ids"]), "int", "designs on both pooled fronts", "len(pooled.shared_design_ids)", [{"artifact": "artifacts/pooled-fronts.json", "pointer": "/shared_design_ids"}])
    m.add("MdoJaccard", "artifacts/pooled-fronts.json", "/jaccard_robust_nominal", "fixed3", "Jaccard index of the robust and nominal fronts")
    m.add("MdoNominalRobustFeasible", "artifacts/pooled-fronts.json", "/nominal/robust_feasible_members", "int", "nominal-front designs that are robust-feasible")
    range_fmt = {"axial_thrust_n": "sig3", "specific_impulse_s": "fixed0", "thruster_electrical_to_beam_efficiency": "fixed3", "anode_input_power_w": "fixed1"}
    range_token = {"axial_thrust_n": "Thrust", "specific_impulse_s": "Isp", "thruster_electrical_to_beam_efficiency": "Eff", "anode_input_power_w": "Power"}
    for front, ftoken in (("robust", "Robust"), ("nominal", "Nominal")):
        for name in OBJECTIVES:
            for bound, btoken in (("minimum", "Min"), ("maximum", "Max")):
                m.add(f"Mdo{ftoken}{range_token[name]}{bound.capitalize()[:3]}", "artifacts/pooled-fronts.json", f"/{front}/objective_ranges/{name}/{bound}", range_fmt[name], f"{front} front {name} {bound}")
    m.add_derived("MdoNominalEffMaxTwo", pooled["nominal"]["objective_ranges"]["thruster_electrical_to_beam_efficiency"]["maximum"], "fixed2", "nominal-front efficiency maximum to two decimals", "same raw value as MdoNominalEffMax", [{"artifact": "artifacts/pooled-fronts.json", "pointer": "/nominal/objective_ranges/thruster_electrical_to_beam_efficiency/maximum"}])
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        block = per_strategy[strategy]
        if len(block["robust"]["design_ids"]) != block["robust"]["front_size"]:
            raise ValueError(f"per-strategy robust front differs from its size for {strategy}")
        m.add(f"MdoPooled{token}RobustFront", "artifacts/per-strategy-fronts.json", f"/{strategy}/robust/front_size", "int", f"pooled robust front size, {strategy}")
        m.add(f"MdoPooled{token}RobustHv", "artifacts/per-strategy-fronts.json", f"/{strategy}/robust/hypervolume", "fixed6", f"pooled robust hypervolume, {strategy}")
        m.add(f"MdoPooled{token}NominalFront", "artifacts/per-strategy-fronts.json", f"/{strategy}/nominal/front_size", "int", f"pooled nominal front size, {strategy}")
        m.add(f"MdoPooled{token}Unique", "artifacts/per-strategy-fronts.json", f"/{strategy}/unique_designs", "int", f"unique designs, {strategy}")
        if metrics["per_strategy_pooled"][strategy]["robust_hypervolume"] != block["robust"]["hypervolume"]:
            raise ValueError(f"per-strategy pooled hypervolume differs from metrics for {strategy}")
    rn_rows = [
        f"feasible pooled designs & {pooled['robust']['candidates']} & {pooled['nominal']['candidates']}\\\\",
        f"front size & {pooled['robust']['front_size']} & {pooled['nominal']['front_size']}\\\\",
        f"hypervolume & {format_value('fixed6', pooled['robust']['hypervolume'])} & {format_value('fixed4', pooled['nominal']['hypervolume'])}\\\\",
        f"designs shared with the other front & {len(shared)} & {len(shared)}\\\\",
        f"nominal-front designs that are robust-feasible & --- & {pooled['nominal']['robust_feasible_members']}\\\\",
    ]
    range_label = {
        "axial_thrust_n": "axial thrust (N)",
        "specific_impulse_s": "specific impulse (s)",
        "thruster_electrical_to_beam_efficiency": "electrical-to-beam efficiency",
        "anode_input_power_w": "anode input power (W)",
    }
    for name in OBJECTIVES:
        r = pooled["robust"]["objective_ranges"][name]
        n = pooled["nominal"]["objective_ranges"][name]
        fmt = range_fmt[name]
        rn_rows.append(
            f"{range_label[name]}, front range & {format_value(fmt, r['minimum'])}--{format_value(fmt, r['maximum'])} & "
            f"{format_value(fmt, n['minimum'])}--{format_value(fmt, n['maximum'])}\\\\"
        )

    # Sensitivity to the cusp prior and fixed scenarios.
    priors = sensitivity["priors"]
    scenarios = sensitivity["scenarios"]
    invariance = reported["design_set_invariance"]
    if [p["cusp_upper"] for p in priors] != protocol["uncertain_inputs"]["sensitivity_priors"]["cusp_upper_bounds"]:
        raise ValueError("sensitivity priors differ from the protocol declaration")
    if len(priors) != len(PRIOR_TOKENS) or len(invariance["per_prior"]) != len(priors):
        raise ValueError("prior count differs from the admitted layout")
    if sensitivity["unique_designs"] != pooled["unique_designs"]:
        raise ValueError("sensitivity unique designs differ from the pooled fronts")
    for prior, gate_prior in zip(priors, invariance["per_prior"], strict=True):
        for key in gate_prior:
            if gate_prior[key] != prior[key]:
                raise ValueError(f"design-set invariance per-prior record differs from sensitivity.json ({key})")
        if prior["feasible"] + prior["infeasible"] != sensitivity["unique_designs"] or len(prior["front_design_ids"]) != prior["front_size"]:
            raise ValueError("prior feasible/infeasible counts or front ids do not reproduce")
    if invariance["passed"] is not all(p["identical_on_common_feasible_set_up_to_ties"] is True for p in priors):
        raise ValueError("design-set invariance flag does not reproduce from the priors")
    if campaign["design_set_invariance"] is not invariance["passed"]:
        raise ValueError("campaign-result invariance flag differs from gates.json")
    campaign_prior = next(p for p in priors if p["cusp_upper"] == upper)
    if campaign_prior["identical_to_campaign_front"] is not True or campaign_prior["front_size"] != pooled["robust"]["front_size"] or campaign_prior["hypervolume"] != pooled["robust"]["hypervolume"]:
        raise ValueError("the campaign prior does not reproduce the pooled robust front")
    m.add_derived("MdoPriorCount", len(priors), "int", "alternative cusp priors evaluated", "len(sensitivity.priors)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add("MdoPriorUppers", "artifacts/protocol.json", "/uncertain_inputs/sensitivity_priors/cusp_upper_bounds", "list_g", "upper bounds of the alternative uniform cusp priors")
    m.add_derived("MdoPriorUpperMin", min(p["cusp_upper"] for p in priors), "g", "smallest alternative prior upper bound", "min(sensitivity.priors[*].cusp_upper)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoPriorUpperMax", max(p["cusp_upper"] for p in priors), "g", "largest alternative prior upper bound", "max(sensitivity.priors[*].cusp_upper)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoInvarianceIdenticalCount", sum(1 for p in priors if p["identical_on_common_feasible_set_up_to_ties"] is True), "int", "priors whose robust nondominated set is identical on the common feasible set", "count(sensitivity.priors[*].identical_on_common_feasible_set_up_to_ties == true)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoInvarianceExactCount", sum(1 for p in priors if p["identical_on_common_feasible_set"] is True), "int", "priors identical on the common feasible set in exact arithmetic", "count(sensitivity.priors[*].identical_on_common_feasible_set == true)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add("MdoInvariancePassed", "artifacts/gates.json", "/reported_not_binding/design_set_invariance/passed", "bool", "design-set invariance outcome")
    m.add("MdoTieTolerance", "artifacts/sensitivity.json", "/priors/0/tie_tolerance_relative", "sci1", "roundoff-aware dominance tolerance")
    m.add_derived("MdoPriorFeasibleMin", min(p["feasible"] for p in priors), "int", "fewest feasible designs under any prior", "min(sensitivity.priors[*].feasible)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoPriorFeasibleMax", max(p["feasible"] for p in priors), "int", "most feasible designs under any prior", "max(sensitivity.priors[*].feasible)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoPriorFrontMin", min(p["front_size"] for p in priors), "int", "smallest front under any prior", "min(sensitivity.priors[*].front_size)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoPriorFrontMax", max(p["front_size"] for p in priors), "int", "largest front under any prior", "max(sensitivity.priors[*].front_size)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add_derived("MdoPriorJaccardMin", min(p["jaccard_with_campaign_front"] for p in priors), "fixed3", "smallest Jaccard index of an alternative-prior front with the campaign front", "min(sensitivity.priors[*].jaccard_with_campaign_front)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/priors"}])
    m.add("MdoSampledSurvivalMin", "artifacts/sensitivity.json", f"/priors/{priors.index(campaign_prior)}/survival_min", "fixed3", "smallest sampled CL-1 survival under the campaign prior")
    m.add("MdoSampledSurvivalMax", "artifacts/sensitivity.json", f"/priors/{priors.index(campaign_prior)}/survival_max", "fixed3", "largest sampled CL-1 survival under the campaign prior")
    m.add("MdoSampledSurvivalMean", "artifacts/sensitivity.json", f"/priors/{priors.index(campaign_prior)}/survival_mean", "fixed3", "mean sampled CL-1 survival under the campaign prior")
    prior_rows: list[str] = []
    for prior, token in zip(priors, PRIOR_TOKENS, strict=True):
        index = priors.index(prior)
        m.add(f"MdoPrior{token}Upper", "artifacts/sensitivity.json", f"/priors/{index}/cusp_upper", "g", f"alternative prior upper bound ({token})")
        m.add(f"MdoPrior{token}Feasible", "artifacts/sensitivity.json", f"/priors/{index}/feasible", "int", f"feasible designs under prior {token}")
        m.add(f"MdoPrior{token}Front", "artifacts/sensitivity.json", f"/priors/{index}/front_size", "int", f"front size under prior {token}")
        m.add(f"MdoPrior{token}Common", "artifacts/sensitivity.json", f"/priors/{index}/common_feasible_designs", "int", f"designs feasible under both prior {token} and the campaign prior")
        m.add(f"MdoPrior{token}Identical", "artifacts/sensitivity.json", f"/priors/{index}/identical_on_common_feasible_set_up_to_ties", "bool", f"front identical on the common feasible set under prior {token}")
        m.add(f"MdoPrior{token}Jaccard", "artifacts/sensitivity.json", f"/priors/{index}/jaccard_with_campaign_front", "fixed3", f"Jaccard index with the campaign front under prior {token}")
        prior_rows.append(
            f"uniform $[{format_value('g', cusp_inputs[0]['lower'])}, {format_value('g', prior['cusp_upper'])}]$ & "
            f"{format_value('fixed3', prior['survival_min'])}--{format_value('fixed3', prior['survival_max'])} & "
            f"{prior['feasible']} / {prior['infeasible']} & {prior['front_size']} & {format_value('sig3', prior['hypervolume'])} & "
            f"{prior['common_feasible_designs']} & {'identical' if prior['identical_on_common_feasible_set_up_to_ties'] else 'differs'} & "
            f"{format_value('fixed3', prior['jaccard_with_campaign_front'])}\\\\"
        )
    if [s["id"] for s in scenarios] != [s["id"] for s in protocol["uncertain_inputs"]["sensitivity_scenarios"]]:
        raise ValueError("scenario ids differ from the protocol declaration")
    if set(SCENARIO_TOKENS) != {s["id"] for s in scenarios}:
        raise ValueError("scenario ids differ from the admitted layout")
    m.add_derived("MdoScenarioCount", len(scenarios), "int", "fixed cusp-probability scenarios evaluated", "len(sensitivity.scenarios)", [{"artifact": "artifacts/sensitivity.json", "pointer": "/scenarios"}])
    scenario_rows: list[str] = []
    for index, scenario in enumerate(scenarios):
        token = SCENARIO_TOKENS[scenario["id"]]
        declared = protocol["uncertain_inputs"]["sensitivity_scenarios"][index]
        if declared["cusp_probabilities"] != scenario["cusp_probabilities"]:
            raise ValueError(f"scenario {scenario['id']} probabilities differ from the protocol")
        survival = math.prod(1.0 - p for p in scenario["cusp_probabilities"])
        if abs(survival - scenario["survival"]) > 1e-15 * max(1.0, abs(survival)):
            raise ValueError(f"scenario {scenario['id']} survival does not reproduce")
        if scenario["pareto_designs_evaluated"] + scenario["pareto_designs_infeasible"] != pooled["robust"]["front_size"]:
            raise ValueError(f"scenario {scenario['id']} does not cover the pooled robust front")
        if len(scenario["rows"]) != scenario["pareto_designs_evaluated"]:
            raise ValueError(f"scenario {scenario['id']} rows differ from the evaluated count")
        for name in OBJECTIVES:
            values = [row["objectives"][name] for row in scenario["rows"]]
            rng = scenario["objective_ranges"][name]
            if min(values) != rng["minimum"] or max(values) != rng["maximum"]:
                raise ValueError(f"scenario {scenario['id']} range does not reproduce for {name}")
        base = f"/scenarios/{index}"
        m.add(f"MdoScenario{token}Id", "artifacts/sensitivity.json", f"{base}/id", "ident", f"scenario identifier ({token})")
        m.add(f"MdoScenario{token}Probabilities", "artifacts/sensitivity.json", f"{base}/cusp_probabilities", "list_g", f"scenario cusp probabilities ({token})")
        m.add(f"MdoScenario{token}Survival", "artifacts/sensitivity.json", f"{base}/survival", "sig3", f"scenario CL-1 survival ({token})")
        m.add(f"MdoScenario{token}Evaluated", "artifacts/sensitivity.json", f"{base}/pareto_designs_evaluated", "int", f"robust-Pareto designs feasible in the scenario ({token})")
        m.add(f"MdoScenario{token}Infeasible", "artifacts/sensitivity.json", f"{base}/pareto_designs_infeasible", "int", f"robust-Pareto designs infeasible in the scenario ({token})")
        m.add(f"MdoScenario{token}ThrustMax", "artifacts/sensitivity.json", f"{base}/objective_ranges/axial_thrust_n/maximum", "sig3", f"scenario thrust maximum (N) ({token})")
        m.add(f"MdoScenario{token}IspMax", "artifacts/sensitivity.json", f"{base}/objective_ranges/specific_impulse_s/maximum", "sig3", f"scenario specific impulse maximum (s) ({token})")
        m.add(f"MdoScenario{token}EffMax", "artifacts/sensitivity.json", f"{base}/objective_ranges/thruster_electrical_to_beam_efficiency/maximum", "sig3", f"scenario efficiency maximum ({token})")
        m.add(f"MdoScenario{token}Hv", "artifacts/sensitivity.json", f"{base}/hypervolume", "sig3", f"scenario hypervolume ({token})")
        scenario_rows.append(
            f"\\texttt{{{format_value('ident', scenario['id'])}}} & {format_value('list_g', scenario['cusp_probabilities'])} & "
            f"{scenario['pareto_designs_evaluated']} / {scenario['pareto_designs_infeasible']} & {format_value('sig3', scenario['survival'])} & "
            f"{format_value('sig3', scenario['objective_ranges']['axial_thrust_n']['maximum'])} & "
            f"{format_value('sig3', scenario['objective_ranges']['specific_impulse_s']['maximum'])} & "
            f"{format_value('sig3', scenario['objective_ranges']['thruster_electrical_to_beam_efficiency']['maximum'])} & "
            f"{format_value('sig3', scenario['hypervolume'])}\\\\"
        )
    jeffreys = next(s for s in scenarios if s["id"] == "v4_per_cell_jeffreys")
    m.add("MdoJeffreysCellOne", "artifacts/sensitivity.json", f"/scenarios/{scenarios.index(jeffreys)}/cusp_probabilities/0", "fixed4", "Jeffreys scenario cell-one probability")
    m.add("MdoJeffreysCellTwo", "artifacts/sensitivity.json", f"/scenarios/{scenarios.index(jeffreys)}/cusp_probabilities/1", "fixed4", "Jeffreys scenario cell-two probability")
    m.add("MdoJeffreysCellThree", "artifacts/sensitivity.json", f"/scenarios/{scenarios.index(jeffreys)}/cusp_probabilities/2", "fixed4", "Jeffreys scenario cell-three probability")
    m.add("MdoJeffreysCellFour", "artifacts/sensitivity.json", f"/scenarios/{scenarios.index(jeffreys)}/cusp_probabilities/3", "fixed4", "Jeffreys scenario cell-four probability")
    m.add("MdoJeffreysRule", "artifacts/protocol.json", f"/uncertain_inputs/sensitivity_scenarios/{scenarios.index(jeffreys)}/rule", "text", "Jeffreys scenario rule")
    if jeffreys["cusp_probabilities"][1] != jeffreys["cusp_probabilities"][2]:
        raise ValueError("Jeffreys scenario cells two and three differ")
    no_wall = next(s for s in scenarios if s["id"] == "no_wall_loss")
    if no_wall["survival"] != 1.0:
        raise ValueError("no_wall_loss scenario survival is not one")

    # Prior-model disclosure recorded in the frozen protocol (parsed with a fixed pattern).
    probe_text = protocol["prior_model_disclosure"]["corrected_four_cell_solver_probe"]
    probe = PROBE_PATTERN.search(probe_text)
    if probe is None:
        raise ValueError("the four-cell solver probe disclosure does not match the fixed pattern")
    probe_inputs = [{"artifact": "artifacts/protocol.json", "pointer": "/prior_model_disclosure/corrected_four_cell_solver_probe"}]
    m.add_derived("MdoProbeClosedCases", int(probe.group("closed")), "int", "four-cell solver probe cases that closed", "regex group 'closed' of PROBE_PATTERN over the protocol disclosure text", probe_inputs)
    m.add_derived("MdoProbeTotalCases", int(probe.group("total")), "int", "four-cell solver probe cases", "regex group 'total' of PROBE_PATTERN over the protocol disclosure text", probe_inputs)
    m.add_derived("MdoProbeResidualMin", float(probe.group("rmin")), "sci1", "smallest residual floor of the non-closing probe cases", "regex group 'rmin' of PROBE_PATTERN over the protocol disclosure text", probe_inputs)
    m.add_derived("MdoProbeResidualMax", float(probe.group("rmax")), "fixed3", "largest residual floor of the non-closing probe cases", "regex group 'rmax' of PROBE_PATTERN over the protocol disclosure text", probe_inputs)
    m.add_derived("MdoProbeSecondsPerSolve", int(probe.group("sec")), "int", "seconds per multistart solve in the probe", "regex group 'sec' of PROBE_PATTERN over the protocol disclosure text", probe_inputs)

    # Shakedown disclosure.
    m.add("MdoShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("MdoShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add("MdoShakedownDisjoint", "artifacts/shakedown.json", "/disjointness/proven", "bool", "shakedown disjointness proven")
    m.add("MdoShakedownBudget", "artifacts/shakedown.json", "/shakedown_plan/evaluations_per_run", "int", "shakedown evaluations per run")
    m.add("MdoShakedownSeeds", "artifacts/shakedown.json", "/shakedown_plan/seeds", "list_int", "shakedown seeds")
    m.add("MdoShakedownRuntimeS", "artifacts/shakedown.json", "/timing_s/runtime_total", "fixed0", "shakedown runtime (s)")

    # Generated TeX.
    lines = [
        "% Generated by paper/scripts/generate_mdo_l0_v1_evidence.py; do not hand edit.",
        f"% Evidence: {RESULTS.as_posix()} at commit {RESULTS_COMMIT_SHA} (manifest SHA-256 {bundle.manifest_sha256}).",
        "% Every macro value traces to an artifact path and JSON pointer recorded in paper/evidence/mdo-l0-v1.json.",
    ]
    for item in m.items:
        lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    artifact_open = f"\\ArtifactClaim{{{ARTIFACT_CLAIM_ID}}}{{{ARTIFACT_ID}}}{{%"
    lines.append("\\newcommand{\\MdoHvTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Final robust hypervolume after \\MdoEvaluationsPerRun{} evaluations per run "
        "(dimensionless all-maximise frame against the declared reference point), fraction of the "
        "\\MdoDenseCount-design dense-reference robust hypervolume \\MdoDenseRobustHv, final Pareto-set size, "
        "constraint-violating evaluations and wall time per run; the last block gives the mean and sample "
        "standard deviation over the \\MdoSeedCount{} seeds. Wall times are diagnostic only.}"
    )
    lines.append("\\label{tab:mdo-l0-v1-hypervolume}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("optimiser & seed & final HV & fraction of dense ref. & Pareto set & infeasible & wall (s)\\\\")
    lines.append("\\midrule")
    lines.extend(hv_rows)
    lines.append("\\midrule")
    lines.extend(summary_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    lines.append("\\newcommand{\\MdoRobustNominalTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Pooled robust (CVaR over the frozen sample) and nominal (prior midpoints) fronts of the "
        "\\MdoUniqueDesigns{} unique evaluated designs. The nominal front is a re-evaluation of the same designs, "
        "not a separate optimisation; objective ranges are the extremes over each front's members.}"
    )
    lines.append("\\label{tab:mdo-l0-v1-robust-nominal}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{lrr}")
    lines.append("\\toprule")
    lines.append("quantity & robust front & nominal front\\\\")
    lines.append("\\midrule")
    lines.extend(rn_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    lines.append("\\newcommand{\\MdoScenarioTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Sensitivity to the cusp probabilities. Upper block: every recorded design re-evaluated under "
        "alternative uniform cusp priors with the same frozen sample rows (survival range, feasible / infeasible "
        "designs, robust front size and hypervolume, designs feasible under both the alternative and the campaign "
        "prior, whether the robust nondominated set restricted to that common set equals the campaign set up to "
        "roundoff ties, Jaccard index of the unrestricted fronts). Lower block: the \\MdoRobustFront{} pooled "
        "robust-Pareto designs evaluated at fixed cusp probabilities with the other closures nominal (survival "
        "$S$, feasible / infeasible designs, maxima of thrust, specific impulse and efficiency, hypervolume).}"
    )
    lines.append("\\label{tab:mdo-l0-v1-sensitivity}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.2cm}rrrrrr}"
    )
    lines.append("\\toprule")
    lines.append(
        "cusp prior & $S$ range & \\shortstack[r]{feasible /\\\\infeasible} & front & HV & "
        "\\shortstack[r]{common\\\\feasible} & \\shortstack[r]{front on\\\\common set} & Jaccard\\\\"
    )
    lines.append("\\midrule")
    lines.extend(prior_rows)
    lines.append("\\midrule")
    lines.append(
        "scenario & $p_k$ per cell & \\shortstack[r]{feasible /\\\\infeasible} & $S$ & "
        "\\shortstack[r]{thrust\\\\max (N)} & \\shortstack[r]{$I_{\\mathrm{sp}}$\\\\max (s)} & "
        "\\shortstack[r]{efficiency\\\\max} & HV\\\\"
    )
    lines.append("\\midrule")
    lines.extend(scenario_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    tex = "\n".join(lines) + "\n"

    evidence = {
        "document_type": "paper-mdo-l0-v1-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "closure": CLOSURE_ID,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "manuscript_integration": {
            "status": "admitted",
            "section_path": SECTION_PATH.as_posix(),
            "section_heading": SECTION_HEADING,
            "section_binding": SECTION_BINDING,
            "generated_tex_path": OUTPUT_PATH.as_posix(),
            "generated_binding": GENERATED_BINDING,
            "manifest_id": MANIFEST_ID,
            "manifest_path": MANIFEST_PATH.as_posix(),
            "gate_id": GATE_ID,
            "gate_kind": "numerical-campaign",
            "artifact_id": ARTIFACT_ID,
            "artifact_claim_id": ARTIFACT_CLAIM_ID,
            "prose_claim_ids": list(PROSE_CLAIM_IDS),
            "rule": (
                "Every number in the section is a macro defined here; each macro is bound below to an "
                "artifact path, JSON pointer, formatter and SHA-256, or to a stated derivation over such "
                "inputs. Claim-bearing sentences are exact EvidenceClaim bodies registered in "
                "paper/evidence/claims.json; the numerical-campaign gate in paper/evidence/result-gates.json "
                "names the typed manifest that admits the section. Every number is conditional on the "
                "declared closure and priors and none is thruster performance."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "MdoHvTable": {"rows": len(hv_rows) + len(summary_rows), "source": "artifacts/metrics.json#/runs, #/hypervolume_table, #/seed_variance"},
            "MdoRobustNominalTable": {"rows": len(rn_rows), "source": "artifacts/pooled-fronts.json#/robust, #/nominal"},
            "MdoScenarioTable": {"rows": len(prior_rows) + len(scenario_rows), "source": "artifacts/sensitivity.json#/priors, #/scenarios"},
        },
        "generator": {
            "path": "paper/scripts/generate_mdo_l0_v1_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_mdo_l0_v1_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_mdo_l0_v1_evidence.py",
        },
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex.encode("utf-8"))},
    }
    if len({item["name"] for item in m.items}) != len(m.items):
        raise ValueError("duplicate macro names")
    return evidence, tex


def render(repo: Path) -> tuple[bytes, bytes, bytes]:
    evidence, tex = build(repo)
    tex_bytes = tex.encode("utf-8")
    build_config = json.loads((repo / "paper/build-config.json").read_text("utf-8"))
    sidecar = {
        "document_type": "paper-generated-artifact-provenance",
        "schema_version": "1.0",
        "artifact_id": ARTIFACT_ID,
        "claim_ids": [ARTIFACT_CLAIM_ID],
        "claim_status": (
            f"authorized by {ARTIFACT_CLAIM_ID} (quantitative-generated-table) in "
            f"paper/evidence/claims.json; admitted through {GATE_ID}"
        ),
        "evidence_revision": RESULTS_COMMIT_SHA,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": evidence["generator"],
        "manifest": {
            "path": EVIDENCE_PATH.as_posix(),
            "sha256": sha256_bytes(canonical_json(evidence)),
            "manifest_id": MANIFEST_ID,
            "gate_manifest_path": MANIFEST_PATH.as_posix(),
        },
        "inputs": [
            {"path": (RESULTS / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["artifacts"].items()
        ],
        "bundle_manifest": {
            "path": evidence["bundle"]["manifest_path"],
            "sha256": evidence["bundle"]["manifest_sha256"],
            "git_blob": evidence["binding"]["manifest_git_blob"],
        },
        "dashboard": evidence["dashboard"],
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex_bytes)},
    }
    return canonical_json(evidence), tex_bytes, canonical_json(sidecar)


def write_generated(repo: Path) -> None:
    evidence, tex, sidecar = render(repo)
    (repo / EVIDENCE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / EVIDENCE_PATH).write_bytes(evidence)
    (repo / OUTPUT_PATH).write_bytes(tex)
    (repo / SIDECAR_PATH).write_bytes(sidecar)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    try:
        write_generated(repo)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"MDO L0 v1 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
