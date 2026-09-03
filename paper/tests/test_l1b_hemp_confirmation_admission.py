"""Adversarial tests for the admission of the L1b/P2 material-aware HEMP confirmation v1.1 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_cusp_topology_v3_1_evidence as ctv  # noqa: E402
import generate_l1a_sweep_v3_evidence as swt  # noqa: E402
import generate_l1b_hemp_confirmation_v1_1_evidence as hmc  # noqa: E402

GATE_ID = hmc.GATE_ID
MANIFEST_ID = hmc.MANIFEST_ID
HEADING = hmc.SECTION_HEADING
SWT_MANIFEST_ID = "L1A-SWEEP-V3-20260903-128-V1"
CTV_MANIFEST_ID = "CUSP-TOPOLOGY-V3-1-20260903-281-V1"
WLH_MANIFEST_ID = "WALL-LOSS-GEOMETRY-SCREENING-V2-20260903-377-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class HempConfirmationAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json(hmc.EVIDENCE_PATH.as_posix())
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_hemp_confirmation(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def _claim(self, claim_id: str, matrix=None):
        return next(c for c in (matrix or self.matrix)["claims"] if c["id"] == claim_id)

    def test_gate_adds_a_sixth_outcome_with_justification(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertEqual(self.gate["kind"], check_paper.SCREENING_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["recorded_outcome"], hmc.RECORDED_OUTCOME)
        # A sixth outcome value: the field is a linear-iron P2 field for every design, not an L1a field.
        self.assertEqual(len(check_paper.SCREENING_OUTCOMES), 6)
        self.assertIn(hmc.RECORDED_OUTCOME, check_paper.SCREENING_OUTCOMES)
        self.assertNotEqual(hmc.RECORDED_OUTCOME, ctv.RECORDED_OUTCOME)
        self.assertNotEqual(hmc.RECORDED_OUTCOME, swt.RECORDED_OUTCOME)
        self.assertIn("sixth outcome", self.gate["recorded_outcome_justification"])
        self.assertIn("as recorded", self.gate["recorded_outcome_justification"])
        self.assertIn(hmc.RECORDED_OUTCOME, kinds["numerical-screening"])
        self.assertIn("linear-iron", kinds["numerical-screening"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-material-aware-confirmation-manifest")
        self.assertEqual(self.gate["evidence_revision"], hmc.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], hmc.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["code_revision"], hmc.CODE_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], hmc.DASHBOARD_COMMIT_SHA)
        self.assertEqual(self.gate["lineage_revisions"]["predecessor_results"], hmc.V1_RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["lineage_revisions"]["predecessor_preregistration"], hmc.V1_PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(set(self.gate["dependencies"]), {"GATE-L1A-SWEEP-V3", "GATE-CUSP-TOPOLOGY-V3-1"})
        for dependency in self.gate["dependencies"]:
            self.assertEqual(next(g for g in self.registry["gates"] if g["id"] == dependency)["status"], "accepted")
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        entry = self.matrix["manifests"][MANIFEST_ID]
        self.assertEqual(entry["recorded_outcome"], hmc.RECORDED_OUTCOME)
        self.assertEqual(entry["lineage_revisions"]["predecessor_results"], hmc.V1_RESULTS_COMMIT_SHA)
        self.assertEqual(entry["posthoc_rejection_disclosure"], hmc.V1_REJECTION_PATH.as_posix())
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-screening")
        self.assertEqual(self.payload["gate_kind"], "numerical-screening")
        self.assertEqual(self.payload["recorded_outcome"], hmc.RECORDED_OUTCOME)
        self.assertEqual(self.payload["classification"], hmc.CLASSIFICATION)
        self.assertEqual(self.payload["topology_label"], hmc.TOPOLOGY_LABEL)
        self.assertEqual(self.payload["verdict"], hmc.VERDICT)
        self.assertEqual(self.payload["screening_model"], hmc.SCREENING_MODEL)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        self.assertIsNone(self.payload["posthoc_audit"])
        self.assertIn("lineage", self.payload["posthoc_audit_note"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        gate_record = self._claim(GATE_ID)
        self.assertEqual(gate_record["kind"], "numerical-screening")
        self.assertEqual(gate_record["recorded_outcome"], hmc.RECORDED_OUTCOME)
        self.assertEqual(gate_record["manifest_id"], MANIFEST_ID)

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))

    def test_manifest_validates_and_binds_every_bundle_file(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, self.payload,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertEqual(errors, [])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-material-aware-confirmation-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-material-aware-confirmation-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]}, set(schema["required_file_roles"]))
        for role in ("design-record", "field-grid", "primary-dataset", "dataset-csv", "binding-gates", "source-binding", "campaign-result"):
            self.assertIn(role, schema["required_file_roles"])
        bundle_paths = {s["path"] for s in self.payload["source_files"] if not s["role"].startswith("preregistered-")}
        on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / hmc.RESULTS).rglob("*") if p.is_file()}
        self.assertEqual(bundle_paths, on_disk)
        self.assertEqual(len(on_disk), 134)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "transition"), 9)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "design-record"), 15)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "field-grid"), 15)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"].startswith("preregistered-")), 4)
        paths = [s["path"] for s in self.payload["source_files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual({f["role"] for f in self.payload["reference_files"]}, {"reference-sweep-v3-catalogue", "reference-sweep-v3-manifest", "reference-sweep-v3-design-authorities", "reference-cusp-topology-protocol"})
        lineage = self.payload["lineage"]
        v1_on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / hmc.V1_RESULTS).rglob("*") if p.is_file()}
        lineage_paths = {f["path"] for f in lineage["files"]}
        self.assertTrue(v1_on_disk <= lineage_paths)
        self.assertEqual(len(v1_on_disk), 104)
        self.assertIn(hmc.V1_REJECTION_PATH.as_posix(), lineage_paths)
        self.assertEqual(lineage["terminal_state"], "development_rejection")
        self.assertEqual(lineage["failed_designs"], ["l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"])
        self.assertIs(lineage["cited_for_numbers"], False)
        self.assertEqual(len(lineage["protocol_paths_changed"]), 12)
        wrong_level = copy.deepcopy(self.payload)
        wrong_level["level"] = "L1"
        errors = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, wrong_level,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertTrue(any("level does not match" in error for error in errors))

    def test_manifest_metrics_equal_the_raw_artifact_values(self) -> None:
        raw = {item["name"]: item["raw"] for item in self.evidence["macros"]}
        for metric, macro in check_paper.HEMP_CONFIRMATION_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.HEMP_CONFIRMATION_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual((metrics["evaluated_count"], metrics["declared_count"], metrics["failed_cases_count"], metrics["matched_cusp_count"]), (15, 15, 0, 37))
        self.assertEqual((metrics["solves_converged"], metrics["solves_total"], metrics["relative_true_residual_maximum"] <= 2.0e-10), (30, 30, True))
        self.assertEqual((metrics["level0_dofs_minimum"], metrics["level0_dofs_maximum"], metrics["level1_dofs_minimum"], metrics["level1_dofs_maximum"]), (24369, 116883, 50037, 466005))
        self.assertEqual((metrics["gate_b_agreeing_strict"], metrics["gate_b_agreeing_boundary_tolerant"], metrics["gate_b_passed"], metrics["gate_c_passed"], metrics["all_designs_bijective"]), (15, 15, True, True, True))
        self.assertAlmostEqual(metrics["gate_c_max_shift_over_tolerance"], 0.8023324858192963)
        self.assertAlmostEqual(metrics["shift_maximum_m"], 0.0003621639692934324)
        self.assertAlmostEqual(metrics["shift_median_m"], 0.0002671743195241487)
        self.assertEqual((metrics["hemp_like_preserved_count"], metrics["hemp_like_lost_count"], metrics["hemp_like_lost_design"]), (14, 1, "028"))
        self.assertAlmostEqual(metrics["lost_design_l1a_min_rho"], 1.5150770513776666)
        self.assertAlmostEqual(metrics["lost_design_p2_min_rho"], 1.463778208091294)
        self.assertAlmostEqual(metrics["rho_ratio_minimum"], 0.942536584332938)
        self.assertAlmostEqual(metrics["rho_ratio_maximum"], 1.4459048971630442)
        self.assertAlmostEqual(metrics["wall_b_ratio_minimum"], 1.0547158205359815)
        self.assertAlmostEqual(metrics["wall_b_ratio_maximum"], 1.5252332413152598)
        self.assertAlmostEqual(metrics["wall_b_ratio_median"], 1.2252323629590833)
        self.assertAlmostEqual(metrics["axis_peak_b_ratio_minimum"], 0.9766238209161009)
        self.assertAlmostEqual(metrics["axis_peak_b_ratio_maximum"], 1.345591831654561)
        self.assertAlmostEqual(metrics["channel_axis_null_shift_maximum_m"], 0.0010724508771368093)
        self.assertAlmostEqual(metrics["separatrix_lean_l1a_maximum_m"], 0.000463299811208474)
        self.assertAlmostEqual(metrics["separatrix_lean_p2_maximum_m"], 0.0011445245108789607)
        self.assertAlmostEqual(metrics["discretisation_shift_maximum_m"], 1.4220098524617836e-06)
        self.assertEqual((metrics["sampling_stable_count"], metrics["discretisation_stable_count"], metrics["channel_axis_null_bijection"], metrics["pooled_axis_null_bijection"]), (15, 15, 6, 0))
        self.assertAlmostEqual(metrics["stage_wall_s"], 3079.2892819000117)
        self.assertAlmostEqual(metrics["assessment_wall_s"], 304.73673110001255)
        self.assertEqual((metrics["worker_pool_size"], metrics["peak_rss_bytes"], metrics["soft_iron_relative_permeability"]), (1, 239702016, 4000.0))
        self.assertEqual((metrics["angle_gate_deg"], metrics["predecessor_angle_gate_deg"]), (5.0, 10.0))
        self.assertEqual((metrics["predecessor_terminal_state"], metrics["predecessor_resolved"], metrics["predecessor_failed"], metrics["predecessor_failure_stage"]), ("development_rejection", 13, 2, "resolve"))
        self.assertEqual((metrics["protocol_paths_changed"], metrics["protocol_declarations_changed"], metrics["protocol_blocks_unchanged"]), (12, 2, True))
        self.assertEqual((metrics["shakedown_designs"], metrics["shakedown_overlap_designs"], metrics["timing_within_budget"], metrics["stage_within_budget"]), (5, 5, False, True))
        self.assertEqual((metrics["verified_file_count"], metrics["tolerated_eol_file_count"], metrics["record_commit_files"], metrics["transition_count"]), (133, 0, 134, 9))
        for metric, rule in self.gate["metric_constraints"].items():
            self.assertIn(metric, metrics, metric)
            self.assertEqual(metrics[metric], rule["equals"], metric)
        self.assertNotIn("stable_multicell_wall_cusp_topology_demonstrated", metrics)

    def test_confirmation_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_tampered_metric_outcome_or_finding_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["hemp_like_preserved_count"] = 15
        payload["metrics"]["matched_cusp_count"] = 37.0  # right value, wrong type
        payload["metrics"]["axis_null_positions_robust_within_tolerance"] = True
        payload["metrics"]["positive_finding_accepted"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("metric 'hemp_like_preserved_count' differs" in e for e in errors))
        self.assertTrue(any("metric 'matched_cusp_count' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'axis_null_positions_robust_within_tolerance'" in e for e in errors))
        self.assertTrue(any("policy metric 'positive_finding_accepted'" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome"] = ctv.RECORDED_OUTCOME
        errors = self._errors(gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        self.assertTrue(any("must not reuse an L1a screening outcome" in e for e in errors))
        gate["recorded_outcome"] = "accepted-finding"
        errors = self._errors(gate=gate)
        self.assertTrue(any("not a recognized screening outcome" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome_justification"] = "a confirmation"
        errors = self._errors(gate=gate)
        self.assertTrue(any("justify the sixth outcome" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["verdict"] = "PARTIALLY_CONFIRMED"
        errors = self._errors(payload=payload)
        self.assertTrue(any("verdict differs" in e for e in errors))

    def test_labels_and_model_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "P2_QUALIFIED_FIELD"
        errors = self._errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["screening_model"] = "saturating iron"
        errors = self._errors(payload=payload)
        self.assertTrue(any("screening_model differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["topology_label"] = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
        errors = self._errors(payload=payload)
        self.assertTrue(any("topology label differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["HmcClassification"]), hmc.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["HmcTopologyLabel"]), hmc.TOPOLOGY_LABEL)
        self.assertEqual(check_paper.tex_unescape(values["HmcRecordedOutcome"]), hmc.RECORDED_OUTCOME)
        self.assertEqual(check_paper.tex_unescape(values["HmcCampaignStatus"]), hmc.CAMPAIGN_STATUS)
        self.assertEqual(values["HmcVerdict"], "CONFIRMED")
        self.assertEqual((values["HmcFieldModelLevelLOneA"], values["HmcTopologyVersion"], values["HmcVersion"], values["HmcVOneVersion"]), ("L1a", "v3.1", "v1.1", "v1"))
        section = (REPO / hmc.SECTION_PATH).read_text(encoding="utf-8")
        for macro in ("\\HmcClassification", "\\HmcTopologyLabel", "\\HmcRecordedOutcome", "\\HmcCampaignStatus", "\\HmcVerdict", "\\HmcLostDesign", "\\HmcChannelNullShiftMaxMm", "\\HmcVOneTerminalState", "\\HmcPaperAdmissionRecord"):
            self.assertIn(macro, section)

    def test_opening_a_level_or_touching_the_dashboard_or_lineage_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = hmc.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        html = next(f for f in payload["dashboard"]["files"] if f["role"] == "dashboard-html")
        html["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"] = {"revision": hmc.CODE_COMMIT_SHA}
        errors = self._errors(payload=payload)
        self.assertTrue(any("binds a post-hoc audit" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["results_commit"] = hmc.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("registered predecessor revisions" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["failed_designs"] = ["l1a-gs-v3-028-f012c0bf33"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("lineage failed_designs differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["files"] = [f for f in payload["lineage"]["files"] if f["path"] != hmc.V1_REJECTION_PATH.as_posix()]
        errors = self._errors(payload=payload)
        self.assertTrue(any("does not bind the post-hoc rejection note" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        note = next(f for f in payload["lineage"]["files"] if f["path"] == hmc.V1_REJECTION_PATH.as_posix())
        note["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        victim = next(f for f in payload["lineage"]["files"] if f["path"].endswith("results/terminal.json"))
        victim["git_blob"] = "0" * 40
        errors = self._errors(payload=payload)
        self.assertTrue(any("Git blob mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["cited_for_numbers"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("never cited for a number" in e for e in errors))

    def test_reference_frozen_and_bundle_files_must_all_be_bound(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["source_files"] = [s for s in payload["source_files"] if s["role"] != "preregistered-design-authorities"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("frozen preregistration files are not all bound" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        frozen = next(s for s in payload["source_files"] if s["role"] == "preregistered-protocol")
        frozen["git_blob"] = "0" * 40
        errors = self._errors(payload=payload)
        self.assertTrue(any("changed after preregistration" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["source_files"] = [s for s in payload["source_files"] if s["role"] != "field-grid"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("does not bind every file of the results tree" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["reference_files"] = [f for f in payload["reference_files"] if f["role"] != "reference-sweep-v3-catalogue"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("file group differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        catalogue = next(f for f in payload["reference_files"] if f["role"] == "reference-sweep-v3-catalogue")
        catalogue["revision"] = hmc.CUSP_TOPOLOGY_RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("revision differs from the evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["results_bundle"]["source_hashes_recomputed_at_preregistration"]["experiment_code_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("recomputed source hashes differ" in e for e in errors))

    def test_missing_non_claim_or_required_wording_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-087", matrix)
        record["non_claims"].append("validated against thruster measurements")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-087 is absent" in e for e in errors))
        for phrase in self._claim("CLM-087")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-089", matrix)
        record["authorized_tex"] = record["authorized_tex"].replace("not about a plasma", "about a plasma")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-089 lacks the required wording 'two field models and not about a plasma'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-090", matrix)
        record["authorized_tex"] = record["authorized_tex"].replace("The axis nulls are not robust", "The axis nulls are robust")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-090 lacks the required wording 'The axis nulls are not robust within the cusp tolerance'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-091", matrix)
        record["authorized_tex"] = record["authorized_tex"].replace("known before the freeze", "known after the freeze")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-091 lacks the required wording 'known before the freeze'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-093", matrix)
        record["claim_class"] = "quantitative-screening-result"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-093 must be an interpretation" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-076", matrix)
        record["manifest_ids"] = [m for m in record["manifest_ids"] if m != MANIFEST_ID]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-076 reads a confirmation macro and must be bound" in e for e in errors))

    def test_unbound_relocated_or_misplaced_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-090", matrix)
        record["manifest_ids"] = [SWT_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-090 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-090 does not allow the section heading" in e for e in errors))
        section = (REPO / hmc.SECTION_PATH).read_text(encoding="utf-8")
        interpretation = self._claim("CLM-093")
        self.assertEqual(interpretation["claim_class"], "interpretation")
        self.assertNotIn("CLM-093", section)
        matrix = copy.deepcopy(self.matrix)
        self._claim("CLM-092", matrix)["claim_class"] = "interpretation"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-092 must not appear inside the results section" in e for e in errors))

    def test_superseded_queued_confirmation_wording_is_rejected(self) -> None:
        stale = self.manuscript.replace(
            "was the confirmation the protocol queued, and\nSection~\\ref{sec:l1b-hemp-confirmation} now reports it",
            "is the confirmation the protocol queues and this paper\ndoes not report; nothing reports it",
        )
        self.assertNotEqual(stale, self.manuscript)
        errors: list[str] = []
        flattened = check_paper.flatten_sections(REPO, stale, errors)
        errors = self._errors(manuscript=stale, flattened=flattened)
        self.assertTrue(any("superseded queued-confirmation wording remains" in e for e in errors))
        limitations_start = self.manuscript.find("\\section{Limitations}")
        limitations_end = self.manuscript.find("\\section{Reproducibility and data availability}")
        limitations = self.manuscript[limitations_start:limitations_end]
        stripped = self.manuscript[:limitations_start] + limitations.replace("\\HmcChannelNullShiftMaxMm~mm", "a millimetre") + self.manuscript[limitations_end:]
        errors = []
        flattened = check_paper.flatten_sections(REPO, stripped, errors)
        errors = self._errors(manuscript=stripped, flattened=flattened)
        self.assertTrue(any("Limitations must carry the confirmation boundary" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("54cd3e82\\allowbreak{}b7c87911", "54cd3e83\\allowbreak{}b7c87911")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(hmc.SECTION_BINDING, hmc.SECTION_BINDING + "\n" + hmc.SECTION_BINDING)
        errors = self._errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(hmc.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + hmc.GENERATED_BINDING
        )
        errors = self._errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = swt.EVIDENCE_PATH.as_posix()
        errors = self._errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["experiment_id"] = hmc.V1_EXPERIMENT_ID
        errors = self._errors(payload=payload)
        self.assertTrue(any("not the registered confirmation study" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-087", "CLM-089", "CLM-090", "CLM-091", "CLM-092"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-086"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-093"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-089", raw_ids)
        self.assertIn("CLM-093", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in hmc.PROSE_CLAIM_IDS:
            self.assertIn(MANIFEST_ID, self._claim(claim_id)["manifest_ids"])
        self.assertEqual(set(self._claim("CLM-093")["manifest_ids"]), {MANIFEST_ID, SWT_MANIFEST_ID, CTV_MANIFEST_ID, WLH_MANIFEST_ID})
        self.assertIn(MANIFEST_ID, self._claim("CLM-076")["manifest_ids"])
        self.assertIn(MANIFEST_ID, self._claim("CLM-028")["manifest_ids"])
        discussion = self._claim("CLM-093")
        for phrase in ("Read as interpretation", "property of the magnet arrangement", "not robust to the material model", "Neither consequence is a design recommendation", "the iron is linear", "confines a plasma"):
            self.assertIn(phrase, check_paper._normalize_tex(discussion["authorized_tex"]))

    def test_scope_and_disclosure_claims_register_the_boundary(self) -> None:
        joined = " ".join(self._claim("CLM-087")["non_claims"])
        for phrase in ("never a probability", "not a positive finding about the thruster", "no design is recommended", "not hardware-valid", "opens no physics level", "reported and never gated"):
            self.assertIn(phrase, joined)
        scope = self._claim("CLM-092")
        self.assertEqual(scope["claim_class"], "screening-scope-limitation")
        for macro in ("\\HmcWhatNotClaimed", "\\HmcWhatConfirmed", "\\HmcNotPTwoQualifiedChain", "\\HmcLinearMaterials", "\\HmcDesignRecommendation"):
            self.assertIn(macro, scope["authorized_tex"])
        results = self._claim("CLM-089")
        self.assertIn("robust to the linear iron within one", check_paper._normalize_tex(results["authorized_tex"]))
        reported = self._claim("CLM-090")
        self.assertIn("The axis nulls are not robust within the cusp tolerance", check_paper._normalize_tex(reported["authorized_tex"]))
        disclosure = self._claim("CLM-091")
        self.assertEqual(disclosure["claim_class"], "screening-disclosure")
        for macro in ("\\HmcVOneTerminalState", "\\HmcVOneFailedDesigns", "\\HmcVOneAngleGateDeg", "\\HmcSliverAMinAngleDeg", "\\HmcSliverBMinAngleDeg", "\\HmcTimingWithinBudget", "\\HmcPaperAdmissionRecord"):
            self.assertIn(macro, disclosure["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == hmc.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [hmc.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 4)
        self.assertEqual(item["generator_module"], "generate_l1b_hemp_confirmation_v1_1_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 4)
        record = self._claim(hmc.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [hmc.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_amended_sentences_are_in_place(self) -> None:
        self.assertIn("Preregistered material-aware confirmation of the HEMP-like sweep designs", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered material-aware confirmation of the HEMP-like sweep designs}", self.manuscript)
        self.assertIn("\\label{sec:l1b-hemp-confirmation}", self.manuscript)
        self.assertNotIn("this paper\ndoes not report", self.manuscript)
        self.assertNotIn("remains open, and whether any cell", self.manuscript)
        self.assertIn("An eighth\nnumerical-screening gate", self.manuscript)
        self.assertIn("material-aware HEMP confirmation at Git revision \\HempConfirmationEvidenceRevision", self.manuscript)
        self.assertIn("was not run within this\ncampaign and is admitted as a separately preregistered campaign", self.manuscript)
        self.assertIn("property of the magnet arrangement", self.manuscript)
        section = (REPO / swt.SECTION_PATH).read_text(encoding="utf-8")
        self.assertIn("had status \\texttt{\\SwtConfirmationStatus} within\nthis campaign (run here: \\SwtMaterialAwareConfirmationRun)", section)
        self.assertIn("Section~\\ref{sec:l1b-hemp-confirmation}", section)
        # The earlier admissions' sentences stay in place.
        self.assertIn("exhausts the collisionless", self.manuscript)
        self.assertIn("published post hoc by a fail-closed recovery", self.manuscript)
        self.assertIn("zero reflections are a launch-position\nresult", self.manuscript)


if __name__ == "__main__":
    unittest.main()
