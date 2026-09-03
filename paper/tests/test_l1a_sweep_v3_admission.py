"""Adversarial tests for the admission of the L1a geometry sweep v3 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_l1a_sweep_v3_evidence as swt  # noqa: E402

GATE_ID = swt.GATE_ID
MANIFEST_ID = swt.MANIFEST_ID
HEADING = swt.SECTION_HEADING
SWEEP_V2_MANIFEST_ID = "L1A-SWEEP-V2-20260902-96-V1"
V4_MANIFEST_ID = "WALL-LOSS-V4-20260902-4608-V1"
TOPOLOGY_MANIFEST_ID = "CUSP-TOPOLOGY-V3-1-20260903-281-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class SweepV3AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json(swt.EVIDENCE_PATH.as_posix())
        cls.section = (REPO / swt.SECTION_PATH).read_text(encoding="utf-8")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_l1a_sweep_v3_screening(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_is_a_numerical_screening_gate_at_the_sweep_outcome(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertIn("accepted-screening (a field-only design-space screening)", kinds["numerical-screening"])
        self.assertEqual(self.gate["kind"], check_paper.SCREENING_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["recorded_outcome"], "accepted-screening")
        self.assertEqual(swt.RECORDED_OUTCOME, "accepted-screening")
        # The outcome vocabulary is unchanged: the sweep reuses the sweep-v2 outcome and justifies it.
        self.assertEqual(len(check_paper.SCREENING_OUTCOMES), 5)
        justification = self.gate["recorded_outcome_justification"]
        for phrase in ("accepted_l1a_sweep_v3", "same kind of object", "did not hold as preregistered", "never that a positive finding, a design recommendation or a material-aware field is accepted"):
            self.assertIn(phrase, justification)
        sweep_v2 = next(g for g in self.registry["gates"] if g["id"] == "GATE-L1A-SWEEP-V2")
        self.assertEqual(sweep_v2["recorded_outcome"], self.gate["recorded_outcome"])
        self.assertNotEqual(sweep_v2["required_manifest_document_type"], self.gate["required_manifest_document_type"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-l1a-regime-screening-manifest")
        self.assertEqual(self.gate["evidence_revision"], swt.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], swt.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], swt.DASHBOARD_COMMIT_SHA)
        self.assertEqual(self.gate["definition_sources_revision"], swt.LITERATURE_COMMIT_SHA)
        self.assertEqual(self.gate["reference_revisions"], {
            "sweep_v2_results": swt.SWEEP_V2_RESULTS_COMMIT_SHA,
            "cusp_topology_v3_1_results": swt.TOPOLOGY_RESULTS_COMMIT_SHA,
            "wall_loss_v4_preregistration": swt.WALL_LOSS_PREREGISTRATION_COMMIT_SHA,
        })
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        entry = self.matrix["manifests"][MANIFEST_ID]
        self.assertEqual(entry["recorded_outcome"], "accepted-screening")
        self.assertEqual(entry["definition_sources_revision"], swt.LITERATURE_COMMIT_SHA)
        self.assertEqual(entry["reference_revisions"], self.gate["reference_revisions"])
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-screening")
        self.assertEqual(self.payload["gate_kind"], "numerical-screening")
        self.assertEqual(self.payload["recorded_outcome"], "accepted-screening")
        self.assertEqual(self.payload["classification"], swt.CLASSIFICATION)
        self.assertEqual(self.payload["topology_label"], swt.TOPOLOGY_LABEL)
        self.assertEqual(self.payload["screening_model"], swt.SCREENING_MODEL)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        self.assertIsNone(self.payload["posthoc_audit"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # The earlier screening gates are untouched, including their frozen-definition flag.
        for other in ("GATE-L1A-SWEEP-V2", "GATE-FOUR-CELL-V2", "GATE-TOPOLOGY-CHAR-V1", "GATE-WALL-LOSS-GEOMETRY-SCREENING-V1"):
            gate = next(g for g in self.registry["gates"] if g["id"] == other)
            self.assertEqual(gate["kind"], self.gate["kind"])
            self.assertIs(gate["metric_constraints"]["stable_multicell_wall_cusp_topology_demonstrated"]["equals"], False)
        gate_record = next(c for c in self.matrix["claims"] if c["id"] == GATE_ID)
        self.assertEqual(gate_record["kind"], "numerical-screening")
        self.assertEqual(gate_record["recorded_outcome"], "accepted-screening")
        self.assertEqual(gate_record["manifest_id"], MANIFEST_ID)

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))
        # The sweep-v2 admission is untouched.
        sweep_v2 = next(g for g in self.registry["gates"] if g["id"] == "GATE-L1A-SWEEP-V2")
        self.assertEqual(sweep_v2["metric_constraints"]["evaluated_count"]["equals"], 96)
        self.assertEqual(sweep_v2["evidence_revision"], swt.SWEEP_V2_RESULTS_COMMIT_SHA)

    def test_manifest_validates_as_a_typed_gate_manifest(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, self.payload,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertEqual(errors, [])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-l1a-regime-screening-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-l1a-regime-screening-manifest"], schema)
        self.assertEqual(self.schemas["manifest_types"], check_paper.EXPECTED_MANIFEST_TYPES)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertTrue({s["role"] for s in self.payload["source_files"]} >= set(schema["required_file_roles"]))
        dataset = _json(f"{swt.RESULTS.as_posix()}/artifacts/sweep-dataset.json")
        bound = sum(1 for d in dataset["designs"] if d["hemp_like_all_cusps"] or d["representative"])
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "design-record"), bound)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "field-grid"), bound)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"].startswith("preregistered-")), 4)
        paths = [s["path"] for s in self.payload["source_files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual({entry["role"] for entry in self.payload["reference_files"]}, {"reference-sweep-manifest", "reference-topology-protocol", "reference-topology-p2-record", "reference-wall-loss-protocol"})
        self.assertEqual({f["role"] for f in self.payload["definition_sources"]["files"]}, {"definition-source-review", "definition-source-check-script", "definition-source-check-output"})
        self.assertEqual(self.payload["definition_sources"]["literature_keys"], ["koch2007", "koch2011"])
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
        for metric, macro in check_paper.SWEEP_V3_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.SWEEP_V3_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual((metrics["evaluated_count"], metrics["sobol_design_count"], metrics["held_out_design_count"]), (224, 128, 96))
        self.assertEqual((metrics["binding_gate_count"], metrics["binding_gates_true"]), (11, 11))
        self.assertEqual(metrics["stable_design_count"], 224)
        self.assertEqual(metrics["hemp_like_count"], 15)
        self.assertEqual(metrics["predicted_hemp_like_count"], 51)
        self.assertEqual(metrics["five_stage_four_cusp_hemp_like_count"], 2)
        self.assertEqual((metrics["sweep_v2_region_design_count"], metrics["sweep_v2_region_hemp_like_count"]), (102, 0))
        self.assertLess(metrics["sweep_v2_region_rho_maximum"], 1.0)
        self.assertEqual((metrics["band_below_threshold_designs"], metrics["band_below_threshold_hemp_like"]), (77, 0))
        self.assertEqual((metrics["end_cusp_count"], metrics["interior_cusp_count"]), (256, 109))
        self.assertEqual((metrics["held_out_passed"], metrics["held_out_nulls"]), (96, 479))
        self.assertLessEqual(metrics["held_out_max_difference_m"], 2.8e-5)
        self.assertLessEqual(metrics["maximum_wall_intersection_shift_m"], 3.4e-5)
        self.assertEqual((metrics["pooled_cusp_count"], metrics["pooled_cusp_is_wall_maximum_count"]), (642, 0))
        self.assertLess(metrics["hypothesis_slope_through_origin"], 0.80)
        self.assertLess(metrics["hypothesis_fraction_within_band"], 0.80)
        self.assertLess(metrics["hypothesis_prediction_accuracy"], 0.85)
        self.assertIs(metrics["hypothesis_h1_held"], False)
        self.assertIs(metrics["hypothesis_h2_held"], False)
        self.assertEqual((metrics["review_launch_cells"], metrics["review_near_centre_cells"], metrics["review_far_cells"]), (16, 7, 9))
        self.assertEqual((metrics["review_near_centre_reflections_maximum"], metrics["review_far_reflections_minimum"], metrics["review_far_reflections_maximum"]), (1, 32, 88))
        self.assertAlmostEqual(metrics["wall_loss_launch_offset_m"], 0.5e-3, places=12)
        self.assertIs(metrics["wall_loss_launch_in_near_class"], True)
        self.assertEqual(metrics["confirmation_status"], "queued_not_run")
        self.assertEqual(metrics["tolerated_eol_file_count"], 0)
        self.assertEqual(metrics["verified_file_count"], 979)
        for metric, rule in self.gate["metric_constraints"].items():
            self.assertIn(metric, metrics, metric)
            self.assertEqual(metrics[metric], rule["equals"], metric)

    def test_screening_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_tampered_metric_or_outcome_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["hemp_like_count"] = 16
        payload["metrics"]["stable_design_count"] = 224.0  # right value, wrong type
        payload["metrics"]["hypothesis_h1_held"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("metric 'hemp_like_count' differs" in e for e in errors))
        self.assertTrue(any("metric 'stable_design_count' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'hypothesis_h1_held'" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome"] = "accepted-topology-screening"
        errors = self._errors(gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        gate["recorded_outcome"] = "accepted-design-space-screening"
        errors = self._errors(gate=gate)
        self.assertTrue(any("not a recognized screening outcome" in e for e in errors))

    def test_classification_and_model_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
        errors = self._errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["screening_model"] = "material-aware field with iron"
        errors = self._errors(payload=payload)
        self.assertTrue(any("screening_model differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["topology_label"] = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
        errors = self._errors(payload=payload)
        self.assertTrue(any("topology label differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["SwtClassification"]), swt.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["SwtTopologyLabel"]), swt.TOPOLOGY_LABEL)
        self.assertEqual(check_paper.tex_unescape(values["SwtRecordedOutcome"]), swt.RECORDED_OUTCOME)
        self.assertEqual(check_paper.tex_unescape(values["SwtCampaignStatus"]), swt.CAMPAIGN_STATUS)
        self.assertEqual(check_paper.tex_unescape(values["SwtConfirmationStatus"]), "queued_not_run")
        for macro in ("\\SwtClassification", "\\SwtTopologyLabel", "\\SwtRecordedOutcome", "\\SwtCampaignStatus", "\\SwtFieldModelLevel", "\\SwtConfirmationStatus", "\\SwtHOneAsPredicted", "\\SwtHTwoAsPredicted"):
            self.assertIn(macro, self.section)

    def test_opening_a_level_or_touching_the_dashboard_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = swt.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        html = next(f for f in payload["dashboard"]["files"] if f["role"] == "dashboard-html")
        html["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"] = {"revision": swt.DASHBOARD_COMMIT_SHA}
        errors = self._errors(payload=payload)
        self.assertTrue(any("post-hoc audit of a bundle that needs none" in e for e in errors))

    def test_reference_and_definition_bindings_are_fail_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["reference_files"] = payload["reference_files"][1:]
        errors = self._errors(payload=payload)
        self.assertTrue(any("reference: file group differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        reference = next(f for f in payload["reference_files"] if f["role"] == "reference-wall-loss-protocol")
        reference["git_blob"] = "0" * 40
        errors = self._errors(payload=payload)
        self.assertTrue(any("Git blob mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        reference = next(f for f in payload["reference_files"] if f["role"] == "reference-sweep-manifest")
        reference["revision"] = swt.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("revision differs from the evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["definition_sources"]["revision"] = swt.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("definition sources must bind the registered literature review revision" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["definition_sources"]["files"] = [f for f in payload["definition_sources"]["files"] if f["role"] != "definition-source-check-output"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("must bind the review, its check script and its committed output" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["definition_sources"]["literature_keys"] = ["koch2007"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("literature keys differ" in e for e in errors))

    def test_frozen_files_must_all_be_bound(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["source_files"] = [s for s in payload["source_files"] if s["role"] != "preregistered-design-authorities"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("frozen preregistration files are not all bound" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        frozen = next(s for s in payload["source_files"] if s["role"] == "preregistered-protocol")
        frozen["git_blob"] = "0" * 40
        errors = self._errors(payload=payload)
        self.assertTrue(any("changed after preregistration" in e for e in errors))

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-070")
        record["non_claims"].append("the HEMP-like designs are recommended")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-070 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-070")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_relocated_or_misplaced_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-072")
        record["manifest_ids"] = [V4_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-072 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-072 does not allow the section heading" in e for e in errors))
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-076")
        self.assertEqual(record["claim_class"], "interpretation")
        self.assertNotIn("CLM-076", self.section)
        matrix = copy.deepcopy(self.matrix)
        next(c for c in matrix["claims"] if c["id"] == "CLM-075")["claim_class"] = "interpretation"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-075 must not appear inside the results section" in e for e in errors))

    def test_hypothesis_and_discussion_wording_is_enforced(self) -> None:
        records = {c["id"]: c for c in self.matrix["claims"]}
        hypothesis = records["CLM-073"]
        for phrase in ("did not hold as preregistered", "upper envelope", "\\SwtEndCuspRhoOverIOneMedian", "\\SwtXStarFromSlope", "the direction the frozen protocol anticipated"):
            self.assertIn(phrase, hypothesis["authorized_tex"])
        self.assertNotIn("confirmed", hypothesis["authorized_tex"])
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-073")
        record["authorized_tex"] = record["authorized_tex"].replace("did not hold as preregistered", "held as preregistered")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-073 lacks the required wording 'did not hold as preregistered'" in e for e in errors))
        earlier_box = records["CLM-074"]
        self.assertTrue(earlier_box["authorized_tex"].startswith("In the field model and under the definition used here"))
        self.assertIn("could not contain a HEMP-like cusp", earlier_box["authorized_tex"])
        self.assertEqual(set(earlier_box["manifest_ids"]), {MANIFEST_ID, SWEEP_V2_MANIFEST_ID})
        discussion = records["CLM-076"]
        self.assertEqual(set(discussion["manifest_ids"]), {MANIFEST_ID, SWEEP_V2_MANIFEST_ID})
        self.assertIn("Read as interpretation", discussion["authorized_tex"])
        self.assertIn("never their ratio", discussion["authorized_tex"])
        self.assertIn("could not contain a HEMP-like", discussion["authorized_tex"])
        self.assertIn("no design of the widened box is recommended here", discussion["authorized_tex"])
        self.assertEqual(set(discussion["bibliography"]), {"Muffatti2017", "Koch2007"})
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-076")
        record["manifest_ids"] = [m for m in record["manifest_ids"] if m != MANIFEST_ID]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("Discussion claim CLM-076 must be an interpretation bound to manifest" in e for e in errors))
        # The manuscript carries the new Discussion paragraph and the Section 14 intro.
        self.assertIn("\\paragraph{The legacy design space could not contain a HEMP-like cusp, because\nthe parameterisation never varied the ratio the HEMP criterion depends on.}", self.manuscript)
        self.assertIn("\\cite{Koch2007}", self.manuscript)
        self.assertIn("is not the outcome it predicted", self.manuscript)

    def test_reflection_statements_are_rescoped_as_a_launch_position_result(self) -> None:
        records = {c["id"]: c for c in self.matrix["claims"]}
        for claim_id in ("CLM-017", "CLM-044", "CLM-052"):
            record = records[claim_id]
            self.assertEqual(record["claim_class"], "interpretation")
            self.assertIn(MANIFEST_ID, record["manifest_ids"])
            self.assertIn("launch-position result", record["authorized_tex"])
            self.assertIn("\\SwtVFourLaunchOffsetMm", record["authorized_tex"])
            self.assertIn(claim_id, swt.PROSE_CLAIM_IDS)
        self.assertEqual(set(records["CLM-017"]["manifest_ids"]), {V4_MANIFEST_ID, MANIFEST_ID})
        self.assertIn("could not test the magnetic-mirror picture", records["CLM-017"]["authorized_tex"])
        self.assertIn("\\SwtPpmLineMaxOverLaunchMax", records["CLM-017"]["authorized_tex"])
        self.assertNotIn("is not supported for this field and design", records["CLM-017"]["authorized_tex"])
        self.assertIn("\\SwtPpmNearReflectionsMax", records["CLM-052"]["authorized_tex"])
        self.assertIn("\\SwtPpmFarReflectionsMin", records["CLM-052"]["authorized_tex"])
        self.assertIn("mirror reflections toward the magnet centres", records["CLM-052"]["authorized_tex"])
        self.assertNotIn("scoped to the field it was made in", records["CLM-052"]["authorized_tex"])
        self.assertNotIn("unsupported for the qualified field", records["CLM-044"]["authorized_tex"])
        self.assertIn("non-adiabatic at every wall cusp", records["CLM-044"]["authorized_tex"])
        # The wall-loss section's own scope claim names the launch planes and defers the reading to the Discussion.
        self.assertIn("at the launch planes of this campaign", records["CLM-016"]["authorized_tex"])
        self.assertIn("\\WlfCellOneZMm", records["CLM-016"]["authorized_tex"])
        self.assertEqual(records["CLM-016"]["manifest_ids"], [V4_MANIFEST_ID])
        # The Limitations carry the non-adiabaticity numbers; the superseded wording is gone.
        for phrase in ("\\SwtPpmMendelAlphaMin", "\\SwtPpmMendelAlphaMax", "\\SwtPpmEpsilonMax", "\\SwtPpmMuOrderedByEpsilon", "cannot be\na loss-cone number"):
            self.assertIn(phrase, self.manuscript)
        for stale in ("that statement is field-specific", "is not supported for this field and design", "unsupported for the qualified field", "a mirror formula the integrated orbits do not support", "does not support, enter as uncertain inputs"):
            self.assertNotIn(stale, self.flattened)
        self.assertIn("could not test at its launch planes", self.manuscript)
        # Removing the wording or the binding is rejected by the checker.
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-052")
        record["authorized_tex"] = record["authorized_tex"].replace("launch-position result", "field-specific result")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-052 lacks the launch-position wording 'launch-position result'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-017")
        record["manifest_ids"] = [V4_MANIFEST_ID]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("Discussion claim CLM-017 must be an interpretation bound to manifest" in e for e in errors))
        stale_manuscript = self.manuscript.replace("cannot be\na loss-cone number", "is\na loss-cone number")
        errors = self._errors(manuscript=stale_manuscript)
        self.assertTrue(any("non-adiabaticity statement" in e for e in errors))
        stale_flattened = self.flattened + "\nthat statement is field-specific\n"
        errors = self._errors(flattened=stale_flattened)
        self.assertTrue(any("superseded mirror-picture wording remains" in e for e in errors))
        # The launch-offset macros recompute from the frozen wall-loss protocol and the topology P2 record.
        raw = {m["name"]: m["raw"] for m in self.evidence["macros"]}
        self.assertAlmostEqual(raw["SwtVFourLaunchOffsetMm"], 0.5e-3, places=12)
        self.assertEqual(raw["SwtVFourLaunchZMm"], [0.0035, 0.0095, 0.0155, 0.0215])
        self.assertEqual(raw["SwtPTwoStageCentresMm"], [0.003, 0.009, 0.015, 0.021])
        self.assertEqual((raw["SwtPpmNearReflectionsMax"], raw["SwtPpmFarReflectionsMin"], raw["SwtPpmFarReflectionsMax"]), (1, 32, 88))
        self.assertLessEqual(raw["SwtPpmLineMaxOverLaunchMax"], 1.0 + 1e-9)

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("2cfe8223\\allowbreak{}630fbef6", "2cfe8224\\allowbreak{}630fbef6")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(swt.SECTION_BINDING, swt.SECTION_BINDING + "\n" + swt.SECTION_BINDING)
        errors = self._errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(swt.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + swt.GENERATED_BINDING
        )
        errors = self._errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = "paper/evidence/l1a-sweep-v2.json"
        errors = self._errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["experiment_id"] = "l1a-geometry-sweep-v2"
        errors = self._errors(payload=payload)
        self.assertTrue(any("not the registered screening study" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-070", "CLM-072", "CLM-073", "CLM-074", "CLM-075"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-069"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-076"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-070", raw_ids)
        self.assertIn("CLM-069", raw_ids)
        self.assertIn("CLM-076", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in swt.PROSE_CLAIM_IDS:
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertIn(MANIFEST_ID, record["manifest_ids"])
        # Claims that compare against the sweep-v2 box are bound to that manifest as well.
        for claim_id in ("CLM-070", "CLM-072", "CLM-074", "CLM-076"):
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertEqual(set(record["manifest_ids"]), {MANIFEST_ID, SWEEP_V2_MANIFEST_ID})

    def test_scope_claims_register_the_boundary(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-070")
        joined = " ".join(record["non_claims"])
        for phrase in (
            "none is a probability",
            "no plasma, wall-loss, confinement or performance quantity",
            "reported and not gated",
            "no HEMP-like design is a design recommendation",
            "opens no physics level",
        ):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-075")
        self.assertEqual(scope["claim_class"], "screening-scope-limitation")
        self.assertIn("\\SwtUsableAs", scope["authorized_tex"])
        self.assertIn("not P2-qualified", scope["authorized_tex"])
        self.assertIn("\\SwtMaterialAwareConfirmationRun", scope["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == swt.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [swt.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 4)
        self.assertEqual(item["generator_module"], "generate_l1a_sweep_v3_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 4)
        record = next(c for c in self.matrix["claims"] if c["id"] == swt.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [swt.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)
        output, sidecar = check_paper._render_sweep_v3_tables(REPO, item)
        self.assertEqual((REPO / item["output_path"]).read_bytes(), output)
        self.assertEqual((REPO / item["sidecar_path"]).read_bytes(), sidecar)

    def test_required_section_and_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Preregistered geometry sweep into the HEMP-like regime", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered geometry sweep into the HEMP-like regime}", self.manuscript)
        self.assertIn("\\label{sec:l1a-sweep-v3}", self.manuscript)
        self.assertIn("L1a geometry sweep v3 at Git revision \\SweepThreeEvidenceRevision", self.manuscript)
        self.assertIn("The admitted geometry sweep into the HEMP-like regime is a field-only\nscreening on the same linear-vacuum fields", self.manuscript)
        self.assertIn("none of its HEMP-like designs is a design recommendation", self.manuscript)
        self.assertIn("no material-aware field and no reduced performance model, and it leaves this gate\nclosed", self.manuscript)
        # The topology admission's sentences stay in place.
        self.assertIn("launches from the catalogue cells is future work and no result of it\nexists", self.manuscript)
        self.assertIn("not the separatrix-bounded\ncells of the catalogue admitted in Section~\\ref{sec:cusp-topology}", self.manuscript)


if __name__ == "__main__":
    unittest.main()
