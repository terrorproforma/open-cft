"""Adversarial tests for the admission of the cusp topology search v3.1 into the manuscript."""

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

GATE_ID = ctv.GATE_ID
MANIFEST_ID = ctv.MANIFEST_ID
HEADING = ctv.SECTION_HEADING
FOUR_CELL_MANIFEST_ID = "FOUR-CELL-V2-20260902-128-V1"
CHAR_MANIFEST_ID = "TOPOLOGY-CHAR-V1-20260902-56-V1"
V4_MANIFEST_ID = "WALL-LOSS-V4-20260902-4608-V1"
CLOSURE_MANIFEST_ID = "FOUR-CELL-CLOSURE-20260903-R27-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class CuspTopologyAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json(ctv.EVIDENCE_PATH.as_posix())
        cls.section = (REPO / ctv.SECTION_PATH).read_text(encoding="utf-8")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_cusp_topology_screening(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_is_a_numerical_screening_gate_at_the_topology_outcome(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertIn("accepted-topology-screening", kinds["numerical-screening"])
        self.assertIn("never plasma confinement or probabilities", kinds["numerical-screening"])
        self.assertIn("changes no earlier frozen-definition record", kinds["numerical-screening"])
        self.assertIn("never accepted physical-orbit", kinds["numerical-screening"])
        self.assertIn("never opens GATE-L1", kinds["numerical-screening"])
        self.assertEqual(self.gate["kind"], check_paper.SCREENING_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["recorded_outcome"], ctv.RECORDED_OUTCOME)
        self.assertIn(ctv.RECORDED_OUTCOME, check_paper.SCREENING_OUTCOMES)
        self.assertEqual(len(check_paper.SCREENING_OUTCOMES), 5)
        self.assertIn("accepted_topology_screening", self.gate["recorded_outcome_justification"])
        self.assertIn("never demonstrated plasma confinement", self.gate["recorded_outcome_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-separatrix-topology-screening-manifest")
        self.assertEqual(self.gate["evidence_revision"], ctv.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], ctv.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], ctv.DASHBOARD_COMMIT_SHA)
        self.assertEqual(self.gate["lineage_results_commit"], ctv.LINEAGE_RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["lineage_preregistration_commit"], ctv.LINEAGE_PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["lineage_posthoc_audit_commit"], ctv.LINEAGE_AUDIT_COMMIT_SHA)
        self.assertEqual(self.gate["definition_sources_revision"], ctv.LITERATURE_COMMIT_SHA)
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        entry = self.matrix["manifests"][MANIFEST_ID]
        self.assertEqual(entry["recorded_outcome"], ctv.RECORDED_OUTCOME)
        self.assertEqual(entry["lineage_results_revision"], ctv.LINEAGE_RESULTS_COMMIT_SHA)
        self.assertEqual(entry["definition_sources_revision"], ctv.LITERATURE_COMMIT_SHA)
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-screening")
        self.assertEqual(self.payload["gate_kind"], "numerical-screening")
        self.assertEqual(self.payload["recorded_outcome"], ctv.RECORDED_OUTCOME)
        self.assertEqual(self.payload["classification"], ctv.CLASSIFICATION)
        self.assertEqual(self.payload["p2_row_classification"], ctv.P2_CLASSIFICATION)
        self.assertEqual(self.payload["screening_model"], ctv.SCREENING_MODEL)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        self.assertIsNone(self.payload["posthoc_audit"])
        self.assertIs(self.payload["lineage"]["cited_for_numbers"], False)
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # The other screening gates keep their manifest types; this one has its own.
        for other in ("GATE-L1A-SWEEP-V2", "GATE-FOUR-CELL-V2", "GATE-TOPOLOGY-CHAR-V1", "GATE-WALL-LOSS-GEOMETRY-SCREENING-V1"):
            gate = next(g for g in self.registry["gates"] if g["id"] == other)
            self.assertEqual(gate["kind"], self.gate["kind"])
            self.assertNotEqual(gate["required_manifest_document_type"], self.gate["required_manifest_document_type"])
            # The earlier gates keep their frozen-definition flag; this admission does not touch it.
            self.assertIs(gate["metric_constraints"]["stable_multicell_wall_cusp_topology_demonstrated"]["equals"], False)
        gate_record = next(c for c in self.matrix["claims"] if c["id"] == GATE_ID)
        self.assertEqual(gate_record["kind"], "numerical-screening")
        self.assertEqual(gate_record["recorded_outcome"], ctv.RECORDED_OUTCOME)
        self.assertEqual(gate_record["manifest_id"], MANIFEST_ID)

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))
        # The frozen-definition records of Section 8 are untouched.
        four_cell = next(g for g in self.registry["gates"] if g["id"] == "GATE-FOUR-CELL-V2")
        self.assertEqual(four_cell["metric_constraints"]["stable_count"]["equals"], 0)
        self.assertEqual(four_cell["recorded_outcome"], "preregistered-null")
        char = next(g for g in self.registry["gates"] if g["id"] == "GATE-TOPOLOGY-CHAR-V1")
        self.assertEqual(char["metric_constraints"]["stable_eligible_cusp_count"]["equals"], 0)

    def test_manifest_validates_as_a_typed_gate_manifest(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, self.payload,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertEqual(errors, [])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-separatrix-topology-screening-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-separatrix-topology-screening-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertTrue({s["role"] for s in self.payload["source_files"]} >= set(schema["required_file_roles"]))
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "design-record"), 14)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "field-grid"), 14)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"].startswith("preregistered-")), 4)
        paths = [s["path"] for s in self.payload["source_files"]]
        self.assertEqual(len(paths), len(set(paths)))
        roles = {entry["role"] for entry in self.payload["lineage_files"]}
        self.assertEqual(roles, {"lineage-rejected-campaign", "lineage-posthoc-audit", "lineage-posthoc-audit-script", "lineage-rejected-preregistration"})
        self.assertEqual(sum(1 for entry in self.payload["lineage_files"] if entry["role"] == "lineage-rejected-campaign"), 62)
        self.assertEqual({entry["role"] for entry in self.payload["reference_files"]}, {"reference-characterization-dataset", "reference-four-cell-dataset", "reference-sweep-manifest"})
        self.assertEqual([f["role"] for f in self.payload["definition_sources"]["files"]], ["definition-source-review"])
        self.assertEqual(self.payload["definition_sources"]["literature_keys"], ["gildea2012", "kornfeld2007", "koch2011", "lewerentz2023", "parnell1996", "haynes2010", "murphy2015"])
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
        for metric, macro in check_paper.CUSP_TOPOLOGY_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.CUSP_TOPOLOGY_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual(metrics["evaluated_count"], 281)
        self.assertEqual(metrics["stable_design_count"], 281)
        self.assertEqual(metrics["binding_gates_true"], 9)
        self.assertEqual(metrics["wall_cusp_count_histogram"], {"0": 6, "1": 140, "2": 36, "3": 56, "4": 25, "5": 6, "6": 6, "7": 6})
        self.assertEqual((metrics["sweep_two_cusp_designs"], metrics["sweep_three_cusp_designs"], metrics["sweep_four_cusp_designs"]), (30, 47, 19))
        self.assertEqual(metrics["sweep_four_wall_cusp_fraction"], 19 / 96)
        self.assertEqual(metrics["sweep_four_cell_fraction"], 47 / 96)
        self.assertEqual(metrics["sweep_n_minus_one_designs"], 83)
        self.assertEqual(metrics["four_cell_one_cusp_designs"], 128)
        self.assertEqual((metrics["held_out_characterization_passed"], metrics["held_out_sweep_passed"]), (56, 96))
        self.assertEqual((metrics["held_out_characterization_nulls"], metrics["held_out_sweep_nulls"]), (180, 479))
        self.assertEqual(metrics["p2_wall_cusp_count"], 3)
        self.assertEqual([round(1e3 * z, 3) for z in metrics["p2_cusp_positions_m"]], [6.028, 12.0, 17.972])
        self.assertLessEqual(metrics["p2_axis_null_to_pic_plane_maximum_m"], 3.2e-5)
        self.assertLessEqual(metrics["maximum_wall_intersection_shift_m"], 3.4e-5)
        self.assertEqual(metrics["lineage_failing_designs"], 14)
        self.assertEqual((metrics["lineage_sealed_axis_clusters"], metrics["lineage_dropped_clusters"], metrics["lineage_dropped_in_channel"]), (206, 26, 22))
        self.assertEqual(metrics["lineage_corrected_passed"], 56)
        self.assertEqual(metrics["characterization_channel_roots"], 200)
        self.assertEqual((metrics["characterization_channel_axis_roots"], metrics["characterization_channel_off_axis_roots"]), (180, 20))
        self.assertEqual(metrics["tolerated_eol_file_count"], 0)
        self.assertEqual(metrics["verified_file_count"], 1211)
        for metric, rule in self.gate["metric_constraints"].items():
            self.assertIn(metric, metrics, metric)
            self.assertEqual(metrics[metric], rule["equals"], metric)

    def test_screening_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_tampered_metric_or_outcome_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["sweep_n_minus_one_designs"] = 84
        payload["metrics"]["stable_design_count"] = 281.0  # right value, wrong type
        payload["metrics"]["confinement_cells_demonstrated"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("metric 'sweep_n_minus_one_designs' differs" in e for e in errors))
        self.assertTrue(any("metric 'stable_design_count' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'confinement_cells_demonstrated'" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome"] = "accepted-screening"
        errors = self._errors(gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        gate["recorded_outcome"] = "accepted-finding"
        errors = self._errors(gate=gate)
        self.assertTrue(any("not a recognized screening outcome" in e for e in errors))

    def test_classification_and_model_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
        errors = self._errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["p2_row_classification"]["equals"] = ctv.CLASSIFICATION
        errors = self._errors(gate=gate)
        self.assertTrue(any("P2 row classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["screening_model"] = "plasma confinement cells"
        errors = self._errors(payload=payload)
        self.assertTrue(any("screening_model differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["CtvClassification"]), ctv.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["CtvPTwoClassification"]), ctv.P2_CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["CtvRecordedOutcome"]), ctv.RECORDED_OUTCOME)
        self.assertEqual(check_paper.tex_unescape(values["CtvCampaignStatus"]), ctv.CAMPAIGN_STATUS)
        self.assertEqual(check_paper.tex_unescape(values["CtvLineageTerminalState"]), ctv.LINEAGE_TERMINAL_STATE)
        for macro in ("\\CtvClassification", "\\CtvPTwoClassification", "\\CtvRecordedOutcome", "\\CtvCampaignStatus", "\\CtvFieldModelLevel", "\\CtvLineageTerminalState"):
            self.assertIn(macro, self.section)

    def test_opening_a_level_or_touching_the_dashboard_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = ctv.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        html = next(f for f in payload["dashboard"]["files"] if f["role"] == "dashboard-html")
        html["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"] = {"revision": ctv.LINEAGE_AUDIT_COMMIT_SHA}
        errors = self._errors(payload=payload)
        self.assertTrue(any("post-hoc audit of the accepted campaign" in e for e in errors))

    def test_lineage_reference_and_definition_bindings_are_fail_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["lineage_files"] = payload["lineage_files"][1:]
        errors = self._errors(payload=payload)
        self.assertTrue(any("lineage: file group differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        audit = next(f for f in payload["lineage_files"] if f["role"] == "lineage-posthoc-audit")
        audit["revision"] = ctv.LINEAGE_RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("revision differs from the evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["cited_for_numbers"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("cited_for_numbers false" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage"]["results_commit"] = ctv.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("lineage results_commit differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        reference = next(f for f in payload["reference_files"] if f["role"] == "reference-characterization-dataset")
        reference["git_blob"] = "0" * 40
        errors = self._errors(payload=payload)
        self.assertTrue(any("Git blob mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["definition_sources"]["revision"] = ctv.LINEAGE_AUDIT_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("definition sources must bind the registered literature review revision" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["definition_sources"]["literature_keys"] = payload["definition_sources"]["literature_keys"][:-1]
        errors = self._errors(payload=payload)
        self.assertTrue(any("literature keys differ" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["lineage_bundle"]["state"] = "accepted_result"
        errors = self._errors(payload=payload)
        self.assertTrue(any("lineage bundle state is not the recorded rejection" in e for e in errors))

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
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-062")
        record["non_claims"].append("cells confine the discharge plasma")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-062 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-062")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_relocated_or_misplaced_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-064")
        record["manifest_ids"] = [V4_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-064 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-064 does not allow the section heading" in e for e in errors))
        # The Discussion interpretations may not be moved into the results section.
        for claim_id in ("CLM-028", "CLM-044"):
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertEqual(record["claim_class"], "interpretation")
            self.assertNotIn(claim_id, self.section)
        matrix = copy.deepcopy(self.matrix)
        next(c for c in matrix["claims"] if c["id"] == "CLM-068")["claim_class"] = "interpretation"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-068 must not appear inside the results section" in e for e in errors))

    def test_discussion_amendments_are_bound_and_worded(self) -> None:
        records = {c["id"]: c for c in self.matrix["claims"]}
        clm028 = records["CLM-028"]
        self.assertIn(MANIFEST_ID, clm028["manifest_ids"])
        self.assertEqual(set(clm028["manifest_ids"]), {FOUR_CELL_MANIFEST_ID, CHAR_MANIFEST_ID, MANIFEST_ID})
        for phrase in ("non-standard", "\\CtvSweepNMinusOne", "\\CtvSweepFourWallCusps", "\\CtvSweepFourCells", "not proof that no such design exists", "at the magnet ends rather than the stage midplanes", "material-aware field model", "not decided here"):
            self.assertIn(phrase, clm028["authorized_tex"])
        self.assertNotIn("undemonstrated, which is a null", clm028["authorized_tex"])
        clm044 = records["CLM-044"]
        # The sweep-v3 admission added its manifest (the zero-reflection finding is a launch-position result).
        self.assertEqual(set(clm044["manifest_ids"]), {CLOSURE_MANIFEST_ID, V4_MANIFEST_ID, FOUR_CELL_MANIFEST_ID, MANIFEST_ID, "L1A-SWEEP-V3-20260903-128-V1"})
        self.assertIn("under the literature definition", clm044["authorized_tex"])
        self.assertIn("\\CtvWithCuspAll", clm044["authorized_tex"])
        self.assertIn("plasma physics", clm044["authorized_tex"])
        self.assertIn("remains undemonstrated", clm044["authorized_tex"])
        self.assertNotIn("never shown to exist", clm044["authorized_tex"])
        self.assertIn("as interpretation only", clm044["authorized_tex"])
        # Removing the amended wording is rejected by the checker.
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-028")
        record["authorized_tex"] = record["authorized_tex"].replace("non-standard", "standard")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-028 lacks the amended wording 'non-standard'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-044")
        record["manifest_ids"] = [m for m in record["manifest_ids"] if m != MANIFEST_ID]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("Discussion claim CLM-044 must be an interpretation bound to manifest" in e for e in errors))
        # The manuscript carries the amended paragraph heading and the Section 8 forward reference.
        self.assertIn("was undemonstrated under the\nfrozen definition and is a geometric property of the screened fields under the\nliterature definition", self.manuscript)
        self.assertNotIn("cells were never shown to exist", self.flattened)
        self.assertNotIn("or\nunder a different cusp and cell definition, remains a question", self.manuscript)
        self.assertIn("differ from the literature's; the same fields\nare screened under the literature definition", self.manuscript)
        self.assertIn("those definitions differ from the literature's", self.manuscript)
        self.assertIn("not the separatrix-bounded\ncells of the catalogue admitted in Section~\\ref{sec:cusp-topology}", self.manuscript)
        self.assertIn("launches from the catalogue cells is admitted in\nSection~\\ref{sec:wall-loss-geometry-screening-v2}", self.manuscript)

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("cec47f12\\allowbreak{}f5909c58", "cec47f13\\allowbreak{}f5909c58")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(ctv.SECTION_BINDING, ctv.SECTION_BINDING + "\n" + ctv.SECTION_BINDING)
        errors = self._errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(ctv.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + ctv.GENERATED_BINDING
        )
        errors = self._errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = "paper/evidence/wall-loss-geometry-screening-v1.json"
        errors = self._errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["experiment_id"] = ctv.LINEAGE_EXPERIMENT_ID
        errors = self._errors(payload=payload)
        self.assertTrue(any("not the registered screening study" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-062", "CLM-064", "CLM-065", "CLM-066", "CLM-067", "CLM-068"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-061"].start), "Abstract")
        for claim_id in ("CLM-028", "CLM-044"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-062", raw_ids)
        self.assertIn("CLM-061", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in ctv.PROSE_CLAIM_IDS:
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertIn(MANIFEST_ID, record["manifest_ids"])
        self.assertEqual(set(next(c for c in self.matrix["claims"] if c["id"] == "CLM-065")["manifest_ids"]), {MANIFEST_ID, FOUR_CELL_MANIFEST_ID, CHAR_MANIFEST_ID})
        self.assertEqual(set(next(c for c in self.matrix["claims"] if c["id"] == "CLM-066")["manifest_ids"]), {MANIFEST_ID, V4_MANIFEST_ID})
        lineage = next(c for c in self.matrix["claims"] if c["id"] == "CLM-067")
        self.assertEqual(lineage["claim_class"], "lineage-disclosure")
        self.assertIn("never cited for a number", lineage["authorized_tex"])
        self.assertIn("recording layer", lineage["authorized_tex"])

    def test_scope_claims_register_the_boundary(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-062")
        joined = " ".join(record["non_claims"])
        for phrase in (
            "field ratios and not probabilities",
            "no plasma, confinement, wall-loss or performance quantity",
            "whether a cell confines a plasma is not decided here",
            "never cited for a number",
            "opens no physics level",
        ):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-068")
        self.assertEqual(scope["claim_class"], "screening-scope-limitation")
        self.assertIn("\\CtvUsableAs", scope["authorized_tex"])
        self.assertIn("consumer contract under its labels", scope["authorized_tex"])
        self.assertIn("not P2-qualified", scope["authorized_tex"])
        construction = next(c for c in self.matrix["claims"] if c["id"] == "CLM-065")
        self.assertIn("construction", construction["authorized_tex"])
        self.assertIn("\\CtvFourCellStrengthRatioMin", construction["authorized_tex"])
        self.assertIn("remain true", construction["authorized_tex"])
        p2 = next(c for c in self.matrix["claims"] if c["id"] == "CLM-066")
        self.assertIn("are not admitted here", p2["authorized_tex"])
        self.assertIn("nothing about a plasma", p2["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == ctv.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [ctv.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 4)
        self.assertEqual(item["generator_module"], "generate_cusp_topology_v3_1_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 4)
        record = next(c for c in self.matrix["claims"] if c["id"] == ctv.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [ctv.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)
        output, sidecar = check_paper._render_cusp_topology_tables(REPO, item)
        self.assertEqual((REPO / item["output_path"]).read_bytes(), output)
        self.assertEqual((REPO / item["sidecar_path"]).read_bytes(), sidecar)

    def test_required_section_and_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Preregistered cusp topology under the literature definition", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered cusp topology under the literature definition}", self.manuscript)
        self.assertIn("\\label{sec:cusp-topology}", self.manuscript)
        self.assertIn("cusp topology screening at Git revision \\CuspTopologyEvidenceRevision", self.manuscript)
        self.assertIn("\\cite{Gildea2012,Kornfeld2007,Koch2011,Lewerentz2023}", self.manuscript)
        # The new Limitations sentence and the L1 note exist; the stale wording is gone.
        # The catalogue's first admitted consumer is the catalogue-cell screening (Section 15).
        self.assertIn("its catalogue is a\nconsumer contract under its labels whose first admitted consumer is the\ncatalogue-cell screening of Section~\\ref{sec:wall-loss-geometry-screening-v2}", self.manuscript)
        self.assertNotIn("no admitted consumer has yet", self.manuscript)
        self.assertIn("The admitted cusp topology screening\ncharacterises prescribed field maps", self.manuscript)
        self.assertNotIn("its cells were never shown to exist", self.flattened)


if __name__ == "__main__":
    unittest.main()
