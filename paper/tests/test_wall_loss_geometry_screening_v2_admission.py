"""Adversarial tests for the admission of the orbit wall-loss geometry screening v2 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_wall_loss_geometry_screening_v1_evidence as geo_v1  # noqa: E402
import generate_wall_loss_geometry_screening_v2_evidence as geo  # noqa: E402

GATE_ID = geo.GATE_ID
MANIFEST_ID = geo.MANIFEST_ID
HEADING = geo.SECTION_HEADING
V1_MANIFEST_ID = "WALL-LOSS-GEOMETRY-SCREENING-V1-20260903-96-V1"
V4_MANIFEST_ID = "WALL-LOSS-V4-20260902-4608-V1"
CTV_MANIFEST_ID = "CUSP-TOPOLOGY-V3-1-20260903-281-V1"
MDB_MANIFEST_ID = "MDO-L0-V2-20260903-1440-V1"
SWT_MANIFEST_ID = "L1A-SWEEP-V3-20260903-128-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class GeometryScreeningV2AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json(geo.EVIDENCE_PATH.as_posix())
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_geometry_screening_v2(
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

    def test_gate_reuses_the_dataset_outcome_with_justification(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertEqual(self.gate["kind"], check_paper.SCREENING_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(geo.RECORDED_OUTCOME, geo_v1.RECORDED_OUTCOME)
        # No sixth outcome value: the v1 outcome is reused and the reuse is justified on the gate.
        self.assertEqual(len(check_paper.SCREENING_OUTCOMES), 5)
        self.assertIn("reused", self.gate["recorded_outcome_justification"])
        self.assertIn("accepted_screening_dataset", self.gate["recorded_outcome_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-orbit-cell-screening-manifest")
        self.assertEqual(self.gate["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], geo.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["disclosure_revision"], geo.DISCLOSURE_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], geo.DASHBOARD_COMMIT_SHA)
        self.assertNotEqual(self.gate["dashboard_revision"], self.gate["evidence_revision"])
        self.assertEqual(set(self.gate["dependencies"]), {"GATE-WALL-LOSS-GEOMETRY-SCREENING-V1", "GATE-CUSP-TOPOLOGY-V3-1", "GATE-WALL-LOSS-V4", "GATE-L1A-SWEEP-V2"})
        for dependency in self.gate["dependencies"]:
            self.assertEqual(next(g for g in self.registry["gates"] if g["id"] == dependency)["status"], "accepted")
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        entry = self.matrix["manifests"][MANIFEST_ID]
        self.assertEqual(entry["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(entry["disclosure_revision"], geo.DISCLOSURE_COMMIT_SHA)
        self.assertEqual(entry["posthoc_finalization_disclosure"], geo.DISCLOSURE_PATH.as_posix())
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-screening")
        self.assertEqual(self.payload["gate_kind"], "numerical-screening")
        self.assertEqual(self.payload["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(self.payload["classification"], geo.CLASSIFICATION)
        self.assertEqual(self.payload["p2_row_label"], geo.P2_LABEL)
        self.assertEqual(self.payload["screening_model"], geo.SCREENING_MODEL)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        self.assertIsNone(self.payload["posthoc_audit"])
        self.assertIn("disclosure source, not as an audit", self.payload["posthoc_audit_note"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        gate_record = self._claim(GATE_ID)
        self.assertEqual(gate_record["kind"], "numerical-screening")
        self.assertEqual(gate_record["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(gate_record["manifest_id"], MANIFEST_ID)

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))

    def test_manifest_validates_as_a_typed_gate_manifest(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, self.payload,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertEqual(errors, [])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-orbit-cell-screening-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-orbit-cell-screening-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertTrue({s["role"] for s in self.payload["source_files"]} >= set(schema["required_file_roles"]))
        for role in ("allocation-decisions", "catalogue-binding", "v1-comparison"):
            self.assertIn(role, schema["required_file_roles"])
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "transition"), 9)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"].startswith("preregistered-")), 4)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "orbit-artifact"), sum(1 for s in self.payload["source_files"] if s["role"] == "case-summary"))
        paths = [s["path"] for s in self.payload["source_files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual({f["role"] for f in self.payload["reference_files"]}, {"reference-cusp-cell-catalogue", "reference-cusp-topology-manifest", "reference-screening-v1-dataset", "reference-screening-v1-manifest", "reference-wall-loss-export", "reference-sweep-manifest"})
        self.assertEqual({f["role"] for f in self.payload["disclosure_sources"]["files"]}, {"disclosure-posthoc-finalization", "disclosure-runtime-recovery-module", "disclosure-runtime-lifecycle-module", "disclosure-runtime-recovery-tests"})
        self.assertIs(self.payload["results_bundle"]["manifest_published_posthoc"], True)
        self.assertIs(self.payload["results_bundle"]["published_inside_locked_attempt"], False)
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
        for metric, macro in check_paper.GEOMETRY_SCREENING_V2_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.GEOMETRY_SCREENING_V2_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual((metrics["evaluated_count"], metrics["sweep_design_count"], metrics["p2_row_count"]), (97, 96, 1))
        self.assertEqual((metrics["cell_count"], metrics["anode_side_cell_count"], metrics["interior_cell_count"], metrics["exit_side_cell_count"], metrics["p2_cell_count"]), (377, 96, 181, 96, 4))
        self.assertEqual((metrics["case_count"], metrics["orbit_count"]), (1105, 104832))
        self.assertEqual((metrics["stage1_launches"], metrics["stage2_launches"], metrics["control_launches"]), (48256, 44928, 11648))
        self.assertEqual((metrics["validator_calls_passed"], metrics["validator_failures"], metrics["timeouts"], metrics["numerical_failures"]), (16549, 0, 0, 0))
        self.assertEqual((metrics["cells_topped_up"], metrics["cells_saturated_after_stage1"]), (117, 260))
        self.assertEqual((metrics["anode_side_cells_topped_up"], metrics["exit_side_cells_topped_up"], metrics["interior_cells_topped_up"]), (32, 83, 0))
        self.assertEqual((metrics["interior_cells_at_one"], metrics["interior_wall_access_minimum"], metrics["interior_designs_all_saturated"]), (181, 1.0, 96))
        self.assertEqual((metrics["anode_side_cells_at_one"], metrics["exit_side_cells_at_one"]), (34, 11))
        self.assertEqual((metrics["cells_surrogate_ready"], metrics["jeffreys_floor_median"], metrics["jeffreys_floor_maximum"]), (294, 0.005492143434474149, 0.02416902492487775))
        self.assertEqual((metrics["total_reflections"], metrics["designs_with_reflections"], metrics["exit_side_cells_with_reflections"], metrics["anode_side_cells_with_reflections"], metrics["interior_cells_with_reflections"]), (10407, 66, 65, 1, 0))
        self.assertEqual((metrics["control_n"], metrics["control_discordant"]), (11648, 2))
        self.assertEqual((metrics["v1_comparison_design_count"], metrics["v1_interval_overlap_launch_weighted"]), (96, 0.4479166666666667))
        self.assertEqual((metrics["p2_anode_side_wall_access"], metrics["p2_exit_side_wall_access"], metrics["p2_exit_side_reflections"]), (0.60546875, 0.169921875, 350))
        self.assertEqual((metrics["disclosed_file_count"], metrics["descriptor_cap"], metrics["pin_cap"]), (16957, 8192, 4096))
        self.assertEqual((metrics["known_defect_zero_count_inexact"], metrics["known_defect_full_count_inexact"]), (734, 1238))
        self.assertEqual((metrics["injector_zone_flagged_cells"], metrics["short_cells"]), (1, 14))
        self.assertEqual((metrics["cross_resolution_design_count"], metrics["tolerated_eol_file_count"], metrics["verified_file_count"]), (97, 0, 16957))
        for metric, rule in self.gate["metric_constraints"].items():
            self.assertIn(metric, metrics, metric)
            self.assertEqual(metrics[metric], rule["equals"], metric)
        self.assertNotIn("stable_multicell_wall_cusp_topology_demonstrated", metrics)

    def test_screening_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_tampered_metric_outcome_or_finding_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["interior_cells_at_one"] = 180
        payload["metrics"]["cells_topped_up"] = 117.0  # right value, wrong type
        payload["metrics"]["access_fraction_is_loss_probability"] = True
        payload["metrics"]["manifest_published_posthoc"] = False
        errors = self._errors(payload=payload)
        self.assertTrue(any("metric 'interior_cells_at_one' differs" in e for e in errors))
        self.assertTrue(any("metric 'cells_topped_up' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'access_fraction_is_loss_probability'" in e for e in errors))
        self.assertTrue(any("policy metric 'manifest_published_posthoc'" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["results_bundle"]["published_inside_locked_attempt"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("must disclose the post-hoc manifest publication" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome"] = "accepted-screening"
        errors = self._errors(gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        gate["recorded_outcome"] = "accepted-finding"
        errors = self._errors(gate=gate)
        self.assertTrue(any("not a recognized screening outcome" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome_justification"] = "a dataset"
        errors = self._errors(gate=gate)
        self.assertTrue(any("justify reusing" in e for e in errors))

    def test_classification_labels_and_model_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "ACCEPTED_PHYSICAL_ORBIT_EVIDENCE"
        errors = self._errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["screening_model"] = "P2 finite-element field"
        errors = self._errors(payload=payload)
        self.assertTrue(any("screening_model differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["p2_row_label"] = "ACCEPTED_V4_REPLICATION"
        errors = self._errors(payload=payload)
        self.assertTrue(any("P2 row label differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["WlhClassification"]), geo.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["WlhPTwoLabel"]), geo.P2_LABEL)
        self.assertEqual(check_paper.tex_unescape(values["WlhRecordedOutcome"]), geo.RECORDED_OUTCOME)
        self.assertEqual(check_paper.tex_unescape(values["WlhCampaignStatus"]), geo.CAMPAIGN_STATUS)
        section = (REPO / geo.SECTION_PATH).read_text(encoding="utf-8")
        for macro in ("\\WlhClassification", "\\WlhPTwoLabel", "\\WlhRecordedOutcome", "\\WlhCampaignStatus", "\\WlhFieldStatus", "\\WlhInteriorAllSaturated", "\\WlhManifestPublishedPosthoc", "\\WlhDescriptorCap", "\\WlhPinCap"):
            self.assertIn(macro, section)

    def test_opening_a_level_or_touching_the_dashboard_or_disclosure_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = geo.RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        html = next(f for f in payload["dashboard"]["files"] if f["role"] == "dashboard-html")
        html["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"] = {"revision": geo.DISCLOSURE_COMMIT_SHA}
        errors = self._errors(payload=payload)
        self.assertTrue(any("binds a post-hoc audit" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["disclosure_sources"]["revision"] = geo.DASHBOARD_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("registered disclosure revision" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        note = next(f for f in payload["disclosure_sources"]["files"] if f["role"] == "disclosure-posthoc-finalization")
        note["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["disclosure_sources"]["files"] = [f for f in payload["disclosure_sources"]["files"] if f["role"] != "disclosure-runtime-recovery-tests"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("must bind the finalization note" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["disclosure_sources"]["verified"]["nothing_rerun_stated"] = False
        errors = self._errors(payload=payload)
        self.assertTrue(any("disclosure verification block differs" in e for e in errors))

    def test_reference_and_frozen_files_must_all_be_bound(self) -> None:
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
        payload["reference_files"] = [f for f in payload["reference_files"] if f["role"] != "reference-cusp-cell-catalogue"]
        errors = self._errors(payload=payload)
        self.assertTrue(any("file group differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        catalogue = next(f for f in payload["reference_files"] if f["role"] == "reference-cusp-cell-catalogue")
        catalogue["revision"] = geo.V1_RESULTS_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("revision differs from the evidence file" in e for e in errors))

    def test_missing_non_claim_or_required_wording_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-078", matrix)
        record["non_claims"].append("validated against thruster measurements")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-078 is absent" in e for e in errors))
        for phrase in self._claim("CLM-078")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-080", matrix)
        record["authorized_tex"] = record["authorized_tex"].replace("never a\nloss probability", "a loss probability")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-080 lacks the required wording 'never a loss probability'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-083", matrix)
        record["authorized_tex"] = record["authorized_tex"].replace("published post hoc", "published")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-083 lacks the required wording 'published post hoc'" in e for e in errors))
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-085", matrix)
        record["claim_class"] = "quantitative-screening-result"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-085 must be an interpretation" in e for e in errors))

    def test_unbound_relocated_or_misplaced_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = self._claim("CLM-081", matrix)
        record["manifest_ids"] = [V1_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-081 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-081 does not allow the section heading" in e for e in errors))
        section = (REPO / geo.SECTION_PATH).read_text(encoding="utf-8")
        interpretation = self._claim("CLM-085")
        self.assertEqual(interpretation["claim_class"], "interpretation")
        self.assertNotIn("CLM-085", section)
        matrix = copy.deepcopy(self.matrix)
        self._claim("CLM-084", matrix)["claim_class"] = "interpretation"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-084 must not appear inside the results section" in e for e in errors))

    def test_superseded_planning_wording_is_rejected(self) -> None:
        stale = self.manuscript.replace(
            "the screening\nthat launches from the catalogue cells is admitted in\nSection~\\ref{sec:wall-loss-geometry-screening-v2}.",
            "a screening\nthat launches from the catalogue cells is future work and no result of it\nexists.",
        )
        self.assertNotEqual(stale, self.manuscript)
        errors: list[str] = []
        flattened = check_paper.flatten_sections(REPO, stale, errors)
        errors = self._errors(manuscript=stale, flattened=flattened)
        self.assertTrue(any("superseded planning wording remains" in e for e in errors))
        no_ref = self.manuscript.replace("Section~\\ref{sec:wall-loss-geometry-screening-v2}", "a later section")
        errors = []
        flattened = check_paper.flatten_sections(REPO, no_ref, errors)
        errors = self._errors(manuscript=no_ref, flattened=flattened)
        self.assertTrue(any("must cite the catalogue-cell screening section" in e for e in errors))
        limitations_start = self.manuscript.find("\\section{Limitations}")
        limitations_end = self.manuscript.find("\\section{Reproducibility and data availability}")
        limitations = self.manuscript[limitations_start:limitations_end]
        stripped = self.manuscript[:limitations_start] + limitations.replace("\\WlhDescriptorCap-descriptor", "descriptor") + self.manuscript[limitations_end:]
        errors = []
        flattened = check_paper.flatten_sections(REPO, stripped, errors)
        errors = self._errors(manuscript=stripped, flattened=flattened)
        self.assertTrue(any("Limitations must carry the post-hoc publication" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("26029b72\\allowbreak{}222e2b40", "26029b73\\allowbreak{}222e2b40")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(geo.SECTION_BINDING, geo.SECTION_BINDING + "\n" + geo.SECTION_BINDING)
        errors = self._errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(geo.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + geo.GENERATED_BINDING
        )
        errors = self._errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = geo_v1.EVIDENCE_PATH.as_posix()
        errors = self._errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["experiment_id"] = geo_v1.EXPERIMENT_ID
        errors = self._errors(payload=payload)
        self.assertTrue(any("not the registered screening study" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-078", "CLM-080", "CLM-081", "CLM-082", "CLM-083", "CLM-084"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-077"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-085"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-078", raw_ids)
        self.assertIn("CLM-085", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in geo.PROSE_CLAIM_IDS:
            self.assertIn(MANIFEST_ID, self._claim(claim_id)["manifest_ids"])
        self.assertEqual(set(self._claim("CLM-080")["manifest_ids"]), {MANIFEST_ID, CTV_MANIFEST_ID, V4_MANIFEST_ID})
        self.assertEqual(set(self._claim("CLM-081")["manifest_ids"]), {MANIFEST_ID, V1_MANIFEST_ID})
        self.assertEqual(set(self._claim("CLM-085")["manifest_ids"]), {MANIFEST_ID, MDB_MANIFEST_ID, SWT_MANIFEST_ID})
        discussion = self._claim("CLM-085")
        for phrase in ("Read as interpretation", "exhausts the collisionless", "deferred", "particle-in-cell", "future work", "not evidence", "\\WlhInteriorAtOne", "\\WlhExitWallSideLastPolarity"):
            self.assertIn(phrase, check_paper._normalize_tex(discussion["authorized_tex"]))
        self.assertIn("sec:mdo-l0-v2", discussion["authorized_tex"])

    def test_scope_claims_register_the_boundary(self) -> None:
        joined = " ".join(self._claim("CLM-078")["non_claims"])
        for phrase in ("never a loss probability", "not a replication", "not demonstrated confinement cells", "declared averages", "no design rule", "opens no physics level", "reported, never gated"):
            self.assertIn(phrase, joined)
        scope = self._claim("CLM-084")
        self.assertEqual(scope["claim_class"], "screening-scope-limitation")
        self.assertIn("\\WlhUsableAs", scope["authorized_tex"])
        self.assertIn("not P2-qualified", scope["authorized_tex"])
        self.assertIn("\\WlhPTwoNotReplication", scope["authorized_tex"])
        headline = self._claim("CLM-080")
        self.assertIn("lost every launch to the dielectric", check_paper._normalize_tex(headline["authorized_tex"]))
        self.assertIn("Structure exists only in the partial cells", headline["authorized_tex"])
        disclosure = self._claim("CLM-083")
        self.assertEqual(disclosure["claim_class"], "screening-disclosure")
        for macro in ("\\WlhDisclosedFileCount", "\\WlhDescriptorCap", "\\WlhPinCap", "\\WlhManifestSha", "\\WlhOrbitsRerun", "\\WlhDefectZeroInexact", "\\WlhInjectorFlaggedCells"):
            self.assertIn(macro, disclosure["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == geo.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [geo.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 5)
        self.assertEqual(item["generator_module"], "generate_wall_loss_geometry_screening_v2_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 5)
        record = self._claim(geo.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [geo.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Preregistered wall-access screening from the catalogue cells of the accepted sweep geometries", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered wall-access screening from the catalogue cells of the accepted sweep geometries}", self.manuscript)
        self.assertIn("\\label{sec:wall-loss-geometry-screening-v2}", self.manuscript)
        # Section 11 cites this section instead of announcing a plan; the topology catalogue names its consumer.
        self.assertNotIn("is planned; no result of it exists", self.manuscript)
        self.assertNotIn("future work and no result of it\nexists", self.manuscript)
        self.assertNotIn("no admitted consumer has yet", self.manuscript)
        self.assertIn("the screening\nthat launches from the catalogue cells is admitted in\nSection~\\ref{sec:wall-loss-geometry-screening-v2}", self.manuscript)
        self.assertIn("whose first admitted consumer is the\ncatalogue-cell screening of Section~\\ref{sec:wall-loss-geometry-screening-v2}", self.manuscript)
        self.assertIn("catalogue-cell wall-access screening at Git revision \\GeometryScreeningTwoEvidenceRevision", self.manuscript)
        self.assertIn("A seventh numerical-screening gate", self.manuscript)
        self.assertIn("exhausts the collisionless", self.manuscript)
        self.assertIn("published post hoc by a fail-closed recovery", self.manuscript)
        # The earlier admissions' sentences stay in place.
        self.assertIn("not the separatrix-bounded\ncells of the catalogue admitted in Section~\\ref{sec:cusp-topology}", self.manuscript)
        self.assertIn("zero reflections are a launch-position\nresult", self.manuscript)


if __name__ == "__main__":
    unittest.main()
