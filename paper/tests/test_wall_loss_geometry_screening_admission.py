"""Adversarial tests for the admission of the orbit wall-loss geometry screening v1 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_wall_loss_geometry_screening_v1_evidence as geo  # noqa: E402

GATE_ID = geo.GATE_ID
MANIFEST_ID = geo.MANIFEST_ID
HEADING = geo.SECTION_HEADING
V4_MANIFEST_ID = "WALL-LOSS-V4-20260902-4608-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class GeometryScreeningAdmissionTests(unittest.TestCase):
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
        check_paper._check_geometry_screening(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_is_a_numerical_screening_gate_at_the_dataset_outcome(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertIn("accepted-screening-dataset", kinds["numerical-screening"])
        self.assertIn("never accepted physical-orbit", kinds["numerical-screening"])
        self.assertEqual(self.gate["kind"], check_paper.SCREENING_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertIn(geo.RECORDED_OUTCOME, check_paper.SCREENING_OUTCOMES)
        # A fifth outcome (accepted-topology-screening) was added by the cusp topology admission.
        self.assertEqual(len(check_paper.SCREENING_OUTCOMES), 5)
        self.assertIn("accepted_screening_dataset", self.gate["recorded_outcome_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-orbit-screening-manifest")
        self.assertEqual(self.gate["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], geo.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], geo.DASHBOARD_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], self.gate["evidence_revision"])
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        self.assertEqual(self.matrix["manifests"][MANIFEST_ID]["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-screening")
        self.assertEqual(self.payload["gate_kind"], "numerical-screening")
        self.assertEqual(self.payload["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(self.payload["classification"], geo.CLASSIFICATION)
        self.assertEqual(self.payload["screening_model"], geo.SCREENING_MODEL)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        self.assertIsNone(self.payload["posthoc_audit"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # The other screening gates keep the l1a manifest type; this one has its own.
        for other in ("GATE-L1A-SWEEP-V2", "GATE-FOUR-CELL-V2", "GATE-TOPOLOGY-CHAR-V1"):
            gate = next(g for g in self.registry["gates"] if g["id"] == other)
            self.assertEqual(gate["kind"], self.gate["kind"])
            self.assertNotEqual(gate["required_manifest_document_type"], self.gate["required_manifest_document_type"])
        gate_record = next(c for c in self.matrix["claims"] if c["id"] == GATE_ID)
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
        wall = next(g for g in self.registry["gates"] if g["id"] == "GATE-WALL-LOSS-V4")
        self.assertEqual(wall["status"], "accepted")
        self.assertEqual(wall["metric_constraints"]["reflected_count"]["equals"], 0)

    def test_manifest_validates_as_a_typed_gate_manifest(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, self.payload,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertEqual(errors, [])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-orbit-screening-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-orbit-screening-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]} >= set(schema["required_file_roles"]), True)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "case-summary"), 18)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "orbit-artifact"), 4)
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"].startswith("preregistered-")), 4)
        paths = [s["path"] for s in self.payload["source_files"]]
        self.assertEqual(len(paths), len(set(paths)))
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
        for metric, macro in check_paper.GEOMETRY_SCREENING_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.GEOMETRY_SCREENING_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        for metric, expected in check_paper.SCREENING_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual(metrics["evaluated_count"], 96)
        self.assertEqual(metrics["case_count"], 196)
        self.assertEqual(metrics["orbit_count"], 100352)
        self.assertEqual(metrics["validator_calls_passed"], 6664)
        self.assertEqual(metrics["validator_failures"], 0)
        self.assertEqual(metrics["converged_design_count"], 96)
        self.assertEqual(metrics["total_reflections"], 22904)
        self.assertEqual(metrics["reflections_reported_timestep"], 11268)
        self.assertEqual((metrics["reflections_per_design_minimum"], metrics["reflections_per_design_maximum"]), (32, 282))
        self.assertEqual(metrics["designs_with_reflections"], 96)
        self.assertEqual(metrics["excluded_design_count"], 0)
        self.assertEqual((metrics["design_cells_saturated_at_one"], metrics["design_cells_saturated_at_zero"]), (94, 0))
        self.assertEqual((metrics["escapes_anode_plane"], metrics["escapes_exit_plane"], metrics["escapes_divergent_radial"]), (1635, 1127, 862))
        self.assertEqual(metrics["wall_hit_probability_minimum"], 0.375)
        self.assertEqual(metrics["wall_hit_probability_maximum"], 0.869140625)
        self.assertEqual(metrics["wall_hit_probability_median"], 0.7021484375)
        self.assertEqual(metrics["maximum_successive_probability_change"], 0.005859375)
        self.assertEqual(metrics["cross_resolution_design_count"], 4)
        self.assertEqual(metrics["tolerated_eol_file_count"], 0)
        self.assertEqual(metrics["verified_file_count"], 2835)
        # Every gate constraint holds against the manifest.
        for metric, rule in self.gate["metric_constraints"].items():
            self.assertIn(metric, metrics, metric)
            self.assertEqual(metrics[metric], rule["equals"], metric)

    def test_screening_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_tampered_metric_or_outcome_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["designs_with_reflections"] = 95
        payload["metrics"]["converged_design_count"] = 96.0  # right value, wrong type
        payload["metrics"]["design_rule_claimed"] = True
        errors = self._errors(payload=payload)
        self.assertTrue(any("metric 'designs_with_reflections' differs" in e for e in errors))
        self.assertTrue(any("metric 'converged_design_count' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'design_rule_claimed'" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["recorded_outcome"] = "accepted-screening"
        errors = self._errors(gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        gate["recorded_outcome"] = "accepted-finding"
        errors = self._errors(gate=gate)
        self.assertTrue(any("not a recognized screening outcome" in e for e in errors))

    def test_classification_and_model_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "ACCEPTED_PHYSICAL_ORBIT_EVIDENCE"
        errors = self._errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["screening_model"] = "P2 finite-element field"
        errors = self._errors(payload=payload)
        self.assertTrue(any("screening_model differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["WlgClassification"]), geo.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["WlgRecordedOutcome"]), geo.RECORDED_OUTCOME)
        self.assertEqual(check_paper.tex_unescape(values["WlgCampaignStatus"]), geo.CAMPAIGN_STATUS)
        section = (REPO / geo.SECTION_PATH).read_text(encoding="utf-8")
        for macro in ("\\WlgClassification", "\\WlgRecordedOutcome", "\\WlgCampaignStatus", "\\WlgFieldStatus"):
            self.assertIn(macro, section)

    def test_opening_a_level_or_touching_the_dashboard_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = geo.PREREGISTRATION_COMMIT_SHA
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        html = next(f for f in payload["dashboard"]["files"] if f["role"] == "dashboard-html")
        html["git_blob_sha256"] = "0" * 64
        errors = self._errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"] = {"revision": geo.RESULTS_COMMIT_SHA}
        errors = self._errors(payload=payload)
        self.assertTrue(any("post-hoc audit the generator does not register" in e for e in errors))

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
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-046")
        record["non_claims"].append("validated against thruster measurements")
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-046 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-046")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_relocated_or_misplaced_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-049")
        record["manifest_ids"] = [V4_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("CLM-049 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-049 does not allow the section heading" in e for e in errors))
        # The Discussion interpretation may not be moved into the results section.
        section = (REPO / geo.SECTION_PATH).read_text(encoding="utf-8")
        interpretation = next(c for c in self.matrix["claims"] if c["id"] == "CLM-052")
        self.assertEqual(interpretation["claim_class"], "interpretation")
        self.assertNotIn("CLM-052", section)
        matrix = copy.deepcopy(self.matrix)
        next(c for c in matrix["claims"] if c["id"] == "CLM-051")["claim_class"] = "interpretation"
        errors = self._errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-051 must not appear inside the results section" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("ab7c2897\\allowbreak{}7963822b", "ab7c2898\\allowbreak{}7963822b")
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
        payload["paper_evidence_file"]["path"] = "paper/evidence/wall-loss-v4.json"
        errors = self._errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["experiment_id"] = "cft-orbit-wall-loss-v4"
        errors = self._errors(payload=payload)
        self.assertTrue(any("not the registered screening study" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-046", "CLM-048", "CLM-049", "CLM-050", "CLM-051"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-045"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-052"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-046", raw_ids)
        self.assertIn("CLM-052", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in geo.PROSE_CLAIM_IDS:
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertIn(MANIFEST_ID, record["manifest_ids"])
        # The reflection contrast and the consumer are also bound to the wall-loss manifest.
        for claim_id in ("CLM-048", "CLM-050"):
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertEqual(set(record["manifest_ids"]), {MANIFEST_ID, V4_MANIFEST_ID})
        # The Discussion interpretation re-scopes the wall-loss campaign's zero reflections as a
        # launch-position result through the analysis bound with the sweep-v3 manifest.
        discussion = next(c for c in self.matrix["claims"] if c["id"] == "CLM-052")
        self.assertEqual(set(discussion["manifest_ids"]), {MANIFEST_ID, V4_MANIFEST_ID, "L1A-SWEEP-V3-20260903-128-V1"})
        self.assertIn("\\WlfPooledReflected", discussion["authorized_tex"])
        self.assertIn("launch-position result", discussion["authorized_tex"])
        self.assertIn("mirror reflections toward the magnet centres", discussion["authorized_tex"])
        self.assertNotIn("scoped to the field it was made in", discussion["authorized_tex"])
        # The consuming optimisation is admitted (Section 12); the "future work" boundary is gone.
        self.assertNotIn("future work", discussion["authorized_tex"])
        self.assertIn("sec:mdo-l0-v2", discussion["authorized_tex"])
        self.assertIn("closure-dependent ranking", discussion["authorized_tex"])
        self.assertIn("Muffatti2017", discussion["bibliography"])

    def test_scope_claims_register_the_boundary(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-046")
        joined = " ".join(record["non_claims"])
        for phrase in (
            "accepted physical-orbit evidence",
            "not a plasma or performance quantity",
            "mirror-formula",
            "protocol positions, not demonstrated confinement cells",
            "observation, not a design rule",
            "opens no physics level",
        ):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-051")
        self.assertEqual(scope["claim_class"], "screening-scope-limitation")
        self.assertIn("\\WlgUsableAs", scope["authorized_tex"])
        self.assertIn("not P2-qualified", scope["authorized_tex"])
        trend = next(c for c in self.matrix["claims"] if c["id"] == "CLM-049")
        self.assertIn("none is a design rule", trend["authorized_tex"])
        self.assertIn("Spearman", trend["authorized_tex"])
        consumer = next(c for c in self.matrix["claims"] if c["id"] == "CLM-050")
        self.assertIn("first consumer", consumer["authorized_tex"])
        self.assertIn("labelled reference row", consumer["authorized_tex"])
        # The wall-loss campaign's own scope claim now records the consumer instead of "no consumer".
        v4_scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-016")
        self.assertNotIn("no consumer model has ingested it", v4_scope["authorized_tex"])
        self.assertIn("sec:wall-loss-geometry-screening", v4_scope["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == geo.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [geo.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 4)
        self.assertEqual(item["generator_module"], "generate_wall_loss_geometry_screening_v1_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 4)
        record = next(c for c in self.matrix["claims"] if c["id"] == geo.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [geo.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Preregistered wall-loss screening across the accepted sweep geometries", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered wall-loss screening across the accepted sweep geometries}", self.manuscript)
        self.assertIn("\\label{sec:wall-loss-geometry-screening}", self.manuscript)
        # Stale boundary sentences of earlier admissions must be gone.
        self.assertNotIn("its coupling export has not been consumed", self.manuscript)
        self.assertNotIn("none of its outputs is admitted here, and until an accepted\nmanifest exists", self.manuscript)
        self.assertNotIn("no consumer model has ingested it", self.flattened)
        self.assertNotIn("statement is field-specific", self.manuscript)
        self.assertIn("zero reflections are a launch-position\nresult", self.manuscript)
        self.assertIn("stratifies its launches\nby catalogue cell is planned; no result of it exists", self.manuscript)
        self.assertIn("planned bridge", self.manuscript)
        self.assertIn("refined-field diagnostic exists\nfor four representative designs only", self.manuscript)
        self.assertIn("wall-loss geometry screening at Git revision \\GeometryScreeningEvidenceRevision", self.manuscript)


if __name__ == "__main__":
    unittest.main()
