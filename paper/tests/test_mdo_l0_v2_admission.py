"""Adversarial tests for the admission of the MDO L0 campaign v2 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_mdo_l0_v1_evidence as mdo_v1  # noqa: E402
import generate_mdo_l0_v2_evidence as mdo  # noqa: E402

GATE_ID = mdo.GATE_ID
MANIFEST_ID = mdo.MANIFEST_ID
HEADING = mdo.SECTION_HEADING
V1_MANIFEST_ID = mdo_v1.MANIFEST_ID
GEO_MANIFEST_ID = "WALL-LOSS-GEOMETRY-SCREENING-V1-20260903-96-V1"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class MdoV2AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json("paper/evidence/mdo-l0-v2.json")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _campaign_errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_mdo_catalogue_campaign(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_reuses_the_numerical_campaign_kind_with_its_own_manifest_type(self) -> None:
        self.assertEqual(self.gate["kind"], check_paper.CAMPAIGN_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertIn("declared component model", self.gate["kind_justification"])
        self.assertIn("acceptance means recorded consistently", self.gate["kind_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-mdo-catalogue-campaign-manifest")
        self.assertEqual(self.gate["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], mdo.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], mdo.DASHBOARD_COMMIT_SHA)
        self.assertEqual(self.gate["prior_campaign_revision"], mdo.V1_RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["prior_posthoc_audit_revision"], mdo.V1_AUDIT_COMMIT_SHA)
        self.assertEqual(self.gate["screening_revision"], mdo.SCREENING_RESULTS_COMMIT_SHA)
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        self.assertEqual(self.matrix["manifests"][MANIFEST_ID]["prior_campaign_manifest_id"], V1_MANIFEST_ID)
        self.assertEqual(self.matrix["manifests"][MANIFEST_ID]["screening_manifest_id"], GEO_MANIFEST_ID)
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-campaign")
        self.assertEqual(self.payload["gate_kind"], "numerical-campaign")
        self.assertEqual(self.payload["classification"], mdo.CLASSIFICATION)
        self.assertEqual(self.payload["closure"], mdo.CLOSURE_ID)
        self.assertEqual(self.payload["sensitivity_closure"], mdo.SENSITIVITY_CLOSURE_ID)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # Three numerical-campaign gates now share the kind; each has its own manifest type.
        kinds = {g["id"]: g["required_manifest_document_type"] for g in self.registry["gates"] if g["kind"] == self.gate["kind"]}
        self.assertEqual(set(kinds), {"GATE-WALL-LOSS-V4", "GATE-MDO-L0-V1", GATE_ID})
        self.assertEqual(len(set(kinds.values())), 3)
        policy = self.registry["acceptance_policy"]["gate_kinds"]["numerical-campaign"]
        self.assertIn("CL-1", policy)
        self.assertIn("catalogue optimisation", policy)

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
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-mdo-catalogue-campaign-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-mdo-catalogue-campaign-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]}, set(schema["required_file_roles"]))
        roles = [s["role"] for s in self.payload["source_files"]]
        self.assertEqual(roles.count("run-artifact"), 9)
        self.assertEqual(roles.count("prior-campaign-artifact"), len(mdo.V1_COMPARISON_ARTIFACTS))
        self.assertEqual(roles.count("screening-dataset"), 1)
        self.assertEqual(roles.count("prior-posthoc-audit"), 1)
        # Every artifact the evidence file read from either bundle is bound in the manifest.
        bound = {s["path"] for s in self.payload["source_files"]}
        for relative in self.evidence["artifacts"]:
            self.assertIn((mdo.RESULTS / relative).as_posix(), bound)
        for relative in self.evidence["v1_artifacts"]:
            self.assertIn((mdo.V1_RESULTS / relative).as_posix(), bound)
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
        for metric, macro in check_paper.MDB_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.MDB_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual(metrics["total_evaluations"], 1440)
        self.assertEqual(metrics["infeasible_evaluations"], 91)
        self.assertEqual((metrics["infeasible_evaluations_qlognehvi"], metrics["infeasible_evaluations_nsga3"], metrics["infeasible_evaluations_lhs"]), (88, 3, 0))
        self.assertEqual(metrics["binding_gate_count"], 12)
        self.assertEqual(metrics["binding_gates_passed"], 12)
        self.assertEqual(metrics["catalogue_size"], 96)
        self.assertEqual(metrics["imported_file_count"], 28)
        self.assertEqual(metrics["nsga3_duplicate_evaluations"], 0)
        self.assertEqual(metrics["bo_beats_random_wins"], 3)
        self.assertEqual(metrics["bo_beats_nsga3_wins"], 3)
        self.assertEqual(metrics["robust_front_catalogue_indices"], [49, 50, 94])
        self.assertEqual(metrics["nominal_front_catalogue_indices"], [49, 50, 74, 94])
        self.assertEqual(metrics["dense_reference_robust_front_catalogue_indices"], [46, 49, 50, 73, 94])
        self.assertEqual((metrics["robust_front_size"], metrics["nominal_front_size"], metrics["shared_designs"]), (96, 86, 75))
        self.assertEqual((metrics["cl2_front_size"], metrics["cl2_front_catalogue_design_count"], metrics["cl2_shared_with_campaign_front"]), (50, 25, 0))
        self.assertEqual(metrics["cl2_jaccard_with_campaign_front"], 0.0)
        self.assertEqual((metrics["width_quarter_front_size"], metrics["width_four_front_size"], metrics["width_point_front_size"]), (15, 91, 94))
        self.assertEqual((metrics["dense_negligible_hypervolume_designs"], metrics["dense_negligible_hypervolume_designs_with_saturated_cell"], metrics["catalogue_saturated_cell_designs"]), (77, 73, 73))
        self.assertEqual((metrics["qlognehvi_first_seed_stall_design"], metrics["qlognehvi_first_seed_missed_design"], metrics["qlognehvi_first_seed_missed_design_evaluations"]), (50, 49, 0))
        self.assertEqual(metrics["v1_audit_disclosures_closed"], 6)
        self.assertEqual(metrics["result_commit_files_outside_results"], 0)
        self.assertEqual(metrics["tolerated_eol_file_count"], 0)
        self.assertIs(metrics["same_reference_frame_as_prior_campaign"], True)

    def test_campaign_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._campaign_errors(), [])

    def test_v1_checker_still_accepts_its_own_gate(self) -> None:
        gate = next(g for g in self.registry["gates"] if g["id"] == mdo_v1.GATE_ID)
        payload = _json(gate["manifest_path"])
        errors: list[str] = []
        check_paper._check_mdo_campaign(REPO, gate, payload, self.manuscript, self.flattened, self.matrix, errors)
        self.assertEqual(errors, [])
        # The v1 checker refuses the v2 manifest and vice versa.
        errors = []
        check_paper._check_mdo_campaign(REPO, gate, self.payload, self.manuscript, self.flattened, self.matrix, errors)
        self.assertTrue(any("not the registered campaign" in e for e in errors))
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("not the registered campaign" in e for e in errors))

    def test_tampered_metric_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["bo_beats_random_wins"] = 2
        payload["metrics"]["robust_front_size"] = 96.0  # right value, wrong type
        payload["metrics"]["robust_front_catalogue_indices"] = [49, 50, 74]
        payload["metrics"]["surrogate_used"] = True
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("metric 'bo_beats_random_wins' differs" in e for e in errors))
        self.assertTrue(any("metric 'robust_front_size' differs" in e for e in errors))
        self.assertTrue(any("metric 'robust_front_catalogue_indices' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'surrogate_used'" in e for e in errors))

    def test_classification_and_closures_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "l0_thruster_performance"
        errors = self._campaign_errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["closure"] = mdo_v1.CLOSURE_ID
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("closure identifier differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["sensitivity_closure"] = mdo.CLOSURE_ID
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("sensitivity closure identifier differs" in e for e in errors))
        values = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(values["MdbClassification"]), mdo.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(values["MdbClosureId"]), mdo.CLOSURE_ID)
        self.assertEqual(check_paper.tex_unescape(values["MdbSensitivityClosureId"]), mdo.SENSITIVITY_CLOSURE_ID)
        self.assertEqual(check_paper.tex_unescape(values["MdbScreeningClassification"]), mdo.SCREENING_CLASSIFICATION)
        self.assertNotEqual(mdo.CLOSURE_ID, mdo_v1.CLOSURE_ID)
        section = (REPO / mdo.SECTION_PATH).read_text(encoding="utf-8")
        for macro in ("\\MdbClassification", "\\MdbClosureId", "\\MdbSensitivityClosureId", "\\MdbScreeningClassification"):
            self.assertIn(macro, section)

    def test_opening_a_level_or_dropping_a_binding_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L1"
        errors = self._campaign_errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["revision"] = mdo.RESULTS_COMMIT_SHA
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("dashboard revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["dashboard"]["files"][1]["git_blob_sha256"] = "0" * 64
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("dashboard-html checkout differs" in e or "SHA-256 mismatch" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["prior_campaign"]["manifest_sha256"] = "0" * 64
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("prior campaign manifest SHA-256 differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["catalogue_binding"]["dataset_git_blob"] = "0" * 40
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("catalogue_binding.dataset_git_blob differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["posthoc_audit"]["disclosures_closed"] = ["F9", "F10"]
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("closed audit disclosures differ" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        del payload["prior_campaign"]
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("lacks the prior_campaign binding" in e for e in errors))

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-054")
        record["non_claims"].append("validated against thruster measurements")
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-054 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-054")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_or_relocated_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-057")
        record["manifest_ids"] = [V1_MANIFEST_ID]
        record["allowed_locations"] = ["Abstract"]
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("CLM-057 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-057 does not allow the section heading" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("a003f766\\allowbreak{}c330d4e5", "a003f767\\allowbreak{}c330d4e5")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._campaign_errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))
        self.assertIn("L0 catalogue optimisation campaign at Git revision \\MdbEvidenceRevision", self.manuscript)

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(mdo.SECTION_BINDING, mdo.SECTION_BINDING + "\n" + mdo.SECTION_BINDING)
        errors = self._campaign_errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(mdo.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + mdo.GENERATED_BINDING
        )
        errors = self._campaign_errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = mdo_v1.EVIDENCE_PATH.as_posix()
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-054", "CLM-056", "CLM-057", "CLM-058", "CLM-059"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-053"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-060"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-054", raw_ids)
        self.assertIn("CLM-060", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        for claim_id in mdo.PROSE_CLAIM_IDS:
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertIn(MANIFEST_ID, record["manifest_ids"])
        # Cross-campaign claims are bound to the manifests they read from.
        bindings = {
            "CLM-056": {MANIFEST_ID, GEO_MANIFEST_ID},
            "CLM-057": {MANIFEST_ID, V1_MANIFEST_ID},
            "CLM-058": {MANIFEST_ID, V1_MANIFEST_ID},
            "CLM-060": {MANIFEST_ID, V1_MANIFEST_ID, GEO_MANIFEST_ID},
            "CLM-055": {MANIFEST_ID, V1_MANIFEST_ID},
        }
        for claim_id, manifests in bindings.items():
            record = next(c for c in self.matrix["claims"] if c["id"] == claim_id)
            self.assertEqual(set(record["manifest_ids"]), manifests, claim_id)
        discussion = next(c for c in self.matrix["claims"] if c["id"] == "CLM-060")
        self.assertIn("closure-dependent", discussion["authorized_tex"])
        self.assertIn("not a demonstrated one", discussion["authorized_tex"])
        self.assertIn("Muffatti2017", discussion["bibliography"])
        self.assertEqual(discussion["claim_class"], "interpretation")

    def test_scope_claim_registers_the_required_non_claims(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-054")
        joined = " ".join(record["non_claims"])
        for phrase in (
            "thruster-performance, plasma or physical-device claim",
            "wins under this closure only",
            "better beyond the recorded budget",
            "nothing is interpolated between them",
        ):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-059")
        self.assertEqual(scope["claim_class"], "campaign-scope-limitation")
        self.assertIn("leaves the benchmark field of the committed campaign policy null", scope["authorized_tex"])
        self.assertIn("\\MdbScreeningClassification", scope["authorized_tex"])
        self.assertIn("\\MdbSurrogateOneOutcome", scope["authorized_tex"])
        audit = next(c for c in self.matrix["claims"] if c["id"] == "CLM-058")
        self.assertEqual(audit["claim_class"], "campaign-audit-closure")
        for token in ("\\MdbAuditNineId", "\\MdbAuditTenId", "\\MdbAuditTwentyTwoId", "\\MdbAuditTwentySixId", "\\MdbAuditTwentySevenId", "\\MdbAuditTwentyEightId"):
            self.assertIn(token, audit["authorized_tex"])
        geometry = next(c for c in self.matrix["claims"] if c["id"] == "CLM-056")
        self.assertIn("exactly the three lowest pooled", geometry["authorized_tex"])
        self.assertIn("\\MdbDenseNegligibleDesigns", geometry["authorized_tex"])
        closure = next(c for c in self.matrix["claims"] if c["id"] == "CLM-057")
        self.assertIn("does not carry over to design-dependent posteriors", closure["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == mdo.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [mdo.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 4)
        self.assertEqual(item["generator_module"], "generate_mdo_l0_v2_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 4)
        record = next(c for c in self.matrix["claims"] if c["id"] == mdo.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [mdo.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_stale_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Preregistered catalogue optimisation of the L0 model over the screened sweep designs", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered catalogue optimisation of the L0 model over the screened sweep designs}", self.manuscript)
        self.assertIn("\\label{sec:mdo-l0-v2}", self.manuscript)
        # The earlier admissions' "future work" / "open link" sentences are gone.
        self.assertNotIn("a design-dependent optimisation that consumes it is future work", self.manuscript)
        self.assertNotIn("the optimisation that consumes it is future work", self.manuscript)
        self.assertNotIn("the geometry-to-performance link stays open", self.manuscript)
        self.assertNotIn("no surrogate or optimisation consuming the dataset is admitted", self.manuscript)
        self.assertNotIn("optimization outcomes beyond that L0\ncampaign", self.manuscript)
        self.assertIn("planned bridge", self.manuscript)
        self.assertIn("recorded successor", self.manuscript)
        self.assertIn("ranking is a property of the closure", self.manuscript)
        # The prior campaign's Discussion reading now points at the catalogue campaign.
        v1_discussion = next(c for c in self.matrix["claims"] if c["id"] == "CLM-035")
        self.assertIn("sec:mdo-l0-v2", v1_discussion["authorized_tex"])
        self.assertNotIn("is open:", v1_discussion["authorized_tex"])
        geo_discussion = next(c for c in self.matrix["claims"] if c["id"] == "CLM-052")
        self.assertIn("sec:mdo-l0-v2", geo_discussion["authorized_tex"])
        self.assertNotIn("future work", geo_discussion["authorized_tex"])


if __name__ == "__main__":
    unittest.main()
