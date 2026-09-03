"""Adversarial tests for the admission of the MDO L0 campaign v1 into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_mdo_l0_v1_evidence as mdo  # noqa: E402

GATE_ID = mdo.GATE_ID
MANIFEST_ID = mdo.MANIFEST_ID
HEADING = mdo.SECTION_HEADING


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class MdoAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json("paper/evidence/mdo-l0-v1.json")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _campaign_errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_mdo_campaign(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_reuses_the_numerical_campaign_kind_and_opens_no_level(self) -> None:
        self.assertEqual(self.gate["kind"], check_paper.CAMPAIGN_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertIn("declared component model", self.gate["kind_justification"])
        self.assertIn("optimiser evidence", self.gate["kind_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-mdo-campaign-manifest")
        self.assertEqual(self.gate["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.gate["preregistration_revision"], mdo.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.gate["dashboard_revision"], mdo.DASHBOARD_COMMIT_SHA)
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-campaign")
        self.assertEqual(self.payload["gate_kind"], "numerical-campaign")
        self.assertEqual(self.payload["classification"], mdo.CLASSIFICATION)
        self.assertEqual(self.payload["closure"], mdo.CLOSURE_ID)
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # The wall-loss gate is the other numerical-campaign gate; both share the kind, not the manifest type.
        wall = next(g for g in self.registry["gates"] if g["id"] == "GATE-WALL-LOSS-V4")
        self.assertEqual(wall["kind"], self.gate["kind"])
        self.assertNotEqual(wall["required_manifest_document_type"], self.gate["required_manifest_document_type"])
        self.assertIn("CL-1", self.registry["acceptance_policy"]["gate_kinds"]["numerical-campaign"])

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
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-mdo-campaign-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-mdo-campaign-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]}, set(schema["required_file_roles"]))
        self.assertEqual(sum(1 for s in self.payload["source_files"] if s["role"] == "run-artifact"), 9)
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
        for metric, macro in check_paper.MDO_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.MDO_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        self.assertEqual(self.payload["metrics"]["total_evaluations"], 864)
        self.assertEqual(self.payload["metrics"]["binding_gates_passed"], 8)
        self.assertEqual(self.payload["metrics"]["bo_beats_random_wins"], 3)
        self.assertEqual(self.payload["metrics"]["bo_beats_nsga3_wins"], 3)
        self.assertEqual(self.payload["metrics"]["robust_front_size"], 114)
        self.assertEqual(self.payload["metrics"]["nominal_front_size"], 62)
        self.assertEqual(self.payload["metrics"]["shared_designs"], 24)
        self.assertEqual(self.payload["metrics"]["no_wall_loss_infeasible_pareto_designs"], 110)
        self.assertEqual(self.payload["metrics"]["tolerated_eol_file_count"], 0)

    def test_campaign_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._campaign_errors(), [])

    def test_tampered_metric_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["bo_beats_random_wins"] = 2
        payload["metrics"]["robust_front_size"] = 114.0  # right value, wrong type
        payload["metrics"]["thruster_performance_claim_forbidden"] = False
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("metric 'bo_beats_random_wins' differs" in e for e in errors))
        self.assertTrue(any("metric 'robust_front_size' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'thruster_performance_claim_forbidden'" in e for e in errors))

    def test_classification_and_closure_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "l0_thruster_performance"
        errors = self._campaign_errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["closure"] = "CL-2"
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("closure identifier differs" in e for e in errors))
        rendered = next(m["value"] for m in self.evidence["macros"] if m["name"] == "MdoClassification")
        self.assertEqual(check_paper.tex_unescape(rendered), mdo.CLASSIFICATION)
        section = (REPO / mdo.SECTION_PATH).read_text(encoding="utf-8")
        self.assertIn("\\MdoClassification", section)
        self.assertIn("\\MdoClosureId", section)

    def test_opening_a_level_or_dropping_the_dashboard_is_rejected(self) -> None:
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

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-030")
        record["non_claims"].append("validated against thruster measurements")
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-030 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-030")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_or_relocated_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-033")
        record["manifest_ids"] = ["L0-SWEEP-20260901-8192-V1"]
        record["allowed_locations"] = ["Abstract"]
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("CLM-033 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-033 does not allow the section heading" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("c553124b\\allowbreak{}7393890d", "c553124c\\allowbreak{}7393890d")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._campaign_errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

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
        payload["paper_evidence_file"]["path"] = "paper/evidence/wall-loss-v4.json"
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-030", "CLM-032", "CLM-033", "CLM-034"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-029"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-035"].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-030", raw_ids)
        self.assertIn("CLM-035", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        # The Discussion interpretation is bound to both the optimisation and the wall-loss manifests.
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-035")
        self.assertEqual(set(record["manifest_ids"]), {MANIFEST_ID, "WALL-LOSS-V4-20260902-4608-V1"})
        self.assertIn("Muffatti2017", record["bibliography"])

    def test_scope_claim_registers_the_required_non_claims(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-030")
        joined = " ".join(record["non_claims"])
        for phrase in (
            "thruster-performance, plasma or physical-device claim",
            "better beyond the recorded budget",
            "geometry",
            "conditional on that closure and on those priors",
        ):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-034")
        self.assertEqual(scope["claim_class"], "campaign-scope-limitation")
        self.assertIn("leaves the benchmark field of the committed campaign policy null", scope["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == mdo.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [mdo.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 3)
        self.assertEqual(item["generator_module"], "generate_mdo_l0_v1_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 3)
        record = next(c for c in self.matrix["claims"] if c["id"] == mdo.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [mdo.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_limitations_are_updated(self) -> None:
        self.assertIn("Preregistered robust multi-objective optimisation of the L0 model", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Preregistered robust multi-objective optimisation of the L0 model}", self.manuscript)
        self.assertNotIn("has no admitted surrogate fit, acquisition", self.manuscript)
        self.assertIn("campaign policy's benchmark field remains null", self.manuscript)
        self.assertIn("planned bridge", self.manuscript)


if __name__ == "__main__":
    unittest.main()
