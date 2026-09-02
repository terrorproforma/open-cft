"""Adversarial tests for the admission of the wall-loss v4 campaign into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_wall_loss_v4_evidence as wl4  # noqa: E402

GATE_ID = wl4.GATE_ID
MANIFEST_ID = wl4.MANIFEST_ID
HEADING = "Collisionless full-orbit electron wall loss in the divergent-exit field"


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class WallLossAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json("paper/evidence/wall-loss-v4.json")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _campaign_errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_wall_loss_campaign(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_campaign_gate_is_a_numerical_campaign_gate_that_opens_no_level(self) -> None:
        self.assertEqual(self.gate["kind"], check_paper.CAMPAIGN_GATE_KIND)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-test-particle-campaign-manifest")
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "numerical-campaign")
        self.assertEqual(self.payload["classification"], wl4.CLASSIFICATION)
        self.assertEqual(self.payload["evidence_revision"], wl4.RESULTS_COMMIT_SHA)
        self.assertEqual(self.payload["preregistration_revision"], wl4.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.payload["evidence_level"]["opens_gate"], None)
        for level in ("L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["kind"], check_paper.PHYSICS_GATE_KIND)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))

    def test_campaign_manifest_validates_as_a_typed_gate_manifest(self) -> None:
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO,
            self.registry["evidence_revision"],
            self.gate,
            self.payload,
            Path(self.gate["manifest_path"]),
            errors,
            require_committed=False,
        )
        self.assertEqual(errors, [])
        wrong_level = copy.deepcopy(self.payload)
        wrong_level["level"] = "L1"
        errors = []
        check_paper._validate_manifest_payload(
            REPO,
            self.registry["evidence_revision"],
            self.gate,
            wrong_level,
            Path(self.gate["manifest_path"]),
            errors,
            require_committed=False,
        )
        self.assertTrue(any("level does not match" in error for error in errors))

    def test_manifest_metrics_equal_the_raw_artifact_values(self) -> None:
        raw = {item["name"]: item["raw"] for item in self.evidence["macros"]}
        for metric, macro in check_paper.WALL_LOSS_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
        for metric, macro in check_paper.WALL_LOSS_CELL_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"]["per_cell_bimodality"][metric], raw[macro])
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-test-particle-campaign-manifest"]
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]}, set(schema["required_file_roles"]))

    def test_campaign_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._campaign_errors(), [])

    def test_tampered_metric_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["reflected_count"] = 1
        payload["metrics"]["per_cell_bimodality"]["exit_cell_escapes"] -= 1
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("metric 'reflected_count' differs" in e for e in errors))
        self.assertTrue(any("per-cell metric 'exit_cell_escapes' differs" in e for e in errors))

    def test_classification_must_agree_everywhere(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "collisionless_pic"
        errors = self._campaign_errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        rendered = next(m["value"] for m in self.evidence["macros"] if m["name"] == "WlfClassification")
        self.assertEqual(check_paper.tex_unescape(rendered), wl4.CLASSIFICATION)
        self.assertIn("\\WlfClassification", (REPO / wl4.SECTION_PATH).read_text(encoding="utf-8"))

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-013")
        record["non_claims"].append("validated against thruster measurements")
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-013 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-013")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_or_relocated_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-015")
        record["manifest_ids"] = ["L0-SWEEP-20260901-8192-V1"]
        record["allowed_locations"] = ["Abstract"]
        errors = self._campaign_errors(matrix=matrix)
        self.assertTrue(any("CLM-015 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-015 does not allow the section heading" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("6922a3cf\\allowbreak{}97d26173", "6922a3cd\\allowbreak{}97d26173")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._campaign_errors(manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        duplicated = self.manuscript.replace(wl4.SECTION_BINDING, wl4.SECTION_BINDING + "\n" + wl4.SECTION_BINDING)
        errors = self._campaign_errors(manuscript=duplicated)
        self.assertTrue(any("section binding must occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(wl4.GENERATED_BINDING + "\n", "") .replace(
            "\\begin{document}", "\\begin{document}\n" + wl4.GENERATED_BINDING
        )
        errors = self._campaign_errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = "paper/evidence/claims.json"
        errors = self._campaign_errors(payload=payload)
        self.assertTrue(any("committed evidence file differs from regeneration" in e for e in errors))

    def test_section_claims_resolve_under_the_section_heading(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-013", "CLM-015", "CLM-016"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-012"].start), "Abstract")
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-017"].start), "Discussion")
        # The raw manuscript alone does not contain the section claims: flattening is what binds them.
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-013", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == wl4.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [wl4.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 2)
        self.assertEqual(item["generator_module"], "generate_wall_loss_v4_evidence")
        record = next(c for c in self.matrix["claims"] if c["id"] == wl4.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [wl4.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)


if __name__ == "__main__":
    unittest.main()
