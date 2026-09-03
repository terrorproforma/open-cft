"""Adversarial tests for the admission of the four-cell power-balance closure analysis."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_four_cell_closure_evidence as fcc  # noqa: E402

GATE_ID = fcc.GATE_ID
MANIFEST_ID = fcc.MANIFEST_ID
HEADING = fcc.SECTION_HEADING


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class FourCellClosureAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.schemas = _json("paper/evidence/manifest-schemas.json")
        cls.gate = next(g for g in cls.registry["gates"] if g["id"] == GATE_ID)
        cls.payload = _json(cls.gate["manifest_path"])
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = _json("paper/evidence/four-cell-closure.json")
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _analysis_errors(self, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_four_cell_closure(
            REPO,
            gate if gate is not None else self.gate,
            payload if payload is not None else self.payload,
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gate_defines_the_analytic_consistency_kind_and_opens_no_level(self) -> None:
        self.assertEqual(self.gate["kind"], "analytic-consistency")
        self.assertEqual(check_paper.ANALYTIC_GATE_KIND, "analytic-consistency")
        self.assertIn("analytic-consistency", check_paper.KNOWN_GATE_KINDS)
        self.assertEqual(self.gate["status"], "accepted")
        self.assertIsNone(self.gate["opens_level"])
        self.assertIn("equation set", self.gate["kind_justification"])
        self.assertIn("PROPOSED_NOT_ACCEPTED", self.gate["kind_justification"])
        self.assertEqual(self.gate["required_manifest_document_type"], "paper-analytic-consistency-manifest")
        self.assertEqual(self.gate["evidence_revision"], fcc.ANALYSIS_COMMIT_SHA)
        self.assertEqual(self.gate["verified_tree_revision"], fcc.VERIFIED_TREE_COMMIT_SHA)
        self.assertEqual(self.gate["mdo_preregistration_revision"], fcc.MDO_PREREGISTRATION_COMMIT_SHA)
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), set(check_paper.KNOWN_GATE_KINDS))
        description = kinds["analytic-consistency"]
        for phrase in ("derivation", "verified numerically", "pinned by committed tests", "opens no physics level", "admitted as recorded", "not a statement about the physical thruster"):
            self.assertIn(phrase, description)
        self.assertIn(MANIFEST_ID, self.matrix["manifests"])
        self.assertEqual(self.payload["manifest_id"], MANIFEST_ID)
        self.assertEqual(self.payload["level"], "analytic-consistency")
        self.assertEqual(self.payload["gate_kind"], "analytic-consistency")
        self.assertEqual(self.payload["classification"], fcc.CLASSIFICATION)
        self.assertEqual(self.payload["correction_status"], "PROPOSED_NOT_ACCEPTED")
        self.assertIsNone(self.payload["evidence_level"]["opens_gate"])
        for level in ("L0", "L1", "L2", "L3"):
            self.assertIn(level, self.payload["evidence_level"]["relation_to_levels"])
        # The kind is new: no other gate carries it and no campaign/screening manifest type is reused.
        others = [g for g in self.registry["gates"] if g["kind"] == "analytic-consistency"]
        self.assertEqual([g["id"] for g in others], [GATE_ID])
        self.assertNotIn(self.gate["required_manifest_document_type"], {g["required_manifest_document_type"] for g in self.registry["gates"] if g["id"] != GATE_ID})

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
        schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-analytic-consistency-manifest"]
        self.assertEqual(self.schemas["manifest_types"]["paper-analytic-consistency-manifest"], schema)
        self.assertEqual(set(schema["required_metrics"]), set(self.payload["metrics"]))
        self.assertEqual({s["role"] for s in self.payload["source_files"]}, set(schema["required_file_roles"]))
        self.assertEqual({s["path"] for s in self.payload["source_files"]}, set(fcc.SOURCE_ROLES))
        wrong_level = copy.deepcopy(self.payload)
        wrong_level["level"] = "numerical-campaign"
        errors = []
        check_paper._validate_manifest_payload(
            REPO, self.registry["evidence_revision"], self.gate, wrong_level,
            Path(self.gate["manifest_path"]), errors, require_committed=False,
        )
        self.assertTrue(any("level does not match" in error for error in errors))

    def test_manifest_metrics_equal_the_evidence_values(self) -> None:
        raw = {item["name"]: item["raw"] for item in self.evidence["macros"]}
        for metric, macro in check_paper.FOUR_CELL_CLOSURE_METRIC_MACROS.items():
            with self.subTest(metric=metric):
                self.assertEqual(self.payload["metrics"][metric], raw[macro])
                self.assertIs(type(self.payload["metrics"][metric]), type(raw[macro]))
        for metric, expected in check_paper.FOUR_CELL_CLOSURE_POLICY_METRICS.items():
            self.assertIs(self.payload["metrics"][metric], expected)
        metrics = self.payload["metrics"]
        self.assertEqual(metrics["correction_status"], "PROPOSED_NOT_ACCEPTED")
        self.assertEqual(metrics["recomputed_jacobian_rank"], 22)
        self.assertEqual(metrics["state_dimension"], 25)
        self.assertEqual(metrics["ledger_row_count"], 28)
        self.assertEqual(metrics["probe_closed_cases"], 13)
        self.assertEqual(metrics["probe_total_cases"], 80)
        self.assertIs(metrics["continuation_branch_found"], False)
        self.assertEqual(metrics["anode_only_closures"], 6)
        self.assertEqual(metrics["recomputed_anode_fall_coefficient"], 2.0)
        self.assertLessEqual(metrics["recomputed_closed_form_relative_difference"], 1e-12)
        self.assertLessEqual(metrics["continuation_max_relative_departure"], 0.25)
        self.assertEqual(metrics["probe_source"], "mdo-protocol-disclosure")
        self.assertEqual(metrics["legacy_ionisation_energy_terms"], 3)
        self.assertEqual(metrics["corrected_rank_if_accepted"], 21)

    def test_analysis_checker_accepts_the_committed_state(self) -> None:
        self.assertEqual(self._analysis_errors(), [])

    def test_tampered_metric_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["metrics"]["recomputed_jacobian_rank"] = 21
        payload["metrics"]["probe_closed_cases"] = 13.0  # right value, wrong type
        payload["metrics"]["proposed_correction_accepted"] = True
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("metric 'recomputed_jacobian_rank' differs" in e for e in errors))
        self.assertTrue(any("metric 'probe_closed_cases' differs" in e for e in errors))
        self.assertTrue(any("policy metric 'proposed_correction_accepted'" in e for e in errors))

    def test_correction_status_and_classification_must_agree_everywhere(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["correction_status"] = "ACCEPTED"
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("correction status differs" in e for e in errors))
        gate = copy.deepcopy(self.gate)
        gate["metric_constraints"]["classification"]["equals"] = "thruster_physics"
        errors = self._analysis_errors(gate=gate)
        self.assertTrue(any("classification differs" in e for e in errors))
        rendered = {m["name"]: m["value"] for m in self.evidence["macros"]}
        self.assertEqual(check_paper.tex_unescape(rendered["FccClassification"]), fcc.CLASSIFICATION)
        self.assertEqual(check_paper.tex_unescape(rendered["FccCorrectionStatus"]), "PROPOSED_NOT_ACCEPTED")
        section = (REPO / fcc.SECTION_PATH).read_text(encoding="utf-8")
        for macro in ("\\FccClassification", "\\FccCorrectionStatus", "\\FccClosedFormRelDiff", "\\FccProbeSource"):
            self.assertIn(macro, section)

    def test_opening_a_level_or_moving_a_revision_is_rejected(self) -> None:
        gate = copy.deepcopy(self.gate)
        gate["opens_level"] = "L2"
        errors = self._analysis_errors(gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["verified_tree_revision"] = fcc.ANALYSIS_COMMIT_SHA
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("verified_tree_revision differs" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        protocol = next(s for s in payload["source_files"] if s["path"] == fcc.PROTOCOL.as_posix())
        protocol["git_blob"] = "0" * 40
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("source binding differs" in e for e in errors))
        self.assertTrue(any("frozen preregistration blob" in e for e in errors))

    def test_executed_package_drift_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["executed_package"]["files"][0]["sha256_lf"] = "0" * 64
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("executed package file differs from the bound blob" in e for e in errors))
        payload = copy.deepcopy(self.payload)
        payload["executed_package"]["matches_bound_blobs"] = False
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("executed package equal to the bound blobs" in e for e in errors))
        # The generator itself refuses a package that differs from the bound blobs.
        bound = {path: {"git_blob_sha256": digest} for path, digest in fcc.executed_package_digests(REPO).items()}
        tampered = dict(fcc.executed_package_digests(REPO))
        first = sorted(tampered)[0]
        tampered[first] = "f" * 64
        with self.assertRaises(ValueError):
            fcc.compare_package(tampered, bound)
        fcc.compare_package(fcc.executed_package_digests(REPO), bound)

    def test_departure_from_the_document_is_refused(self) -> None:
        summary = self.evidence["recomputed_summary"]
        documented = {
            "anode_fall_coefficient": 2.0,
            "continuation_epsilons": tuple(fcc.CONTINUATION_EPSILONS),
            "continuation_floors": tuple(self.evidence["documented_summary"]["continuation_floors"]),
            "dm92_misfit": self.evidence["documented_summary"]["dm92_misfit"],
            "relaxed_root_depth_v": summary["relaxed_root_depth_v"],
            "jacobian_rank": 22,
            "jacobian_condition_max": 200.0,
        }
        ladder = [
            {
                "epsilon": eps, "floor": floor, "converged": False, "reason": "iteration_limit",
                "dominant_row": 27, "jacobian_rank": 22, "jacobian_condition": 20.0,
                "anode_only_converged": True, "anode_only_residual": 1e-13, "anode_only_phi4_minus_ua": 0.0,
            }
            for eps, floor in zip(fcc.CONTINUATION_EPSILONS, summary["continuation_floors"], strict=True)
        ]
        recomputed = {
            "closed_form_relative_difference": summary["closed_form_relative_difference"],
            "manifold_normalized_residual": summary["manifold_normalized_residual"],
            "anode_fall_coefficient": 2.0,
            "ladder": ladder,
            "dm92_misfit": summary["dm92_misfit"],
            "relaxed_root_depth_v": summary["relaxed_root_depth_v"],
            "relaxed_root_residual": 1e-16,
            "relaxed_root_feasible": False,
            "relaxed_root_anode_margin_v": -summary["relaxed_root_depth_v"],
        }
        departures = fcc.check_against_document(recomputed, documented, fcc.TOLERANCES)
        self.assertLessEqual(departures["continuation_max_relative_departure"], fcc.TOLERANCES["continuation_floor_relative"])
        # A floor that departs by more than the tolerance is refused.
        bad = copy.deepcopy(recomputed)
        bad["ladder"][2]["floor"] = documented["continuation_floors"][2] * 1.5
        with self.assertRaises(ValueError):
            fcc.check_against_document(bad, documented, fcc.TOLERANCES)
        # A closing interior rung (a branch) is refused.
        bad = copy.deepcopy(recomputed)
        bad["ladder"][0]["converged"] = True
        with self.assertRaises(ValueError):
            fcc.check_against_document(bad, documented, fcc.TOLERANCES)
        # A closed form that no longer agrees with the residual is refused.
        bad = copy.deepcopy(recomputed)
        bad["closed_form_relative_difference"] = 1e-6
        with self.assertRaises(ValueError):
            fcc.check_against_document(bad, documented, fcc.TOLERANCES)
        # A relaxed root that the admissible region accepts is refused.
        bad = copy.deepcopy(recomputed)
        bad["relaxed_root_feasible"] = True
        with self.assertRaises(ValueError):
            fcc.check_against_document(bad, documented, fcc.TOLERANCES)
        # A different Jacobian rank at a floor point is refused.
        bad = copy.deepcopy(recomputed)
        bad["ladder"][3]["jacobian_rank"] = 21
        with self.assertRaises(ValueError):
            fcc.check_against_document(bad, documented, fcc.TOLERANCES)

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-037")
        record["non_claims"].append("validated against the thruster")
        errors = self._analysis_errors(matrix=matrix)
        self.assertTrue(any("non-claim of CLM-037 is absent" in e for e in errors))
        for phrase in next(c for c in self.matrix["claims"] if c["id"] == "CLM-037")["non_claims"]:
            self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_relocated_or_misclassed_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-038")
        record["manifest_ids"] = ["L0-SWEEP-20260901-8192-V1"]
        record["allowed_locations"] = ["Abstract"]
        errors = self._analysis_errors(matrix=matrix)
        self.assertTrue(any("CLM-038 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-038 does not allow the section heading" in e for e in errors))
        # An interpretation may not be smuggled into the results section.
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-040")
        record["claim_class"] = "interpretation"
        errors = self._analysis_errors(matrix=matrix)
        self.assertTrue(any("interpretation claim CLM-040 must not appear inside the results section" in e for e in errors))
        # Dropping the labelled interpretation altogether is also rejected.
        matrix = copy.deepcopy(self.matrix)
        for claim_id in ("CLM-043", "CLM-044"):
            next(c for c in matrix["claims"] if c["id"] == claim_id)["claim_class"] = "commentary"
        errors = self._analysis_errors(matrix=matrix)
        self.assertTrue(any("labelled interpretation" in e for e in errors))

    def test_revision_macro_must_spell_the_analysis_revision(self) -> None:
        tampered = self.manuscript.replace("266d8a99\\allowbreak{}ce75fe35", "266d8a98\\allowbreak{}ce75fe35")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._analysis_errors(manuscript=tampered)
        self.assertTrue(any("does not spell the analysis revision" in e for e in errors))

    def test_section_binding_and_displayed_closed_form(self) -> None:
        duplicated = self.manuscript.replace(fcc.SECTION_BINDING, fcc.SECTION_BINDING + "\n" + fcc.SECTION_BINDING)
        errors = self._analysis_errors(manuscript=duplicated)
        self.assertTrue(any("occur exactly once" in e for e in errors))
        moved = self.manuscript.replace(fcc.GENERATED_BINDING + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + fcc.GENERATED_BINDING
        )
        errors = self._analysis_errors(manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))
        # The coefficient of the displayed closed form must be a macro, not a typed digit.
        typed = self.manuscript.replace("R_{\\FccGlobalRowIndex} = \\FccAnodeFallCoefficient\\,", "R_{\\FccGlobalRowIndex} = 2\\,")
        self.assertNotEqual(typed, self.manuscript)
        errors = self._analysis_errors(manuscript=typed)
        self.assertTrue(any("does not use \\FccAnodeFallCoefficient" in e or "not macro-bound" in e for e in errors))
        without = self.manuscript.replace("\\begin{equation}\n  R_{\\FccGlobalRowIndex}", "\\begin{align}\n  R_{\\FccGlobalRowIndex}")
        errors = self._analysis_errors(manuscript=without)
        self.assertTrue(any("does not display the closed form as an equation" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["paper_evidence_file"]["path"] = "paper/evidence/mdo-l0-v1.json"
        errors = self._analysis_errors(payload=payload)
        self.assertTrue(any("differs from the registered evidence file" in e for e in errors))

    def test_section_claims_resolve_under_their_headings(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for claim_id in ("CLM-037", "CLM-038", "CLM-040", "CLM-041", "CLM-042"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), HEADING)
        self.assertEqual(check_paper._heading_at(self.flattened, by_id["CLM-036"].start), "Abstract")
        for claim_id in ("CLM-043", "CLM-044"):
            self.assertEqual(check_paper._heading_at(self.flattened, by_id[claim_id].start), "Discussion")
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        self.assertNotIn("CLM-037", raw_ids)
        self.assertIn("CLM-043", raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        records = {c["id"]: c for c in self.matrix["claims"]}
        self.assertEqual(records["CLM-043"]["claim_class"], "interpretation")
        self.assertEqual(records["CLM-044"]["claim_class"], "interpretation")
        # The cusp topology admission bound CLM-044 to its manifest as well (cells exist under the literature definition).
        self.assertEqual(
            set(records["CLM-044"]["manifest_ids"]),
            {MANIFEST_ID, "WALL-LOSS-V4-20260902-4608-V1", "FOUR-CELL-V2-20260902-128-V1", "CUSP-TOPOLOGY-V3-1-20260903-281-V1"},
        )
        self.assertIn("Kornfeld2007", records["CLM-040"]["bibliography"])
        self.assertIn("SRC-AUDIT", records["CLM-043"]["evidence"])
        # The legacy-study reading is worded as interpretation and the audit's rule is what it says.
        self.assertIn("The reading that follows is interpretation", records["CLM-043"]["authorized_tex"])
        self.assertIn("\\FccAuditAcceptedFlags", records["CLM-043"]["authorized_tex"])
        self.assertIn("no numerical value of that run is claimed or recomputed", records["CLM-043"]["authorized_tex"])
        self.assertIn("as interpretation only", records["CLM-044"]["authorized_tex"])
        self.assertIn("whose results are not admitted here", records["CLM-044"]["authorized_tex"])

    def test_scope_claim_registers_the_required_non_claims(self) -> None:
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-037")
        joined = " ".join(record["non_claims"])
        for phrase in ("physical thruster", "proposed correction is right", "accepts no correction", "not of the solver", "not recomputed", "opens none"):
            self.assertIn(phrase, joined)
        scope = next(c for c in self.matrix["claims"] if c["id"] == "CLM-042")
        self.assertEqual(scope["claim_class"], "analytic-scope-limitation")
        self.assertIn("opens none of the field-resolved, coupled or experimental gates", scope["authorized_tex"])
        correction = next(c for c in self.matrix["claims"] if c["id"] == "CLM-041")
        self.assertEqual(correction["correction_status"], "PROPOSED_NOT_ACCEPTED")
        self.assertIn("is not accepted here", correction["authorized_tex"])
        self.assertIn(GATE_ID, {c["gate_registry_id"] for c in self.matrix["claims"] if c.get("status") == "evidence-gate"})
        self.assertIn("analytic_consistency_rule", self.matrix["policy"])

    def test_generated_tables_are_a_verified_contract_item(self) -> None:
        item = next(i for i in self.contract["items"] if i["id"] == fcc.ARTIFACT_ID)
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["required_gate"], GATE_ID)
        self.assertEqual(item["claim_ids"], [fcc.ARTIFACT_CLAIM_ID])
        self.assertEqual(item["artifact_claim_count"], 2)
        self.assertEqual(item["generator_module"], "generate_four_cell_closure_evidence")
        self.assertEqual(len(item["manuscript_labels"]), 2)
        record = next(c for c in self.matrix["claims"] if c["id"] == fcc.ARTIFACT_CLAIM_ID)
        self.assertEqual(record["authorized_artifact_ids"], [fcc.ARTIFACT_ID])
        self.assertNotIn("authorized_tex", record)

    def test_required_section_and_boundary_sentences_are_updated(self) -> None:
        self.assertIn("Consistency of the four-cell power balance", check_paper.REQUIRED_SECTIONS)
        self.assertIn("\\section{Consistency of the four-cell power balance}", self.manuscript)
        self.assertIn("\\label{eq:four-cell-closed-form}", self.manuscript)
        self.assertIn("four-cell power-balance closure analysis at Git revision \\ClosureEvidenceRevision", self.manuscript)
        normalized = check_paper._normalize_tex(self.manuscript)
        self.assertIn("accepts no correction of the ledger", normalized)
        self.assertIn("The reading that the legacy performance values were residual-floor artefacts is interpretation", normalized)
        self.assertIn("it accepts no correction of the ledger, and it leaves this gate closed", normalized)
        # Kornfeld is cited from the manuscript section (the macro-only section may type no digit).
        self.assertIn("\\cite{Kornfeld2007}", self.manuscript[self.manuscript.find("\\section{Consistency of the four-cell power balance}"):])


if __name__ == "__main__":
    unittest.main()
