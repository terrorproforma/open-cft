"""Adversarial tests for the admission of the three topology-screening studies into the manuscript."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_topology_screening_evidence as topo  # noqa: E402


def _json(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


class TopologyScreeningAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _json("paper/evidence/result-gates.json")
        cls.matrix = _json("paper/evidence/claims.json")
        cls.contract = _json("paper/evidence/figure-table-contract.json")
        cls.gates = {spec.key: next(g for g in cls.registry["gates"] if g["id"] == spec.gate_id) for spec in topo.EXPERIMENTS.values()}
        cls.payloads = {key: _json(gate["manifest_path"]) for key, gate in cls.gates.items()}
        cls.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        cls.evidence = {key: _json(spec.evidence_path.as_posix()) for key, spec in topo.EXPERIMENTS.items()}
        errors: list[str] = []
        cls.flattened = check_paper.flatten_sections(REPO, cls.manuscript, errors)
        assert errors == []

    def _errors(self, key: str, *, gate=None, payload=None, manuscript=None, flattened=None, matrix=None):
        errors: list[str] = []
        check_paper._check_topology_screening(
            REPO,
            gate if gate is not None else self.gates[key],
            payload if payload is not None else self.payloads[key],
            manuscript if manuscript is not None else self.manuscript,
            flattened if flattened is not None else self.flattened,
            matrix if matrix is not None else self.matrix,
            errors,
        )
        return errors

    def test_gates_are_numerical_screening_gates_that_open_no_level(self) -> None:
        kinds = self.registry["acceptance_policy"]["gate_kinds"]
        self.assertEqual(set(kinds), check_paper.KNOWN_GATE_KINDS)
        self.assertIn("never opens GATE-L1", kinds["numerical-screening"])
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                gate, payload = self.gates[key], self.payloads[key]
                self.assertEqual(gate["kind"], check_paper.SCREENING_GATE_KIND)
                self.assertEqual(gate["status"], "accepted")
                self.assertIsNone(gate["opens_level"])
                self.assertEqual(gate["recorded_outcome"], spec.recorded_outcome)
                self.assertIn(gate["recorded_outcome"], check_paper.SCREENING_OUTCOMES)
                self.assertEqual(gate["required_manifest_document_type"], "paper-l1a-screening-manifest")
                self.assertIn(spec.manifest_id, self.matrix["manifests"])
                self.assertEqual(self.matrix["manifests"][spec.manifest_id]["recorded_outcome"], spec.recorded_outcome)
                self.assertEqual(payload["manifest_id"], spec.manifest_id)
                self.assertEqual(payload["level"], "numerical-screening")
                self.assertEqual(payload["gate_kind"], "numerical-screening")
                self.assertEqual(payload["recorded_outcome"], spec.recorded_outcome)
                self.assertEqual(payload["classification"], spec.classification)
                self.assertEqual(payload["evidence_revision"], spec.results_commit)
                self.assertEqual(payload["preregistration_revision"], spec.preregistration_commit)
                self.assertIsNone(payload["evidence_level"]["opens_gate"])
                for level in ("L1", "L2", "L3"):
                    self.assertIn(level, payload["evidence_level"]["relation_to_levels"])
                self.assertFalse(payload["metrics"]["stable_multicell_wall_cusp_topology_demonstrated"])
                self.assertFalse(payload["metrics"]["permanent_magnet_material_model"])
                gate_record = next(c for c in self.matrix["claims"] if c["id"] == spec.gate_id)
                self.assertEqual(gate_record["kind"], "numerical-screening")
                self.assertEqual(gate_record["recorded_outcome"], spec.recorded_outcome)

    def test_physics_level_gates_remain_closed(self) -> None:
        for gate_id in sorted(check_paper.PHYSICS_GATE_IDS):
            gate = next(g for g in self.registry["gates"] if g["id"] == gate_id)
            self.assertEqual(gate["kind"], check_paper.PHYSICS_GATE_KIND)
            self.assertEqual(gate["status"], "closed")
            self.assertIsNone(gate["manifest_path"])
        visible = {m.arguments[0] for m in check_paper.extract_macros(self.flattened, "EvidenceGate", 2)}
        self.assertEqual(visible, set(check_paper.PHYSICS_GATE_IDS))

    def test_manifests_validate_as_typed_gate_manifests(self) -> None:
        for key in topo.EXPERIMENTS:
            with self.subTest(study=key):
                errors: list[str] = []
                check_paper._validate_manifest_payload(
                    REPO, self.registry["evidence_revision"], self.gates[key], self.payloads[key],
                    Path(self.gates[key]["manifest_path"]), errors, require_committed=False,
                )
                self.assertEqual(errors, [])
                wrong = copy.deepcopy(self.payloads[key])
                wrong["level"] = "L1"
                errors = []
                check_paper._validate_manifest_payload(
                    REPO, self.registry["evidence_revision"], self.gates[key], wrong,
                    Path(self.gates[key]["manifest_path"]), errors, require_committed=False,
                )
                self.assertTrue(any("level does not match" in error for error in errors))
                schema = check_paper.EXPECTED_MANIFEST_TYPES["paper-l1a-screening-manifest"]
                self.assertTrue(set(schema["required_metrics"]) <= set(self.payloads[key]["metrics"]))
                self.assertTrue(set(schema["required_file_roles"]) <= {s["role"] for s in self.payloads[key]["source_files"]})

    def test_manifest_metrics_equal_the_raw_artifact_values(self) -> None:
        for key in topo.EXPERIMENTS:
            raw = {item["name"]: item["raw"] for item in self.evidence[key]["macros"]}
            for metric, macro in check_paper.SCREENING_METRIC_MACROS[key].items():
                with self.subTest(study=key, metric=metric):
                    self.assertEqual(self.payloads[key]["metrics"][metric], raw[macro])
            for metric, expected in check_paper.SCREENING_POLICY_METRICS.items():
                self.assertIs(self.payloads[key]["metrics"][metric], expected)

    def test_screening_checker_accepts_the_committed_state(self) -> None:
        for key in topo.EXPERIMENTS:
            with self.subTest(study=key):
                self.assertEqual(self._errors(key), [])

    def test_tampered_metric_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payloads["four-cell-v2"])
        payload["metrics"]["stable_count"] = 1
        payload["metrics"]["gpu_replay_pass_count"] = 4
        errors = self._errors("four-cell-v2", payload=payload)
        self.assertTrue(any("metric 'stable_count' differs" in e for e in errors))
        self.assertTrue(any("metric 'gpu_replay_pass_count' differs" in e for e in errors))
        payload = copy.deepcopy(self.payloads["l1a-sweep-v2"])
        payload["metrics"]["axis_cusp_count_maximum"] = 5  # int where the artifact holds 5.0
        errors = self._errors("l1a-sweep-v2", payload=payload)
        self.assertTrue(any("metric 'axis_cusp_count_maximum' differs" in e for e in errors))

    def test_policy_metric_and_outcome_tampering_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payloads["topology-characterization-v1"])
        payload["metrics"]["stable_multicell_wall_cusp_topology_demonstrated"] = True
        errors = self._errors("topology-characterization-v1", payload=payload)
        self.assertTrue(any("policy metric 'stable_multicell_wall_cusp_topology_demonstrated'" in e for e in errors))
        gate = copy.deepcopy(self.gates["four-cell-v2"])
        gate["recorded_outcome"] = "accepted-screening"
        errors = self._errors("four-cell-v2", gate=gate)
        self.assertTrue(any("recorded_outcome differs" in e for e in errors))
        gate = copy.deepcopy(self.gates["four-cell-v2"])
        gate["opens_level"] = "L1"
        errors = self._errors("four-cell-v2", gate=gate)
        self.assertTrue(any("cannot open a physics level" in e for e in errors))

    def test_classification_must_agree_everywhere(self) -> None:
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                gate = copy.deepcopy(self.gates[key])
                gate["metric_constraints"]["classification"]["equals"] = "something_else"
                errors = self._errors(key, gate=gate)
                self.assertTrue(any("classification differs" in e for e in errors))
                rendered = next(m["value"] for m in self.evidence[key]["macros"] if m["name"] == f"{spec.macro_prefix}Classification")
                self.assertEqual(check_paper.tex_unescape(rendered), spec.classification)

    def test_missing_non_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-022")
        record["non_claims"].append("proves that no four-cell design exists")
        errors = self._errors("four-cell-v2", matrix=matrix)
        self.assertTrue(any("non-claim of CLM-022 is absent" in e for e in errors))
        for claim_id in ("CLM-019", "CLM-022", "CLM-025"):
            for phrase in next(c for c in self.matrix["claims"] if c["id"] == claim_id)["non_claims"]:
                self.assertIn(check_paper._normalize_tex(phrase), check_paper._normalize_tex(self.flattened))

    def test_unbound_or_relocated_claim_is_rejected(self) -> None:
        matrix = copy.deepcopy(self.matrix)
        record = next(c for c in matrix["claims"] if c["id"] == "CLM-025")
        record["manifest_ids"] = ["L0-SWEEP-20260901-8192-V1"]
        record["allowed_locations"] = ["Abstract"]
        errors = self._errors("topology-characterization-v1", matrix=matrix)
        self.assertTrue(any("CLM-025 is not bound to manifest" in e for e in errors))
        self.assertTrue(any("CLM-025 does not allow the section heading" in e for e in errors))

    def test_revision_macro_must_spell_the_manifest_revision(self) -> None:
        tampered = self.manuscript.replace("7120e8ed\\allowbreak{}cb74c02c", "7120e8ef\\allowbreak{}cb74c02c")
        self.assertNotEqual(tampered, self.manuscript)
        errors = self._errors("four-cell-v2", manuscript=tampered)
        self.assertTrue(any("does not spell the manifest revision" in e for e in errors))

    def test_section_binding_must_occur_exactly_once(self) -> None:
        spec = topo.SWEEP
        duplicated = self.manuscript.replace(spec.section_binding, spec.section_binding + "\n" + spec.section_binding)
        errors = self._errors(spec.key, manuscript=duplicated)
        self.assertTrue(any("occur exactly once in manuscript.tex" in e for e in errors))
        moved = self.manuscript.replace(spec.generated_binding + "\n", "").replace(
            "\\begin{document}", "\\begin{document}\n" + spec.generated_binding
        )
        errors = self._errors(spec.key, manuscript=moved)
        self.assertTrue(any("input exactly once in the preamble" in e for e in errors))

    def test_evidence_file_substitution_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payloads["l1a-sweep-v2"])
        payload["paper_evidence_file"]["path"] = "paper/evidence/four-cell-v2.json"
        errors = self._errors("l1a-sweep-v2", payload=payload)
        self.assertTrue(any("paper_evidence_file.path differs" in e for e in errors))
        payload = copy.deepcopy(self.payloads["l1a-sweep-v2"])
        payload["experiment_id"] = topo.FOUR_CELL.experiment_id
        errors = self._errors("l1a-sweep-v2", payload=payload)
        self.assertTrue(any("not a registered screening study of this gate" in e for e in errors))

    def test_eol_tolerance_binding_is_checked_against_the_audit_modules(self) -> None:
        for spec in (topo.SWEEP, topo.FOUR_CELL):
            with self.subTest(study=spec.key):
                self.assertEqual(check_paper._eol_audited_file_errors(REPO, spec, "test"), [])
                (relative, audited), = spec.audited_eol_files.items()
                module = (REPO / audited.audit_module).read_text(encoding="utf-8")
                self.assertIn(audited.lf_sha256, module)
                self.assertIn(audited.recorded_sha256, module)
                fake = copy.deepcopy(spec)
                broken = topo.AuditedEolFile(
                    lf_sha256=audited.lf_sha256, recorded_sha256="0" * 64,
                    recorded_bytes=audited.recorded_bytes, audit_module=audited.audit_module,
                )
                fake = topo.ExperimentSpec(**{**spec.__dict__, "audited_eol_files": {relative: broken}})
                errors = check_paper._eol_audited_file_errors(REPO, fake, "test")
                self.assertTrue(any("does not reproduce the recorded CRLF digest" in e for e in errors))
                self.assertTrue(any("does not bind the audited digests" in e for e in errors))
        payload = copy.deepcopy(self.payloads["four-cell-v2"])
        payload["results_bundle"]["tolerated_eol_files"] = []
        errors = self._errors("four-cell-v2", payload=payload)
        self.assertTrue(any("tolerated end-of-line file list differs" in e for e in errors))

    def test_lineage_files_are_bound_but_never_evidence(self) -> None:
        payload = self.payloads["four-cell-v2"]
        roles = {entry["role"] for entry in payload["lineage_files"]}
        self.assertEqual(roles, {"lineage-superseded-search", "lineage-failed-validation"})
        self.assertEqual(
            {entry["path"] for entry in payload["lineage_files"]},
            set(self.evidence["four-cell-v2"]["lineage_artifacts"]["files"]),
        )
        record = next(c for c in self.matrix["claims"] if c["id"] == "CLM-024")
        self.assertIn("are lineage only and are not evidence", record["authorized_tex"])
        tampered = copy.deepcopy(payload)
        tampered["lineage_files"] = tampered["lineage_files"][1:]
        errors = self._errors("four-cell-v2", payload=tampered)
        self.assertTrue(any("lineage files differ" in e for e in errors))
        self.assertNotIn("lineage_files", self.payloads["l1a-sweep-v2"])
        self.assertNotIn("lineage_files", self.payloads["topology-characterization-v1"])

    def test_section_claims_resolve_under_their_headings(self) -> None:
        macros = check_paper.extract_macros(self.flattened, "EvidenceClaim", 2)
        by_id = {m.arguments[0]: m for m in macros}
        for spec in topo.EXPERIMENTS.values():
            for claim_id in spec.prose_claim_ids:
                location = check_paper._heading_at(self.flattened, by_id[claim_id].start)
                if claim_id == "CLM-018":
                    self.assertEqual(location, "Abstract")
                elif claim_id == "CLM-028":
                    self.assertEqual(location, "Discussion")
                else:
                    self.assertEqual(location, spec.section_heading, claim_id)
        raw_ids = {m.arguments[0] for m in check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)}
        for claim_id in ("CLM-019", "CLM-022", "CLM-025"):
            self.assertNotIn(claim_id, raw_ids)
        self.assertEqual(check_paper.find_unregistered_claims(self.flattened), [])
        # the Discussion now cites the nulls as evidence rather than as an open question
        discussion = self.flattened[self.flattened.find("\\section{Discussion}"):self.flattened.find("\\section{Limitations}")]
        self.assertIn("\\EvidenceClaim{CLM-028}", discussion)
        self.assertNotIn("not admitted to", discussion)
        self.assertIn("not proof that no such design exists", discussion)

    def test_generated_tables_are_verified_contract_items(self) -> None:
        for spec in topo.EXPERIMENTS.values():
            with self.subTest(study=spec.key):
                item = next(i for i in self.contract["items"] if i["id"] == spec.artifact_id)
                self.assertEqual(item["status"], "verified")
                self.assertEqual(item["required_gate"], spec.gate_id)
                self.assertEqual(item["claim_ids"], [spec.artifact_claim_id])
                self.assertEqual(item["artifact_claim_count"], len(spec.table_macros))
                self.assertEqual(item["generator_module"], "generate_topology_screening_evidence")
                self.assertEqual(item["evidence_file"], spec.evidence_path.as_posix())
                record = next(c for c in self.matrix["claims"] if c["id"] == spec.artifact_claim_id)
                self.assertEqual(record["authorized_artifact_ids"], [spec.artifact_id])
                self.assertNotIn("authorized_tex", record)
                output, sidecar = check_paper._render_topology_screening_tables(REPO, item)
                self.assertEqual((REPO / item["output_path"]).read_bytes(), output)
                self.assertEqual((REPO / item["sidecar_path"]).read_bytes(), sidecar)


if __name__ == "__main__":
    unittest.main()
