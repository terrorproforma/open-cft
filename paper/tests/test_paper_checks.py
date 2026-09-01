"""Adversarial regression tests for the paper evidence boundary."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_tables  # noqa: E402


class PaperPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")

    def test_repository_passes_all_checks(self) -> None:
        self.assertEqual(check_paper.collect_errors(REPO), [])

    def test_experimental_accuracy_variants_are_rejected(self) -> None:
        variants = (
            "The model is experimentally accurate!",
            "Excellent agreement—with measurements.",
            "Validated, against experimental observations.",
            "Measured performance was accurately predicted.",
        )
        for prose in variants:
            with self.subTest(prose=prose):
                findings = check_paper.find_unregistered_claims(prose)
                self.assertIn(
                    "unregistered experimental accuracy or validation claim",
                    findings,
                )

    def test_cuda_performance_variants_are_rejected(self) -> None:
        variants = (
            "CUDA delivered a 10x acceleration.",
            "A ten-fold GPU speedup was measured.",
            "GPU: 10 × faster.",
            "Throughput on CUDA was dramatically faster.",
        )
        for prose in variants:
            with self.subTest(prose=prose):
                findings = check_paper.find_unregistered_claims(prose)
                self.assertIn("unregistered GPU/CUDA performance claim", findings)

    def test_existing_id_cannot_authorize_different_text(self) -> None:
        macros = check_paper.extract_macros(self.manuscript, "EvidenceClaim", 2)
        target = next(macro for macro in macros if macro.arguments[0] == "CLM-007")
        replacement = (
            r"\EvidenceClaim{CLM-007}{CUDA delivered a 10x acceleration "
            r"with experimental accuracy.}"
        )
        tampered = (
            self.manuscript[: target.start]
            + replacement
            + self.manuscript[target.end :]
        )
        citation_keys = {
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", tampered)
            for key in group.split(",")
        }
        errors: list[str] = []
        check_paper._check_claims(REPO, tampered, citation_keys, errors)
        self.assertTrue(
            any("claim CLM-007 body is not authorized" in error for error in errors)
        )

    def test_detached_existing_id_is_rejected(self) -> None:
        tampered = self.manuscript + "\nCLM-007\n"
        citation_keys = {
            key.strip()
            for group in re.findall(r"\\cite\{([^}]+)\}", tampered)
            for key in group.split(",")
        }
        errors: list[str] = []
        check_paper._check_claims(REPO, tampered, citation_keys, errors)
        self.assertTrue(
            any("claim ID appears outside a structured claim" in error for error in errors)
        )

    def test_readme_cannot_masquerade_as_l1_manifest(self) -> None:
        registry = json.loads(
            (REPO / "paper/evidence/result-gates.json").read_text(encoding="utf-8")
        )
        gate = next(g for g in registry["gates"] if g["id"] == "GATE-L1")
        payload = {
            "document_type": "paper-L1-result-manifest",
            "schema_version": "1.0",
            "level": "L1",
            "status": "accepted",
            "manifest_id": "fake-readme",
            "evidence_revision": registry["evidence_revision"],
            "source_files": [],
            "metrics": {},
        }
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO,
            registry["evidence_revision"],
            gate,
            payload,
            Path("paper/README.md"),
            errors,
            require_committed=False,
        )
        self.assertTrue(any("paper/evidence/manifests" in error for error in errors))

    def test_deadbeef_manifest_revision_is_rejected(self) -> None:
        registry = json.loads(
            (REPO / "paper/evidence/result-gates.json").read_text(encoding="utf-8")
        )
        gate = next(g for g in registry["gates"] if g["id"] == "GATE-L1")
        payload = {
            "document_type": "paper-L1-result-manifest",
            "schema_version": "1.0",
            "level": "L1",
            "status": "accepted",
            "manifest_id": "fake-deadbeef",
            "evidence_revision": "deadbeef",
            "source_files": [],
            "metrics": {},
        }
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO,
            registry["evidence_revision"],
            gate,
            payload,
            Path("paper/evidence/manifests/fake.json"),
            errors,
            require_committed=False,
        )
        self.assertTrue(any("resolvable 40-hex commit" in error for error in errors))

    def test_wrong_manifest_type_and_missing_metrics_are_rejected(self) -> None:
        registry = json.loads(
            (REPO / "paper/evidence/result-gates.json").read_text(encoding="utf-8")
        )
        gate = next(g for g in registry["gates"] if g["id"] == "GATE-L1")
        payload = {
            "document_type": "paper-L3-result-manifest",
            "schema_version": "1.0",
            "level": "L3",
            "status": "accepted",
            "manifest_id": "wrong-type",
            "evidence_revision": registry["evidence_revision"],
            "source_files": [],
            "metrics": {},
        }
        errors: list[str] = []
        check_paper._validate_manifest_payload(
            REPO,
            registry["evidence_revision"],
            gate,
            payload,
            Path("paper/evidence/manifests/wrong.json"),
            errors,
            require_committed=False,
        )
        self.assertTrue(any("unrecognized document_type" in error for error in errors))
        self.assertTrue(any("required metric" in error for error in errors))
        self.assertTrue(any("missing required file roles" in error for error in errors))

    def test_generated_table_and_sidecar_match_generator(self) -> None:
        table, sidecar = generate_tables.render(REPO)
        self.assertEqual(
            (REPO / generate_tables.OUTPUT_PATH).read_bytes(),
            table,
        )
        self.assertEqual(
            (REPO / generate_tables.SIDECAR_PATH).read_bytes(),
            generate_tables.canonical_json(sidecar),
        )

    def test_manual_table_change_would_invalidate_hash(self) -> None:
        table, sidecar = generate_tables.render(REPO)
        changed = table.replace(b"0.00188384225", b"0.00188384226")
        self.assertNotEqual(
            check_paper.sha256_bytes(changed),
            sidecar["output"]["sha256"],
        )

    def test_author_and_human_submission_gates(self) -> None:
        gates = json.loads(
            (REPO / "paper/evidence/submission-gates.json").read_text(encoding="utf-8")
        )
        records = {gate["id"]: gate for gate in gates["gates"]}
        self.assertEqual(records["AUTHOR-IDENTITY"]["value"], "Angus Muffatti")
        for gate_id in (
            "COAUTHOR-APPROVAL",
            "CONTRIBUTION-STATEMENT-APPROVAL",
            "AFFILIATION-APPROVAL",
            "CORRESPONDING-AUTHOR-APPROVAL",
        ):
            self.assertEqual(records[gate_id]["status"], "human-approval-required")


if __name__ == "__main__":
    unittest.main()
