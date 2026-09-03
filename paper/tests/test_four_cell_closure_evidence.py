"""Regression tests for the bound and recomputed four-cell closure paper evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_four_cell_closure_evidence as fcc  # noqa: E402

EVIDENCE = REPO / fcc.EVIDENCE_PATH
GENERATED = REPO / fcc.OUTPUT_PATH
SIDECAR = REPO / fcc.SIDECAR_PATH
SECTION = REPO / fcc.SECTION_PATH
STANDALONE = REPO / "paper/sections/four-cell-closure-standalone.tex"


def _git_show(revision: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=REPO, check=True, capture_output=True).stdout


class FourCellClosureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = fcc.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.document = _git_show(fcc.ANALYSIS_COMMIT_SHA, fcc.DOCUMENT.as_posix()).decode("utf-8")

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = fcc.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE, REPO / "paper/scripts/generate_four_cell_closure_evidence.py"):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_analysis_revision_and_the_executed_package(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-four-cell-closure-evidence")
        self.assertEqual(self.evidence["evidence_revision"], fcc.ANALYSIS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], fcc.CLASSIFICATION)
        self.assertEqual(self.evidence["correction_status"], "PROPOSED_NOT_ACCEPTED")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        for commit in (fcc.ANALYSIS_COMMIT_SHA, fcc.VERIFIED_TREE_COMMIT_SHA, fcc.MDO_PREREGISTRATION_COMMIT_SHA):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=False).returncode, 0)
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", fcc.ANALYSIS_COMMIT_SHA, fcc.VERIFIED_TREE_COMMIT_SHA], cwd=REPO, check=False).returncode, 0)
        sources = {s["path"]: s for s in self.evidence["sources"]}
        self.assertEqual(set(sources), set(fcc.SOURCE_ROLES))
        for path, source in sources.items():
            self.assertEqual(source["role"], fcc.SOURCE_ROLES[path])
            blob = subprocess.run(["git", "rev-parse", f"{fcc.ANALYSIS_COMMIT_SHA}:{path}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(source["git_blob"], blob, path)
            content = _git_show(fcc.ANALYSIS_COMMIT_SHA, path)
            self.assertEqual(source["git_blob_sha256"], hashlib.sha256(content).hexdigest(), path)
            self.assertEqual(source["bytes"], len(content), path)
            later = subprocess.run(["git", "rev-parse", f"{fcc.VERIFIED_TREE_COMMIT_SHA}:{path}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(later, blob, path)
        frozen = subprocess.run(["git", "rev-parse", f"{fcc.MDO_PREREGISTRATION_COMMIT_SHA}:{fcc.PROTOCOL.as_posix()}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(sources[fcc.PROTOCOL.as_posix()]["git_blob"], frozen)
        # The legacy lineage blob is the one recorded in the task brief and the manifest.
        self.assertEqual(sources[fcc.LEGACY.as_posix()]["git_blob"], "8eeca9c61cce5b1f9157b7db5a7c9e6c21d63d80")
        executed = {entry["path"]: entry for entry in self.evidence["executed_package"]["files"]}
        self.assertEqual(set(executed), {(fcc.PACKAGE_DIR / name).as_posix() for name in fcc.PACKAGE_FILES})
        for path, entry in executed.items():
            on_disk = hashlib.sha256((REPO / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            self.assertEqual(entry["sha256_lf"], on_disk)
            self.assertEqual(entry["git_blob_sha256"], on_disk)
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], fcc.GATE_ID)
        self.assertEqual(integration["gate_kind"], "analytic-consistency")
        self.assertEqual(integration["manifest_id"], fcc.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], fcc.SECTION_HEADING)
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(fcc.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(fcc.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(fcc.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_traces_to_a_bound_file_or_a_recorded_recomputation(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 150)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        artifacts = self.evidence["artifacts"]
        ledger = json.loads(_git_show(fcc.ANALYSIS_COMMIT_SHA, fcc.LEDGER.as_posix()))
        protocol = json.loads(_git_show(fcc.ANALYSIS_COMMIT_SHA, fcc.PROTOCOL.as_posix()))
        recomputed_count = 0
        documented_count = 0
        for item in macros:
            with self.subTest(macro=item["name"]):
                self.assertTrue(item["name"].isalpha())
                self.assertTrue(item["name"].startswith("Fcc"))
                self.assertEqual(fcc.format_value(item["format"], item["raw"]), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
                if item["derived"]:
                    self.assertTrue(item["derivation"])
                    self.assertTrue(item["inputs"])
                    for source in item["inputs"]:
                        self.assertIn(source["artifact"], artifacts)
                    if item["recomputed"]:
                        recomputed_count += 1
                        self.assertTrue(any(src["artifact"].startswith(fcc.PACKAGE_DIR.as_posix()) or src["artifact"] == fcc.LEGACY.as_posix() for src in item["inputs"]))
                    continue
                documented_count += 1
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                pointer = source["pointer"]
                if pointer.startswith("regex:"):
                    name = pointer[len("regex:"):]
                    if name.startswith("MDO_PROBE_PATTERN"):
                        group = name[name.index("[") + 1:name.index("]")]
                        text = fcc.resolve_pointer(protocol, name[name.index("@") + 1:])
                        match = fcc.MDO_PROBE_PATTERN.search(text)
                        self.assertIsNotNone(match)
                        self.assertEqual(item["raw"], type(item["raw"])(match.group(group)))
                    elif name.startswith("LEDGER_CLOSED_FORM"):
                        match = fcc.LEDGER_CLOSED_FORM.search(ledger["global_row_consistency"]["closed_form_on_manifold"])
                        self.assertEqual(item["raw"], int(match.group("coefficient")))
                    else:
                        pattern_name, group = name[:-1].split("[")
                        value = fcc.document_value(self.document, pattern_name, group)
                        if isinstance(item["raw"], list):
                            self.assertEqual(item["raw"], [float(x) for x in value.split(",")])
                        elif isinstance(item["raw"], int):
                            self.assertEqual(item["raw"], int(value.replace(" ", "")))
                        elif isinstance(item["raw"], float):
                            self.assertEqual(item["raw"], float(value.replace(" ", "")))
                        else:
                            self.assertEqual(item["raw"], value)
                else:
                    self.assertEqual(source["artifact"], fcc.LEDGER.as_posix())
                    self.assertEqual(fcc.resolve_pointer(ledger, pointer), item["raw"])
        self.assertGreater(recomputed_count, 30)
        self.assertGreater(documented_count, 60)

    def test_recomputed_values_reproduce_the_analysis_document(self) -> None:
        b = self.by_name
        summary = self.evidence["recomputed_summary"]
        documented = self.evidence["documented_summary"]
        # Closed form against the full residual: below the declared bound and of the documented order.
        self.assertLessEqual(b["FccClosedFormRelDiff"]["raw"], fcc.TOLERANCES["closed_form_relative_difference_upper_bound"])
        self.assertLess(b["FccClosedFormRelDiff"]["raw"], 10 * b["FccDocClosedFormRelDiff"]["raw"])
        self.assertEqual(b["FccDocClosedFormRelDiff"]["raw"], 1.9e-13)
        self.assertEqual(b["FccClosedFormSamples"]["raw"], 400)
        self.assertLessEqual(b["FccManifoldMaxResidual"]["raw"], 1e-11)
        self.assertEqual(b["FccAnodeFallCoefficient"]["raw"], 2.0)
        self.assertEqual(b["FccLedgerCoefficient"]["raw"], 2)
        # Continuation ladder: documented and recomputed floors within the declared tolerance, linear in eps.
        self.assertEqual(documented["continuation_floors"], [1.28e-6, 1.28e-5, 1.30e-4, 3.99e-4, 1.43e-3, 5.78e-3])
        self.assertEqual(len(summary["continuation_floors"]), 6)
        for recomputed, doc in zip(summary["continuation_floors"], documented["continuation_floors"], strict=True):
            self.assertLessEqual(abs(recomputed - doc) / doc, fcc.TOLERANCES["continuation_floor_relative"])
        slopes = [f / e for f, e in zip(summary["continuation_floors"], fcc.CONTINUATION_EPSILONS, strict=True)]
        self.assertLessEqual(max(slopes) / min(slopes), fcc.TOLERANCES["continuation_slope_spread_maximum"])
        self.assertEqual(summary["continuation_reasons"], ["iteration_limit"] * 6)
        self.assertEqual(summary["continuation_dominant_rows"], [27] * 6)
        self.assertEqual(summary["jacobian_ranks"], [22] * 6)
        self.assertIs(b["FccBranchFound"]["raw"], False)
        self.assertEqual(b["FccAnodeOnlyClosed"]["raw"], 6)
        self.assertTrue(all(r <= fcc.RESIDUAL_TOLERANCE for r in summary["anode_only_residuals"]))
        self.assertEqual(b["FccAnodeOnlyPhiFourGap"]["raw"], 0.0)
        # Published-state misfit and relaxed root reproduce the document.
        self.assertEqual(b["FccDocDmMisfit"]["raw"], 1.47e-3)
        self.assertAlmostEqual(b["FccDmMisfit"]["raw"], 1.47e-3, delta=0.02 * 1.47e-3)
        self.assertEqual(b["FccLedgerDmMisfit"]["raw"], 0.0014866093499999807)
        self.assertEqual(b["FccDocRelaxedDepthThreeHundred"]["raw"], 1.18)
        self.assertAlmostEqual(b["FccRelaxedDepth"]["raw"], 1.18, delta=0.02 * 1.18)
        self.assertIs(b["FccRelaxedFeasible"]["raw"], False)
        self.assertLessEqual(b["FccRelaxedResidual"]["raw"], 1e-10)
        # Probe: read from the frozen protocol, reproduced by the document, not recomputed.
        self.assertEqual((b["FccProbeClosed"]["raw"], b["FccProbeTotal"]["raw"]), (13, 80))
        self.assertEqual((b["FccDocProbeClosed"]["raw"], b["FccDocProbeTotal"]["raw"]), (13, 80))
        self.assertEqual(b["FccProbeSource"]["raw"], "mdo-protocol-disclosure")
        self.assertIn("the 80-case solver probe", " ".join(self.evidence["recomputation_protocol"]["not_recomputed"]))
        # Global search: documented only.
        self.assertEqual(b["FccDocDeEvaluations"]["raw"], 205312)
        self.assertEqual(b["FccDocDeBest"]["raw"], 2.06e-2)
        self.assertEqual((b["FccDocLmStarts"]["raw"], b["FccDocLmClosed"]["raw"]), (200, 0))
        self.assertEqual(b["FccDocLmFloorMin"]["raw"], 1.82e-3)
        self.assertEqual((b["FccDocRelaxedDepthMin"]["raw"], b["FccDocRelaxedDepthMax"]["raw"]), (0.6, 12.6))
        # Attribution and correction status.
        self.assertEqual(b["FccKornfeldAssumption"]["raw"], 8)
        self.assertEqual(b["FccKornfeldId"]["raw"], "IEPC-2007-108")
        self.assertEqual((b["FccLegacyCuspLine"]["raw"], b["FccLegacyAnodeLine"]["raw"], b["FccLegacyIeTerms"]["raw"]), (136, 137, 3))
        self.assertEqual(b["FccDocLegacyLine"]["raw"], 137)
        self.assertEqual(b["FccCorrectionStatus"]["raw"], "PROPOSED_NOT_ACCEPTED")
        self.assertEqual((b["FccDocCorrectedRankBefore"]["raw"], b["FccDocCorrectedRankAfter"]["raw"], b["FccDocCorrectedNullity"]["raw"]), (22, 21, 4))
        self.assertEqual((b["FccDocZeroCuspBefore"]["raw"], b["FccDocZeroCuspAfter"]["raw"], b["FccDocZeroCuspTotal"]["raw"]), (13, 16, 16))
        self.assertEqual(b["FccAuditAcceptedFlags"]["raw"], "1-3")
        self.assertEqual(b["FccAuditRejectedFlag"]["raw"], 4)
        self.assertEqual(b["FccAuditTolFun"]["raw"], "1e-50")
        # Rendered forms.
        self.assertEqual(b["FccClosedFormRelDiff"]["value"], "$2.0\\times10^{-13}$")
        self.assertEqual(b["FccDocFloorMin"]["value"], "$1.28\\times10^{-6}$")
        self.assertEqual(b["FccFloorDepartureMax"]["value"], "6\\%")
        self.assertEqual(b["FccRelaxedDepth"]["value"], "1.18")

    def test_legacy_blob_carries_the_same_two_power_row_terms(self) -> None:
        legacy = _git_show(fcc.ANALYSIS_COMMIT_SHA, fcc.LEGACY.as_posix()).decode("utf-8", errors="replace").splitlines()
        cusp = legacy[self.by_name["FccLegacyCuspLine"]["raw"] - 1]
        anode = legacy[self.by_name["FccLegacyAnodeLine"]["raw"] - 1]
        self.assertTrue(fcc.LEGACY_CUSP_LINE.match(cusp))
        self.assertTrue(fcc.LEGACY_ANODE_LINE.match(anode))
        self.assertEqual(cusp.count("+IE"), 3)
        self.assertIn("(x(9)-Ua+x(13))", anode)
        # The executable ledger still carries +EI in Pcusp and the proposal drops it.
        ledger = json.loads(_git_show(fcc.ANALYSIS_COMMIT_SHA, fcc.LEDGER.as_posix()))
        cusp_expression = next(p for p in ledger["power_expressions"] if p["id"] == "Pcusp")["expression"]
        self.assertIn("+EI", cusp_expression.replace(" ", ""))
        proposals = {p["id"]: p for p in ledger["global_row_consistency"]["proposed_corrections"]}
        self.assertNotIn("+EI", proposals["Pcusp"]["proposed_expression"].replace(" ", ""))
        self.assertIn("Ua-phi_4+T4", proposals["Panode_e"]["proposed_expression"].replace(" ", ""))

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Fcc[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Fcc[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        for table in fcc.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Fcc"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n13 of 80 cases\n", "Fcc"), ["1", "3", "8", "0"])
        self.assertIn(f"\\subsection{{{fcc.SECTION_HEADING}}}", self.section)
        for heading in ("Derivation and numerical verification.", "Solution sub-region, continuation and global search.", "Attribution.", "Proposed correction (not accepted).", "Scope."):
            self.assertIn(f"\\paragraph{{{heading}}}", self.section)
        self.assertIn("Claim boundary", self.section)
        self.assertNotIn("\\cite{", self.section)
        for pattern in check_paper.PLACEHOLDERS.values():
            self.assertIsNone(pattern.search(self.section))
            self.assertIsNone(pattern.search(self.tex))
        for pattern in check_paper.FORBIDDEN_MODEL_WORDING.values():
            self.assertIsNone(pattern.search(self.section))
        self.assertEqual(check_paper.find_unregistered_claims(self.section), [])
        self.assertEqual(check_paper.find_unregistered_claims(self.tex), [])
        self.assertNotIn("CLM-", "\n".join(line for line in self.section.splitlines() if line.lstrip().startswith("%")))

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in fcc.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        summary = self.evidence["recomputed_summary"]
        documented = self.evidence["documented_summary"]
        for eps, doc, interior, rec in zip(fcc.CONTINUATION_EPSILONS, documented["continuation_floors"], documented["continuation_interior_floors"], summary["continuation_floors"], strict=True):
            row = f"{eps:g} & {fcc.format_value('sci2', doc)} & {fcc.format_value('sci2', interior)} & {fcc.format_value('sci2', rec)} & "
            self.assertIn(row, self.tex)
        self.assertIn("205{,}312 evaluations", self.tex)
        self.assertIn("13 / 80 & frozen protocol", self.tex)
        self.assertIn("& documented\\\\", self.tex)
        self.assertIn("& recomputed\\\\", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], fcc.ANALYSIS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [fcc.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["executed_package"], self.evidence["executed_package"])
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 2)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (fcc.ARTIFACT_CLAIM_ID, fcc.ARTIFACT_ID))

    def test_tampered_macro_or_document_pattern_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\FccProbeClosed}{13}", "\\newcommand{\\FccProbeClosed}{14}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        # Every fixed document pattern matches the bound blob; an edited number stops matching.
        for name in fcc.DOCUMENT_PATTERNS:
            self.assertIsNotNone(fcc.DOCUMENT_PATTERNS[name].search(self.document), name)
        edited = self.document.replace("| 1.28e-6 | 1.28e-5 |", "| 1.28e-6 || 1.28e-5 |")
        self.assertNotEqual(edited, self.document)
        with self.assertRaises(ValueError):
            fcc.document_value(edited, "continuation_all_cells", "a")
        # The generator refuses a plasma package imported from outside the checkout.
        module = fcc._plasma(REPO)
        self.assertTrue(Path(module.__file__).resolve().is_relative_to((REPO / "modern/src").resolve()))

    def test_standalone_section_compiles_when_pdflatex_is_available(self) -> None:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            self.skipTest("pdflatex is not installed")
        with tempfile.TemporaryDirectory() as scratch:
            env = dict(os.environ)
            env.update({"SOURCE_DATE_EPOCH": "1788270043", "MIKTEX_ENABLE_INSTALLER": "0"})
            flags = ["--disable-installer"] if "miktex" in pdflatex.casefold() else []
            for _ in range(2):
                completed = subprocess.run(
                    [pdflatex, *flags, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                     f"-output-directory={scratch}", "sections/four-cell-closure-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "four-cell-closure-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "four-cell-closure-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
