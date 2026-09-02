"""Regression tests for the hash-bound wall-loss v4 paper evidence."""

from __future__ import annotations

import hashlib
import json
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
import generate_wall_loss_v4_evidence as wl4  # noqa: E402

EVIDENCE = REPO / wl4.EVIDENCE_PATH
GENERATED = REPO / wl4.OUTPUT_PATH
SIDECAR = REPO / wl4.SIDECAR_PATH
SECTION = REPO / wl4.SECTION_PATH
STANDALONE = REPO / "paper/sections/wall-loss-v4-standalone.tex"
RESULTS = REPO / wl4.RESULTS


def _load_artifact(relative: str):
    path = RESULTS / relative
    raw = path.read_bytes()
    if relative.endswith(".gz"):
        import gzip

        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


class WallLossEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = wl4.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = wl4.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_committed_results_revision(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-wall-loss-v4-evidence")
        self.assertEqual(self.evidence["evidence_revision"], wl4.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], wl4.CLASSIFICATION)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", wl4.RESULTS_COMMIT_SHA, head],
            cwd=REPO, check=False, capture_output=True,
        ).returncode
        self.assertEqual(ancestor, 0)
        manifest_rel = (wl4.RESULTS / "manifest.json").as_posix()
        committed = subprocess.run(
            ["git", "rev-parse", f"{wl4.RESULTS_COMMIT_SHA}:{manifest_rel}"],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(self.evidence["binding"]["manifest_git_blob"], committed)
        self.assertEqual(
            self.evidence["bundle"]["manifest_sha256"],
            hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 387)
        self.assertEqual(
            self.evidence["bundle"]["tolerated_crlf_sidecars"],
            sorted(f"artifacts/orbits/{case}.json.sha256" for case in wl4.CASES),
        )
        self.assertEqual(
            self.evidence["manuscript_integration"]["status"],
            "draft-section-not-wired-into-manuscript",
        )
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertNotIn("wall-loss-v4", manuscript)

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 100)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        artifacts = self.evidence["artifacts"]
        for relative, meta in artifacts.items():
            raw = (RESULTS / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
            self.assertEqual(len(raw), meta["bytes"], relative)
        loaded = {relative: _load_artifact(relative) for relative in artifacts}
        derived_count = 0
        for item in macros:
            with self.subTest(macro=item["name"]):
                self.assertTrue(item["name"].isalpha())
                if item["derived"]:
                    derived_count += 1
                    self.assertTrue(item["derivation"])
                    self.assertTrue(item["inputs"])
                    for source in item["inputs"]:
                        if source["artifact"] == "manifest.json":
                            self.assertTrue((RESULTS / "manifest.json").is_file())
                            continue
                        self.assertIn(source["artifact"], artifacts)
                        wl4.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                    self.assertEqual(wl4.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                raw = wl4.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(wl4.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 10)

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        by_name = {item["name"]: item for item in self.evidence["macros"]}
        campaign = _load_artifact("artifacts/campaign-result.json")
        wall = sum(campaign["campaigns"][c]["termination_counts"]["wall_hit"] for c in wl4.CASES)
        escape = sum(campaign["campaigns"][c]["termination_counts"]["domain_escape"] for c in wl4.CASES)
        trials = sum(campaign["campaigns"][c]["trial_count"] for c in wl4.CASES)
        self.assertEqual(by_name["WlfPooledWall"]["raw"], wall)
        self.assertEqual(by_name["WlfPooledEscape"]["raw"], escape)
        self.assertEqual(by_name["WlfPooledTrials"]["raw"], trials)
        self.assertEqual(by_name["WlfPooledIncomplete"]["raw"], 0)
        self.assertEqual(by_name["WlfPooledReflected"]["raw"], 0)
        self.assertEqual(by_name["WlfPooledWallP"]["value"], f"{wall / trials:.3f}")
        gates = _load_artifact("artifacts/gates.json")
        self.assertEqual(by_name["WlfGateCount"]["raw"], len(gates["checks"]))
        self.assertEqual(by_name["WlfGatesTrue"]["raw"], sum(1 for v in gates["checks"].values() if v is True))
        self.assertEqual(by_name["WlfGateCount"]["raw"], by_name["WlfGatesTrue"]["raw"])
        cells = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
        tolerance_close = 0
        for case in wl4.CASES:
            summary = _load_artifact(f"artifacts/summaries/{case}.json")
            tolerance_close += summary["diagnostics"]["tolerance_close_event_count"]
            for stratum in summary["strata"]:
                cell = int(stratum["cell_id"].rsplit("-", 1)[1])
                cells[cell][0] += stratum["termination_counts"]["wall_hit"]
                cells[cell][1] += stratum["trials"]
        self.assertEqual(by_name["WlfCellTwoThreeWall"]["raw"], cells[2][0] + cells[3][0])
        self.assertEqual(by_name["WlfCellTwoThreeTrials"]["raw"], cells[2][1] + cells[3][1])
        self.assertEqual(cells[2][0] + cells[3][0], cells[2][1] + cells[3][1])
        self.assertEqual(by_name["WlfCellFourEscape"]["raw"], cells[4][1] - cells[4][0])
        self.assertEqual(cells[4][0], 0)
        self.assertEqual(by_name["WlfToleranceCloseCount"]["raw"], tolerance_close)
        self.assertEqual(by_name["WlfToleranceCloseShare"]["value"], f"{100 * tolerance_close / trials:.1f}\\%")
        convergence = _load_artifact("artifacts/probability-convergence.json")
        changes = [c for chain in convergence["timestep"] + convergence["cross_map"] for c in chain["successive_changes"]]
        self.assertEqual(by_name["WlfMaxSuccessiveChange"]["raw"], max(changes))
        self.assertLessEqual(max(changes), by_name["WlfGateThreshold"]["raw"])

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Wlf[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Wlf[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertIn("WlfCaseTable", used)
        self.assertIn("WlfCellTable", used)
        body = re.sub(r"(?m)^%.*$", "", self.section)
        body = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", body)
        body = re.sub(r"\\Wlf[A-Za-z]+", "", body)
        body = re.sub(r"\bP2\b", "", body)
        body = re.sub(r"\\begin\{minipage\}\{[^}]*\}", "", body)  # layout width, not a result
        self.assertEqual(re.findall(r"\d", body), [], "hand-typed digits in the section")
        self.assertIn("Collisionless full-orbit electron wall loss in the divergent-exit field", self.section)
        for heading in ("Method.", "Results.", "Numerical convergence.", "Interpretation."):
            self.assertIn(f"\\paragraph{{{heading}}}", self.section)
        self.assertIn("Model-bounded interpretation", self.section)
        for pattern in check_paper.PLACEHOLDERS.values():
            self.assertIsNone(pattern.search(self.section))
            self.assertIsNone(pattern.search(self.tex))
        for pattern in check_paper.FORBIDDEN_MODEL_WORDING.values():
            self.assertIsNone(pattern.search(self.section))
        self.assertEqual(check_paper.find_unregistered_claims(self.tex), [])

    def test_generated_tex_tables_are_data_bound(self) -> None:
        self.assertEqual(self.tex.count("\\newcommand{\\WlfCaseTable}"), 1)
        self.assertEqual(self.tex.count("\\newcommand{\\WlfCellTable}"), 1)
        campaign = _load_artifact("artifacts/campaign-result.json")
        for case in wl4.CASES:
            role, policy = case.split("-")
            block = campaign["campaigns"][case]
            row = (
                f"{role} & {policy} & {block['wall_hit']['successes']} & "
                f"{block['wall_hit']['probability']:.4f} & [{block['wall_hit']['lower']:.3f}, "
                f"{block['wall_hit']['upper']:.3f}] & {block['escaped']['successes']} & "
            )
            self.assertIn(row, self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], wl4.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [])

    def test_tampered_bundle_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\WlfPooledWall}{2962}", "\\newcommand{\\WlfPooledWall}{2963}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(
            hashlib.sha256(changed.encode("utf-8")).hexdigest(),
            json.loads(self.sidecar_bytes)["output"]["sha256"],
        )
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / wl4.RESULTS
            shutil.copytree(RESULTS, target)
            shutil.copy(REPO / "paper/build-config.json", (repo / "paper").mkdir(parents=True, exist_ok=True) or repo / "paper/build-config.json")
            victim = target / "artifacts" / "campaign-result.json"
            victim.write_bytes(victim.read_bytes().replace(b'"wall_hit":329', b'"wall_hit":330', 1))
            with self.assertRaises(ValueError):
                wl4.Bundle(repo)
            victim.write_bytes((RESULTS / "artifacts" / "campaign-result.json").read_bytes())
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                wl4.Bundle(repo)

    def test_standalone_section_compiles_when_pdflatex_is_available(self) -> None:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            self.skipTest("pdflatex is not installed")
        with tempfile.TemporaryDirectory() as scratch:
            env = dict(__import__("os").environ)
            env.update({"SOURCE_DATE_EPOCH": "1788270043", "MIKTEX_ENABLE_INSTALLER": "0"})
            flags = ["--disable-installer"] if "miktex" in pdflatex.casefold() else []
            for _ in range(2):
                completed = subprocess.run(
                    [pdflatex, *flags, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                     f"-output-directory={scratch}", "sections/wall-loss-v4-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "wall-loss-v4-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "wall-loss-v4-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
