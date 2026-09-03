"""Regression tests for the hash-bound MDO L0 campaign v1 paper evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_mdo_l0_v1_evidence as mdo  # noqa: E402

EVIDENCE = REPO / mdo.EVIDENCE_PATH
GENERATED = REPO / mdo.OUTPUT_PATH
SIDECAR = REPO / mdo.SIDECAR_PATH
SECTION = REPO / mdo.SECTION_PATH
STANDALONE = REPO / "paper/sections/mdo-l0-v1-standalone.tex"
RESULTS = REPO / mdo.RESULTS


def _load_artifact(relative: str):
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


class MdoEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = mdo.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = mdo.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_committed_revisions_and_the_dashboard(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-mdo-l0-v1-evidence")
        self.assertEqual(self.evidence["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], mdo.CLASSIFICATION)
        self.assertEqual(self.evidence["closure"], mdo.CLOSURE_ID)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        for commit in (mdo.PREREGISTRATION_COMMIT_SHA, mdo.RESULTS_COMMIT_SHA, mdo.DASHBOARD_COMMIT_SHA):
            self.assertEqual(
                subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=False).returncode, 0
            )
        # prereg -> results -> dashboard, strictly.
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", mdo.PREREGISTRATION_COMMIT_SHA, mdo.RESULTS_COMMIT_SHA], cwd=REPO, check=False).returncode, 0)
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", mdo.RESULTS_COMMIT_SHA, mdo.DASHBOARD_COMMIT_SHA], cwd=REPO, check=False).returncode, 0)
        manifest_rel = (mdo.RESULTS / "manifest.json").as_posix()
        committed = subprocess.run(
            ["git", "rev-parse", f"{mdo.RESULTS_COMMIT_SHA}:{manifest_rel}"], cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(self.evidence["binding"]["manifest_git_blob"], committed)
        self.assertEqual(
            self.evidence["bundle"]["manifest_sha256"],
            hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381")
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 137)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        # The dashboard is bound by LF-normalised SHA-256 equal to the blob committed at its revision.
        for key, path in (("generator_sha256_lf", mdo.DASHBOARD_GENERATOR), ("html_sha256_lf", mdo.DASHBOARD_HTML)):
            blob = subprocess.run(
                ["git", "show", f"{mdo.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True,
            ).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob).hexdigest())
        self.assertEqual(self.evidence["dashboard"]["payload_manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], mdo.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-campaign")
        self.assertEqual(integration["manifest_id"], mdo.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], mdo.SECTION_HEADING)
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(mdo.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(mdo.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(mdo.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 300)
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
                        mdo.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                    self.assertEqual(mdo.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                raw = mdo.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(mdo.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 40)

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        b = self.by_name
        metrics = _load_artifact("artifacts/metrics.json")
        dense = _load_artifact("artifacts/dense-reference-summary.json")
        for strategy, token in mdo.STRATEGY_TOKENS.items():
            hv = [metrics["runs"][f"{strategy}:{seed}"]["final_hypervolume"] for seed in (101, 202, 303)]
            self.assertEqual(b[f"MdoHv{token}Mean"]["raw"], metrics["seed_variance"][strategy]["mean"])
            self.assertAlmostEqual(b[f"MdoHv{token}Mean"]["raw"], statistics.mean(hv), places=15)
            self.assertAlmostEqual(b[f"MdoHv{token}Std"]["raw"], statistics.stdev(hv), places=15)
            attained = [hv_i / dense["robust_hypervolume"] for hv_i in hv]
            self.assertAlmostEqual(b[f"MdoAttained{token}Mean"]["raw"], statistics.mean(attained), places=12)
            self.assertEqual(b[f"MdoAttained{token}Min"]["raw"], min(metrics["hypervolume_table"][f"{strategy}:{s}"]["attained_fraction_of_dense_reference"] for s in (101, 202, 303)))
        self.assertEqual(b["MdoHvBoA"]["value"], "0.003863")
        self.assertEqual(b["MdoHvBoB"]["value"], "0.003877")
        self.assertEqual(b["MdoHvBoC"]["value"], "0.003860")
        self.assertEqual(b["MdoAttainedBoMeanTwo"]["value"], "1.02")
        self.assertEqual(b["MdoDenseToBudgetRatio"]["raw"], 8192 / 96)
        gates = _load_artifact("artifacts/gates.json")
        self.assertEqual(b["MdoGateCount"]["raw"], len(gates["binding"]))
        self.assertEqual(b["MdoGatesPassed"]["raw"], 8)
        self.assertEqual(b["MdoBoBeatsRandomWins"]["raw"], 3)
        self.assertEqual(b["MdoBoBeatsNsgaWins"]["raw"], 3)
        pooled = _load_artifact("artifacts/pooled-fronts.json")
        shared = set(pooled["robust"]["design_ids"]) & set(pooled["nominal"]["design_ids"])
        self.assertEqual(b["MdoSharedDesigns"]["raw"], len(shared))
        self.assertAlmostEqual(b["MdoJaccard"]["raw"], len(shared) / (114 + 62 - len(shared)), places=15)
        self.assertEqual(b["MdoJaccard"]["value"], "0.158")
        self.assertEqual(b["MdoNominalEffMaxTwo"]["value"], "0.89")
        self.assertEqual(b["MdoNominalIspMax"]["value"], "725")
        self.assertEqual(b["MdoRobustEffMax"]["value"], "0.255")
        self.assertEqual(b["MdoRobustIspMax"]["value"], "413")
        protocol = _load_artifact("artifacts/protocol.json")
        upper = protocol["uncertain_inputs"]["inputs"][0]["upper"]
        self.assertAlmostEqual(b["MdoImpliedPriorSurvival"]["raw"], (1 - upper / 2) ** 4, places=15)
        v4 = protocol["authority"]["wall_loss_v4"]["pooled_wall_hit"]
        self.assertAlmostEqual(b["MdoVFourSurvival"]["raw"], 1 - v4["successes"] / v4["trials"], places=15)
        self.assertEqual(b["MdoImpliedPriorSurvival"]["value"], "0.361")
        self.assertEqual(b["MdoVFourSurvival"]["value"], "0.357")
        sensitivity = _load_artifact("artifacts/sensitivity.json")
        for scenario in sensitivity["scenarios"]:
            token = mdo.SCENARIO_TOKENS[scenario["id"]]
            self.assertAlmostEqual(b[f"MdoScenario{token}Survival"]["raw"], math.prod(1 - p for p in scenario["cusp_probabilities"]), places=20)
            self.assertEqual(scenario["pareto_designs_evaluated"] + scenario["pareto_designs_infeasible"], b["MdoRobustFront"]["raw"])
        self.assertEqual(b["MdoScenarioNoWallLossInfeasible"]["raw"], 110)
        self.assertEqual(b["MdoScenarioJeffreysSurvival"]["value"], "$6.86\\times10^{-8}$")
        self.assertEqual(b["MdoInvarianceIdenticalCount"]["raw"], sum(1 for p in sensitivity["priors"] if p["identical_on_common_feasible_set_up_to_ties"]))
        self.assertEqual(b["MdoInvarianceIdenticalCount"]["raw"], b["MdoPriorCount"]["raw"])
        self.assertEqual(b["MdoProbeClosedCases"]["raw"], 13)
        self.assertEqual(b["MdoProbeTotalCases"]["raw"], 80)
        self.assertEqual(b["MdoLifecycleWallMin"]["value"], "27.3")
        self.assertEqual(b["MdoFailedRuns"]["raw"], 0)
        self.assertEqual(b["MdoToleratedEolFiles"]["raw"], 0)

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Mdo[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Mdo[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        for table in mdo.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Mdo"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n96 evaluations\n", "Mdo"), ["9", "6"])
        self.assertIn(f"\\subsection{{{mdo.SECTION_HEADING}}}", self.section)
        for heading in ("Method.", "Results.", "Robust versus nominal fronts.", "Sensitivity to the cusp prior.", "Interpretation."):
            self.assertIn(f"\\paragraph{{{heading}}}", self.section)
        self.assertIn("Model-bounded interpretation", self.section)
        for pattern in check_paper.PLACEHOLDERS.values():
            self.assertIsNone(pattern.search(self.section))
            self.assertIsNone(pattern.search(self.tex))
        for pattern in check_paper.FORBIDDEN_MODEL_WORDING.values():
            self.assertIsNone(pattern.search(self.section))
        self.assertEqual(check_paper.find_unregistered_claims(self.section), [])
        self.assertEqual(check_paper.find_unregistered_claims(self.tex), [])
        self.assertNotIn("CLM-", "\n".join(line for line in self.section.splitlines() if line.lstrip().startswith("%")))

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in mdo.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        metrics = _load_artifact("artifacts/metrics.json")
        for strategy in mdo.STRATEGIES:
            for seed in (101, 202, 303):
                run = metrics["runs"][f"{strategy}:{seed}"]
                row = f"{mdo.STRATEGY_LABELS[strategy]} & {seed} & {run['final_hypervolume']:.6f} & "
                self.assertIn(row, self.tex)
        sensitivity = _load_artifact("artifacts/sensitivity.json")
        for scenario in sensitivity["scenarios"]:
            self.assertIn(f" & {scenario['pareto_designs_evaluated']} / {scenario['pareto_designs_infeasible']} & ", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [mdo.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 3)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (mdo.ARTIFACT_CLAIM_ID, mdo.ARTIFACT_ID))

    def test_tampered_bundle_dashboard_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\MdoTotalEvaluations}{864}", "\\newcommand{\\MdoTotalEvaluations}{865}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(
            hashlib.sha256(changed.encode("utf-8")).hexdigest(),
            json.loads(self.sidecar_bytes)["output"]["sha256"],
        )
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / mdo.RESULTS
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            victim.write_bytes(original.replace(b'"total_evaluations":864', b'"total_evaluations":865', 1))
            with self.assertRaises(ValueError):
                mdo.Bundle(repo)
            victim.write_bytes(original)
            # A CRLF rewrite of any bundle file is a byte mismatch: no tolerance exists for this bundle.
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                mdo.Bundle(repo)
        # A dashboard whose payload names another manifest is refused before any macro is written.
        html = (REPO / mdo.DASHBOARD_HTML).read_bytes()
        payload = mdo.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(mdo.dashboard_payload(tampered)["identity"]["manifest_sha256"], payload["identity"]["manifest_sha256"])

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
                     f"-output-directory={scratch}", "sections/mdo-l0-v1-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "mdo-l0-v1-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "mdo-l0-v1-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
