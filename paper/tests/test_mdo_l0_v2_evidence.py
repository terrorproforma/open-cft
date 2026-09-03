"""Regression tests for the hash-bound MDO L0 campaign v2 paper evidence."""

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
import generate_mdo_l0_v1_evidence as mdo_v1  # noqa: E402
import generate_mdo_l0_v2_evidence as mdo  # noqa: E402

EVIDENCE = REPO / mdo.EVIDENCE_PATH
GENERATED = REPO / mdo.OUTPUT_PATH
SIDECAR = REPO / mdo.SIDECAR_PATH
SECTION = REPO / mdo.SECTION_PATH
STANDALONE = REPO / "paper/sections/mdo-l0-v2-standalone.tex"
RESULTS = REPO / mdo.RESULTS
V1_RESULTS = REPO / mdo.V1_RESULTS


def _load(root: Path, relative: str):
    return json.loads((root / relative).read_bytes().decode("utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=REPO, check=False).returncode == 0


class MdoV2EvidenceTests(unittest.TestCase):
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
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE, REPO / "paper/scripts/generate_mdo_l0_v2_evidence.py"):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_both_bundles_the_dashboard_the_dataset_and_the_audit(self) -> None:
        e = self.evidence
        self.assertEqual(e["document_type"], "paper-mdo-l0-v2-evidence")
        self.assertEqual(e["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(e["classification"], mdo.CLASSIFICATION)
        self.assertEqual(e["closure"], mdo.CLOSURE_ID)
        self.assertEqual(e["sensitivity_closure"], mdo.SENSITIVITY_CLOSURE_ID)
        head = _git("rev-parse", "HEAD")
        for commit in (mdo.PREREGISTRATION_COMMIT_SHA, mdo.RESULTS_COMMIT_SHA, mdo.DASHBOARD_COMMIT_SHA, mdo.V1_RESULTS_COMMIT_SHA, mdo.V1_AUDIT_COMMIT_SHA, mdo.SCREENING_RESULTS_COMMIT_SHA):
            self.assertTrue(_is_ancestor(commit, head), commit)
        # prereg -> results -> dashboard, strictly; audit and screening precede the preregistration.
        self.assertTrue(_is_ancestor(mdo.PREREGISTRATION_COMMIT_SHA, mdo.RESULTS_COMMIT_SHA))
        self.assertTrue(_is_ancestor(mdo.RESULTS_COMMIT_SHA, mdo.DASHBOARD_COMMIT_SHA))
        self.assertTrue(_is_ancestor(mdo.V1_AUDIT_COMMIT_SHA, mdo.PREREGISTRATION_COMMIT_SHA))
        self.assertTrue(_is_ancestor(mdo.SCREENING_RESULTS_COMMIT_SHA, mdo.PREREGISTRATION_COMMIT_SHA))
        self.assertTrue(_is_ancestor(mdo.V1_RESULTS_COMMIT_SHA, mdo.V1_AUDIT_COMMIT_SHA))
        manifest_rel = (mdo.RESULTS / "manifest.json").as_posix()
        self.assertEqual(e["binding"]["manifest_git_blob"], _git("rev-parse", f"{mdo.RESULTS_COMMIT_SHA}:{manifest_rel}"))
        self.assertEqual(e["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(e["bundle"]["manifest_sha256"], mdo.EXPECTED_MANIFEST_SHA256)
        self.assertEqual(e["bundle"]["verified_file_count"], 147)
        self.assertEqual(e["bundle"]["tolerated_eol_files"], [])
        # The results commit adds files under results/ only (v1 audit F9).
        files = _git("diff-tree", "--no-commit-id", "--name-only", "-r", mdo.RESULTS_COMMIT_SHA).splitlines()
        self.assertEqual(len(files), e["binding"]["result_commit_file_count"])
        self.assertEqual(len(files), 148)
        self.assertEqual(e["binding"]["result_commit_files_outside_results"], [])
        self.assertTrue(all(f.startswith(mdo.RESULTS.as_posix() + "/") for f in files))
        self.assertEqual(e["binding"]["preregistration_commit_file_count"], 1)
        # The prior campaign's bundle.
        v1 = e["v1_bundle"]
        self.assertEqual(v1["manifest_sha256"], mdo.V1_MANIFEST_SHA256)
        self.assertEqual(v1["manifest_sha256"], hashlib.sha256((V1_RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(v1["manifest_git_blob"], _git("rev-parse", f"{mdo.V1_RESULTS_COMMIT_SHA}:{(mdo.V1_RESULTS / 'manifest.json').as_posix()}"))
        self.assertEqual(v1["verified_file_count"], 137)
        self.assertEqual(set(e["v1_artifacts"]), set(mdo.V1_COMPARISON_ARTIFACTS))
        # The dashboard is bound by LF-normalised SHA-256 equal to the blob committed at its revision.
        for key, path in (("generator_sha256_lf", mdo.DASHBOARD_GENERATOR), ("html_sha256_lf", mdo.DASHBOARD_HTML)):
            blob = subprocess.run(["git", "show", f"{mdo.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(e["dashboard"][key], hashlib.sha256(blob).hexdigest())
        self.assertEqual(e["dashboard"]["payload_manifest_sha256"], e["bundle"]["manifest_sha256"])
        self.assertEqual(e["dashboard"]["payload_v1_manifest_sha256"], mdo.V1_MANIFEST_SHA256)
        # The screening dataset behind the catalogue.
        cb = e["catalogue_binding"]
        dataset = REPO / mdo.SCREENING_DATASET_PATH
        self.assertEqual(cb["dataset_sha256"], hashlib.sha256(dataset.read_bytes()).hexdigest())
        self.assertEqual(cb["dataset_bytes"], dataset.stat().st_size)
        self.assertEqual(cb["dataset_git_blob"], _git("rev-parse", f"{mdo.SCREENING_RESULTS_COMMIT_SHA}:{mdo.SCREENING_DATASET_PATH.as_posix()}"))
        self.assertEqual(cb["dataset_git_blob"], _git("rev-parse", f"HEAD:{mdo.SCREENING_DATASET_PATH.as_posix()}"))
        self.assertEqual(cb["screening_classification"], mdo.SCREENING_CLASSIFICATION)
        # The prior campaign's post-hoc audit.
        audit = e["audit"]
        self.assertEqual(audit["git_blob"], _git("rev-parse", f"{mdo.V1_AUDIT_COMMIT_SHA}:{mdo.V1_AUDIT_PATH.as_posix()}"))
        self.assertEqual(audit["git_blob"], _git("rev-parse", f"HEAD:{mdo.V1_AUDIT_PATH.as_posix()}"))
        self.assertEqual(audit["disclosures"], ["F9", "F10", "F22", "F26", "F27", "F28"])
        audit_text = (REPO / mdo.V1_AUDIT_PATH).read_text(encoding="utf-8")
        self.assertEqual(mdo.AUDIT_PATTERN.search(audit_text).group(1), "F9, F10, F22, F26, F27, F28")
        integration = e["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], mdo.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-campaign")
        self.assertEqual(integration["manifest_id"], mdo.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], mdo.SECTION_HEADING)
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(mdo.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(mdo.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(mdo.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact_of_the_right_bundle(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 550)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Mdb") for name in names))
        self.assertFalse(any(name.startswith("Mdo") for name in names))
        roots = {"v2": (RESULTS, self.evidence["artifacts"]), "v1": (V1_RESULTS, self.evidence["v1_artifacts"])}
        loaded: dict[tuple[str, str], object] = {}
        for bundle, (root, artifacts) in roots.items():
            for relative, meta in artifacts.items():
                raw = (root / relative).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
                self.assertEqual(len(raw), meta["bytes"], relative)
                loaded[(bundle, relative)] = json.loads(raw)
        derived_count = 0
        v1_count = 0
        for item in macros:
            with self.subTest(macro=item["name"]):
                self.assertTrue(item["name"].isalpha())
                if item["derived"]:
                    derived_count += 1
                    self.assertTrue(item["derivation"])
                    self.assertTrue(item["inputs"])
                    for source in item["inputs"]:
                        bundle = source.get("bundle", "v2")
                        if source["artifact"] == "manifest.json":
                            self.assertTrue((roots[bundle][0] / "manifest.json").is_file())
                            continue
                        self.assertIn(source["artifact"], roots[bundle][1])
                        mdo.resolve_pointer(loaded[(bundle, source["artifact"])], source["pointer"])
                    self.assertEqual(mdo.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                bundle = source.get("bundle", "v2")
                if bundle == "v1":
                    v1_count += 1
                self.assertIn(source["artifact"], roots[bundle][1])
                raw = mdo.resolve_pointer(loaded[(bundle, source["artifact"])], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(mdo.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 100)
        self.assertGreater(v1_count, 20)
        # Every macro read from the prior campaign is named Prior...
        for item in macros:
            if not item["derived"] and item["source"].get("bundle") == "v1":
                self.assertTrue(item["name"].startswith("MdbPrior"), item["name"])

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        b = self.by_name
        metrics = _load(RESULTS, "artifacts/metrics.json")
        dense = _load(RESULTS, "artifacts/dense-reference-summary.json")
        for strategy, token in mdo.STRATEGY_TOKENS.items():
            hv = [metrics["runs"][f"{strategy}:{seed}"]["final_hypervolume"] for seed in (101, 202, 303)]
            self.assertEqual(b[f"MdbHv{token}Mean"]["raw"], metrics["seed_variance"][strategy]["mean"])
            self.assertAlmostEqual(b[f"MdbHv{token}Mean"]["raw"], statistics.mean(hv), places=15)
            self.assertAlmostEqual(b[f"MdbHv{token}Std"]["raw"], statistics.stdev(hv), places=15)
            attained = [hv_i / dense["robust_hypervolume"] for hv_i in hv]
            self.assertAlmostEqual(b[f"MdbAttained{token}Mean"]["raw"], statistics.mean(attained), places=12)
            self.assertEqual(b[f"MdbInfeasible{token}Total"]["raw"], sum(metrics["runs"][f"{strategy}:{s}"]["infeasible_evaluations"] for s in (101, 202, 303)))
        self.assertEqual(b["MdbHvBoA"]["value"], "$9.269\\times10^{-4}$")
        self.assertEqual(b["MdbHvBoB"]["value"], "$2.159\\times10^{-3}$")
        self.assertEqual(b["MdbHvBoC"]["value"], "$2.151\\times10^{-3}$")
        self.assertEqual((b["MdbAttainedBoA"]["value"], b["MdbAttainedBoB"]["value"], b["MdbAttainedBoC"]["value"]), ("0.49", "1.13", "1.13"))
        self.assertEqual(b["MdbDenseRobustHv"]["value"], "$1.907\\times10^{-3}$")
        self.assertEqual(b["MdbDenseToBudgetRatio"]["raw"], 98304 / 160)
        self.assertEqual((b["MdbInfeasibleBoTotal"]["raw"], b["MdbInfeasibleNsgaTotal"]["raw"], b["MdbInfeasibleLhsTotal"]["raw"]), (88, 3, 0))
        gates = _load(RESULTS, "artifacts/gates.json")
        self.assertEqual(b["MdbGateCount"]["raw"], len(gates["binding"]))
        self.assertEqual(b["MdbGatesPassed"]["raw"], 12)
        self.assertEqual(b["MdbNsgaDuplicates"]["raw"], sum(r["duplicates"] for r in gates["binding"]["nsga3_duplicates_eliminated"]["runs"].values()))
        self.assertEqual(b["MdbLabelChecks"]["raw"], len(gates["binding"]["labels_consistent"]["checks"]))
        self.assertEqual(b["MdbImportedFiles"]["raw"], 28)
        pooled = _load(RESULTS, "artifacts/pooled-fronts.json")
        shared = set(pooled["robust"]["design_ids"]) & set(pooled["nominal"]["design_ids"])
        self.assertEqual(b["MdbSharedDesigns"]["raw"], len(shared))
        self.assertAlmostEqual(b["MdbJaccard"]["raw"], len(shared) / (96 + 86 - len(shared)), places=15)
        self.assertEqual(b["MdbJaccard"]["value"], "0.70")
        self.assertEqual(b["MdbRobustFrontDesigns"]["raw"], [49, 50, 94])
        self.assertEqual(b["MdbNominalOnlyDesign"]["raw"], 74)
        # The robust-front designs are the three lowest pooled wall-hit probabilities of the catalogue.
        catalogue = _load(RESULTS, "artifacts/catalogue.json")
        order = sorted(catalogue["designs"], key=lambda d: (d["pooled"]["probability"], d["catalogue_index"]))
        self.assertEqual(sorted(d["catalogue_index"] for d in order[:3]), [49, 50, 94])
        self.assertIs(b["MdbRobustFrontLowestRanks"]["raw"], True)
        self.assertEqual((b["MdbDesignFortyNineRank"]["raw"], b["MdbDesignNinetyFourRank"]["raw"], b["MdbDesignFiftyRank"]["raw"]), (1, 2, 3))
        self.assertEqual((b["MdbDesignFortyNineMembers"]["raw"], b["MdbDesignFiftyMembers"]["raw"], b["MdbDesignNinetyFourMembers"]["raw"]), (60, 19, 17))
        self.assertEqual(b["MdbDesignFortyNinePooledP"]["value"], "0.375")
        self.assertEqual((b["MdbDesignFortyNinePooledLo"]["value"], b["MdbDesignFortyNinePooledHi"]["value"]), ("0.334", "0.418"))
        self.assertEqual((b["MdbDesignFortyNineLengthMm"]["value"], b["MdbDesignFortyNineRadiusMm"]["value"], b["MdbDesignFortyNinePitchMm"]["value"]), ("29.4", "1.80", "5.9"))
        self.assertEqual((b["MdbDesignFiftyLengthMm"]["value"], b["MdbDesignFiftyRadiusMm"]["value"]), ("20.4", "1.91"))
        self.assertEqual((b["MdbDesignNinetyFourLengthMm"]["value"], b["MdbDesignNinetyFourRadiusMm"]["value"]), ("28.8", "2.14"))
        # Jeffreys means, Wilson intervals and survivals recompute exactly from the counts.
        for design in catalogue["designs"]:
            for estimate in (*design["cells"], design["pooled"]):
                self.assertEqual(estimate["posterior_mean"], (estimate["wall_hits"] + 0.5) / (estimate["trials"] + 1))
                self.assertEqual(tuple(estimate["wilson_95"]), mdo.wilson(estimate["wall_hits"], estimate["trials"]))
            self.assertAlmostEqual(design["nominal_survival_cl1"], math.prod(1 - c["posterior_mean"] for c in design["cells"]), places=15)
        dense_full = _load(RESULTS, "artifacts/dense-reference.json")
        negligible = [row["catalogue_index"] for row in dense_full["per_design"] if row["robust_hypervolume"] < mdo.DENSE_NEGLIGIBLE_HYPERVOLUME]
        saturated = {d["catalogue_index"] for d in catalogue["designs"] if any(c["wall_hits"] == c["trials"] for c in d["cells"])}
        self.assertEqual(b["MdbDenseNegligibleDesigns"]["raw"], len(negligible))
        self.assertEqual(len(negligible), 77)
        self.assertEqual(b["MdbSaturatedDesigns"]["raw"], len(saturated))
        self.assertEqual(len(saturated), 73)
        self.assertTrue(saturated <= set(negligible))
        self.assertEqual(b["MdbDenseNegligibleSaturated"]["raw"], 73)
        self.assertEqual(b["MdbDenseNegligibleUnsaturated"]["raw"], 4)
        sensitivity = _load(RESULTS, "artifacts/sensitivity.json")
        cl2 = sensitivity["closure_cl2"]
        self.assertEqual(b["MdbClTwoShared"]["raw"], len(set(cl2["front_design_ids"]) & set(pooled["robust"]["design_ids"])))
        self.assertEqual(b["MdbClTwoShared"]["raw"], 0)
        self.assertEqual(b["MdbClTwoJaccard"]["raw"], 0.0)
        self.assertEqual(b["MdbClTwoFrontDesignCount"]["raw"], len({m["catalogue_index"] for m in cl2["front_members"]}))
        self.assertEqual((b["MdbClTwoFront"]["raw"], b["MdbClTwoFrontDesignCount"]["raw"]), (50, 25))
        widths = {w["width_scale"]: w for w in sensitivity["widths"]}
        self.assertEqual((b["MdbWidthQuarterFront"]["raw"], b["MdbWidthFourFront"]["raw"], b["MdbWidthPointFront"]["raw"]), (15, 91, 94))
        self.assertEqual((b["MdbWidthQuarterJaccard"]["value"], b["MdbWidthFourJaccard"]["value"], b["MdbWidthPointJaccard"]["value"]), ("0.03", "0.82", "0.79"))
        self.assertEqual(b["MdbWidthIdenticalCount"]["raw"], sum(1 for w in widths.values() if w["identical_on_common_feasible_set_up_to_ties"] and not w["is_campaign_posterior"]))
        self.assertEqual(b["MdbWidthIdenticalCount"]["raw"], 1)
        self.assertEqual(b["MdbCampaignSurvivalMax"]["raw"], widths[1.0]["survival_max"])
        # Seed 101 stalled on design 50 and never evaluated design 49.
        run = _load(RESULTS, "artifacts/runs/qlognehvi-101.json")
        indices = [r["design"]["catalogue_index"] for r in run["records"]]
        self.assertEqual(b["MdbBoAStallDesign"]["raw"], 50)
        self.assertEqual(b["MdbBoAStallEvaluations"]["raw"], indices.count(50))
        self.assertEqual(b["MdbBoAStallEvaluations"]["raw"], 119)
        self.assertEqual(b["MdbBoAMissedDesign"]["raw"], 49)
        self.assertNotIn(49, indices)
        self.assertIs(b["MdbBoAMissedInInitial"]["raw"], False)
        self.assertIs(b["MdbBoBMissedInInitial"]["raw"], True)
        self.assertIs(b["MdbBoCMissedInInitial"]["raw"], True)
        # Comparison with the prior campaign (its own bundle, same reference frame).
        v1_dense = _load(V1_RESULTS, "artifacts/dense-reference-summary.json")
        self.assertIs(b["MdbSameReferenceFrame"]["raw"], True)
        self.assertAlmostEqual(b["MdbPriorToThisDenseHvRatio"]["raw"], v1_dense["robust_hypervolume"] / dense["robust_hypervolume"], places=15)
        self.assertEqual(b["MdbPriorToThisDenseHvRatio"]["value"], "2.0")
        self.assertEqual(b["MdbPriorSurvivalMax"]["value"], "0.704")
        self.assertEqual(b["MdbCampaignSurvivalMax"]["value"], "0.180")
        self.assertEqual(b["MdbPriorHvBoMean"]["value"], "$3.867\\times10^{-3}$")
        self.assertEqual(b["MdbPriorEvaluationsPerRun"]["raw"], 96)
        self.assertEqual(b["MdbAuditDisclosureIds"]["value"], "F9, F10, F22, F26, F27, F28")
        self.assertEqual(b["MdbAuditDisclosuresClosed"]["raw"], 6)
        self.assertEqual(b["MdbLifecycleWallMin"]["value"], "82.8")
        self.assertEqual(b["MdbFailedRuns"]["raw"], 0)
        self.assertEqual(b["MdbToleratedEolFiles"]["raw"], 0)
        self.assertEqual(b["MdbResultCommitOutsideResults"]["raw"], 0)
        self.assertEqual(b["MdbClosureShort"]["value"], "CL-1")
        self.assertEqual(b["MdbSensitivityClosureShort"]["value"], "CL-2")
        self.assertEqual(b["MdbTieTolerance"]["raw"], 1e-9)

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Mdb[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Mdb[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertEqual(re.findall(r"\\Mdo[A-Za-z]+", self.section), [], "the v2 section must not borrow v1 macros")
        for table in mdo.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Mdb"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n160 evaluations\n", "Mdb"), ["1", "6", "0"])
        self.assertIn(f"\\subsection{{{mdo.SECTION_HEADING}}}", self.section)
        for heading in ("Method.", "Results.", "Catalogue designs on the robust front.", "Closure dependence and uncertainty width.", "Closure of the prior campaign's audit disclosures.", "Interpretation."):
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
        metrics = _load(RESULTS, "artifacts/metrics.json")
        for strategy in mdo.STRATEGIES:
            for seed in (101, 202, 303):
                run = metrics["runs"][f"{strategy}:{seed}"]
                row = f"{mdo.STRATEGY_LABELS[strategy]} & {seed} & {mdo.format_value('sci3', run['final_hypervolume'])} & "
                self.assertIn(row, self.tex)
        catalogue = _load(RESULTS, "artifacts/catalogue.json")
        for index in (46, 49, 50, 73, 94):
            design = catalogue["designs"][index]
            self.assertIn(f"& {design['pooled']['probability']:.3f} [{design['pooled']['wilson_95'][0]:.3f}, {design['pooled']['wilson_95'][1]:.3f}] &", self.tex)
        sensitivity = _load(RESULTS, "artifacts/sensitivity.json")
        for width in sensitivity["widths"]:
            self.assertIn(f" & {width['front_size']} & {mdo.format_value('list_int', width['front_catalogue_indices'])} & {width['shared_with_campaign_front']} & ", self.tex)
        self.assertIn("CL-2 & $w = 1$ &", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], mdo.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [mdo.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertEqual(sidecar["v1_bundle_manifest"]["sha256"], mdo.V1_MANIFEST_SHA256)
        input_paths = {entry["path"] for entry in sidecar["inputs"]}
        self.assertTrue(any(path.startswith(mdo.V1_RESULTS.as_posix()) for path in input_paths))
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 4)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (mdo.ARTIFACT_CLAIM_ID, mdo.ARTIFACT_ID))

    def test_tampered_bundle_dashboard_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\MdbTotalEvaluations}{1440}", "\\newcommand{\\MdbTotalEvaluations}{1441}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / mdo.RESULTS
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            victim.write_bytes(original.replace(b'"total_evaluations":1440', b'"total_evaluations":1441', 1))
            with self.assertRaises(ValueError):
                mdo.Bundle(repo, mdo.RESULTS, mdo.EXPERIMENT_ID)
            victim.write_bytes(original)
            # A CRLF rewrite of any bundle file is a byte mismatch: no tolerance exists for this bundle.
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                mdo.Bundle(repo, mdo.RESULTS, mdo.EXPERIMENT_ID)
            # The same class refuses a bundle whose experiment identity is not the one asked for.
            with self.assertRaises(ValueError):
                mdo.Bundle(REPO, mdo.RESULTS, mdo_v1.EXPERIMENT_ID)
        # A dashboard whose payload names another manifest is refused before any macro is written.
        html = (REPO / mdo.DASHBOARD_HTML).read_bytes()
        payload = mdo.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        self.assertEqual(payload["v1"]["manifest_sha256"], mdo.V1_MANIFEST_SHA256)
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(mdo.dashboard_payload(tampered)["identity"]["manifest_sha256"], payload["identity"]["manifest_sha256"])
        # A catalogue whose Wilson bound or posterior mean is off by one ulp is refused.
        catalogue = _load(RESULTS, "artifacts/catalogue.json")
        protocol = _load(RESULTS, "artifacts/protocol.json")
        authorities = _load(RESULTS, "artifacts/authorities.json")
        mdo.verify_catalogue(catalogue, protocol, authorities)
        broken = json.loads(json.dumps(catalogue))
        broken["designs"][49]["cells"][0]["posterior_mean"] = 0.2596
        with self.assertRaises(ValueError):
            mdo.verify_catalogue(broken, protocol, authorities)
        broken = json.loads(json.dumps(catalogue))
        broken["designs"][49]["pooled"]["wilson_95"][0] = 0.334
        with self.assertRaises(ValueError):
            mdo.verify_catalogue(broken, protocol, authorities)
        # The audit's disclosure list is parsed with a fixed pattern and must equal the protocol's closed list.
        short = mdo.AUDIT_PATTERN.search("Six disclosures are recorded below (F9, F10).")
        self.assertNotEqual(tuple(short.group(1).split(", ")), mdo.AUDIT_DISCLOSURES)
        full = mdo.AUDIT_PATTERN.search("Six disclosures are recorded below (F9, F10, F22, F26, F27, F28)")
        self.assertEqual(tuple(full.group(1).split(", ")), mdo.AUDIT_DISCLOSURES)
        self.assertIsNone(mdo.AUDIT_PATTERN.search("Five disclosures are recorded below (F9, F10)."))

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
                     f"-output-directory={scratch}", "sections/mdo-l0-v2-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "mdo-l0-v2-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "mdo-l0-v2-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
