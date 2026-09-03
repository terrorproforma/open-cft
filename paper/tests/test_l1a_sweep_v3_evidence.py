"""Regression tests for the hash-bound L1a geometry sweep v3 paper evidence."""

from __future__ import annotations

import gzip
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
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_l1a_sweep_v3_evidence as swt  # noqa: E402

EVIDENCE = REPO / swt.EVIDENCE_PATH
GENERATED = REPO / swt.OUTPUT_PATH
SIDECAR = REPO / swt.SIDECAR_PATH
SECTION = REPO / swt.SECTION_PATH
STANDALONE = REPO / "paper/sections/l1a-sweep-v3-standalone.tex"
RESULTS = REPO / swt.RESULTS


def _load_artifact(relative: str):
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, check=False).returncode == 0


class SweepV3EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = swt.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.dataset = _load_artifact("artifacts/sweep-dataset.json")
        cls.designs = cls.dataset["designs"]
        cls.sobol = [d for d in cls.designs if d["set_id"] == "sobol_v3"]
        cls.held_out = [d for d in cls.designs if d["set_id"] == "sweep_v2"]

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = swt.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE, REPO / swt.MANIFEST_PATH, REPO / "paper/scripts/generate_l1a_sweep_v3_evidence.py"):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_committed_revisions_references_and_dashboard(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-l1a-sweep-v3-evidence")
        self.assertEqual(self.evidence["evidence_revision"], swt.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], swt.CLASSIFICATION)
        self.assertEqual(self.evidence["topology_label"], swt.TOPOLOGY_LABEL)
        self.assertEqual(self.evidence["recorded_outcome"], "accepted-screening")
        self.assertEqual(self.evidence["campaign_status"], "accepted_l1a_sweep_v3")
        head = _git("rev-parse", "HEAD")
        for commit in (swt.PREREGISTRATION_COMMIT_SHA, swt.RESULTS_COMMIT_SHA, swt.DASHBOARD_COMMIT_SHA, swt.LITERATURE_COMMIT_SHA, swt.SWEEP_V2_RESULTS_COMMIT_SHA, swt.TOPOLOGY_RESULTS_COMMIT_SHA, swt.WALL_LOSS_PREREGISTRATION_COMMIT_SHA):
            self.assertTrue(_ancestor(commit, head), commit)
        chain = (swt.TOPOLOGY_RESULTS_COMMIT_SHA, swt.LITERATURE_COMMIT_SHA, swt.PREREGISTRATION_COMMIT_SHA, swt.RESULTS_COMMIT_SHA, swt.DASHBOARD_COMMIT_SHA)
        for earlier, later in zip(chain, chain[1:]):
            self.assertTrue(_ancestor(earlier, later), (earlier, later))
            self.assertNotEqual(earlier, later)
        results_rel = swt.RESULTS.as_posix()
        parent_has_results = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{swt.RESULTS_COMMIT_SHA}^:{results_rel}"], cwd=REPO, check=False, capture_output=True,
        ).returncode
        self.assertNotEqual(parent_has_results, 0, "results tree exists before the record commit")
        tree = _git("rev-parse", f"{swt.RESULTS_COMMIT_SHA}:{results_rel}")
        self.assertEqual(self.evidence["binding"]["results_tree"], tree)
        self.assertEqual(_git("rev-parse", f"HEAD:{results_rel}"), tree)
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 979)
        self.assertEqual(self.evidence["bundle"]["artifact_count"], 990)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        for name in swt.FROZEN_FILES:
            relative = (swt.EXPERIMENT / name).as_posix()
            blobs = [_git("rev-parse", f"{commit}:{relative}") for commit in (swt.PREREGISTRATION_COMMIT_SHA, swt.RESULTS_COMMIT_SHA)]
            self.assertEqual(blobs[0], blobs[1], name)
            self.assertEqual(_git("hash-object", "--", relative), blobs[0], name)
        for key, path in (
            ("generator_sha256_lf", swt.DASHBOARD_GENERATOR),
            ("template_sha256_lf", swt.DASHBOARD_TEMPLATE),
            ("html_sha256_lf", swt.DASHBOARD_HTML),
        ):
            blob = subprocess.run(["git", "show", f"{swt.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob).hexdigest())
        self.assertEqual(self.evidence["dashboard"]["payload_manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        # Reference files hash to the sealed-source identity the bundle recorded and to their blobs.
        references = self.evidence["reference_artifacts"]["files"]
        self.assertEqual(set(references), {swt.SWEEP_V2_MANIFEST.as_posix(), swt.TOPOLOGY_PROTOCOL.as_posix(), swt.TOPOLOGY_P2_RECORD.as_posix(), swt.WALL_LOSS_PROTOCOL.as_posix()})
        self.assertEqual(references[swt.SWEEP_V2_MANIFEST.as_posix()]["sha256"], self.dataset["sealed_sources"]["sweep_v2"]["manifest_file_sha256"])
        self.assertEqual(references[swt.SWEEP_V2_MANIFEST.as_posix()]["revision"], swt.SWEEP_V2_RESULTS_COMMIT_SHA)
        self.assertEqual(references[swt.TOPOLOGY_PROTOCOL.as_posix()]["revision"], swt.TOPOLOGY_RESULTS_COMMIT_SHA)
        self.assertEqual(references[swt.WALL_LOSS_PROTOCOL.as_posix()]["revision"], swt.WALL_LOSS_PREREGISTRATION_COMMIT_SHA)
        definition = self.evidence["definition_sources"]
        self.assertEqual(definition["revision"], swt.LITERATURE_COMMIT_SHA)
        self.assertEqual(set(definition["files"]), {swt.LITERATURE_REVIEW.as_posix(), swt.PPM_CHECK_SCRIPT.as_posix(), swt.PPM_CHECK_OUTPUT.as_posix()})
        self.assertEqual(definition["literature_keys"], ["koch2007", "koch2011"])
        for path, meta in {**references, **definition["files"]}.items():
            raw = (REPO / path).read_bytes()
            self.assertEqual(len(raw), meta["bytes"], path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], path)
            committed = subprocess.run(["git", "cat-file", "blob", meta["git_blob"]], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(hashlib.sha256(committed).hexdigest(), meta["git_blob_sha256"], path)
            self.assertEqual(committed.replace(b"\r\n", b"\n"), raw.replace(b"\r\n", b"\n"), path)
        # The shakedown ran at the literature-review commit; the review is named by the frozen protocol.
        authorities = _load_artifact("artifacts/authorities.json")
        self.assertEqual(authorities["shakedown_git_head"], swt.LITERATURE_COMMIT_SHA)
        protocol = _load_artifact("artifacts/protocol.json")
        self.assertIn(swt.LITERATURE_REVIEW.as_posix(), protocol["purpose"])
        self.assertIn(swt.LITERATURE_COMMIT_SHA[:8], protocol["purpose"])
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], swt.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-screening")
        self.assertEqual(integration["manifest_id"], swt.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], swt.SECTION_HEADING)
        self.assertEqual(integration["prose_claim_ids"], list(swt.PROSE_CLAIM_IDS))
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(swt.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(swt.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(swt.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 330)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Swt") for name in names))
        artifacts = self.evidence["artifacts"]
        self.assertGreater(len(artifacts), 460)
        for relative, meta in artifacts.items():
            raw = (RESULTS / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
            self.assertEqual(len(raw), meta["bytes"], relative)
        loaded = {relative: _load_artifact(relative) for relative in artifacts if relative.endswith(".json")}
        derived_count = 0
        for item in macros:
            with self.subTest(macro=item["name"]):
                self.assertTrue(item["name"].isalpha())
                if item["derived"]:
                    derived_count += 1
                    self.assertTrue(item["derivation"])
                    self.assertTrue(item["inputs"])
                    for source in item["inputs"]:
                        artifact = source["artifact"]
                        if artifact == "manifest.json":
                            self.assertTrue((RESULTS / "manifest.json").is_file())
                            continue
                        if artifact.startswith("reference:") or artifact.startswith("definition-source:"):
                            self.assertTrue((REPO / artifact.split(":", 1)[1]).is_file(), artifact)
                            continue
                        self.assertIn(artifact, artifacts)
                        if source["pointer"]:
                            swt.resolve_pointer(loaded[artifact], source["pointer"])
                    self.assertEqual(swt.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                raw = swt.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(swt.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 120)
        with self.assertRaises(ValueError):
            swt.format_value("symbol", "\\input{evil}")

    def test_headline_hemp_like_and_hypothesis_recompute_from_the_rows(self) -> None:
        b = self.by_name
        sobol = self.sobol
        histogram = Counter(d["wall_cusp_count"] for d in sobol)
        self.assertEqual({int(k): v for k, v in self.dataset["estimands"]["sobol_v3"]["wall_cusp_count_histogram"].items()}, dict(histogram))
        self.assertEqual([b[f"SwtSobolHist{t}"]["raw"] for t in ("Two", "Three", "Four")], [histogram[2], histogram[3], histogram[4]])
        self.assertEqual(b["SwtSobolHistogramText"]["value"], "2:47 / 3:53 / 4:28")
        hemp = [d for d in sobol if d["hemp_like_all_cusps"]]
        self.assertEqual(len(hemp), 15)
        self.assertEqual(b["SwtHempLikeCount"]["raw"], 15)
        self.assertEqual(b["SwtHempLikeFraction"]["value"], "11.7\\%")
        for d in sobol:
            self.assertIs(d["hemp_like_all_cusps"], bool(d["rho"]) and all(r["rho_conservative"] >= 1.5 for r in d["rho"]))
        self.assertEqual(sorted(swt._short_id(d["design_id"]) for d in sobol if d["five_stage_four_cusp_hemp_like"]), ["005", "106"])
        self.assertEqual(b["SwtFiveStageFourCuspIds"]["value"], "\\texttt{005}, \\texttt{106}")
        x_star = swt.i1_root(1.5)
        self.assertAlmostEqual(x_star, 1.9373184746065641, places=12)
        self.assertAlmostEqual(swt.bessel_i(1, x_star), 1.5, places=12)
        self.assertEqual(b["SwtXStar"]["value"], "1.937318")
        self.assertEqual(b["SwtRwOverLStar"]["value"], "0.616668")
        bands = [(0.0, x_star), (x_star, 2.5), (2.5, 3.0), (3.0, 3.9)]
        expected = [(77, 0), (30, 5), (13, 4), (8, 6)]
        for (lo, hi), token, (designs, hemp_like) in zip(bands, ("BelowStar", "One", "Two", "Three"), expected):
            members = [d for d in sobol if lo <= d["x_w"] < hi]
            self.assertEqual(len(members), designs, token)
            self.assertEqual(sum(1 for d in members if d["hemp_like_all_cusps"]), hemp_like, token)
            self.assertEqual((b[f"SwtBand{token}Designs"]["raw"], b[f"SwtBand{token}HempLike"]["raw"]), (designs, hemp_like))
        end = [r["rho_conservative"] / d["ppm_prediction"]["i1_x_w"] for d in sobol for i, r in enumerate(d["rho"]) if i in (0, len(d["rho"]) - 1)]
        interior = [r["rho_conservative"] / d["ppm_prediction"]["i1_x_w"] for d in sobol for i, r in enumerate(d["rho"]) if 0 < i < len(d["rho"]) - 1]
        self.assertEqual((len(end), len(interior)), (256, 109))
        self.assertEqual((b["SwtEndCuspCount"]["raw"], b["SwtInteriorCuspCount"]["raw"]), (256, 109))
        self.assertEqual(b["SwtEndCuspRhoOverIOneMedian"]["raw"], statistics.median(end))
        self.assertEqual(b["SwtInteriorCuspRhoOverIOneMedian"]["raw"], statistics.median(interior))
        self.assertEqual((b["SwtEndCuspRhoOverIOneMedian"]["value"], b["SwtInteriorCuspRhoOverIOneMedian"]["value"]), ("0.80", "0.87"))
        self.assertLess(max(end), 1.0)
        self.assertEqual(b["SwtPredictedOnlyEndCuspFailures"]["raw"], 28)
        test = self.dataset["estimands"]["sobol_v3"]["hypothesis_test"]
        recomputed = swt.hypothesis_test(sobol, 0.25, x_star)
        self.assertEqual(recomputed["confusion_predicted_i1_vs_realised"], {"predicted_and_realised": 15, "predicted_not_realised": 36, "not_predicted_but_realised": 0, "neither": 77})
        self.assertTrue(math.isclose(recomputed["slope_through_origin"], test["slope_through_origin"], rel_tol=1e-9))
        self.assertTrue(math.isclose(recomputed["r_squared"], test["r_squared"], rel_tol=1e-9))
        self.assertTrue(math.isclose(recomputed["x_star_from_fitted_slope"], test["x_star_from_fitted_slope"], rel_tol=1e-9))
        self.assertEqual((b["SwtSlope"]["value"], b["SwtRSquared"]["value"], b["SwtBandFraction"]["value"], b["SwtAccuracy"]["value"]), ("0.689", "0.39", "70\\%", "0.72"))
        self.assertEqual((b["SwtXStarFromSlope"]["value"], b["SwtRwOverLStarFromSlope"]["value"]), ("2.34", "0.745"))
        self.assertIs(b["SwtHOneAsPredicted"]["raw"], False)
        self.assertIs(b["SwtHTwoAsPredicted"]["raw"], False)
        self.assertIs(b["SwtHTwoNoRegionHempLike"]["raw"], True)
        region = [d for d in self.designs if d["inside_sweep_v2_box"]]
        self.assertEqual(len(region), 102)
        self.assertEqual(sum(1 for d in region if d["hemp_like_all_cusps"]), 0)
        self.assertEqual(b["SwtRegionMaxRho"]["raw"], max(r["rho_conservative"] for d in region for r in d["rho"]))
        self.assertEqual(b["SwtRegionMaxRho"]["value"], "0.993")
        self.assertEqual(b["SwtRegionDesigns"]["raw"], 102)
        self.assertEqual(b["SwtSobolInsideVTwoBox"]["raw"], 6)
        self.assertEqual(b["SwtPooledCuspCount"]["raw"], 642)
        self.assertEqual(b["SwtCuspIsWallMaximumCount"]["raw"], 0)
        self.assertTrue(all(r["rho_wall"] < 1.0 for d in self.designs for r in d["rho"]))
        self.assertEqual((b["SwtHeldOutPassed"]["raw"], b["SwtHeldOutNulls"]["raw"], b["SwtHeldOutMaxUm"]["value"]), (96, 479, "27.3"))
        self.assertEqual((b["SwtBindingGateCount"]["raw"], b["SwtBindingGatesTrue"]["raw"]), (11, 11))
        self.assertEqual((b["SwtStableDesigns"]["raw"], b["SwtHempFlagStableDesigns"]["raw"]), (224, 224))
        self.assertEqual((b["SwtRhoSensitivityMedian"]["value"], b["SwtRhoSensitivityMax"]["value"]), ("0.9\\%", "8.0\\%"))
        self.assertEqual(b["SwtWallBThreeMedian"]["value"], "0.030")
        self.assertEqual((b["SwtVersion"]["raw"], b["SwtPriorVersion"]["raw"], b["SwtTopologyVersion"]["raw"], b["SwtWallLossVersion"]["raw"], b["SwtFieldModelLevel"]["raw"], b["SwtMaterialLevel"]["raw"]), ("v3", "v2", "v3.1", "v4", "L1a", "L1b"))
        for set_id, rows in (("sobol_v3", self.sobol), ("sweep_v2", self.held_out), ("pooled_all", self.designs), ("sweep_v2_region_pooled", region)):
            swt._compare_estimands(swt.set_estimands(rows, 0.25, x_star), self.dataset["estimands"][set_id], set_id)

    def test_review_launch_position_macros_recompute_from_the_committed_output(self) -> None:
        b = self.by_name
        output = json.loads((REPO / swt.PPM_CHECK_OUTPUT).read_bytes().replace(b"\r\n", b"\n"))
        cells = [c for block in output["reflections"].values() for c in block["cells"]]
        near = [c for c in cells if c["dist_to_centre_over_pitch"] <= 0.17]
        far = [c for c in cells if c["dist_to_centre_over_pitch"] >= 0.22]
        self.assertEqual((len(cells), len(near), len(far)), (16, 7, 9))
        self.assertEqual((b["SwtPpmLaunchCells"]["raw"], b["SwtPpmNearCells"]["raw"], b["SwtPpmFarCells"]["raw"]), (16, 7, 9))
        self.assertEqual((min(c["reflected"] for c in near), max(c["reflected"] for c in near)), (0, 1))
        self.assertEqual((min(c["reflected"] for c in far), max(c["reflected"] for c in far)), (32, 88))
        self.assertEqual((b["SwtPpmNearReflectionsMin"]["raw"], b["SwtPpmNearReflectionsMax"]["raw"], b["SwtPpmFarReflectionsMin"]["raw"], b["SwtPpmFarReflectionsMax"]["raw"]), (0, 1, 32, 88))
        self.assertEqual(b["SwtPpmOrbitsPerCell"]["raw"], 128)
        self.assertEqual((b["SwtPpmFarMinPitch"]["value"], b["SwtPpmFarMaxPitch"]["value"]), ("0.22", "0.48"))
        alphas = [e["mendel_alpha"] for r in output["results"] for e in r["electrons"]]
        self.assertEqual((b["SwtPpmMendelAlphaMin"]["raw"], b["SwtPpmMendelAlphaMax"]["raw"]), (min(alphas), max(alphas)))
        self.assertEqual((b["SwtPpmMendelAlphaMin"]["value"], b["SwtPpmMendelAlphaMax"]["value"]), ("9.93", "1190"))
        self.assertGreater(min(alphas), 0.66)
        self.assertEqual((b["SwtPpmEpsilonMin"]["value"], b["SwtPpmEpsilonMax"]["value"]), ("0.05", "0.75"))
        self.assertGreater(b["SwtPpmEpsilonMin"]["raw"], 0.03)
        self.assertIs(b["SwtPpmMuOrderedByEpsilon"]["raw"], True)
        self.assertEqual((b["SwtPpmMuMedianMin"]["value"], b["SwtPpmMuMedianMax"]["value"]), ("0.11", "0.42"))
        self.assertEqual(set(output["reflections"]), {d["design_id"] for d in self.held_out if d["representative"]})
        protocol = json.loads((REPO / swt.WALL_LOSS_PROTOCOL).read_bytes())
        launch = sorted({s["position_m"][2] for s in protocol["launches"]["position_seeds"]})
        record = json.loads((REPO / swt.TOPOLOGY_P2_RECORD).read_bytes())
        centres = record["geometry"]["stage_centres_m"]
        self.assertEqual(b["SwtVFourLaunchZMm"]["raw"], launch)
        self.assertEqual(b["SwtPTwoStageCentresMm"]["raw"], centres)
        self.assertEqual(b["SwtVFourLaunchZMm"]["value"], "3.5, 9.5, 15.5, 21.5")
        self.assertEqual(b["SwtPTwoStageCentresMm"]["value"], "3.0, 9.0, 15.0, 21.0")
        offsets = [min(abs(z - c) for c in centres) for z in launch]
        self.assertTrue(all(math.isclose(o, 0.5e-3, abs_tol=1e-12) for o in offsets))
        self.assertEqual(b["SwtVFourLaunchOffsetMm"]["value"], "0.5")
        self.assertEqual(b["SwtVFourLaunchOffsetPitch"]["value"], "0.083")
        self.assertIs(b["SwtVFourLaunchInNearClass"]["raw"], True)
        # The wall-loss campaign's own evidence macros carry the same launch planes.
        wlf = json.loads((REPO / "paper/evidence/wall-loss-v4.json").read_text(encoding="utf-8"))
        wlf_by = {m["name"]: m for m in wlf["macros"]}
        self.assertEqual([wlf_by[n]["raw"] for n in ("WlfCellOneZMm", "WlfCellTwoZMm", "WlfCellThreeZMm", "WlfCellFourZMm")], launch)

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Swt[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Swt[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertGreater(len(used), 150)
        for table in swt.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Swt"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n128 designs\n", "Swt"), ["1", "2", "8"])
        self.assertIn(f"\\subsection{{{swt.SECTION_HEADING}}}", self.section)
        for heading in ("Design space and method.", "Execution and integrity.", "Results: the HEMP-like regime.", "The preregistered hypothesis.", "The earlier design box.", "Scope."):
            self.assertIn(f"\\paragraph{{{heading}}}", self.section)
        self.assertIn("Model-bounded scope", self.section)
        for pattern in check_paper.PLACEHOLDERS.values():
            self.assertIsNone(pattern.search(self.section))
            self.assertIsNone(pattern.search(self.tex))
        for pattern in check_paper.FORBIDDEN_MODEL_WORDING.values():
            self.assertIsNone(pattern.search(self.section))
        self.assertEqual(check_paper.find_unregistered_claims(self.section), [])
        self.assertEqual(check_paper.find_unregistered_claims(self.tex), [])
        self.assertNotIn("CLM-", "\n".join(line for line in self.section.splitlines() if line.lstrip().startswith("%") and "claim records bound" not in line))
        for foreign in ("\\Swp", "\\Fcn", "\\Tch", "\\Wlg", "\\Wlf", "\\Mdo", "\\Mdb", "\\Fcc", "\\Ctv"):
            self.assertNotIn(foreign, self.section, "the section stays self-contained")
        # The section reports the hypothesis as recorded, never as confirmed.
        self.assertIn("did not hold as preregistered", self.section)
        self.assertIn("upper envelope", self.section)
        self.assertNotIn("confirmed the hypothesis", self.section)

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in swt.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        self.assertIn("wall radius $r_w$ (mm) & 1.40--2.20 & 1.40--4.20\\\\", self.tex)
        self.assertIn("stage pitch $L$ (mm) & 3.80--6.50 & 3.40--6.50\\\\", self.tex)
        self.assertIn("$r_w/L$ implied by the box & 0.215--0.579 & 0.215--1.235\\\\", self.tex)
        self.assertIn("designs (stages $3/4/5$) & 96 (26/45/25) & 128 (43/43/42)\\\\", self.tex)
        self.assertIn("HEMP-like designs ($\\rho \\ge 1.5$ at every cusp) & 0 & 15\\\\", self.tex)
        self.assertIn("$[0, x^*)$ & 0.39--1.49 & 77 & 0 & 0 & 219 &", self.tex)
        self.assertIn("$[x^*, 2.50)$ & 1.53--2.48 & 30 & 30 & 5 & 85 &", self.tex)
        self.assertIn("$[2.50, 3.00)$ & 2.55--3.88 & 13 & 13 & 4 & 40 &", self.tex)
        self.assertIn("$[3.00, 3.90)$ & 4.06--6.45 & 8 & 8 & 6 & 21 &", self.tex)
        self.assertIn("sweep-v2 region (102 designs) & 0.38--1.03 & 102 & 0 & 0 & 293 &", self.tex)
        self.assertIn("H1: slope of $\\rho$ on $I_1(x_w)$ through the origin & $[0.80, 1.00]$ & 0.689 & no\\\\", self.tex)
        self.assertIn("H2: prediction accuracy over 128 designs & $\\ge 0.85$ & 0.72 & no\\\\", self.tex)
        self.assertIn("H2: HEMP-like designs inside the sweep-v2 region & $0$ & 0 of 102 & yes\\\\", self.tex)
        self.assertIn("predicted and realised / predicted only / realised only / neither & reported & 15 / 36 / 0 / 77 & --\\\\", self.tex)
        self.assertIn("\\texttt{005} & 5 & 4 &", self.tex)
        self.assertIn("\\texttt{106} & 5 & 4 &", self.tex)
        self.assertEqual(self.tex.count("& yes\\\\"), 3)  # two five-stage four-cusp designs and the H2 region row
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], swt.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [swt.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertIn(swt.RECORDED_OUTCOME, sidecar["claim_status"])
        self.assertEqual(len(sidecar["reference_inputs"]), 4)
        self.assertEqual(len(sidecar["definition_source_inputs"]), 3)
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 4)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (swt.ARTIFACT_CLAIM_ID, swt.ARTIFACT_ID))
        self.assertEqual(self.evidence["tables"]["SwtHempLikeTable"]["rows"], 15)
        self.assertEqual(self.evidence["tables"]["SwtBandTable"]["rows"], 7)

    def test_tampered_bundle_dashboard_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\SwtHempLikeCount}{15}", "\\newcommand{\\SwtHempLikeCount}{16}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / swt.RESULTS
            target.parent.mkdir(parents=True)
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            self.assertIn(b'"sobol_hemp_like_count":15', original)
            victim.write_bytes(original.replace(b'"sobol_hemp_like_count":15', b'"sobol_hemp_like_count":16', 1))
            with self.assertRaises(ValueError):
                swt.Bundle(repo, swt.RESULTS, experiment_id=swt.EXPERIMENT_ID, expected_state="accepted_result")
            victim.write_bytes(original)
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                swt.Bundle(repo, swt.RESULTS, experiment_id=swt.EXPERIMENT_ID, expected_state="accepted_result")
        html = (REPO / swt.DASHBOARD_HTML).read_bytes()
        payload = swt.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_file_sha256"], self.evidence["bundle"]["manifest_sha256"])
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(swt.dashboard_payload(tampered)["identity"]["manifest_file_sha256"], payload["identity"]["manifest_file_sha256"])
        # A hypothesis record that claims a different slope is refused by the recomputation.
        rows = json.loads(json.dumps(self.sobol))
        rows[0]["rho"][0]["rho_conservative"] *= 1.5
        recomputed = swt.hypothesis_test(rows, 0.25, swt.i1_root(1.5))
        with self.assertRaises(ValueError):
            swt._compare_hypothesis(recomputed, self.dataset["estimands"]["sobol_v3"]["hypothesis_test"], "tampered")

    def test_field_grids_hash_to_the_design_records(self) -> None:
        for design in self.designs:
            if not (design["representative"] or design["hemp_like_all_cusps"]):
                continue
            record = _load_artifact(design["record_path"])
            raw = gzip.decompress((RESULTS / record["accepted_grid_path"]).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["accepted_grid_payload_sha256"])
            grid = json.loads(raw)
            self.assertEqual(grid["identity"]["accepted_field_identity_sha256"], design["identity"]["accepted_field_identity_sha256"])
            self.assertEqual([c["z_c_m"] for c in record["accepted"]["topology"]["wall_cusps"]], [c["z_c_m"] for c in design["wall_cusps"]])
            self.assertEqual([r["rho_conservative"] for r in record["descriptors"]["accepted"]["cusps"]], [r["rho_conservative"] for r in design["rho"]])

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
                     f"-output-directory={scratch}", "sections/l1a-sweep-v3-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "l1a-sweep-v3-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "l1a-sweep-v3-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
