"""Regression tests for the hash-bound L1b/P2 material-aware HEMP confirmation v1.1 paper evidence."""

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
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_l1b_hemp_confirmation_v1_1_evidence as hmc  # noqa: E402

EVIDENCE = REPO / hmc.EVIDENCE_PATH
GENERATED = REPO / hmc.OUTPUT_PATH
SIDECAR = REPO / hmc.SIDECAR_PATH
SECTION = REPO / hmc.SECTION_PATH
STANDALONE = REPO / "paper/sections/l1b-hemp-confirmation-v1-1-standalone.tex"
RESULTS = REPO / hmc.RESULTS
V1_RESULTS = REPO / hmc.V1_RESULTS


def _load_artifact(root: Path, relative: str):
    return json.loads((root / relative).read_bytes().decode("utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


class HempConfirmationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = hmc.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.dataset = _load_artifact(RESULTS, "artifacts/confirmation-dataset.json")
        cls.rows = cls.dataset["designs"]
        cls.pairs = [pair for row in cls.rows for pair in row["comparison"]["matched_cusps"]]

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = hmc.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_revisions_the_references_and_the_lineage(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-l1b-hemp-confirmation-v1-1-evidence")
        self.assertEqual(self.evidence["evidence_revision"], hmc.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], hmc.CLASSIFICATION)
        self.assertEqual(self.evidence["topology_label"], hmc.TOPOLOGY_LABEL)
        self.assertEqual(self.evidence["recorded_outcome"], hmc.RECORDED_OUTCOME)
        self.assertEqual(self.evidence["campaign_status"], hmc.CAMPAIGN_STATUS)
        self.assertEqual(self.evidence["verdict"], "CONFIRMED")
        head = _git("rev-parse", "HEAD")
        chain = (
            hmc.V1_CODE_COMMIT_SHA, hmc.V1_PREREGISTRATION_COMMIT_SHA, hmc.V1_RESULTS_COMMIT_SHA, hmc.CODE_COMMIT_SHA,
            hmc.PREREGISTRATION_COMMIT_SHA, hmc.RESULTS_COMMIT_SHA, hmc.DASHBOARD_COMMIT_SHA,
        )
        for earlier, later in zip(chain, chain[1:] + (head,), strict=True):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later], cwd=REPO, check=False).returncode, 0, (earlier, later))
        for commit in (hmc.SWEEP_V3_RESULTS_COMMIT_SHA, hmc.CUSP_TOPOLOGY_RESULTS_COMMIT_SHA):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", commit, hmc.V1_CODE_COMMIT_SHA], cwd=REPO, check=False).returncode, 0, commit)
        # Both record commits carry only their results trees, which are unchanged at HEAD.
        for commit, results in ((hmc.RESULTS_COMMIT_SHA, hmc.RESULTS), (hmc.V1_RESULTS_COMMIT_SHA, hmc.V1_RESULTS)):
            results_rel = results.as_posix()
            parent_has_results = subprocess.run(
                ["git", "rev-parse", "--verify", "-q", f"{commit}^:{results_rel}"], cwd=REPO, check=False, capture_output=True,
            ).returncode
            self.assertNotEqual(parent_has_results, 0, "results tree exists before the record commit")
            changed = _git("diff", "--name-only", f"{commit}~1", commit).split()
            self.assertTrue(changed and all(p.startswith(results_rel + "/") for p in changed))
            self.assertEqual(_git("rev-parse", f"HEAD:{results_rel}"), _git("rev-parse", f"{commit}:{results_rel}"))
        self.assertEqual(len(_git("diff", "--name-only", f"{hmc.RESULTS_COMMIT_SHA}~1", hmc.RESULTS_COMMIT_SHA).split()), 134)
        self.assertEqual(len(_git("diff", "--name-only", f"{hmc.V1_RESULTS_COMMIT_SHA}~1", hmc.V1_RESULTS_COMMIT_SHA).split()), 104)
        self.assertEqual(self.evidence["binding"]["results_tree"], _git("rev-parse", f"{hmc.RESULTS_COMMIT_SHA}:{hmc.RESULTS.as_posix()}"))
        manifest_rel = (hmc.RESULTS / "manifest.json").as_posix()
        self.assertEqual(self.evidence["binding"]["manifest_git_blob"], _git("rev-parse", f"{hmc.RESULTS_COMMIT_SHA}:{manifest_rel}"))
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 133)
        self.assertEqual(self.evidence["bundle"]["artifact_count"], 142)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        # The sealed source hashes recompute from the blobs at the rebased preregistration commit.
        recomputed = self.evidence["binding"]["source_hashes_recomputed_at_preregistration"]
        authorities = _load_artifact(RESULTS, "artifacts/authorities.json")
        for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
            self.assertEqual(recomputed[key], authorities[key])
        source_binding = _load_artifact(RESULTS, "artifacts/source-binding.json")
        self.assertEqual(hmc._committed_source_hash(REPO, hmc.PREREGISTRATION_COMMIT_SHA, source_binding["experiment_code_files"], hmc.EXPERIMENT.as_posix() + "/"), authorities["experiment_code_sha256"])
        self.assertNotEqual(hmc._committed_source_hash(REPO, hmc.V1_PREREGISTRATION_COMMIT_SHA, source_binding["experiment_code_files"], hmc.V1_EXPERIMENT.as_posix() + "/"), authorities["experiment_code_sha256"])
        # The lock names the pre-rebase preregistration commit, recorded as a string and never resolved.
        lock = _load_artifact(RESULTS, "execution-lock.json")
        self.assertNotEqual(lock["commit"], hmc.PREREGISTRATION_COMMIT_SHA)
        self.assertEqual(self.by_name["HmcLockCommit"]["raw"], lock["commit"])
        # Frozen files: same blob at preregistration and results revisions, for both campaigns.
        for experiment, prereg, results in ((hmc.EXPERIMENT, hmc.PREREGISTRATION_COMMIT_SHA, hmc.RESULTS_COMMIT_SHA), (hmc.V1_EXPERIMENT, hmc.V1_PREREGISTRATION_COMMIT_SHA, hmc.V1_RESULTS_COMMIT_SHA)):
            for name in hmc.FROZEN_FILES:
                relative = (experiment / name).as_posix()
                blobs = [_git("rev-parse", f"{commit}:{relative}") for commit in (prereg, results)]
                self.assertEqual(blobs[0], blobs[1], name)
        # Reference and lineage files are bound at their revisions and equal the checkout.
        for group, lf in (("reference_artifacts", False), ("lineage", True)):
            for path, meta in self.evidence[group]["files"].items():
                blob = subprocess.run(["git", "show", f"{meta['revision']}:{path}"], cwd=REPO, check=True, capture_output=True).stdout
                working = (REPO / path).read_bytes()
                self.assertEqual(meta["git_blob_sha256"], hashlib.sha256(blob).hexdigest(), path)
                self.assertEqual(meta["sha256"], hashlib.sha256(working).hexdigest(), path)
                if lf:
                    self.assertEqual(blob.replace(b"\r\n", b"\n"), working.replace(b"\r\n", b"\n"), path)
                else:
                    self.assertEqual(blob, working, path)
        sealed = self.dataset["sealed_sources"]["l1a_geometry_sweep_v3"]
        self.assertEqual(self.evidence["reference_artifacts"]["files"][hmc.SWEEP_V3_CATALOGUE_PATH.as_posix()]["sha256"], sealed["catalogue_byte_sha256"])
        self.assertEqual(self.evidence["reference_artifacts"]["files"][hmc.SWEEP_V3_MANIFEST_PATH.as_posix()]["sha256"], sealed["manifest_file_sha256"])
        lineage = self.evidence["lineage"]
        self.assertEqual(lineage["terminal_state"], "development_rejection")
        self.assertEqual((lineage["resolved_design_count"], lineage["failed_design_count"]), (13, 2))
        self.assertEqual(lineage["failed_designs"], ["l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"])
        self.assertEqual(lineage["angle_gate_deg"], 10.0)
        self.assertEqual(tuple(lineage["protocol_paths_changed"]), hmc.ALLOWED_PROTOCOL_CHANGES)
        self.assertEqual(lineage["manifest_sha256"], hashlib.sha256((V1_RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(lineage["verified_file_count"], 103)
        self.assertIs(lineage["cited_for_numbers"], False)
        # The dashboard is bound by LF-normalised SHA-256 equal to the blob committed at its revision.
        for key, path in (("generator_sha256_lf", hmc.DASHBOARD_GENERATOR), ("template_sha256_lf", hmc.DASHBOARD_TEMPLATE), ("html_sha256_lf", hmc.DASHBOARD_HTML)):
            blob = subprocess.run(["git", "show", f"{hmc.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest())
        self.assertEqual(self.evidence["dashboard"]["payload_manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        self.assertEqual(self.evidence["dashboard"]["payload_predecessor_manifest_sha256"], lineage["manifest_sha256"])
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], hmc.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-screening")
        self.assertEqual(integration["manifest_id"], hmc.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], hmc.SECTION_HEADING)
        self.assertEqual(integration["prose_claim_ids"], list(hmc.PROSE_CLAIM_IDS))
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(hmc.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(hmc.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(hmc.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 300)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Hmc") for name in names))
        roots = {"results": (RESULTS, self.evidence["artifacts"]), "lineage": (V1_RESULTS, self.evidence["lineage_artifacts"])}
        for root, artifacts in roots.values():
            for relative, meta in artifacts.items():
                raw = (root / relative).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
                self.assertEqual(len(raw), meta["bytes"], relative)
        self.assertEqual(len(self.evidence["artifacts"]), 56)
        loaded = {
            scope: {relative: _load_artifact(root, relative) for relative in artifacts if relative.endswith(".json")}
            for scope, (root, artifacts) in roots.items()
        }
        for scope, (root, _artifacts) in roots.items():
            loaded[scope]["manifest.json"] = _load_artifact(root, "manifest.json")
        derived_count = 0
        for item in macros:
            with self.subTest(macro=item["name"]):
                self.assertTrue(item["name"].isalpha())
                self.assertIn(item["bundle"], roots)
                root, artifacts = roots[item["bundle"]]
                if item["derived"]:
                    derived_count += 1
                    self.assertTrue(item["derivation"])
                    self.assertTrue(item["inputs"])
                    for source in item["inputs"]:
                        if source["artifact"] != "manifest.json":
                            self.assertIn(source["artifact"], artifacts)
                        hmc.resolve_pointer(loaded[item["bundle"]][source["artifact"]], source["pointer"])
                    self.assertEqual(hmc.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                if source["artifact"] != "manifest.json":
                    self.assertIn(source["artifact"], artifacts)
                raw = hmc.resolve_pointer(loaded[item["bundle"]][source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(hmc.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 100)
        self.assertEqual(sum(1 for item in macros if item["bundle"] == "lineage"), 18)

    def test_verdict_gates_and_headline_recompute_from_the_rows(self) -> None:
        b = self.by_name
        rows = self.rows
        pairs = self.pairs
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(pairs), 37)
        self.assertEqual((b["HmcDesignCount"]["raw"], b["HmcMatchedCusps"]["raw"], b["HmcSolvesConverged"]["raw"]), (15, 37, 30))
        strict = sum(row["comparison"]["p2_wall_cusp_count"] == row["comparison"]["l1a_wall_cusp_count"] for row in rows)
        self.assertEqual((strict, b["HmcGateBAgreeStrict"]["raw"], b["HmcGateBAgreeTolerant"]["raw"]), (15, 15, 15))
        shifts = [pair["shift_m"] for pair in pairs]
        over = [pair["shift_over_tolerance"] for pair in pairs]
        self.assertEqual(b["HmcShiftMaxMm"]["raw"], max(shifts))
        self.assertEqual(b["HmcShiftMedianMm"]["raw"], statistics.median(shifts))
        self.assertEqual(b["HmcGateCMaxShiftOverTol"]["raw"], max(over))
        self.assertEqual((b["HmcShiftMaxMm"]["value"], b["HmcShiftMedianMm"]["value"], b["HmcGateCMaxShiftOverTol"]["value"], b["HmcShiftMinUm"]["value"]), ("0.362", "0.267", "0.80", "1.3"))
        self.assertEqual((b["HmcMaxShiftDesign"]["raw"], b["HmcMinShiftDesign"]["raw"]), ("076", "015"))
        design_pairs = [(design, pair) for design in rows for pair in design["comparison"]["matched_cusps"]]
        for design, pair in design_pairs:
            tolerance = max(design["geometry"]["wall_radius_m"] / 8, 0.0004513888888888889)
            self.assertAlmostEqual(design["comparison"]["cusp_position_tolerance_m"], tolerance, places=15)
            self.assertAlmostEqual(pair["shift_m"], abs(pair["p2_z_c_m"] - pair["l1a_z_c_m"]), places=15)
            self.assertLessEqual(pair["shift_m"], tolerance)
            self.assertGreater(pair["shift_m"], design["p2_discretisation"]["max_wall_intersection_shift_m"])
            self.assertAlmostEqual(pair["wall_b_ratio_p2_over_l1a"], pair["p2_wall_b_t"] / pair["l1a_wall_b_t"], places=12)
            self.assertAlmostEqual(pair["rho_conservative_ratio_p2_over_l1a"], pair["p2_rho_conservative"] / pair["l1a_rho_conservative"], places=12)
        self.assertEqual((b["HmcToleranceMinMm"]["value"], b["HmcToleranceMaxMm"]["value"], b["HmcLOneADzMm"]["value"]), ("0.45", "0.52", "0.451"))
        self.assertEqual((b["HmcShiftsAboveStability"]["raw"], b["HmcShiftsAboveDiscretisation"]["raw"]), (22, 37))
        # Verdict by the predeclared rule.
        b_passed = b["HmcGateBAgreeTolerant"]["raw"] / 15 >= 1.0
        c_passed = b["HmcAllBijective"]["raw"] and b["HmcGateCMaxShiftOverTol"]["raw"] <= 1.0
        self.assertTrue(b_passed and c_passed)
        self.assertEqual(b["HmcVerdict"]["raw"], "CONFIRMED")
        # Reported (d): Koch ratios recompute from the wall field and the adjacent axis peaks.
        for row in rows:
            for item in row["p2_rho"] + row["l1a"]["rho"]:
                self.assertAlmostEqual(item["rho_conservative"], item["wall_b_t"] / max(item["upstream_axis_peak_t"], item["downstream_axis_peak_t"]), places=12)
            p2_hemp = all(item["rho_conservative"] >= 1.5 for item in row["p2_rho"])
            self.assertIs(row["comparison"]["p2_hemp_like_all_cusps"], p2_hemp)
        lost = [row["design_id"] for row in rows if not row["comparison"]["p2_hemp_like_all_cusps"]]
        self.assertEqual(lost, ["l1a-gs-v3-028-f012c0bf33"])
        self.assertEqual((b["HmcPreservedCount"]["raw"], b["HmcLostDesign"]["raw"], b["HmcLostDesignLOneARho"]["value"], b["HmcLostDesignPTwoRho"]["value"]), (14, "028", "1.515", "1.464"))
        wall = [pair["wall_b_ratio_p2_over_l1a"] for pair in pairs]
        rho = [pair["rho_conservative_ratio_p2_over_l1a"] for pair in pairs]
        self.assertEqual((b["HmcWallBRatioMin"]["raw"], b["HmcWallBRatioMedian"]["raw"], b["HmcWallBRatioMax"]["raw"]), (min(wall), statistics.median(wall), max(wall)))
        self.assertEqual((b["HmcWallBRatioMin"]["value"], b["HmcWallBRatioMedian"]["value"], b["HmcWallBRatioMax"]["value"]), ("1.05", "1.23", "1.53"))
        self.assertEqual((b["HmcWallBRaiseMinPct"]["value"], b["HmcWallBRaiseMaxPct"]["value"]), ("$+$5\\%", "$+$53\\%"))
        self.assertEqual((b["HmcRhoRatioMin"]["value"], b["HmcRhoRatioMedian"]["value"], b["HmcRhoRatioMax"]["value"]), ("0.94", "1.06", "1.45"))
        self.assertEqual((b["HmcRhoRatioBelowOneCusps"]["raw"], b["HmcRhoRatioBelowOneDesigns"]["raw"]), (sum(1 for r in rho if r < 1.0), 8))
        self.assertEqual((b["HmcAxisPeakRatioMin"]["value"], b["HmcAxisPeakRatioMax"]["value"]), ("0.98", "1.35"))
        # Axis nulls and lean.
        channel_shifts = [s for row in rows for s in row["comparison"]["channel_axis_nulls"]["sorted_shifts_m"]]
        self.assertEqual(b["HmcChannelNullShiftMaxMm"]["raw"], max(channel_shifts))
        self.assertEqual((b["HmcChannelNullShiftMaxMm"]["value"], b["HmcChannelNullBijection"]["raw"], b["HmcChannelNullsBeyondTolDesigns"]["raw"], b["HmcPooledNullBijection"]["raw"]), ("1.07", 6, 9, 0))
        self.assertEqual((b["HmcLeanLOneAMaxMm"]["value"], b["HmcLeanPTwoMaxMm"]["value"]), ("0.46", "1.14"))
        self.assertEqual((b["HmcLOneAOutsideNulls"]["raw"], b["HmcPTwoOutsideNulls"]["raw"], b["HmcMorePTwoNullDesigns"]["raw"]), (11, 29, 9))
        # Solve evidence.
        self.assertEqual((b["HmcLevelZeroDofsMin"]["value"], b["HmcLevelZeroDofsMax"]["value"], b["HmcLevelOneDofsMin"]["value"], b["HmcLevelOneDofsMax"]["value"]), ("24{,}369", "116{,}883", "50{,}037", "466{,}005"))
        self.assertEqual(b["HmcResidualMax"]["value"], "$2.00\\times10^{-10}$")
        self.assertLessEqual(b["HmcResidualMax"]["raw"], 2.0e-10)
        self.assertEqual((b["HmcStageWallS"]["value"], b["HmcAssessmentWallS"]["value"], b["HmcPeakRssMb"]["value"], b["HmcWorkerPool"]["raw"]), ("3079", "305", "240", 1))
        self.assertEqual((b["HmcDiscShiftMaxUm"]["value"], b["HmcSamplingStable"]["raw"], b["HmcDiscStable"]["raw"]), ("1.4", 15, 15))
        self.assertEqual((b["HmcIronMuR"]["value"], b["HmcMagnetRecoilMuR"]["value"], b["HmcAngleGateDeg"]["value"], b["HmcVOneAngleGateDeg"]["value"]), ("4000", "1.05", "5", "10"))
        self.assertEqual((b["HmcXwMin"]["value"], b["HmcXwMax"]["value"], b["HmcRwOverLMin"]["value"], b["HmcRwOverLMax"]["value"]), ("2.25", "3.24", "0.715", "1.032"))
        x_w = [math.pi * row["geometry"]["wall_radius_m"] / row["derived"]["represented_stage_pitch_m"] for row in rows]
        self.assertAlmostEqual(b["HmcXwMin"]["raw"], min(x_w), places=12)

    def test_lineage_disclosure_recomputes_from_the_predecessor_bundle(self) -> None:
        b = self.by_name
        terminal = _load_artifact(V1_RESULTS, "terminal.json")
        failures = _load_artifact(V1_RESULTS, "artifacts/design-failures.json")
        self.assertEqual(terminal["state"], "development_rejection")
        self.assertEqual((terminal["payload"]["resolved_design_count"], terminal["payload"]["failed_design_count"]), (13, 2))
        self.assertEqual([f["key"].split(":")[1] for f in failures["failed"]], ["l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"])
        self.assertTrue(all(f["stage"] == "resolve" for f in failures["failed"]))
        self.assertEqual((b["HmcVOneResolved"]["raw"], b["HmcVOneFailed"]["raw"], b["HmcVOneFailedDesigns"]["value"], b["HmcVOneStageWallMin"]["value"]), (13, 2, "028, 048", "45.6"))
        self.assertEqual((b["HmcVOneAssessmentAccess"]["raw"], b["HmcVOneRecords"]["raw"], b["HmcVOneVerifiedFiles"]["raw"]), (0, 13, 103))
        v1_protocol = _load_artifact(V1_RESULTS, "artifacts/protocol.json")
        protocol = _load_artifact(RESULTS, "artifacts/protocol.json")
        diff = hmc._diff_paths(v1_protocol, protocol)
        self.assertEqual(tuple(diff), hmc.ALLOWED_PROTOCOL_CHANGES)
        self.assertEqual((b["HmcProtocolChangedPaths"]["raw"], b["HmcProtocolDeclarationsChanged"]["raw"], b["HmcProtocolBlocksUnchanged"]["raw"]), (12, 2, True))
        for block in ("comparison", "gates", "definition_v3_import", "design_sets", "claim_boundary"):
            self.assertEqual(v1_protocol[block], protocol[block], block)
        self.assertEqual((v1_protocol["p2"]["mesh"]["reject_below_angle_deg"], protocol["p2"]["mesh"]["reject_below_angle_deg"]), (10.0, 5.0))
        self.assertEqual(protocol["predecessor"]["preregistration_commit"], _load_artifact(V1_RESULTS, "execution-lock.json")["commit"])
        # Sliver record of the two rejected designs on the v1.1 meshes (the recorded minimum angles, not the note's population angle).
        self.assertEqual((b["HmcSliverDesignA"]["raw"], b["HmcSliverAMinAngleDeg"]["value"], b["HmcSliverABelowTen"]["value"]), ("028", "5.3", "3"))
        self.assertEqual((b["HmcSliverDesignB"]["raw"], b["HmcSliverBMinAngleDeg"]["value"], b["HmcSliverBBelowTen"]["value"]), ("048", "5.6", "13{,}816"))
        for design_id, macro in (("l1a-gs-v3-028-f012c0bf33", "HmcSliverAMinAngleDeg"), ("l1a-gs-v3-048-aabacb3a59", "HmcSliverBMinAngleDeg")):
            record = _load_artifact(RESULTS, next(row["record_path"] for row in self.rows if row["design_id"] == design_id))
            self.assertEqual(record["evidence"]["p2"]["levels"][0]["mesh_quality"]["minimum_angle_deg"], b[macro]["raw"])
            self.assertLess(b[macro]["raw"], 10.0)
            self.assertGreaterEqual(b[macro]["raw"], 5.0)
        shakedown = _load_artifact(RESULTS, "artifacts/shakedown.json")
        self.assertEqual(sorted(shakedown["mesh_preflight"]["designs_with_elements_below_10deg"]), ["l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59"])
        self.assertEqual((b["HmcPreflightDesigns"]["raw"], b["HmcPreflightPassed"]["raw"], b["HmcPreflightMinAngleDeg"]["value"]), (15, 15, "5.3"))
        self.assertEqual((b["HmcShakedownDesigns"]["raw"], b["HmcShakedownOverlapDesigns"]["raw"], b["HmcShakedownDesignIds"]["value"]), (5, 5, "015, 036, 106, 028, 048"))
        self.assertEqual((b["HmcTimingProjectedMin"]["value"], b["HmcTimingBudgetMin"]["value"], b["HmcTimingWithinBudget"]["raw"], b["HmcStageWithinBudget"]["raw"]), ("100.3", "90.0", False, True))
        self.assertEqual(b["HmcPaperAdmissionRecord"]["raw"], "NOT in scope of this campaign; the result records what a numerical-screening admission would state")
        note = (REPO / hmc.V1_REJECTION_PATH).read_text(encoding="utf-8")
        for phrase in ("`b9449ee5`", "`978c71be`", "13/15 designs resolved", "No assessment, gates, verdict or dashboard exist for v1", "2738 s"):
            self.assertIn(phrase, note)

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Hmc[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Hmc[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertGreater(len(used), 200)
        for table in hmc.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Hmc"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n15 designs\n", "Hmc"), ["1", "5"])
        self.assertIn(f"\\subsection{{{hmc.SECTION_HEADING}}}", self.section)
        for heading in ("Method.", "Execution and integrity.", "Results: cusp count and positions.", "Reported: field ratios, the HEMP-like flag and the axis nulls.", "Disclosures.", "Scope."):
            self.assertIn(f"\\paragraph{{{heading}}}", self.section)
        self.assertIn("Model-bounded scope", self.section)
        for pattern in check_paper.PLACEHOLDERS.values():
            self.assertIsNone(pattern.search(self.section))
            self.assertIsNone(pattern.search(self.tex))
        for pattern in check_paper.FORBIDDEN_MODEL_WORDING.values():
            self.assertIsNone(pattern.search(self.section))
        self.assertEqual(check_paper.find_unregistered_claims(self.section), [])
        self.assertEqual(check_paper.find_unregistered_claims(self.tex), [])
        self.assertNotIn("CLM-", "\n".join(line for line in self.section.splitlines() if line.lstrip().startswith("%")))
        for other in ("\\Wlf", "\\Wlg", "\\Wlh", "\\Ctv", "\\Swt"):
            self.assertNotIn(other, self.section, "the section stays self-contained")
        for phrase in ("never a probability", "not a positive finding about the thruster", "no design is recommended", "opens no physics level", "Three disclosures", "not robust", "re-preregistration of a predecessor", "not the campaign's"):
            self.assertIn(phrase, check_paper._normalize_tex(self.section))

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in hmc.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        self.assertIn("\\texttt{028} & 3 & 2.33 & 0.741 & 2 / 2 & 2 & 0.349 (0.77) & 0.51 & 1.12--1.14 & 1.17 & 1.515 $\\to$ 1.464 & no & 81{,}607\\\\", self.tex)
        self.assertIn("\\texttt{076} & 3 & 2.38 & 0.756 & 2 / 2 & 2 & 0.362 (0.80) & 0.54 & 1.12 & 1.16 & 1.676 $\\to$ 1.615 & yes & 209{,}453\\\\", self.tex)
        self.assertIn("(a) P2 solves converged at relative true residual $\\le$ $2.0\\times10^{-10}$ & 30 of 30 (largest residual $2.00\\times10^{-10}$)\\\\", self.tex)
        self.assertIn("(c) largest shift in tolerance units (threshold 1.0) & 0.80 (design \\texttt{076}); passed: yes\\\\", self.tex)
        self.assertIn("verdict by the predeclared rule & \\texttt{CONFIRMED}\\\\", self.tex)
        self.assertIn("(d) designs HEMP-like under P2 (reported, not gated) & 14 of 15 (lost: \\texttt{028}, $\\rho_{\\min}$ 1.515 $\\to$ 1.464)\\\\", self.tex)
        self.assertIn("channel axis nulls: designs with equal count / in bijection within the cusp tolerance & 15 / 6 of 15\\\\", self.tex)
        self.assertIn("soft-iron poles (one per inter-magnet gap) and return yoke: relative permeability & 4000 (linear; no saturation, no B-H curve)\\\\", self.tex)
        self.assertIn("(ii) angle gate v1 $\\to$ v1.1 & 10$^\\circ$ $\\to$ 5$^\\circ$ (disclosed in the frozen protocol)\\\\", self.tex)
        self.assertIn("(iii) shakedown timing projection / budget (min); within budget & 100.3 / 90.0; no (contention factor 1.5)\\\\", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], hmc.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [hmc.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertEqual(len(sidecar["reference_inputs"]), 4)
        self.assertEqual(len(sidecar["lineage_inputs"]), 16)
        self.assertIn(hmc.RECORDED_OUTCOME, sidecar["claim_status"])
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 4)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (hmc.ARTIFACT_CLAIM_ID, hmc.ARTIFACT_ID))
        self.assertEqual(self.evidence["tables"]["HmcDesignTable"]["rows"], 15)
        self.assertEqual(self.evidence["tables"]["HmcDisclosureTable"]["rows"], 15)

    def test_tampered_bundle_grid_or_dashboard_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\HmcMatchedCusps}{37}", "\\newcommand{\\HmcMatchedCusps}{38}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / hmc.RESULTS
            target.parent.mkdir(parents=True)
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            self.assertIn(b'"verdict":"CONFIRMED"', original)
            victim.write_bytes(original.replace(b'"verdict":"CONFIRMED"', b'"verdict":"DISCONFIRMD"', 1))
            with self.assertRaises(ValueError):
                hmc.Bundle(repo, hmc.RESULTS, hmc.EXPERIMENT_ID, "accepted_result")
            victim.write_bytes(original)
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                hmc.Bundle(repo, hmc.RESULTS, hmc.EXPERIMENT_ID, "accepted_result")
            sidecar_victim.write_bytes((RESULTS / "artifacts" / "gates.json.sha256.json").read_bytes())
            (target / "artifacts" / "stray.json").write_bytes(b"{}")
            with self.assertRaises(ValueError):
                hmc.Bundle(repo, hmc.RESULTS, hmc.EXPERIMENT_ID, "accepted_result")
        # A field grid whose payload hash differs from the record binding is refused.
        row = self.rows[0]
        record = _load_artifact(RESULTS, row["record_path"])
        bundle = hmc.Bundle(REPO, hmc.RESULTS, hmc.EXPERIMENT_ID, "accepted_result")
        grid_rel = row.get("accepted_grid_path", record["accepted_grid_path"])
        payload = gzip.decompress((RESULTS / grid_rel).read_bytes())
        self.assertEqual(hashlib.sha256(payload).hexdigest(), record["accepted_grid_payload_sha256"])
        with self.assertRaises(ValueError):
            bundle.load_gz(grid_rel, "0" * 64)
        # A dashboard whose payload names another manifest is refused before any macro is written.
        html = (REPO / hmc.DASHBOARD_HTML).read_bytes()
        payload = hmc.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_file_sha256"], self.evidence["bundle"]["manifest_sha256"])
        self.assertEqual(payload["predecessor"]["manifest_file_sha256"], self.evidence["lineage"]["manifest_sha256"])
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(hmc.dashboard_payload(tampered)["identity"]["manifest_file_sha256"], payload["identity"]["manifest_file_sha256"])
        # The protocol diff refuses an undeclared change.
        v1_protocol = _load_artifact(V1_RESULTS, "artifacts/protocol.json")
        protocol = _load_artifact(RESULTS, "artifacts/protocol.json")
        protocol["comparison"]["l1a_dz_m"] = 0.0005
        self.assertNotEqual(tuple(hmc._diff_paths(v1_protocol, protocol)), hmc.ALLOWED_PROTOCOL_CHANGES)

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
                     f"-output-directory={scratch}", "sections/l1b-hemp-confirmation-v1-1-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "l1b-hemp-confirmation-v1-1-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "l1b-hemp-confirmation-v1-1-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
