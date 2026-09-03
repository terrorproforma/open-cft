"""Regression tests for the hash-bound cusp topology search v3.1 paper evidence."""

from __future__ import annotations

import hashlib
import json
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
import generate_cusp_topology_v3_1_evidence as ctv  # noqa: E402

EVIDENCE = REPO / ctv.EVIDENCE_PATH
GENERATED = REPO / ctv.OUTPUT_PATH
SIDECAR = REPO / ctv.SIDECAR_PATH
SECTION = REPO / ctv.SECTION_PATH
STANDALONE = REPO / "paper/sections/cusp-topology-v3-1-standalone.tex"
RESULTS = REPO / ctv.RESULTS
LINEAGE_RESULTS = REPO / ctv.LINEAGE_RESULTS


def _load_artifact(relative: str):
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


class CuspTopologyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = ctv.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.dataset = _load_artifact("artifacts/topology-dataset.json")

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = ctv.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_committed_revisions_lineage_and_dashboard(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-cusp-topology-v3-1-evidence")
        self.assertEqual(self.evidence["evidence_revision"], ctv.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], ctv.CLASSIFICATION)
        self.assertEqual(self.evidence["p2_row_classification"], ctv.P2_CLASSIFICATION)
        self.assertEqual(self.evidence["recorded_outcome"], ctv.RECORDED_OUTCOME)
        self.assertEqual(self.evidence["campaign_status"], ctv.CAMPAIGN_STATUS)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()

        def ancestor(a: str, b: str) -> bool:
            return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, check=False).returncode == 0

        for commit in (ctv.PREREGISTRATION_COMMIT_SHA, ctv.RESULTS_COMMIT_SHA, ctv.DASHBOARD_COMMIT_SHA, ctv.LINEAGE_RESULTS_COMMIT_SHA, ctv.LINEAGE_AUDIT_COMMIT_SHA, ctv.LITERATURE_COMMIT_SHA):
            self.assertTrue(ancestor(commit, head), commit)
        chain = (ctv.LITERATURE_COMMIT_SHA, ctv.LINEAGE_PREREGISTRATION_COMMIT_SHA, ctv.LINEAGE_RESULTS_COMMIT_SHA, ctv.LINEAGE_AUDIT_COMMIT_SHA, ctv.PREREGISTRATION_COMMIT_SHA, ctv.RESULTS_COMMIT_SHA, ctv.DASHBOARD_COMMIT_SHA)
        for earlier, later in zip(chain, chain[1:]):
            self.assertTrue(ancestor(earlier, later), (earlier, later))
            self.assertNotEqual(earlier, later)
        results_rel = ctv.RESULTS.as_posix()
        parent_has_results = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{ctv.RESULTS_COMMIT_SHA}^:{results_rel}"], cwd=REPO, check=False, capture_output=True,
        ).returncode
        self.assertNotEqual(parent_has_results, 0, "results tree exists before the record commit")
        tree = subprocess.run(["git", "rev-parse", f"{ctv.RESULTS_COMMIT_SHA}:{results_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(self.evidence["binding"]["results_tree"], tree)
        self.assertEqual(subprocess.run(["git", "rev-parse", f"HEAD:{results_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip(), tree)
        lineage_rel = ctv.LINEAGE_RESULTS.as_posix()
        lineage_tree = subprocess.run(["git", "rev-parse", f"{ctv.LINEAGE_RESULTS_COMMIT_SHA}:{lineage_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(self.evidence["binding"]["lineage"]["results_tree"], lineage_tree)
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 1211)
        self.assertEqual(self.evidence["bundle"]["artifact_count"], 1226)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        self.assertEqual(self.evidence["lineage_bundle"]["manifest_sha256"], hashlib.sha256((LINEAGE_RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["lineage_bundle"]["state"], "assessment_rejection")
        for name in ctv.FROZEN_FILES:
            relative = (ctv.EXPERIMENT / name).as_posix()
            blobs = [
                subprocess.run(["git", "rev-parse", f"{commit}:{relative}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
                for commit in (ctv.PREREGISTRATION_COMMIT_SHA, ctv.RESULTS_COMMIT_SHA)
            ]
            self.assertEqual(blobs[0], blobs[1], name)
        for key, path in (
            ("generator_sha256_lf", ctv.DASHBOARD_GENERATOR),
            ("template_sha256_lf", ctv.DASHBOARD_TEMPLATE),
            ("html_sha256_lf", ctv.DASHBOARD_HTML),
        ):
            blob = subprocess.run(["git", "show", f"{ctv.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob).hexdigest())
        self.assertEqual(self.evidence["dashboard"]["payload_manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        self.assertEqual(self.evidence["dashboard"]["payload_lineage_manifest_sha256"], self.evidence["lineage_bundle"]["manifest_sha256"])
        # Reference files hash to the sealed-source identities the bundle recorded.
        sealed = self.dataset["sealed_sources"]
        references = self.evidence["reference_artifacts"]["files"]
        self.assertEqual(references[ctv.V1_DATASET.as_posix()]["sha256"], sealed["characterization_v1"]["dataset_file_sha256"])
        self.assertEqual(references[ctv.V2_DATASET.as_posix()]["sha256"], sealed["four_cell_v2"]["dataset_file_sha256"])
        self.assertEqual(references[ctv.SWEEP_MANIFEST.as_posix()]["sha256"], sealed["sweep_v2"]["manifest_file_sha256"])
        for path, meta in {**references, **self.evidence["lineage_artifacts"]["files"], **self.evidence["definition_sources"]["files"]}.items():
            raw = (REPO / path).read_bytes()
            self.assertEqual(len(raw), meta["bytes"], path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], path)
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], ctv.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-screening")
        self.assertEqual(integration["manifest_id"], ctv.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], ctv.SECTION_HEADING)
        self.assertEqual(integration["prose_claim_ids"], list(ctv.PROSE_CLAIM_IDS))
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(ctv.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(ctv.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(ctv.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 400)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Ctv") for name in names))
        artifacts = self.evidence["artifacts"]
        self.assertGreater(len(artifacts), 570)
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
                        if artifact.startswith("lineage:") or artifact.startswith("reference:"):
                            self.assertTrue((REPO / artifact.split(":", 1)[1]).is_file(), artifact)
                            continue
                        self.assertIn(artifact, artifacts)
                        if source["pointer"]:
                            ctv.resolve_pointer(loaded[artifact], source["pointer"])
                    self.assertEqual(ctv.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                raw = ctv.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(ctv.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 120)

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        b = self.by_name
        designs = self.dataset["designs"]
        histogram = Counter(d["wall_cusp_count"] for d in designs)
        self.assertEqual({int(k): v for k, v in self.dataset["headline"]["wall_cusp_count_histogram"].items()}, dict(histogram))
        self.assertEqual([b[f"CtvHist{t}"]["raw"] for t in ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven")], [histogram[i] for i in range(8)])
        self.assertEqual(b["CtvHistogramText"]["value"], "0:6 / 1:140 / 2:36 / 3:56 / 4:25 / 5:6 / 6:6 / 7:6")
        self.assertEqual(b["CtvWithCuspAll"]["raw"], sum(1 for d in designs if d["wall_cusp_count"] >= 1))
        self.assertEqual(b["CtvWithTwoCuspsAll"]["raw"], sum(1 for d in designs if d["wall_cusp_count"] >= 2))
        self.assertEqual((b["CtvWithCuspAll"]["raw"], b["CtvWithTwoCuspsAll"]["raw"]), (275, 135))
        sweep = [d for d in designs if d["set_id"] == "sweep_v2"]
        stages = [len(d["geometry"]["stage_centres_m"]) for d in sweep]
        self.assertEqual(b["CtvSweepNMinusOne"]["raw"], sum(1 for d, n in zip(sweep, stages) if d["wall_cusp_count"] == n - 1))
        self.assertEqual((b["CtvSweepNMinusOne"]["raw"], b["CtvSweepNMinusTwo"]["raw"], b["CtvSweepNPlusOne"]["raw"]), (83, 12, 1))
        self.assertEqual(b["CtvSweepNPlusOneDesign"]["raw"], "l1a-gs-v2-088-54d047707b")
        self.assertEqual(b["CtvSweepCuspsEqualChannelNulls"]["raw"], 95)
        gaps = [c["distance_to_nearest_stage_gap_m"] for d in sweep for c in d["wall_cusps"]]
        self.assertEqual(b["CtvSweepGapMedianMm"]["raw"], statistics.median(gaps))
        self.assertEqual((b["CtvSweepGapMedianMm"]["value"], b["CtvSweepGapMaxMm"]["value"]), ("0.14", "0.26"))
        interior = [c for d in sweep for c in d["cells"] if c["kind"] == "interior"]
        self.assertEqual(b["CtvSweepInteriorCells"]["raw"], len(interior))
        self.assertEqual((b["CtvSweepInteriorWallMirrorMin"]["value"], b["CtvSweepInteriorWallMirrorMax"]["value"]), ("1.000", "1.017"))
        self.assertEqual((b["CtvSweepInteriorAxisMirrorMin"]["value"], b["CtvSweepInteriorAxisMirrorMax"]["value"]), ("0.20", "1.15"))
        self.assertEqual((b["CtvSweepInteriorLengthPitchMin"]["value"], b["CtvSweepInteriorLengthPitchMax"]["value"]), ("0.90", "1.12"))
        self.assertEqual(b["CtvSweepAngleMedianDeg"]["value"], "0.7")
        self.assertEqual((b["CtvSweepFourWallCuspFraction"]["value"], b["CtvSweepFourCellFraction"]["value"]), ("0.198", "0.490"))
        for stages_count, token, expected in ((3, "Three", (26, 25)), (4, "Four", (45, 40)), (5, "Five", (25, 18))):
            self.assertEqual((b[f"CtvSweepStage{token}Designs"]["raw"], b[f"CtvSweepStage{token}NMinusOne"]["raw"]), expected)
            self.assertEqual(b[f"CtvSweepStage{token}Designs"]["raw"], sum(1 for n in stages if n == stages_count))
        four_cell = [d for d in designs if d["set_id"] == "four_cell_v2"]
        self.assertEqual(b["CtvFourCellOneCusp"]["raw"], sum(1 for d in four_cell if d["wall_cusp_count"] == 1))
        self.assertEqual(b["CtvFourCellOneCusp"]["raw"], 128)
        self.assertEqual((b["CtvFourCellStrengthRatioMin"]["value"], b["CtvFourCellStrengthRatioMax"]["value"]), ("16\\%", "42\\%"))
        char = [d for d in designs if d["set_id"] == "characterization_v1"]
        self.assertEqual(b["CtvCharVNMinusOne"]["raw"], sum(1 for d in char if d["wall_cusp_count"] == len(d["geometry"]["stage_centres_m"]) - 1))
        self.assertEqual(b["CtvCharVCuspsEqualChannelNulls"]["raw"], 56)
        self.assertEqual((b["CtvCharVInteriorAxisMirrorMin"]["value"], b["CtvCharVInteriorAxisMirrorMax"]["value"]), ("5.1", "174"))
        p2 = next(d for d in designs if d["set_id"] == "p2_divergent_exit")
        self.assertEqual(b["CtvPTwoCuspPositionsMm"]["raw"], [c["z_c_m"] for c in p2["wall_cusps"]])
        self.assertEqual(b["CtvPTwoCuspPositionsMm"]["value"], "6.028, 12.000, 17.972")
        self.assertEqual(b["CtvPTwoNullToPicMaxUm"]["value"], "31")
        self.assertEqual([b[f"CtvPTwoCusp{t}ToDashboardMm"]["value"] for t in ("One", "Two", "Three")], ["0.02", "0.05", "0.18"])
        self.assertEqual(b["CtvPTwoThirdCuspInsideEndUm"]["raw"], p2["geometry"]["straight_z_max_m"] - p2["wall_cusps"][-1]["z_c_m"])
        self.assertEqual(b["CtvPTwoThirdCuspInsideEndUm"]["value"], "27.9")
        self.assertIs(b["CtvPTwoCuspThreeAmbiguous"]["raw"], True)
        self.assertEqual(b["CtvMaxWallShiftUm"]["value"], "33.3")
        self.assertEqual((b["CtvHeldOutCharMaxUm"]["value"], b["CtvHeldOutSweepMaxUm"]["value"]), ("17.6", "27.3"))
        self.assertEqual((b["CtvHeldOutCharNulls"]["raw"], b["CtvHeldOutSweepNulls"]["raw"]), (180, 479))
        self.assertEqual(b["CtvFieldModelLevel"]["raw"], "L1a")
        self.assertEqual((b["CtvCampaignVersion"]["raw"], b["CtvLineageVersion"]["raw"], b["CtvFourCellVersion"]["raw"], b["CtvCharacterizationVersion"]["raw"]), ("v3.1", "v3", "v2", "v1"))
        self.assertEqual(b["CtvVerifiedFiles"]["value"], "1{,}211")
        self.assertEqual(b["CtvToleratedEolFiles"]["raw"], 0)
        self.assertEqual(b["CtvRecordedOutcome"]["raw"], ctv.RECORDED_OUTCOME)
        self.assertEqual(b["CtvWallTraceCount"]["raw"], 804)
        self.assertLess(b["CtvFluxRootMaxDiff"]["raw"], 1e-8)

    def test_frozen_definition_references_recompute_from_the_sealed_datasets(self) -> None:
        b = self.by_name
        v1 = json.loads((REPO / ctv.V1_DATASET).read_bytes())
        channel = axis = 0
        off_axis = []
        for case in v1["cases"]:
            for root in case["maps"]["primary"]["roots"]:
                if root["finite_box_boundary"] or root["geometry_association"]["zone"] not in ctv.V1_CHANNEL_ZONES:
                    continue
                channel += 1
                if any(m["method"] in ctv.AXIS_METHODS for m in root["members"]):
                    axis += 1
                else:
                    off_axis.append((case["case_id"], root["r_m"] / case["chamber_radius_m"], root["exclusion_reason"], root["eligible_cusp"]))
        self.assertEqual((b["CtvVOneChannelRoots"]["raw"], b["CtvVOneChannelAxisRoots"]["raw"], b["CtvVOneChannelOffAxisRoots"]["raw"]), (channel, axis, len(off_axis)))
        self.assertEqual((channel, axis, len(off_axis)), (200, 180, 20))
        self.assertEqual(b["CtvVOneOffAxisCases"]["raw"], len({c for c, *_ in off_axis}))
        self.assertEqual(b["CtvVOneOffAxisRadiusFractionMax"]["raw"], max(r for _, r, *_ in off_axis))
        self.assertLess(b["CtvVOneOffAxisRadiusFractionMax"]["raw"], 0.6)
        self.assertEqual({e for *_, e, _ in off_axis}, {"no_cell_bounding_separatrix"})
        self.assertFalse(any(el for *_, el in off_axis))
        self.assertEqual(axis, self.dataset["held_out"]["characterization_v1"]["reference_null_count"])
        v2 = json.loads((REPO / ctv.V2_DATASET).read_bytes())
        ratios = [case["sampling"]["values"]["alternating_strength_ratio"] for case in v2["cases"]]
        self.assertEqual((b["CtvFourCellStrengthRatioMin"]["raw"], b["CtvFourCellStrengthRatioMax"]["raw"]), (min(ratios), max(ratios)))
        self.assertEqual(b["CtvFourCellReferenceStable"]["raw"], v2["summary"]["stable_count"])
        self.assertEqual(b["CtvFourCellReferenceStable"]["raw"], 0)
        self.assertEqual((b["CtvCharVReferenceEligibleCusps"]["raw"], b["CtvCharVReferenceEligibleCells"]["raw"]), (0, 0))

    def test_lineage_audit_reproduces_the_recorded_rejection(self) -> None:
        lineage = ctv.Bundle(REPO, ctv.LINEAGE_RESULTS, experiment_id=ctv.LINEAGE_EXPERIMENT_ID, expected_state=ctv.LINEAGE_TERMINAL_STATE)
        v1 = json.loads((REPO / ctv.V1_DATASET).read_bytes())
        audit = ctv.reproduce_lineage_audit(lineage, v1, self.dataset["definition_v3"]["held_out_tolerance_m"])
        self.assertEqual((audit["sealed_axis_clusters"], audit["dropped_by_recorded_filter"], audit["dropped_in_channel"]), (206, 26, 22))
        self.assertEqual(len(audit["recorded_failing_designs"]), 14)
        self.assertEqual(audit["corrected_filter_pass_count"], 56)
        self.assertAlmostEqual(audit["corrected_filter_max_difference_m"], 1.7564011821423475e-05, places=15)
        self.assertLess(audit["max_dropped_centroid_r_m"], 2e-8)
        self.assertIs(audit["failures_explained_by_dropped_clusters"], True)
        self.assertEqual(self.evidence["lineage_artifacts"]["audit"], audit)
        documented = self.evidence["lineage_artifacts"]["documented"]
        self.assertEqual((documented["dropped"], documented["clusters"], documented["dropped_in_channel"], documented["failing_designs"]), (26, 206, 22, 14))
        self.assertEqual((documented["corrected_passed"], documented["corrected_total"]), (56, 56))
        text = (REPO / ctv.LINEAGE_AUDIT).read_text(encoding="utf-8")
        self.assertIsNotNone(ctv.AUDIT_ROOT_CAUSE_PATTERN.search(text))
        self.assertIsNotNone(ctv.AUDIT_CORRECTED_PATTERN.search(text))
        b = self.by_name
        self.assertEqual((b["CtvLineageHeldOutCharPassed"]["raw"], b["CtvLineageHeldOutCharRefNulls"]["raw"]), (42, 158))
        self.assertEqual(b["CtvLineageGatesTrue"]["raw"], 8)
        self.assertIs(b["CtvLineageHistogramEqual"]["raw"], True)
        gates = json.loads((LINEAGE_RESULTS / "artifacts/gates.json").read_bytes())
        self.assertEqual(sorted(gates["failing_designs"]["held_out_correspondence"]), audit["recorded_failing_designs"])
        # The audit refuses a reference that does not explain the recorded failures.
        tampered = json.loads(json.dumps(v1))
        for case in tampered["cases"]:
            for root in case["maps"]["primary"]["roots"]:
                root["r_m"] = 0.0
        with self.assertRaises(ValueError):
            ctv.reproduce_lineage_audit(lineage, tampered, self.dataset["definition_v3"]["held_out_tolerance_m"])

    def test_headline_and_estimands_recompute_from_the_rows(self) -> None:
        designs = self.dataset["designs"]
        for set_id in ctv.SET_IDS:
            rows = [d for d in designs if d["set_id"] == set_id]
            est = self.dataset["estimands"][set_id]
            self.assertEqual(ctv._histogram([d["wall_cusp_count"] for d in rows]), est["wall_cusp_count_histogram"])
            self.assertEqual(ctv._histogram([d["cell_count"] for d in rows]), est["cell_count_histogram"])
            self.assertEqual(sum(d["four_wall_cusps"] for d in rows), est["four_wall_cusp_count"])
            interior = [c["wall_mirror_ratio"] for d in rows for c in d["cells"] if c["kind"] == "interior"]
            self.assertEqual(ctv._distribution(interior), est["interior_wall_mirror_ratio"])
        headline = self.dataset["headline"]
        self.assertEqual(headline["stable_design_count"], sum(1 for d in designs if d["stability"]["stable"]))
        self.assertEqual(headline["max_wall_intersection_shift_m"], max(d["stability"]["max_wall_intersection_shift_m"] for d in designs if d["stability"]["max_wall_intersection_shift_m"] is not None))
        bijection, difference = ctv._match_sorted([1.0, 2.0], [2.0, 1.00001], 1e-4)
        self.assertIs(bijection, True)
        self.assertEqual(difference, abs(1.00001 - 1.0))
        self.assertEqual(ctv._match_sorted([1.0, 2.0], [2.0], 1e-4), (False, 0.0))
        self.assertEqual(ctv._match_sorted([1.0, 2.0], [2.0, 1.5], 1e-4), (False, 0.0))

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Ctv[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Ctv[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertGreater(len(used), 200)
        for table in ctv.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Ctv"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n281 designs\n", "Ctv"), ["2", "8", "1"])
        self.assertIn(f"\\subsection{{{ctv.SECTION_HEADING}}}", self.section)
        for heading in ("Definition and method.", "Design sets and execution.", "Results.", "Geometry sweep: cusps at the inter-magnet gaps.", "Four-cell candidates and characterization cases.", "The P2 row.", "Lineage: the recorded rejection of the predecessor.", "Scope."):
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
        for foreign in ("\\Swp", "\\Fcn", "\\Tch", "\\Wlg", "\\Wlf", "\\Mdo", "\\Mdb", "\\Fcc"):
            self.assertNotIn(foreign, self.section, "the section stays self-contained")

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in ctv.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        headline = self.dataset["headline"]
        for count in range(8):
            cells = [str(headline["wall_cusp_count_histogram_by_set"][s].get(str(count), 0)) for s in ctv.SET_IDS]
            self.assertIn(f"{count} & {' & '.join(cells)} & {headline['wall_cusp_count_histogram'].get(str(count), 0)}\\\\", self.tex)
        self.assertIn("designs & 96 & 128 & 56 & 1 & 281\\\\", self.tex)
        self.assertIn("3 & 26 & 2:25 / 4:1 & 25 & 1 & 0 & 28 &", self.tex)
        self.assertIn("all & 96 & 2:30 / 3:47 / 4:19 & 83 & 19 & 47 & 181 &", self.tex)
        self.assertIn("1 & 6.028 & 6.031 & 6.00 & 31.5 & 6.05 & 0.02 &", self.tex)
        self.assertIn("3 & 17.972 & 17.969 & 17.95 & 18.7 & 18.15 & 0.18 &", self.tex)
        self.assertIn("failing gate (designs) & \\texttt{held\\_\\allowbreak{}out\\_\\allowbreak{}correspondence} (14) & none\\\\", self.tex)
        self.assertIn("sealed axis clusters kept & 180 of 206 (22 in-channel dropped) & 206 of 206\\\\", self.tex)
        self.assertIn("cited for any number & no & yes\\\\", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], ctv.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [ctv.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertIn(ctv.RECORDED_OUTCOME, sidecar["claim_status"])
        self.assertEqual(len(sidecar["lineage_inputs"]), len(self.evidence["lineage_artifacts"]["files"]))
        self.assertEqual(len(sidecar["reference_inputs"]), 3)
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 4)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (ctv.ARTIFACT_CLAIM_ID, ctv.ARTIFACT_ID))
        self.assertEqual(self.evidence["tables"]["CtvHistogramTable"]["rows"], 14)
        self.assertEqual(self.evidence["tables"]["CtvPTwoTable"]["rows"], 3)
        self.assertEqual(self.evidence["tables"]["CtvSweepStageTable"]["rows"], 4)

    def test_tampered_bundle_dashboard_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\CtvSweepNMinusOne}{83}", "\\newcommand{\\CtvSweepNMinusOne}{84}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / ctv.RESULTS
            target.parent.mkdir(parents=True)
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            self.assertIn(b'"design_count":281', original)
            victim.write_bytes(original.replace(b'"design_count":281', b'"design_count":282', 1))
            with self.assertRaises(ValueError):
                ctv.Bundle(repo, ctv.RESULTS, experiment_id=ctv.EXPERIMENT_ID, expected_state="accepted_result")
            victim.write_bytes(original)
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                ctv.Bundle(repo, ctv.RESULTS, experiment_id=ctv.EXPERIMENT_ID, expected_state="accepted_result")
        # The lineage bundle is refused when read as an accepted result.
        with self.assertRaises(ValueError):
            ctv.Bundle(REPO, ctv.LINEAGE_RESULTS, experiment_id=ctv.LINEAGE_EXPERIMENT_ID, expected_state="accepted_result")
        html = (REPO / ctv.DASHBOARD_HTML).read_bytes()
        payload = ctv.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_file_sha256"], self.evidence["bundle"]["manifest_sha256"])
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(ctv.dashboard_payload(tampered)["identity"]["manifest_file_sha256"], payload["identity"]["manifest_file_sha256"])

    def test_field_grids_hash_to_the_design_records(self) -> None:
        import gzip

        for design in self.dataset["designs"]:
            if not design["representative"]:
                continue
            record = _load_artifact(design["record_path"])
            raw = gzip.decompress((RESULTS / record["accepted_grid_path"]).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["accepted_grid_payload_sha256"])
            grid = json.loads(raw)
            self.assertEqual(grid["identity"]["accepted_field_identity_sha256"], design["identity"]["accepted_field_identity_sha256"])
            self.assertEqual([c["z_c_m"] for c in record["accepted"]["topology"]["wall_cusps"]], [c["z_c_m"] for c in design["wall_cusps"]])

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
                     f"-output-directory={scratch}", "sections/cusp-topology-v3-1-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "cusp-topology-v3-1-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "cusp-topology-v3-1-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
