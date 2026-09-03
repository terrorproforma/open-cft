"""Regression tests for the hash-bound orbit wall-loss geometry screening v2 paper evidence."""

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
import generate_wall_loss_geometry_screening_v2_evidence as geo  # noqa: E402

EVIDENCE = REPO / geo.EVIDENCE_PATH
GENERATED = REPO / geo.OUTPUT_PATH
SIDECAR = REPO / geo.SIDECAR_PATH
SECTION = REPO / geo.SECTION_PATH
STANDALONE = REPO / "paper/sections/wall-loss-geometry-screening-v2-standalone.tex"
RESULTS = REPO / geo.RESULTS


def _load_artifact(relative: str):
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


class GeometryScreeningV2EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = geo.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.dataset = _load_artifact("artifacts/geometry-wall-loss-dataset-v2.json")
        cls.sweep_cells = [c for d in cls.dataset["designs"] if d["set_id"] == "sweep_v2" for c in d["cells"]]

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        again = geo.render(REPO)
        self.assertEqual(again, (self.evidence_bytes, self.tex_bytes, self.sidecar_bytes))
        self.assertEqual(EVIDENCE.read_bytes(), self.evidence_bytes)
        self.assertEqual(GENERATED.read_bytes(), self.tex_bytes)
        self.assertEqual(SIDECAR.read_bytes(), self.sidecar_bytes)
        for path in (EVIDENCE, GENERATED, SIDECAR, SECTION, STANDALONE):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)
        for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
            self.assertNotIn(forbidden, self.tex)
            self.assertNotIn(forbidden, self.evidence_bytes.decode("utf-8"))

    def test_evidence_binds_the_committed_revisions_the_references_and_the_disclosure(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-wall-loss-geometry-screening-v2-evidence")
        self.assertEqual(self.evidence["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], geo.CLASSIFICATION)
        self.assertEqual(self.evidence["p2_row_label"], geo.P2_LABEL)
        self.assertEqual(self.evidence["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(self.evidence["campaign_status"], geo.CAMPAIGN_STATUS)
        self.assertEqual(self.evidence["screening_model"], geo.SCREENING_MODEL)
        head = _git("rev-parse", "HEAD")
        chain = (geo.PREREGISTRATION_COMMIT_SHA, geo.RESULTS_COMMIT_SHA, geo.DISCLOSURE_COMMIT_SHA, geo.DASHBOARD_COMMIT_SHA)
        for earlier, later in zip(chain, chain[1:] + (head,)):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", earlier, later], cwd=REPO, check=False).returncode, 0, (earlier, later))
        for commit in (geo.CATALOGUE_RESULTS_COMMIT_SHA, geo.V1_RESULTS_COMMIT_SHA, geo.V4_RESULTS_COMMIT_SHA, geo.SWEEP_V2_RESULTS_COMMIT_SHA):
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", commit, geo.PREREGISTRATION_COMMIT_SHA], cwd=REPO, check=False).returncode, 0, commit)
        # The results tree first exists at the record commit (results only) and is unchanged at HEAD.
        results_rel = geo.RESULTS.as_posix()
        parent_has_results = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{geo.RESULTS_COMMIT_SHA}^:{results_rel}"], cwd=REPO, check=False, capture_output=True,
        ).returncode
        self.assertNotEqual(parent_has_results, 0, "results tree exists before the record commit")
        changed = _git("diff", "--name-only", f"{geo.RESULTS_COMMIT_SHA}~1", geo.RESULTS_COMMIT_SHA).split()
        self.assertTrue(changed and all(p.startswith(results_rel + "/") for p in changed))
        tree = _git("rev-parse", f"{geo.RESULTS_COMMIT_SHA}:{results_rel}")
        self.assertEqual(self.evidence["binding"]["results_tree"], tree)
        self.assertEqual(_git("rev-parse", f"HEAD:{results_rel}"), tree)
        manifest_rel = (geo.RESULTS / "manifest.json").as_posix()
        self.assertEqual(self.evidence["binding"]["manifest_git_blob"], _git("rev-parse", f"{geo.RESULTS_COMMIT_SHA}:{manifest_rel}"))
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], "876dc7e1ca76b33d1975a51c7fe749e2e271ab0d42ecdcaf158ecfa31fa0a30c")
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 16957)
        self.assertEqual(self.evidence["bundle"]["artifact_count"], 16968)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        # The disclosure commit changes Markdown only under the experiment.
        disclosure_changes = _git("diff", "--name-only", geo.RESULTS_COMMIT_SHA, geo.DISCLOSURE_COMMIT_SHA, "--", geo.EXPERIMENT.as_posix()).split()
        self.assertTrue(disclosure_changes and all(p.endswith(".md") for p in disclosure_changes))
        self.assertEqual(sorted(disclosure_changes), self.evidence["binding"]["disclosure_commit_experiment_files_changed"])
        verified = self.evidence["disclosure_sources"]["verified"]
        self.assertIs(verified["manifest_sha256_matches_bundle"], True)
        self.assertIs(verified["terminal_sha256_matches_bundle"], True)
        self.assertEqual((verified["file_count"], verified["descriptor_cap"], verified["pin_cap"], verified["validate_artifact_count"]), (16957, 8192, 4096, 16968))
        self.assertEqual(verified["results_commit_prefix"], geo.RESULTS_COMMIT_SHA[:8])
        self.assertIs(verified["nothing_rerun_stated"], True)
        # Frozen files: same blob at preregistration and results revisions.
        for name in geo.FROZEN_FILES:
            relative = (geo.EXPERIMENT / name).as_posix()
            blobs = [_git("rev-parse", f"{commit}:{relative}") for commit in (geo.PREREGISTRATION_COMMIT_SHA, geo.RESULTS_COMMIT_SHA)]
            self.assertEqual(blobs[0], blobs[1], name)
        # Reference and disclosure files are bound at their revisions and equal the checkout.
        for group, lf in (("reference_artifacts", False), ("disclosure_sources", True)):
            for path, meta in self.evidence[group]["files"].items():
                blob = subprocess.run(["git", "show", f"{meta['revision']}:{path}"], cwd=REPO, check=True, capture_output=True).stdout
                working = (REPO / path).read_bytes()
                self.assertEqual(meta["git_blob_sha256"], hashlib.sha256(blob).hexdigest(), path)
                self.assertEqual(meta["sha256"], hashlib.sha256(working).hexdigest(), path)
                if lf:
                    self.assertEqual(blob.replace(b"\r\n", b"\n"), working.replace(b"\r\n", b"\n"), path)
                else:
                    self.assertEqual(blob, working, path)
        self.assertEqual(self.evidence["reference_artifacts"]["files"][geo.CATALOGUE_PATH.as_posix()]["sha256"], self.dataset["catalogue_file_sha256"])
        # The dashboard is bound by LF-normalised SHA-256 equal to the blob committed at its revision.
        for key, path in (("generator_sha256_lf", geo.DASHBOARD_GENERATOR), ("template_sha256_lf", geo.DASHBOARD_TEMPLATE), ("html_sha256_lf", geo.DASHBOARD_HTML)):
            blob = subprocess.run(["git", "show", f"{geo.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest())
        self.assertEqual(self.evidence["dashboard"]["payload_manifest_sha256"], self.evidence["bundle"]["manifest_sha256"])
        integration = self.evidence["manuscript_integration"]
        self.assertEqual(integration["status"], "admitted")
        self.assertEqual(integration["gate_id"], geo.GATE_ID)
        self.assertEqual(integration["gate_kind"], "numerical-screening")
        self.assertEqual(integration["manifest_id"], geo.MANIFEST_ID)
        self.assertEqual(integration["section_heading"], geo.SECTION_HEADING)
        self.assertEqual(integration["prose_claim_ids"], list(geo.PROSE_CLAIM_IDS))
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        self.assertEqual(manuscript.count(geo.SECTION_BINDING), 1)
        self.assertEqual(manuscript.count(geo.GENERATED_BINDING), 1)
        self.assertLess(manuscript.find(geo.GENERATED_BINDING), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        macros = self.evidence["macros"]
        self.assertGreater(len(macros), 350)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Wlh") for name in names))
        artifacts = self.evidence["artifacts"]
        self.assertGreater(len(artifacts), 3000)
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
                        if source["artifact"] == "manifest.json":
                            self.assertTrue((RESULTS / "manifest.json").is_file())
                            continue
                        self.assertIn(source["artifact"], artifacts)
                        geo.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                    self.assertEqual(geo.format_value(item["format"], item["raw"]), item["value"])
                    continue
                source = item["source"]
                self.assertIn(source["artifact"], artifacts)
                raw = geo.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                self.assertEqual(raw, item["raw"])
                self.assertEqual(geo.format_value(item["format"], raw), item["value"])
                self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex)
        self.assertGreater(derived_count, 120)

    def test_headline_macros_recompute_from_the_cells(self) -> None:
        b = self.by_name
        cells = self.sweep_cells
        interior = [c for c in cells if c["position_class"] == "interior"]
        anode = [c for c in cells if c["position_class"] == "anode_side"]
        exit_side = [c for c in cells if c["position_class"] == "exit_side"]
        self.assertEqual((len(anode), len(interior), len(exit_side)), (96, 181, 96))
        self.assertEqual((b["WlhAnodeCells"]["raw"], b["WlhInteriorCells"]["raw"], b["WlhExitCells"]["raw"]), (96, 181, 96))
        self.assertTrue(all(c["final"]["wall_hit"] == c["final"]["trials"] == 128 for c in interior))
        self.assertEqual(b["WlhInteriorAtOne"]["raw"], 181)
        self.assertIs(b["WlhInteriorAllSaturated"]["raw"], True)
        self.assertEqual(b["WlhInteriorDesignsAllSaturated"]["raw"], 96)
        self.assertEqual(b["WlhInteriorPMin"]["value"], "1.000")
        anode_p = [c["final"]["p_wall"]["probability"] for c in anode]
        self.assertEqual(b["WlhAnodePMedian"]["raw"], statistics.median(anode_p))
        self.assertEqual((b["WlhAnodePMin"]["value"], b["WlhAnodePMedian"]["value"], b["WlhAnodeAtOne"]["raw"]), ("0.307", "0.984", 34))
        exit_p = [c["final"]["p_wall"]["probability"] for c in exit_side]
        self.assertEqual((b["WlhExitPMin"]["value"], b["WlhExitPMedian"]["value"], b["WlhExitPMax"]["value"]), ("0.248", "0.500", "1.000"))
        self.assertEqual(b["WlhExitPMedian"]["raw"], statistics.median(exit_p))
        self.assertEqual((b["WlhCellsToppedUp"]["raw"], b["WlhCellsSaturated"]["raw"]), (117, 260))
        self.assertEqual((b["WlhAnodeToppedUp"]["raw"], b["WlhExitToppedUp"]["raw"], b["WlhInteriorToppedUp"]["raw"]), (32, 83, 0))
        all_cells = [c for d in self.dataset["designs"] for c in d["cells"]]
        self.assertEqual(len(all_cells), 377)
        self.assertEqual(b["WlhCellsToppedUp"]["raw"], sum(c["topped_up"] for c in all_cells))
        floors = [c["final"]["jeffreys_floor"] for c in all_cells]
        self.assertEqual((b["WlhFloorMedian"]["raw"], b["WlhFloorMax"]["raw"]), (statistics.median(floors), max(floors)))
        self.assertEqual((b["WlhFloorMedian"]["value"], b["WlhFloorMax"]["value"]), ("0.0055", "0.0242"))
        self.assertEqual(b["WlhCellsReady"]["raw"], sum(c["final"]["surrogate_ready"] for c in all_cells))
        self.assertEqual((b["WlhCellsReady"]["raw"], b["WlhFractionReady"]["value"]), (294, "78.0\\%"))
        self.assertEqual(b["WlhFloorHalfAtFinal"]["raw"], geo.jeffreys_floor(256, 512))
        self.assertEqual(b["WlhFloorFullAtStageOne"]["raw"], geo.jeffreys_floor(128, 128))
        self.assertEqual(b["WlhReflectionsFinal"]["raw"], sum(c["final"]["reflected"] for c in all_cells))
        self.assertEqual((b["WlhReflectionsFinal"]["value"], b["WlhReflectionShareFinal"]["value"]), ("10{,}407", "11.2\\%"))
        self.assertEqual((b["WlhExitCellsWithReflections"]["raw"], b["WlhAnodeCellsWithReflections"]["raw"], b["WlhInteriorCellsWithReflections"]["raw"]), (65, 1, 0))
        self.assertEqual((b["WlhDesignsWithReflections"]["raw"], b["WlhPTwoReflections"]["raw"]), (66, 350))
        self.assertEqual((b["WlhOrbitCount"]["value"], b["WlhStageOneLaunches"]["value"], b["WlhStageTwoLaunches"]["value"], b["WlhControlLaunches"]["value"]), ("104{,}832", "48{,}256", "44{,}928", "11{,}648"))
        self.assertEqual((b["WlhCaseCount"]["raw"], b["WlhValidatorsPassed"]["raw"], b["WlhValidatorsFailed"]["raw"]), (1105, 16549, 0))
        self.assertEqual((b["WlhTimeouts"]["raw"], b["WlhNumericalFailures"]["raw"], b["WlhEnergyErrorMax"]["raw"]), (0, 0, 0.0))
        self.assertEqual((b["WlhCrossResolutionDesigns"]["raw"], b["WlhCrossResolutionRmsMax"]["value"], b["WlhInterpolationRmsMax"]["value"]), (97, "1.04\\%", "0.87\\%"))
        self.assertEqual(b["WlhExecutionWallMin"]["value"], "85.9")
        self.assertEqual((b["WlhInjectorFlaggedCells"]["raw"], b["WlhInjectorFlaggedDesign"]["raw"], b["WlhInjectorFlaggedLengthMm"]["value"]), (1, "l1a-gs-v2-088-54d047707b", "0.16"))
        self.assertEqual((b["WlhShortCells"]["raw"], b["WlhShortSweepExitCells"]["raw"], b["WlhShortSweepAnodeCells"]["raw"], b["WlhShortPTwoCells"]["raw"], b["WlhShortCellLengthMinUm"]["value"]), (14, 12, 1, 1, "28"))
        p2 = next(d for d in self.dataset["designs"] if d["set_id"] == "p2_divergent_exit")
        self.assertEqual([c["final"]["p_wall"]["probability"] for c in p2["cells"]], [b["WlhPTwoAnodeP"]["raw"], 1.0, 1.0, b["WlhPTwoExitP"]["raw"]])
        self.assertEqual((b["WlhPTwoAnodeP"]["value"], b["WlhPTwoExitP"]["value"], b["WlhPTwoExitReflections"]["raw"], b["WlhPTwoExitLengthUm"]["value"]), ("0.605", "0.170", 350, "28"))
        self.assertEqual(b["WlhVFourP"]["value"], "0.645")
        self.assertIs(b["WlhPTwoNotReplication"]["raw"], True)

    def test_control_and_comparison_macros_recompute(self) -> None:
        b = self.by_name
        designs = self.dataset["designs"]
        n = sum(d["control"]["n_control"] for d in designs)
        wall_n = sum(d["control"]["wall_N"] for d in designs)
        wall_2n = sum(d["control"]["wall_2N"] for d in designs)
        discordant = sum(d["control"]["discordant"] for d in designs)
        self.assertEqual((b["WlhControlN"]["raw"], b["WlhControlWallN"]["raw"], b["WlhControlWallTwoN"]["raw"], b["WlhControlDiscordant"]["raw"]), (n, wall_n, wall_2n, discordant))
        self.assertEqual((n, discordant), (11648, 2))
        self.assertEqual(b["WlhControlBias"]["raw"], (wall_2n - wall_n) / n)
        paired: list[float] = []
        for d in designs:
            plus = d["control"]["wall_2N"] - min(d["control"]["wall_2N"], d["control"]["wall_N"])
            minus = d["control"]["wall_N"] - min(d["control"]["wall_2N"], d["control"]["wall_N"])
            paired.extend([1.0] * plus + [-1.0] * minus + [0.0] * (d["control"]["n_control"] - plus - minus))
        self.assertAlmostEqual(b["WlhControlBiasSe"]["raw"], statistics.pstdev(paired) / math.sqrt(len(paired)), places=18)
        self.assertEqual((b["WlhControlBias"]["value"], b["WlhControlBiasSe"]["value"], b["WlhControlDiscordanceRate"]["value"]), ("$-$$8.6\\times10^{-5}$", "$8.6\\times10^{-5}$", "0.017\\%"))
        self.assertIs(b["WlhControlPassed"]["raw"], True)
        rows = [d for d in designs if d["set_id"] == "sweep_v2"]
        v1 = [d["v1_comparison"]["v1_probability"] for d in rows]
        v2 = [d["pooled"]["launches"]["probability"] for d in rows]
        self.assertAlmostEqual(b["WlhSpearmanLaunch"]["raw"], geo.spearman(v1, v2), places=15)
        self.assertEqual((b["WlhSpearmanLaunch"]["value"], b["WlhMeanDiffLaunch"]["value"], b["WlhMeanAbsDiffLaunch"]["value"], b["WlhOverlapLaunch"]["value"]), ("$+$0.15", "$+$0.038", "0.113", "45\\%"))
        self.assertEqual((b["WlhSpearmanArea"]["value"], b["WlhOverlapArea"]["value"]), ("$+$0.35", "0\\%"))
        self.assertEqual(b["WlhVOnePooledMedian"]["value"], "0.702")
        # Design pooled values recompute with the experiment's weighting.
        for design in rows[:5]:
            for weight in ("wall_area", "launches"):
                expected = geo.design_pooled(design["cells"], weight)
                self.assertAlmostEqual(design["pooled"][weight]["probability"], expected["probability"], places=15)
                self.assertAlmostEqual(design["pooled"][weight]["standard_uncertainty"], expected["standard_uncertainty"], places=15)
        # Spearman self-checks.
        self.assertAlmostEqual(geo.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]), 1.0, places=15)
        self.assertAlmostEqual(geo.spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]), -1.0, places=15)
        # Direction split of the divergent-exit designs.
        self.assertEqual((b["WlhDivergentDesigns"]["raw"], b["WlhStraightDesigns"]["raw"]), (90, 6))
        self.assertEqual((b["WlhExitWallSideLastPolarity"]["raw"], b["WlhExitWallSideNotLastPolarity"]["raw"]), (82, 8))
        self.assertEqual((b["WlhExitDirPlusPooled"]["value"], b["WlhExitDirMinusPooled"]["value"]), ("0.566", "0.418"))
        self.assertEqual((b["WlhDivergentExitAtOne"]["raw"], b["WlhStraightExitAtOne"]["raw"], b["WlhExitAtOne"]["raw"]), (7, 4, 11))

    def test_allocation_rule_floors_and_control_replay_from_the_cases(self) -> None:
        checked = 0
        for design in self.dataset["designs"]:
            for cell in design["cells"]:
                stage1 = design["cases"][f"{design['design_key']}--{cell['cell_id']}--stage1-N"]
                width = geo.wilson_width(stage1["termination_counts"]["wall_hit"], stage1["trial_count"])
                self.assertEqual(cell["topped_up"], width > 0.1)
                self.assertEqual(cell["final"]["trials"], 512 if cell["topped_up"] else 128)
                self.assertEqual(cell["final"]["jeffreys_floor"], geo.jeffreys_floor(cell["final"]["wall_hit"], cell["final"]["trials"]))
                self.assertEqual(cell["final"]["binomial_floor"], geo.binomial_floor(cell["final"]["wall_hit"], cell["final"]["trials"]))
                self.assertEqual(cell["final"]["surrogate_ready"], cell["final"]["jeffreys_floor"] <= 0.02)
                self.assertEqual(cell["control"]["n_control"], cell["final"]["trials"] // 8)
                for estimand in ("p_wall", "p_reflected", "p_escape", "p_timeout"):
                    estimate = cell["final"][estimand]
                    self.assertEqual((estimate["probability"], estimate["lower"], estimate["upper"]), geo.wilson(estimate["successes"], estimate["trials"]))
                checked += 1
        self.assertEqual(checked, 377)
        # The paired control of one topped-up cell replays orbit by orbit from the endpoint tables.
        design = next(d for d in self.dataset["designs"] if d["set_id"] == "p2_divergent_exit")
        cell = next(c for c in design["cells"] if c["topped_up"])
        terminations: dict[str, str] = {}
        for stage in ("stage1", "stage2b1", "stage2b2", "stage2b3"):
            key = f"{design['design_key']}--{cell['cell_id']}--{stage}-N"
            raw = gzip.decompress((RESULTS / "artifacts/endpoints" / f"{key}.json.gz").read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), design["cases"][key]["endpoints_payload_sha256"])
            for row in json.loads(raw)["rows"]:
                terminations[row["launch_key"]] = row["termination"]
        control_key = f"{design['design_key']}--{cell['cell_id']}--control-2N"
        control_rows = json.loads(gzip.decompress((RESULTS / "artifacts/endpoints" / f"{control_key}.json.gz").read_bytes()))["rows"]
        self.assertEqual(len(control_rows), 64)
        wall_n = sum(terminations[r["launch_key"]] == "wall_hit" for r in control_rows)
        wall_2n = sum(r["termination"] == "wall_hit" for r in control_rows)
        discordant = sum(terminations[r["launch_key"]] != r["termination"] for r in control_rows)
        self.assertEqual((cell["control"]["wall_N"], cell["control"]["wall_2N"], cell["control"]["discordant"]), (wall_n, wall_2n, discordant))
        # Wilson exactness of the frozen case sizes, recomputed.
        scan = geo.known_defect_scan(4000)
        self.assertEqual((scan["zero_count_lower_inexact"], scan["full_count_upper_inexact"]), (734, 1238))
        self.assertTrue(all(scan["exact_at_both_ends"].values()))
        self.assertTrue(scan["n512_full_inexact"] and not scan["n512_zero_inexact"] and scan["n384_zero_inexact"])
        self.assertEqual(geo.wilson(330, 512)[1:], (0.6021349532568827, 0.6847749053232215))

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Wlh[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Wlh[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertGreater(len(used), 250)
        for table in geo.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Wlh"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n97 designs\n", "Wlh"), ["9", "7"])
        self.assertIn(f"\\subsection{{{geo.SECTION_HEADING}}}", self.section)
        for heading in ("Method.", "Results.", "Wall access by cell class.", "Reflections, control and the fixed-fraction screening.", "Floors and readiness.", "Disclosures.", "Scope."):
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
        for other in ("\\Wlf", "\\Wlg", "\\Ctv", "\\Swt"):
            self.assertNotIn(other, self.section, "the section stays self-contained")
        for phrase in ("never a loss probability", "not a replication", "declared averages", "no design rule", "opens no physics level", "lost every launch to the dielectric", "published post hoc"):
            self.assertIn(phrase, check_paper._normalize_tex(self.section))

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in geo.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        self.assertIn("interior & 181 & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 & 181 & 0 & 0 & 181 & 0.0055 & 0.0055 & 0\\\\", self.tex)
        self.assertIn("anode-side & 96 & 0.307 & 0.826 & 0.984 & 1.000 & 1.000 & 0.907 & 34 & 0 & 32 & 82 &", self.tex)
        self.assertIn("exit-side & 96 & 0.248 & 0.484 & 0.500 & 0.508 & 1.000 & 0.548 & 11 & 0 & 83 & 28 &", self.tex)
        self.assertIn("electron orbits integrated (stage 1 / stage 2 / control) & 104{,}832 (48{,}256 / 44{,}928 / 11{,}648)\\\\", self.tex)
        self.assertIn("cells topped up to 512 (anode-side / exit-side / P2) & 117 (32 / 83 / 2)\\\\", self.tex)
        self.assertIn("discordant orbits (termination differs between N and 2N) & 2 (0.017\\%)\\\\", self.tex)
        self.assertIn("Spearman rank correlation of v2 with v1 & $+$0.15 & $+$0.35\\\\", self.tex)
        self.assertIn("(i) files in the bundle / Windows C-runtime descriptor cap & 16{,}957 / 8{,}192\\\\", self.tex)
        self.assertIn("& 734\\\\", self.tex)
        self.assertIn("& 1238\\\\", self.tex)
        self.assertIn("\\texttt{876dc7e1ca76}", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [geo.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertEqual(len(sidecar["reference_inputs"]), 6)
        self.assertEqual(len(sidecar["disclosure_inputs"]), 4)
        self.assertIn(geo.RECORDED_OUTCOME, sidecar["claim_status"])
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 5)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (geo.ARTIFACT_CLAIM_ID, geo.ARTIFACT_ID))
        self.assertEqual(self.evidence["tables"]["WlhCellClassTable"]["rows"], 5)
        self.assertEqual(self.evidence["tables"]["WlhDisclosureTable"]["rows"], 14)

    def test_tampered_bundle_disclosure_or_dashboard_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\WlhOrbitCount}{104{,}832}", "\\newcommand{\\WlhOrbitCount}{104{,}833}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(hashlib.sha256(changed.encode("utf-8")).hexdigest(), json.loads(self.sidecar_bytes)["output"]["sha256"])
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / geo.RESULTS
            target.parent.mkdir(parents=True)
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            self.assertIn(b'"orbit_count":104832', original)
            victim.write_bytes(original.replace(b'"orbit_count":104832', b'"orbit_count":104833', 1))
            with self.assertRaises(ValueError):
                geo.Bundle(repo)
            victim.write_bytes(original)
            sidecar_victim = target / "artifacts" / "gates.json.sha256.json"
            sidecar_victim.write_bytes(sidecar_victim.read_bytes().replace(b"\n", b"\r\n") + b"\r\n")
            with self.assertRaises(ValueError):
                geo.Bundle(repo)
        # A dashboard whose payload names another manifest is refused before any macro is written.
        html = (REPO / geo.DASHBOARD_HTML).read_bytes()
        payload = geo.dashboard_payload(html)
        self.assertEqual(payload["identity"]["manifest_file_sha256"], self.evidence["bundle"]["manifest_sha256"])
        tampered = html.replace(self.evidence["bundle"]["manifest_sha256"].encode(), b"0" * 64, 1)
        self.assertNotEqual(geo.dashboard_payload(tampered)["identity"]["manifest_file_sha256"], payload["identity"]["manifest_file_sha256"])
        # The disclosure must name the committed manifest hash: a tampered note is refused by the generator.
        disclosure = (REPO / geo.DISCLOSURE_PATH).read_text(encoding="utf-8")
        self.assertIn(f"manifest_byte_sha256 {self.evidence['bundle']['manifest_sha256']}", disclosure)
        self.assertIn("terminal_byte_sha256 a495d12bc83241e6c2b84623b2e0c75e760176b9c0796854724aea467e195b6a", disclosure)
        self.assertIn("No orbit was re-integrated and no experiment code was changed", disclosure)
        self.assertRegex((REPO / geo.LIFECYCLE_MODULE_PATH).read_text(encoding="utf-8"), r"(?m)^MAX_PINNED_DESCRIPTORS = 4096$")
        # The Wilson check refuses a sealed estimate whose bound was altered.
        estimate = dict(self.dataset["designs"][0]["cells"][0]["final"]["p_wall"])
        geo._check_estimate(estimate, estimate["successes"], estimate["trials"], "intact")
        estimate["upper"] = estimate["upper"] - 1e-9
        with self.assertRaises(ValueError):
            geo._check_estimate(estimate, estimate["successes"], estimate["trials"], "tampered")

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
                     f"-output-directory={scratch}", "sections/wall-loss-geometry-screening-v2-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "wall-loss-geometry-screening-v2-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "wall-loss-geometry-screening-v2-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
