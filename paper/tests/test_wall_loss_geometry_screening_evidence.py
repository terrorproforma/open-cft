"""Regression tests for the hash-bound orbit wall-loss geometry screening v1 paper evidence."""

from __future__ import annotations

import gzip
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
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_wall_loss_geometry_screening_v1_evidence as geo  # noqa: E402

EVIDENCE = REPO / geo.EVIDENCE_PATH
GENERATED = REPO / geo.OUTPUT_PATH
SIDECAR = REPO / geo.SIDECAR_PATH
SECTION = REPO / geo.SECTION_PATH
STANDALONE = REPO / "paper/sections/wall-loss-geometry-screening-v1-standalone.tex"
RESULTS = REPO / geo.RESULTS


def _load_artifact(relative: str):
    return json.loads((RESULTS / relative).read_bytes().decode("utf-8"))


class GeometryScreeningEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence_bytes, cls.tex_bytes, cls.sidecar_bytes = geo.render(REPO)
        cls.evidence = json.loads(cls.evidence_bytes)
        cls.tex = cls.tex_bytes.decode("utf-8")
        cls.section = SECTION.read_text(encoding="utf-8")
        cls.by_name = {item["name"]: item for item in cls.evidence["macros"]}
        cls.dataset = _load_artifact("artifacts/geometry-wall-loss-dataset.json")

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

    def test_evidence_binds_the_committed_revisions_and_the_dashboard(self) -> None:
        self.assertEqual(self.evidence["document_type"], "paper-wall-loss-geometry-screening-v1-evidence")
        self.assertEqual(self.evidence["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(self.evidence["classification"], geo.CLASSIFICATION)
        self.assertEqual(self.evidence["recorded_outcome"], geo.RECORDED_OUTCOME)
        self.assertEqual(self.evidence["campaign_status"], geo.CAMPAIGN_STATUS)
        self.assertEqual(self.evidence["screening_model"], geo.SCREENING_MODEL)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        for commit in (geo.PREREGISTRATION_COMMIT_SHA, geo.RESULTS_COMMIT_SHA):
            self.assertEqual(
                subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=False).returncode, 0
            )
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", geo.PREREGISTRATION_COMMIT_SHA, geo.RESULTS_COMMIT_SHA], cwd=REPO, check=False).returncode, 0)
        self.assertEqual(geo.DASHBOARD_COMMIT_SHA, geo.RESULTS_COMMIT_SHA)
        # The results tree first exists at the record commit and is unchanged at HEAD.
        results_rel = geo.RESULTS.as_posix()
        parent_has_results = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{geo.RESULTS_COMMIT_SHA}^:{results_rel}"], cwd=REPO, check=False, capture_output=True,
        ).returncode
        self.assertNotEqual(parent_has_results, 0, "results tree exists before the record commit")
        tree = subprocess.run(["git", "rev-parse", f"{geo.RESULTS_COMMIT_SHA}:{results_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(self.evidence["binding"]["results_tree"], tree)
        self.assertEqual(subprocess.run(["git", "rev-parse", f"HEAD:{results_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip(), tree)
        manifest_rel = (geo.RESULTS / "manifest.json").as_posix()
        committed = subprocess.run(
            ["git", "rev-parse", f"{geo.RESULTS_COMMIT_SHA}:{manifest_rel}"], cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(self.evidence["binding"]["manifest_git_blob"], committed)
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.evidence["bundle"]["manifest_sha256"], "39bd52133fefd3adae45e9593e7312b8e9027322ca3142d59912bbc13e2e027a")
        self.assertEqual(self.evidence["bundle"]["verified_file_count"], 2835)
        self.assertEqual(self.evidence["bundle"]["artifact_count"], 2846)
        self.assertEqual(self.evidence["bundle"]["tolerated_eol_files"], [])
        # Frozen files: same blob at preregistration and results revisions.
        for name in geo.FROZEN_FILES:
            relative = (geo.EXPERIMENT / name).as_posix()
            blobs = [
                subprocess.run(["git", "rev-parse", f"{commit}:{relative}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
                for commit in (geo.PREREGISTRATION_COMMIT_SHA, geo.RESULTS_COMMIT_SHA)
            ]
            self.assertEqual(blobs[0], blobs[1], name)
        # The dashboard is bound by LF-normalised SHA-256 equal to the blob committed at its revision.
        for key, path in (
            ("generator_sha256_lf", geo.DASHBOARD_GENERATOR),
            ("template_sha256_lf", geo.DASHBOARD_TEMPLATE),
            ("html_sha256_lf", geo.DASHBOARD_HTML),
        ):
            blob = subprocess.run(
                ["git", "show", f"{geo.DASHBOARD_COMMIT_SHA}:{path.as_posix()}"], cwd=REPO, check=True, capture_output=True,
            ).stdout
            self.assertEqual(self.evidence["dashboard"][key], hashlib.sha256(blob).hexdigest())
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
        self.assertGreater(len(macros), 250)
        names = [item["name"] for item in macros]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("Wlg") for name in names))
        artifacts = self.evidence["artifacts"]
        self.assertGreater(len(artifacts), 500)
        for relative, meta in artifacts.items():
            raw = (RESULTS / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
            self.assertEqual(len(raw), meta["bytes"], relative)
        loaded = {
            relative: _load_artifact(relative)
            for relative in artifacts
            if relative.endswith(".json")
        }
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
        self.assertGreater(derived_count, 100)

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        b = self.by_name
        designs = self.dataset["designs"]
        wall = [d["reported"]["wall_hit"]["probability"] for d in designs]
        self.assertEqual(b["WlgWallPMin"]["raw"], min(wall))
        self.assertEqual(b["WlgWallPMax"]["raw"], max(wall))
        self.assertEqual(b["WlgWallPMedian"]["raw"], statistics.median(wall))
        self.assertEqual((b["WlgWallPMin"]["value"], b["WlgWallPMax"]["value"], b["WlgWallPMedian"]["value"]), ("0.375", "0.869", "0.702"))
        self.assertEqual(b["WlgWallPMean"]["value"], "0.697")
        reflections_2n = [d["cases"]["accepted-2N"]["termination_counts"]["reflected"] for d in designs]
        self.assertEqual(b["WlgReflectionsTwoN"]["raw"], sum(reflections_2n))
        self.assertEqual((b["WlgReflectionsMin"]["raw"], b["WlgReflectionsMax"]["raw"]), (min(reflections_2n), max(reflections_2n)))
        self.assertEqual((b["WlgReflectionsMin"]["raw"], b["WlgReflectionsMax"]["raw"]), (32, 282))
        self.assertEqual(b["WlgDesignsWithReflections"]["raw"], sum(1 for r in reflections_2n if r > 0))
        self.assertEqual(b["WlgDesignsWithReflections"]["raw"], 96)
        all_reflections = sum(c["termination_counts"]["reflected"] for d in designs for c in d["cases"].values())
        self.assertEqual(b["WlgReflectionsTotal"]["raw"], all_reflections)
        self.assertEqual(b["WlgReflectionsTotal"]["raw"], 22904)
        self.assertEqual(b["WlgReflectionShareAll"]["value"], "22.8\\%")
        self.assertEqual(b["WlgReflectionShareTwoN"]["value"], "22.9\\%")
        self.assertEqual(b["WlgReflectionsTotal"]["value"], "22{,}904")
        self.assertEqual(b["WlgOrbitCount"]["value"], "100{,}352")
        escapes = [d["reported"]["domain_escape"]["probability"] for d in designs]
        self.assertEqual((b["WlgEscapePMin"]["raw"], b["WlgEscapePMax"]["raw"], b["WlgEscapePMedian"]["raw"]), (min(escapes), max(escapes), statistics.median(escapes)))
        self.assertEqual((b["WlgEscapePMin"]["value"], b["WlgEscapePMax"]["value"], b["WlgEscapePMedian"]["value"]), ("0.000", "0.215", "0.069"))
        subclasses = {"upstream_anode_plane": 0, "exit_plane": 0, "divergent_section_radial": 0}
        for d in designs:
            for key, value in d["reported"]["domain_escape_subclasses"].items():
                subclasses[key] += value
        self.assertEqual((b["WlgEscapeAnode"]["raw"], b["WlgEscapeExit"]["raw"], b["WlgEscapeDivergent"]["raw"]), (subclasses["upstream_anode_plane"], subclasses["exit_plane"], subclasses["divergent_section_radial"]))
        self.assertEqual((b["WlgEscapeAnode"]["raw"], b["WlgEscapeExit"]["raw"], b["WlgEscapeDivergent"]["raw"]), (1635, 1127, 862))
        self.assertEqual(b["WlgEscapeUnclassified"]["raw"], 0)
        changes = [abs(d["convergence"]["successive_change"]) for d in designs]
        self.assertEqual(b["WlgMaxSuccessiveChange"]["raw"], max(changes))
        self.assertEqual(b["WlgMaxSuccessiveChange"]["value"], "0.0059")
        self.assertAlmostEqual(b["WlgMeanSuccessiveChange"]["raw"], statistics.fmean(changes), places=18)
        self.assertEqual(b["WlgMeanSuccessiveChange"]["value"], "$3.46\\times10^{-4}$")
        self.assertEqual(b["WlgConvergedDesigns"]["raw"], sum(1 for d in designs if d["convergence"]["converged"]))
        self.assertEqual(b["WlgConvergedDesigns"]["raw"], 96)
        refined = [d["convergence"]["field_resolution_sensitivity"]["change"] for d in designs if d["representative"]]
        self.assertEqual(len(refined), 4)
        self.assertEqual(b["WlgRefinedSensitivityMax"]["raw"], max(refined))
        self.assertEqual(b["WlgRefinedSensitivityMax"]["value"], "0.0078")
        cross = [d["field"]["cross_resolution_b_relative_rms"] for d in designs if d["field"]["cross_resolution_b_relative_rms"] is not None]
        self.assertEqual(b["WlgCrossResolutionDesigns"]["raw"], len(cross))
        self.assertEqual(b["WlgCrossResolutionDesigns"]["raw"], 4)
        self.assertEqual(b["WlgCrossResolutionRmsMax"]["raw"], max(cross))
        self.assertEqual(b["WlgCrossResolutionRmsMax"]["value"], "0.66\\%")
        self.assertEqual(b["WlgInterpolationRmsMax"]["raw"], max(d["field"]["interpolation_b_relative_rms"] for d in designs))
        self.assertEqual(b["WlgInterpolationRmsMax"]["value"], "0.87\\%")
        cells = ["gs1-cell-1", "gs1-cell-2", "gs1-cell-3", "gs1-cell-4"]
        tokens = ["One", "Two", "Three", "Four"]
        saturated = 0
        for cell, token in zip(cells, tokens):
            values = [d["per_cell"]["accepted-2N"][cell]["wall_hit"]["probability"] for d in designs]
            self.assertAlmostEqual(b[f"WlgCell{token}Mean"]["raw"], statistics.fmean(values), places=15)
            self.assertEqual(b[f"WlgCell{token}Saturated"]["raw"], sum(1 for v in values if v == 1.0))
            saturated += sum(1 for v in values if v == 1.0)
            self.assertEqual(sum(1 for v in values if v == 0.0), 0)
        self.assertEqual([b[f"WlgCell{t}Mean"]["value"] for t in tokens], ["0.65", "0.82", "0.77", "0.55"])
        self.assertEqual(b["WlgCellsSaturatedOne"]["raw"], saturated)
        self.assertEqual((b["WlgCellsSaturatedOne"]["raw"], b["WlgCellsSaturatedZero"]["raw"], b["WlgDesignCells"]["raw"]), (94, 0, 384))
        ordered = sorted(designs, key=lambda d: d["reported"]["wall_hit"]["probability"])
        self.assertEqual([b[f"WlgLeast{t}Id"]["raw"] for t in ("One", "Two", "Three")], [d["case_id"] for d in ordered[:3]])
        self.assertEqual([b[f"WlgMost{t}Id"]["raw"] for t in ("One", "Two", "Three")], [d["case_id"] for d in ordered[-3:][::-1]])
        self.assertEqual(b["WlgLeastOneId"]["raw"], "l1a-gs-v2-049-cf0a7d1028")
        self.assertEqual(b["WlgMostOneId"]["raw"], "l1a-gs-v2-091-1aab0b78cb")
        self.assertEqual(b["WlgLeastLengthMaxMm"]["value"], "29.4")
        self.assertEqual(b["WlgLeastRadiusMaxMm"]["value"], "2.14")
        lengths = [d["geometry"]["chamber_length_m"] for d in designs]
        self.assertAlmostEqual(b["WlgRhoLength"]["raw"], geo.spearman(lengths, wall), places=15)
        self.assertEqual(b["WlgRhoLength"]["value"], "$-$0.05")
        self.assertEqual(b["WlgRhoRadius"]["value"], "$-$0.12")
        self.assertEqual(b["WlgRhoPitch"]["value"], "$+$0.36")
        self.assertEqual(b["WlgRhoReflected"]["value"], "$-$0.79")
        # Spearman self-checks: perfect monotone relations and a tie-aware rank.
        self.assertAlmostEqual(geo.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]), 1.0, places=15)
        self.assertAlmostEqual(geo.spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]), -1.0, places=15)
        self.assertEqual(geo._rank([2.0, 1.0, 2.0, 3.0]), [2.5, 1.0, 2.5, 4.0])
        mu = [d["diagnostics"]["magnetic_moment_variation"]["median"] for d in designs]
        self.assertEqual((b["WlgMuMedianMin"]["value"], b["WlgMuMedianMax"]["value"]), ("0.11", "0.47"))
        self.assertEqual((b["WlgMuMedianMin"]["raw"], b["WlgMuMedianMax"]["raw"]), (min(mu), max(mu)))
        consumer = _load_artifact("artifacts/coupling-consumer-record.json")
        self.assertEqual(b["WlgConsumedVerified"]["raw"], sum(1 for c in consumer["screening_designs_consumed"] if c["consumption_status"] == "consumed_verified_handoff"))
        self.assertEqual(b["WlgConsumedVerified"]["raw"], 96)
        self.assertEqual(b["WlgVFourP"]["value"], "0.645")
        self.assertIs(b["WlgVFourInScreeningSet"]["raw"], False)
        self.assertEqual(b["WlgValidatorsPassed"]["raw"], 6664)
        self.assertEqual(b["WlgValidatorsFailed"]["raw"], 0)
        self.assertEqual(b["WlgCaseCount"]["raw"], 196)
        self.assertEqual(b["WlgOrbitCount"]["raw"], 100352)
        self.assertEqual(b["WlgTotalOrbits"]["raw"], sum(c["trial_count"] for d in designs for c in d["cases"].values()))
        self.assertEqual(b["WlgEnergyErrorMax"]["raw"], 0.0)
        self.assertEqual(b["WlgTimeouts"]["raw"], 0)
        self.assertEqual(b["WlgNumericalFailures"]["raw"], 0)
        self.assertEqual(b["WlgToleratedEolFiles"]["raw"], 0)
        self.assertEqual(b["WlgRecordedOutcome"]["raw"], geo.RECORDED_OUTCOME)
        self.assertEqual(b["WlgFieldModelLevel"]["raw"], "L1a")

    def test_wilson_intervals_recompute_exactly(self) -> None:
        checked = 0
        for design in self.dataset["designs"]:
            for case in design["cases"].values():
                for estimand in ("wall_hit", "domain_escape", "reflected", "timeout"):
                    estimate = case[estimand]
                    p, lower, upper = geo.wilson(estimate["successes"], estimate["trials"])
                    self.assertEqual((estimate["probability"], estimate["lower"], estimate["upper"]), (p, lower, upper))
                    checked += 1
        self.assertEqual(checked, 196 * 4)
        self.assertEqual(geo.wilson(330, 512)[1:], (0.6021349532568827, 0.6847749053232215))
        with self.assertRaises(ValueError):
            geo.wilson(5, 4)

    def test_section_uses_only_generated_macros_and_types_no_numbers(self) -> None:
        defined = set(re.findall(r"\\newcommand\{\\(Wlg[A-Za-z]+)\}", self.tex))
        used = set(re.findall(r"\\(Wlg[A-Za-z]+)", self.section))
        self.assertTrue(used, "section must use evidence macros")
        self.assertEqual(used - defined, set(), "section uses undefined macros")
        self.assertGreater(len(used), 150)
        for table in geo.TABLE_MACROS:
            self.assertIn(table, used)
        self.assertEqual(check_paper.section_literal_digits(self.section, "Wlg"), [], "hand-typed digits in the section")
        self.assertEqual(check_paper.section_literal_digits(self.section + "\n96 designs\n", "Wlg"), ["9", "6"])
        self.assertIn(f"\\subsection{{{geo.SECTION_HEADING}}}", self.section)
        for heading in ("Method.", "Results.", "Reflections, escapes and cell structure.", "Geometry and the wall-hit probability.", "Coupling consumer.", "Scope."):
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
        self.assertNotIn("\\Wlf", self.section, "the section stays self-contained; the wall-loss contrast is bound through the protocol disclosure")

    def test_generated_tex_tables_are_data_bound(self) -> None:
        for table in geo.TABLE_MACROS:
            self.assertEqual(self.tex.count(f"\\newcommand{{\\{table}}}"), 1)
        designs = self.dataset["designs"]
        ordered = sorted(designs, key=lambda d: d["reported"]["wall_hit"]["probability"])
        for design in ordered[:3] + ordered[-3:]:
            geometry = design["geometry"]
            reported = design["reported"]["wall_hit"]
            row = (
                f"& {geometry['stage_count']} & {1e3 * geometry['chamber_length_m']:.1f} & {1e3 * geometry['exit_start_m']:.1f} & "
                f"{1e3 * geometry['wall_radius_m']:.2f} & {1e3 * geometry['stage_pitch_m']:.2f} & "
                f"{'yes' if geometry['has_divergent_exit'] else 'no'} & {reported['probability']:.3f} [{reported['lower']:.3f}, {reported['upper']:.3f}] &"
            )
            self.assertIn(row, self.tex)
        for cell in ("gs1-cell-1", "gs1-cell-2", "gs1-cell-3", "gs1-cell-4"):
            wall = sum(d["per_cell"]["accepted-2N"][cell]["counts"]["wall_hit"] for d in designs)
            trials = sum(d["per_cell"]["accepted-2N"][cell]["trials"] for d in designs)
            self.assertIn(f"& {wall} & ", self.tex)
            self.assertEqual(trials, 96 * 128)
        self.assertIn("\\texttt{reflected} & 11268 & 22.9\\% & 22904 & 22.8\\%\\\\", self.tex)
        self.assertIn("\\quad\\texttt{upstream\\_\\allowbreak{}anode\\_\\allowbreak{}plane} & 1635 & ", self.tex)
        self.assertIn("total & 49152 & 100.0\\% & 100352 & 100.0\\%\\\\", self.tex)
        self.assertIn("electron orbits integrated & 100{,}352\\\\", self.tex)
        sidecar = json.loads(self.sidecar_bytes)
        self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.tex_bytes).hexdigest())
        self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.evidence_bytes).hexdigest())
        self.assertEqual(sidecar["evidence_revision"], geo.RESULTS_COMMIT_SHA)
        self.assertEqual(sidecar["claim_ids"], [geo.ARTIFACT_CLAIM_ID])
        self.assertEqual(sidecar["dashboard"], self.evidence["dashboard"])
        self.assertIn(geo.RECORDED_OUTCOME, sidecar["claim_status"])
        artifact_macros = check_paper.extract_macros(self.tex, "ArtifactClaim", 3)
        self.assertEqual(len(artifact_macros), 4)
        for macro in artifact_macros:
            self.assertEqual(macro.arguments[:2], (geo.ARTIFACT_CLAIM_ID, geo.ARTIFACT_ID))
        self.assertEqual(self.evidence["tables"]["WlgExtremeTable"]["rows"], 6)
        self.assertEqual(self.evidence["tables"]["WlgCellTable"]["rows"], 4)

    def test_tampered_bundle_dashboard_or_macro_is_rejected(self) -> None:
        changed = self.tex.replace("\\newcommand{\\WlgOrbitCount}{100{,}352}", "\\newcommand{\\WlgOrbitCount}{100{,}353}")
        self.assertNotEqual(changed, self.tex)
        self.assertNotEqual(
            hashlib.sha256(changed.encode("utf-8")).hexdigest(),
            json.loads(self.sidecar_bytes)["output"]["sha256"],
        )
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            target = repo / geo.RESULTS
            target.parent.mkdir(parents=True)
            # Copy only what the Bundle constructor reads: every manifest-listed file.
            shutil.copytree(RESULTS, target)
            victim = target / "artifacts" / "campaign-result.json"
            original = victim.read_bytes()
            self.assertIn(b'"orbit_count":100352', original)
            victim.write_bytes(original.replace(b'"orbit_count":100352', b'"orbit_count":100353', 1))
            with self.assertRaises(ValueError):
                geo.Bundle(repo)
            victim.write_bytes(original)
            # A CRLF rewrite of any bundle file is a byte mismatch: no tolerance exists for this bundle.
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
        # The Wilson check refuses a sealed estimate whose bound was altered.
        estimate = dict(self.dataset["designs"][0]["reported"]["wall_hit"])
        geo._check_estimate(estimate, "intact")
        estimate["upper"] = estimate["upper"] + 1e-9
        with self.assertRaises(ValueError):
            geo._check_estimate(estimate, "tampered")

    def test_representative_endpoint_tables_hash_to_the_dataset(self) -> None:
        for design in self.dataset["designs"]:
            if not design["representative"]:
                continue
            for key, case in design["cases"].items():
                raw = gzip.decompress((RESULTS / "artifacts/endpoints" / f"{design['case_id']}--{key}.json.gz").read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), case["endpoints_payload_sha256"])
                rows = json.loads(raw)["rows"]
                self.assertEqual(len(rows), case["trial_count"])
                self.assertEqual(sum(1 for r in rows if r["termination"] == "reflected"), case["termination_counts"]["reflected"])
                self.assertEqual(sum(1 for r in rows if r["termination"] == "wall_hit"), case["termination_counts"]["wall_hit"])

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
                     f"-output-directory={scratch}", "sections/wall-loss-geometry-screening-v1-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "wall-loss-geometry-screening-v1-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "wall-loss-geometry-screening-v1-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
