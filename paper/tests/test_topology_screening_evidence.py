"""Regression tests for the hash-bound topology-screening paper evidence (three studies)."""

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
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "paper/scripts"))

import check_paper  # noqa: E402
import generate_topology_screening_evidence as topo  # noqa: E402

STANDALONE = REPO / "paper/sections/topology-screening-standalone.tex"


def _load(spec: topo.ExperimentSpec, relative: str):
    return json.loads((REPO / spec.experiment_path / relative).read_bytes().decode("utf-8"))


class TopologyScreeningEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = {key: topo.render(REPO, spec) for key, spec in topo.EXPERIMENTS.items()}
        cls.evidence = {key: json.loads(value[0]) for key, value in cls.rendered.items()}
        cls.tex = {key: value[1].decode("utf-8") for key, value in cls.rendered.items()}
        cls.sections = {key: (REPO / spec.section_path).read_text(encoding="utf-8") for key, spec in topo.EXPERIMENTS.items()}

    def test_regeneration_is_deterministic_and_committed_files_are_current(self) -> None:
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                again = topo.render(REPO, spec)
                self.assertEqual(again, self.rendered[key])
                evidence_bytes, tex_bytes, sidecar_bytes = self.rendered[key]
                self.assertEqual((REPO / spec.evidence_path).read_bytes(), evidence_bytes)
                self.assertEqual((REPO / spec.output_path).read_bytes(), tex_bytes)
                self.assertEqual((REPO / spec.sidecar_path).read_bytes(), sidecar_bytes)
                for path in (spec.evidence_path, spec.output_path, spec.sidecar_path, spec.section_path):
                    self.assertNotIn(b"\r", (REPO / path).read_bytes(), path.as_posix())
                for forbidden in ("AppData", "C:\\", "C:/", "/Users/", "http://", "https://"):
                    self.assertNotIn(forbidden, self.tex[key])
                    self.assertNotIn(forbidden, evidence_bytes.decode("utf-8"))
        self.assertNotIn(b"\r", STANDALONE.read_bytes())

    def test_evidence_binds_the_committed_results_revisions(self) -> None:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        manuscript = (REPO / "paper/manuscript.tex").read_text(encoding="utf-8")
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                evidence = self.evidence[key]
                self.assertEqual(evidence["document_type"], spec.document_type)
                self.assertEqual(evidence["evidence_revision"], spec.results_commit)
                self.assertEqual(evidence["classification"], spec.classification)
                self.assertEqual(evidence["recorded_outcome"], spec.recorded_outcome)
                self.assertEqual(evidence["screening_model"], topo.SCREENING_MODEL)
                for commit in (spec.results_commit, spec.preregistration_commit):
                    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=REPO, check=False, capture_output=True).returncode
                    self.assertEqual(ancestor, 0, commit)
                manifest_rel = (spec.experiment_path / "results/manifest.json").as_posix()
                committed = subprocess.run(["git", "rev-parse", f"{spec.results_commit}:{manifest_rel}"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
                self.assertEqual(evidence["binding"]["manifest_git_blob"], committed)
                self.assertEqual(evidence["bundle"]["manifest_sha256"], hashlib.sha256((REPO / manifest_rel).read_bytes()).hexdigest())
                self.assertEqual(evidence["bundle"]["tolerated_eol_files"], sorted(spec.audited_eol_files))
                integration = evidence["manuscript_integration"]
                self.assertEqual(integration["status"], "admitted")
                self.assertEqual(integration["gate_kind"], "numerical-screening")
                self.assertEqual(integration["gate_id"], spec.gate_id)
                self.assertEqual(integration["manifest_id"], spec.manifest_id)
                self.assertEqual(integration["manifest_path"], spec.manifest_path.as_posix())
                self.assertEqual(manuscript.count(spec.section_binding), 1)
                self.assertEqual(manuscript.count(spec.generated_binding), 1)
                self.assertLess(manuscript.find(spec.generated_binding), manuscript.find("\\begin{document}"))

    def test_every_macro_value_traces_to_a_hashed_artifact(self) -> None:
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                evidence = self.evidence[key]
                macros = evidence["macros"]
                self.assertGreater(len(macros), 60)
                names = [item["name"] for item in macros]
                self.assertEqual(len(names), len(set(names)))
                artifacts = evidence["artifacts"]
                root = REPO / spec.experiment_path
                for relative, meta in artifacts.items():
                    raw = (root / relative).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
                    self.assertEqual(len(raw), meta["bytes"], relative)
                for relative, meta in evidence.get("lineage_artifacts", {}).get("files", {}).items():
                    raw = (REPO / relative).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), meta["sha256"], relative)
                loaded = {}
                for relative in artifacts:
                    if relative.endswith(".json"):
                        loaded[relative] = _load(spec, relative)
                derived = 0
                for item in macros:
                    self.assertTrue(item["name"].isalpha() and item["name"].startswith(spec.macro_prefix))
                    self.assertEqual(topo.format_value(item["format"], item["raw"]), item["value"])
                    self.assertIn(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}", self.tex[key])
                    if item["derived"]:
                        derived += 1
                        self.assertTrue(item["derivation"] and item["inputs"])
                        for source in item["inputs"]:
                            if source["artifact"].startswith("lineage:"):
                                self.assertIn(source["artifact"][len("lineage:"):], evidence["lineage_artifacts"]["files"])
                            else:
                                self.assertIn(source["artifact"], artifacts)
                        continue
                    source = item["source"]
                    self.assertIn(source["artifact"], artifacts)
                    raw = topo.resolve_pointer(loaded[source["artifact"]], source["pointer"])
                    self.assertEqual(raw, item["raw"])
                self.assertGreater(derived, 8)

    def test_derived_macros_recompute_from_the_artifacts(self) -> None:
        sweep = {m["name"]: m["raw"] for m in self.evidence["l1a-sweep-v2"]["macros"]}
        raw = _load(topo.SWEEP, "results/raw-results.json")
        summary = _load(topo.SWEEP, "results/summary.json")
        positions = [z for case in raw["cases"] for z in case["qois"]["axis_cusp_positions_m"]]
        hist = Counter(int(case["qois"]["axis_cusp_count"]) for case in raw["cases"])
        self.assertEqual(sweep["SwpAxisCuspTotal"], len(positions))
        self.assertEqual(sweep["SwpAxisCuspZMinMm"], min(positions))
        self.assertEqual(sweep["SwpAxisCuspZMaxMm"], max(positions))
        self.assertEqual((sweep["SwpCuspThreeDesigns"], sweep["SwpCuspFourDesigns"], sweep["SwpCuspFiveDesigns"]), (hist[3], hist[4], hist[5]))
        self.assertEqual(sweep["SwpGateCount"], len(summary["terminal_gates"]))
        self.assertEqual(sweep["SwpGatesPassed"], sum(1 for g in summary["terminal_gates"] if g["passed"]))
        self.assertEqual(sweep["SwpGatesPassed"], 7)
        self.assertEqual(sweep["SwpNondominated"], summary["nondominated_count"])

        fcn = {m["name"]: m["raw"] for m in self.evidence["four-cell-v2"]["macros"]}
        dataset = _load(topo.FOUR_CELL, "results/dataset.json")
        counts = [len(case["maps"]["primary"]["interior_cusp_z_m"]) for case in dataset["cases"]]
        self.assertEqual((fcn["FcnPrimaryCuspMin"], fcn["FcnPrimaryCuspMax"], fcn["FcnPrimaryCuspTotal"]), (min(counts), max(counts), sum(counts)))
        self.assertEqual(fcn["FcnStable"], 0)
        self.assertEqual(fcn["FcnTopologyCountFailures"], dataset["summary"]["failure_counts"]["TOPOLOGY_COUNT"])
        self.assertEqual(fcn["FcnGpuReplayFailed"], sum(1 for r in dataset["gpu_replay"] if not r["passed"]))
        self.assertEqual(fcn["FcnGpuReplayPassed"] + fcn["FcnGpuReplayFailed"], fcn["FcnGpuReplayRequired"])
        self.assertEqual(fcn["FcnLineageVOneCompatible"], 2)

        tch = {m["name"]: m["raw"] for m in self.evidence["topology-characterization-v1"]["macros"]}
        char = _load(topo.CHARACTERIZATION, "results/dataset.json")
        classes = Counter(root["local_topology"]["classification"] for case in char["cases"] for root in case["maps"]["primary"]["roots"])
        self.assertEqual((tch["TchXRoots"], tch["TchORoots"], tch["TchDegenerateRoots"]), (classes["X"], classes["O"], classes["degenerate"]))
        self.assertEqual(tch["TchClusteredRoots"], sum(classes.values()))
        self.assertEqual(tch["TchStableEligibleCusps"], 0)
        self.assertEqual(tch["TchStableEligibleCells"], 0)
        self.assertEqual(tch["TchChannelRoots"], tch["TchChannelXRoots"], tch["TchChannelUnresolved"])

    def test_sections_use_only_generated_macros_and_type_no_numbers(self) -> None:
        for key, spec in topo.EXPERIMENTS.items():
            with self.subTest(study=key):
                section = self.sections[key]
                defined = set(re.findall(rf"\\newcommand\{{\\({spec.macro_prefix}[A-Za-z]+)\}}", self.tex[key]))
                used = set(re.findall(rf"\\({spec.macro_prefix}[A-Za-z]+)", section))
                self.assertTrue(used)
                self.assertEqual(used - defined, set())
                for macro in (*spec.table_macros, f"{spec.macro_prefix}Classification"):
                    self.assertIn(macro, used)
                self.assertEqual(check_paper.section_literal_digits(section, spec.macro_prefix), [])
                self.assertIn(f"\\subsection{{{spec.section_heading}}}", section)
                self.assertEqual(check_paper.find_unregistered_claims(section), [])
                self.assertEqual(check_paper.find_unregistered_claims(self.tex[key]), [])
                for pattern in check_paper.PLACEHOLDERS.values():
                    self.assertIsNone(pattern.search(section))
                    self.assertIsNone(pattern.search(self.tex[key]))
                artifact_macros = check_paper.extract_macros(self.tex[key], "ArtifactClaim", 3)
                self.assertEqual(len(artifact_macros), len(spec.table_macros))
                for macro in artifact_macros:
                    self.assertEqual(macro.arguments[:2], (spec.artifact_claim_id, spec.artifact_id))
                sidecar = json.loads(self.rendered[key][2])
                self.assertEqual(sidecar["output"]["sha256"], hashlib.sha256(self.rendered[key][1]).hexdigest())
                self.assertEqual(sidecar["manifest"]["sha256"], hashlib.sha256(self.rendered[key][0]).hexdigest())
                self.assertEqual(sidecar["claim_ids"], [spec.artifact_claim_id])

    def test_eol_tolerance_applies_to_exactly_the_audited_file(self) -> None:
        for spec in (topo.SWEEP, topo.FOUR_CELL):
            with self.subTest(study=spec.key):
                (relative, audited), = spec.audited_eol_files.items()
                bundle = topo.Bundle(REPO, spec)
                raw = bundle.verify(relative, audited.recorded_sha256)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), audited.lf_sha256)
                self.assertEqual(bundle.tolerated, [relative])
                # any other byte change of the audited file is refused
                with tempfile.TemporaryDirectory() as scratch:
                    repo = Path(scratch)
                    target = repo / spec.experiment_path
                    target.mkdir(parents=True)
                    (target / Path(relative)).parent.mkdir(parents=True, exist_ok=True)
                    (target / relative).write_bytes(raw + b"\n")
                    with self.assertRaises(ValueError):
                        topo.Bundle(repo, spec).verify(relative, audited.recorded_sha256)
                    (target / relative).write_bytes(raw.replace(b"\n", b"\r\n"))
                    # CRLF-restored bytes hash to the recorded digest and take the ordinary byte-exact path
                    scratch_bundle = topo.Bundle(repo, spec)
                    scratch_bundle.verify(relative, audited.recorded_sha256)
                    self.assertEqual(scratch_bundle.tolerated, [])
                    # the rule never applies to another file with the same bytes
                    (target / "other.json").write_bytes(raw)
                    with self.assertRaises(ValueError):
                        topo.Bundle(repo, spec).verify("other.json", audited.recorded_sha256)
        self.assertEqual(topo.CHARACTERIZATION.audited_eol_files, {})
        self.assertEqual(self.evidence["topology-characterization-v1"]["bundle"]["tolerated_eol_files"], [])

    def test_tampered_bundle_is_rejected(self) -> None:
        spec = topo.FOUR_CELL
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch)
            results = repo / spec.experiment_path / "results"
            results.mkdir(parents=True)
            for name in ("manifest.json", "manifest.json.sha256", "dataset.json", "dataset.json.sha256"):
                shutil.copy(REPO / spec.experiment_path / "results" / name, results / name)
            manifest = json.loads((results / "manifest.json").read_bytes())
            entry = next(item for item in manifest["artifacts"] if item["path"] == "dataset.json")
            untampered = topo.Bundle(repo, spec)
            untampered.verify("results/dataset.json", entry["sha256"], entry["bytes"])
            victim = results / "dataset.json"
            original = victim.read_bytes()
            tampered = original.replace(b'"stable_count": 0', b'"stable_count": 1', 1)
            self.assertNotEqual(tampered, original)
            victim.write_bytes(tampered)
            with self.assertRaises(ValueError):
                topo.Bundle(repo, spec).verify("results/dataset.json", entry["sha256"], entry["bytes"])
            with self.assertRaises(ValueError):
                topo.verify_sealed(json.loads(tampered), "tampered dataset")
            # a CRLF rewrite of a non-audited file is a mismatch, never a tolerated EOL difference
            victim.write_bytes(original.replace(b"\n", b"\r\n"))
            with self.assertRaises(ValueError):
                topo.Bundle(repo, spec).verify("results/dataset.json", entry["sha256"], entry["bytes"])

    def test_standalone_sections_compile_when_pdflatex_is_available(self) -> None:
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
                     f"-output-directory={scratch}", "sections/topology-screening-standalone.tex"],
                    cwd=REPO / "paper", env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:])
            log = (Path(scratch) / "topology-screening-standalone.log").read_text(encoding="utf-8", errors="replace")
            for marker in ("LaTeX Error:", "There were undefined references", "Overfull \\hbox", "Overfull \\vbox"):
                self.assertNotIn(marker, log)
            self.assertTrue((Path(scratch) / "topology-screening-standalone.pdf").stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
