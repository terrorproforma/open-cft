"""Build ``xenon-cross-sections-v2.json`` (e-Xe set with FOUR resolved excitation levels; model v2.3.0 / R3a).

Why a v2 file
-------------
``xenon-cross-sections-v1.json`` lumps the four Biagi-v7.1 excitation levels (8.315 / 9.447 /
9.917 / 11.7 eV) into ONE channel with an 8.32 eV energy loss (Szabo 2001 convention).  The physics
completeness audit (``modern/docs/pic2d-physics-completeness-audit.md`` section 4.e, gap e1) grades
that a medium gap: the total excitation frequency is right but the per-event energy loss is
under-estimated by up to 3.4 eV for the upper levels (inelastic power ~15 % low).  v2 keeps every
level as its own process with its own threshold, so the MCC removes the level's energy and the
inelastic ledger sums ``W x sum_k n_k E_k``.

Data source (unchanged database, same bytes)
--------------------------------------------
The four levels are the Biagi-v7.1 (Magboltz 7.1) ``EXCITATION`` blocks of the LXCat export that
v1 already binds (``sources/lxcat_biagi-v7.1_xe_extract.txt``, sha256 pinned below and in the v1
builder).  The audit's "four levels" ARE the Biagi-v7.1 levels; the other LXCat sets in the same
export are NOT four-level sets (Biagi = Magboltz 8.9 lineage: 33 levels; Hayashi: 6 partial
levels; Morgan: 1 lumped; Puech: not extracted here), so choosing Biagi-v7.1 keeps elastic,
excitation and ionisation in ONE self-consistent swarm-validated database.  The cross-check
values of the other databases (summed excitation at fixed energies) are recorded in the
provenance so the factor-2..4 spread stays visible.

``--verify-upstream`` re-downloads the pinned upstream export (1.97 MB), checks its sha256 against
the v1 pin, re-derives the extract and compares it byte-for-byte, and recomputes the cross-check
sums; the retrieval timestamp is written into the provenance.  Without the flag the build reads the
bound extract only (fully offline, deterministic).

Grid and rounding are those of v1 (``inelastic_grid`` per level: the exact threshold point plus 81
points log-spaced in ``E - E_k`` from 0.01 eV to 1000 eV; 6 significant figures).  Consistency
check written into the file: the sum of the four v2 levels equals the v1 lumped channel to
``max_relative_deviation`` (interpolation on different grids; ~1e-3) at every v1 grid point.

Usage (from the repository root)::

    python modern/spec/pic2d/build_xenon_cross_sections_v2.py
    python modern/spec/pic2d/build_xenon_cross_sections_v2.py --verify-upstream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_xenon_cross_sections as v1  # noqa: E402  (same directory; the v1 pipeline is reused verbatim)

OUT_JSON = HERE / "xenon-cross-sections-v2.json"
OUT_SHA = HERE / "xenon-cross-sections-v2.json.sha256"
V1_JSON = HERE / "xenon-cross-sections-v1.json"

# Biagi-v7.1 e/Xe excitation levels (LXCat block names ``Xe -> Xe*(<threshold>eV)``), in threshold order.
LEVELS = ((8.315, "8p315"), (9.447, "9p447"), (9.917, "9p917"), (11.7, "11p7"))
CROSS_CHECK_ENERGIES_EV = (10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 300.0)
# databases of the same export used as cross-checks of the summed excitation (none is a 4-level set)
CROSS_CHECK_DATABASES = ("Biagi (transcription of data from SF Biagi's Fortran code, Magboltz.)", "Hayashi database",
                         "Morgan (Kinema Research  Software)", "BSR (Quantum-mechanical calculations by O. Zatsarinny and K. Bartschat)")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def summed_excitation(blocks: list[dict], energies: np.ndarray) -> np.ndarray:
    total = np.zeros_like(energies)
    for block in blocks:
        if block["E"][-1] >= energies.max():
            total += v1.evaluate(block, energies)
        else:   # a table ending below the probe energies: hold its last value (cross-check only)
            total += np.interp(energies, block["E"], block["sigma"], left=0.0, right=block["sigma"][-1])
    return total


def cross_checks_from_export(upstream: bytes) -> dict[str, dict[str, object]]:
    """Summed e/Xe excitation of the other LXCat databases in the export at the probe energies (m2)."""

    text = upstream.decode("utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith("DATABASE:")]
    out: dict[str, dict[str, object]] = {}
    energies = np.asarray(CROSS_CHECK_ENERGIES_EV)
    for k, start in enumerate(starts):
        name = lines[start].partition(":")[2].strip()
        if name not in CROSS_CHECK_DATABASES:
            continue
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        blocks = [b for b in v1.parse_lxcat_blocks("\n".join(lines[start:end])) if b["meta"].get("SPECIES") == "e / Xe"]
        excitation = [b for b in blocks if b["kind"] == "EXCITATION"]
        if not excitation:
            continue
        total = summed_excitation(excitation, energies)
        out[name] = {
            "excitation_levels": len(excitation),
            "thresholds_ev": sorted({float(b["param"]) for b in excitation}),
            "summed_excitation_m2": {f"{e:g}": float(f"{s:.4e}") for e, s in zip(CROSS_CHECK_ENERGIES_EV, total, strict=True)},
        }
    return out


def verify_upstream() -> tuple[str, dict[str, object]]:
    data = v1.download(v1.UPSTREAM_URL_PINNED)
    retrieved = utc_now()
    digest = v1.sha256_bytes(data)
    if digest != v1.UPSTREAM_SHA256 or len(data) != v1.UPSTREAM_BYTES:
        raise SystemExit(f"upstream changed: sha256 {digest} ({len(data)} B) != pinned {v1.UPSTREAM_SHA256} ({v1.UPSTREAM_BYTES} B)")
    extract = v1.make_extract(data)
    if v1.sha256_bytes(extract) != v1.EXTRACT_SHA256:
        raise SystemExit("re-derived extract differs from the bound extract")
    return retrieved, cross_checks_from_export(data)


def build(*, reverified_utc: str | None, cross_checks: dict[str, object] | None) -> dict:
    extract_bytes = v1.EXTRACT_PATH.read_bytes()
    extract_sha = v1.sha256_bytes(extract_bytes)
    if v1.EXTRACT_SHA256 and extract_sha != v1.EXTRACT_SHA256:
        raise SystemExit(f"extract sha256 {extract_sha} != pinned {v1.EXTRACT_SHA256}")
    blocks = v1.parse_lxcat_blocks(extract_bytes.decode("utf-8"))
    elastic = [b for b in blocks if b["kind"] == "ELASTIC"]
    excit = sorted((b for b in blocks if b["kind"] == "EXCITATION"), key=lambda b: float(b["param"]))
    ioniz = [b for b in blocks if b["kind"] == "IONIZATION"]
    assert len(elastic) == 1 and len(excit) == 4 and len(ioniz) == 1, "unexpected Biagi-v7.1 Xe block set"
    for block, (threshold, _) in zip(excit, LEVELS, strict=True):
        assert abs(float(block["param"]) - threshold) < 1e-9, (block["name"], threshold)
    source_file = v1.EXTRACT_PATH.relative_to(HERE).as_posix()
    source_label = "LXCat Biagi-v7.1 (Magboltz 7.1) e/Xe set; LXCat export of 21 May 2023 mirrored in lanl/ThunderBoltz"
    tail_formula = (
        "Tabulated LXCat data, linear interpolation in (E, sigma). For E above the last tabulated "
        "energy E_N (only 965-977 eV < E <= 1000 eV): sigma(E) = sigma_N * (E/E_N)^p with "
        "p = ln(sigma_N/sigma_{N-1}) / ln(E_N/E_{N-1})."
    )

    def tail_note(block: dict) -> str:
        return f"{block['name']}: last point {block['E'][-1]:.2f} eV, p={v1.tail_exponent(block['E'], block['sigma']):+.3f}"

    # v1 processes are reused for elastic and ionisation (identical grids and values: same builder functions)
    v1_document = json.loads(V1_JSON.read_text(encoding="utf-8"))
    v1_by_id = {p["id"]: p for p in v1_document["processes"]}
    e_el = v1.elastic_grid()
    energy_el = v1.round_sig(e_el, 10)
    sigma_el = v1.round_sig(v1.evaluate(elastic[0], e_el), 6)
    assert energy_el == v1_by_id["elastic"]["energy_ev"] and sigma_el == v1_by_id["elastic"]["cross_section_m2"]
    e_io = v1.inelastic_grid(v1.THRESHOLD_IONIZATION)
    s_io = v1.evaluate(ioniz[0], e_io)
    s_io[0] = 0.0
    energy_io = v1.round_sig(e_io, 10)
    sigma_io = v1.round_sig(s_io, 6)
    assert energy_io == v1_by_id["ionization"]["energy_ev"] and sigma_io == v1_by_id["ionization"]["cross_section_m2"]

    processes: list[dict] = [dict(v1_by_id["elastic"])]
    level_processes = []
    for block, (threshold, tag) in zip(excit, LEVELS, strict=True):
        grid = v1.inelastic_grid(threshold)
        sigma = v1.evaluate(block, grid)
        sigma[0] = 0.0
        energy = v1.round_sig(grid, 10)
        values = v1.round_sig(sigma, 6)
        v1.check_table(energy, values)
        level_processes.append({
            "id": f"excitation_{tag}",
            "kind": "excitation",
            "threshold_ev": threshold,
            "energy_ev": energy,
            "cross_section_m2": values,
            "source": source_label,
            "source_bytes_sha256": extract_sha,
            "source_file": source_file,
            "lxcat_block": block["name"],
            "formula": tail_formula,
            "notes": (
                f"Biagi-v7.1 effective excitation level '{block['name']}' (LXCat EXCITATION block, threshold {threshold} eV = the "
                f"energy removed from the electron per event). Grid: exact threshold point (sigma=0) plus 81 points log-spaced in "
                f"(E - {threshold} eV) from 0.01 eV to 1000 eV (16/decade). Tail extension {tail_note(block)}. Magboltz 7.1 groups "
                "the xenon manifold into these four effective levels (6s metastable+resonance manifold -> 8.315; 6s' -> 9.447; "
                "6p/5d group -> 9.917; higher Rydberg + 'EX HIGH' -> 11.7); the level identities are Biagi's, not spectroscopic terms."
            ),
        })
    processes.extend(level_processes)
    processes.append(dict(v1_by_id["ionization"]))

    # consistency with the v1 lumped channel: sum of the levels on the v1 grid
    v1_exc = v1_by_id["excitation"]
    grid_v1 = np.asarray(v1_exc["energy_ev"])
    lumped = np.asarray(v1_exc["cross_section_m2"])
    total = np.zeros_like(grid_v1)
    for process in level_processes:
        total += np.interp(grid_v1, np.asarray(process["energy_ev"]), np.asarray(process["cross_section_m2"]), left=0.0)
    mask = lumped > 1e-23
    relative = np.abs(total[mask] - lumped[mask]) / lumped[mask]
    above_10 = grid_v1[mask] > 10.0
    absolute = np.abs(total - lumped)
    lumped_reference = {
        "file": V1_JSON.name,
        "payload_sha256": v1_document["integrity"]["payload_sha256"],
        "lumped_threshold_ev": v1.THRESHOLD_EXCITATION,
        "max_relative_deviation_of_summed_levels_vs_lumped": float(f"{float(relative.max()):.3e}"),
        "max_relative_deviation_energy_ev": float(grid_v1[mask][int(np.argmax(relative))]),
        "max_relative_deviation_above_10_ev": float(f"{float(relative[above_10].max()):.3e}"),
        "max_absolute_deviation_m2": float(f"{float(absolute.max()):.3e}"),
        "max_absolute_deviation_energy_ev": float(grid_v1[int(np.argmax(absolute))]),
        "lumped_peak_m2": float(lumped.max()),
        "comparison_points": int(mask.sum()),
        "note": (
            "sum_k sigma_k(E) of the four v2 levels equals the v1 lumped channel at every v1 grid point above 1e-23 m2 to the "
            "deviations recorded here (the levels are tabulated on their own threshold-anchored grids, so the residual is "
            "piecewise-linear interpolation of the v1 grid across the sharp 9.447 / 9.917 eV onsets, not data: it exceeds 1 % "
            "only below 10 eV where sigma < 2e-21 m2). The total excitation frequency of the v2 set is therefore the v1 one; "
            "only the per-event energy loss changes (8.32 eV -> the level threshold)."
        ),
    }

    provenance = {
        "status": "lxcat-tabulated",
        "database": "Biagi-v7.1",
        "database_version": "Magboltz 7.1 (June 2004), LXCat database 'Biagi-v7.1 (Magboltz version 7.1)'",
        "lxcat_export": "LXCat 'Download for offline use' export generated 21 May 2023 (lxcat/assortment.txt in github.com/lanl/ThunderBoltz)",
        "retrieved_utc": v1.UPSTREAM_RETRIEVED_UTC,
        "upstream_url": v1.UPSTREAM_URL_PINNED,
        "upstream_sha256": v1.UPSTREAM_SHA256,
        "upstream_bytes": v1.UPSTREAM_BYTES,
        "upstream_reverified_utc": reverified_utc,
        "extract_file": source_file,
        "extract_sha256": extract_sha,
        "citations": v1.CITATIONS + [
            "Model v2.3.0 (R3a of modern/docs/pic2d-physics-completeness-audit.md, 2026-09-05): four resolved excitation levels "
            "replace the lumped 8.32 eV channel of xenon-cross-sections-v1.json; same database, same bytes.",
        ],
        "level_choice": (
            "The audit's four levels are the Biagi-v7.1 levels (8.315, 9.447, 9.917, 11.7 eV). Alternatives in the same LXCat export "
            "were inspected and rejected as the production set: 'Biagi' (Magboltz 8.9 lineage) resolves 33 levels whose sum is ~2x LOWER "
            "than Biagi-v7.1 above 20 eV (1.35e-20 vs 2.95e-20 m2 peak); 'Hayashi' carries 6 partial levels summing to ~1/3 of "
            "Biagi-v7.1; 'Morgan' is one lumped total ~1.3x higher; BSR (quantum) has 47 levels to 80-400 eV only. Keeping Biagi-v7.1 "
            "keeps elastic, excitation and ionisation in one self-consistent swarm-validated set (Bordage et al. 2013)."
        ),
        "cross_checks_other_databases_same_export": cross_checks,
        "notes": (
            "Same source bytes and pipeline as v1 (build_xenon_cross_sections.py: parse, linear interpolation in (E, sigma), "
            "power-law tail 965-977 -> 1000 eV, 6 s.f.); elastic and ionisation tables are byte-identical to v1. The only change is "
            "the excitation representation: four processes, each zero below its own threshold, each removing its own threshold energy."
        ),
    }
    payload = {
        "schema": "cft.pic2d.xenon-cross-sections.v2",
        "target": "Xe",
        "projectile": "e-",
        "units": {"energy": "eV", "cross_section": "m2"},
        "provenance": provenance,
        "lumped_reference": lumped_reference,
        "processes": processes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8",
        "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return payload


def write_outputs(payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    OUT_JSON.write_bytes(data)
    file_sha = v1.sha256_bytes(data)
    OUT_SHA.write_bytes(f"{file_sha}  {OUT_JSON.name}\n".encode("utf-8"))
    print(f"wrote {OUT_JSON.name}: {len(data)} bytes")
    print(f"file sha256    : {file_sha}")
    print(f"payload sha256 : {payload['integrity']['payload_sha256']}")


def sanity_report(payload: dict) -> None:
    probe = np.array([9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 300.0])
    print("\nsigma(E) [m2]        " + "".join(f"{e:>10.0f} eV" for e in probe))
    total = np.zeros_like(probe)
    for process in payload["processes"]:
        e, s = np.asarray(process["energy_ev"]), np.asarray(process["cross_section_m2"])
        values = np.interp(probe, e, s, left=0.0)
        if process["kind"] == "excitation":
            total += values
        print(f"{process['id']:20s}" + "".join(f"{v:13.3e}" for v in values))
    print(f"{'sum excitation':20s}" + "".join(f"{v:13.3e}" for v in total))
    ref = payload["lumped_reference"]
    print(f"\nlevels vs lumped v1: max relative deviation {ref['max_relative_deviation_of_summed_levels_vs_lumped']:.2e} at "
          f"{ref['max_relative_deviation_energy_ev']:.3f} eV; above 10 eV {ref['max_relative_deviation_above_10_ev']:.2e}; "
          f"max absolute {ref['max_absolute_deviation_m2']:.2e} m2 ({ref['max_absolute_deviation_m2'] / ref['lumped_peak_m2']:.2e} of the peak)")
    if ref["max_relative_deviation_above_10_ev"] > 5e-3 or ref["max_absolute_deviation_m2"] > 1e-22:
        raise SystemExit("summed levels deviate from the lumped v1 channel by more than 0.5 % above 10 eV or 1e-22 m2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-upstream", action="store_true",
                        help="re-download the pinned LXCat export, verify sha256 + extract, record the retrieval time and cross-checks")
    args = parser.parse_args(argv)
    reverified: str | None = None
    cross_checks: dict[str, object] | None = None
    if args.verify_upstream:
        reverified, cross_checks = verify_upstream()
        print(f"upstream re-verified at {reverified}: sha256 {v1.UPSTREAM_SHA256} unchanged; extract byte-identical")
    else:
        # offline build: keep the recorded cross-checks / re-verification of the previous build, if any
        if OUT_JSON.exists():
            previous = json.loads(OUT_JSON.read_text(encoding="utf-8"))["provenance"]
            reverified = previous.get("upstream_reverified_utc")
            cross_checks = previous.get("cross_checks_other_databases_same_export")
    payload = build(reverified_utc=reverified, cross_checks=cross_checks)
    write_outputs(payload)
    sanity_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
