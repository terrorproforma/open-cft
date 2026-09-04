"""Build ``xenon-ion-neutral-cross-sections-v1.json`` (Xe+ + Xe charge exchange and momentum transfer; model v2.3.0 / R3b).

Processes
---------
``cex`` - resonant symmetric charge exchange Xe+ + Xe -> Xe + Xe+.  Cross section: the guided-ion-beam
    fit of Miller, Pullins, Levandier, Chiu and Dressler, J. Appl. Phys. 91, 984 (2002), doi:10.1063/1.1426246,
    ``sigma_CEX(E) = (87.3 - 13.6 log10(E / eV)) A^2`` (their Eq. 4; measured 1-300 eV, in agreement with earlier
    beam data and ~30 % above the Rapp-Francis model).  Kinematics in the MCC: the ion takes the sampled thermal
    velocity of the atom, the atom leaves as a fast neutral with the ion's velocity (= backscattering through pi
    in the centre of mass for equal masses).
``mex`` - momentum transfer (elastic, isotropic in the centre of mass).  Cross section: the ISOTROPIC component of
    A. V. Phelps' two-component (isotropic + backscatter) Xe+/Xe elastic model (Phelps database on LXCat, from
    Piscitelli, Phelps, de Urquijo, Basurto and Pitchford, Phys. Rev. E 68, 046408 (2003), doi:10.1103/PhysRevE.68.046408),
    taken from the copy redistributed in BLAST-WarpX/warpx-data ``MCC_cross_sections/Xe/ion_scattering.dat`` (a
    0.01 eV resample, 0-750 eV; the file states the origin).  The table is exactly ``3.39e-19 E^-1/2 m2``
    (polarisation / Langevin form) to 5 significant figures at every probed energy, which is the analytic form
    Phelps uses for the isotropic part; above 750 eV it is extended with that power law.
``cross_check`` (NOT a process): Phelps' BACKSCATTER component (``ion_back_scatter.dat``, "A. V. Phelps database
    (available on LxCat) from Piscitelli et al 2003"), which in Phelps' decomposition plays the role of the charge
    transfer cross section; it lies 17 % (10 eV) to 41 % (300 eV) ABOVE the Miller 2002 fit.  Miller's direct
    measurement is used for ``cex``; the Phelps backscatter table is recorded so the spread is visible.

Energy convention (recorded in the file and enforced by the MCC operator)
------------------------------------------------------------------------
``energy_ev`` is the relative kinetic energy ``E = 1/2 M |v_ion - v_atom|^2`` with the XENON mass ``M`` - the ion's
laboratory energy for a target at rest, the frame of Miller 2002's beam data (their "ion energy") and of Phelps'
compilations.  The centre-of-mass energy of the symmetric pair is ``E/2``.  (WarpX evaluates its copies of the same
tables at the centre-of-mass energy; that is a documented factor-2 difference in the energy argument, worth <= 10 %
of sigma on these log-slow curves.)  Below 0.1 eV the Miller fit is held at its 0.1 eV value (100.9 A^2): the fit
diverges logarithmically as E -> 0 and its data start at ~1 eV; the drift-tube value for thermal Xe+(2P3/2) is of the
same order (Hegerberg, Elford and Skullerud 1986, doi:10.1088/0022-3700/19/4/012).

Grid: ``E = 0`` (held value) plus 16 points per decade from 0.01 eV to 2000 eV; 6 significant figures.

Sources handling
----------------
``--refresh-source`` downloads the two WarpX files at the pinned commit, verifies their sha256 against the pins
below and writes ``sources/warpx_phelps_xe_ion_extract.txt``: the upstream lines at the grid energies (verbatim
rows of the 0.01 eV resample, nearest row at or below each grid energy) for both tables, with the upstream
sha256s and the retrieval time in the header.  The default build reads the extract only (offline, deterministic).

Usage (from the repository root)::

    python modern/spec/pic2d/build_xenon_ion_neutral_cross_sections.py
    python modern/spec/pic2d/build_xenon_ion_neutral_cross_sections.py --refresh-source
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES_DIR = HERE / "sources"
EXTRACT_PATH = SOURCES_DIR / "warpx_phelps_xe_ion_extract.txt"
OUT_JSON = HERE / "xenon-ion-neutral-cross-sections-v1.json"
OUT_SHA = HERE / "xenon-ion-neutral-cross-sections-v1.json.sha256"

# --- pinned upstream (BLAST-WarpX/warpx-data, commit that added the Xe MCC tables, 2021-07-16) -------------------
UPSTREAM_COMMIT = "c42f106f96fe6415c7cc89dd31d1027ff5c8c68e"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/BLAST-WarpX/warpx-data/{UPSTREAM_COMMIT}/MCC_cross_sections/Xe/"
UPSTREAM_FILES = {
    "ion_scattering.dat": ("5f196141e10b5986b2dc1ad017dd4f3c3d417eb6c17ee1189863e668fce874ab", 1875000),
    "ion_back_scatter.dat": ("1d47da856088f69d6682c55e00a010883803e83f954e8ea3399d2a3212c2cb3b", 1875000),
}
UPSTREAM_README_SHA256 = "1022835e125b39e150dc9cf70d6ee73e753c44af1938b920b0c35bb33e57ea80"
# sha256 of the extract written by --refresh-source (pinned after the first generation; None = not yet pinned)
EXTRACT_SHA256: str | None = "7bab80addce401c7c22959ac7c38944bfedb1f5bf0a6f6d9240309b0c548f085"

# --- physics constants ------------------------------------------------------------------------------------------
MILLER_A_A2 = 87.3        # A^2
MILLER_B_A2 = 13.6        # A^2 per decade
MILLER_FLOOR_EV = 0.1     # the fit is held below this energy
MILLER_RANGE_EV = (1.0, 300.0)
A2_TO_M2 = 1.0e-20
E_MAX = 2000.0
GRID = (-2.0, math.log10(E_MAX), 16)   # log10 range and points per decade


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cft-pic2d cross-section builder"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (pinned https URL)
        return resp.read()


def energy_grid() -> np.ndarray:
    lo, hi, per_decade = GRID
    count = int(round((hi - lo) * per_decade)) + 1
    grid = np.logspace(lo, hi, count)
    grid[-1] = E_MAX
    return np.concatenate([[0.0], grid])


def round_sig(values: np.ndarray, sig: int) -> list[float]:
    return [0.0 if v == 0 else float(f"{v:.{sig - 1}e}") for v in values]


# ---------------------------------------------------------------------------------------------------------------
# source extract
# ---------------------------------------------------------------------------------------------------------------
def make_extract(files: dict[str, bytes], retrieved_utc: str) -> bytes:
    grid = energy_grid()[1:]
    header = [
        "EXTRACT of BLAST-WarpX/warpx-data MCC_cross_sections/Xe ion tables (not verbatim upstream files).",
        f"Upstream commit : {UPSTREAM_COMMIT} (2021-07-16 'Added MCC xsections for argon and xenon (#2)')",
        f"Retrieved (UTC) : {retrieved_utc}",
        "Upstream README (sha256 %s): 'ion neutral isotropic scattering: Piscitelli et al Phys. Rev. E 68, 046408 (2003)';" % UPSTREAM_README_SHA256,
        "  'ion back scattering: A. V. Phelps database (available on LxCat) from Piscitelli et al Phys. Rev. E 68, 046408 (2003)'.",
        "Each upstream file: 75000 rows '<energy eV>  <sigma m2>' at 0.01 eV spacing (1e-4, 1.01e-2, ..., 749.99 eV).",
        "Rows below are VERBATIM upstream rows: the first row and, for each grid energy of the v1 spec (16/decade, 0.01-750 eV),",
        "the row at or immediately below it.  Energies are the ion energy for a stationary atom (Phelps' laboratory frame).",
        "",
    ]
    body: list[str] = []
    for name, data in files.items():
        digest, size = UPSTREAM_FILES[name]
        lines = data.decode("utf-8").splitlines()
        energies = np.array([float(line.split()[0]) for line in lines])
        body.append(f"FILE {name}  sha256 {digest}  bytes {size}  rows {len(lines)}")
        emitted = {0}
        body.append(lines[0])
        for e in grid:
            if e > energies[-1]:
                break
            index = int(np.searchsorted(energies, e, side="right") - 1)
            if index not in emitted:      # the 0.01 eV resample is coarser than the grid below ~0.02 eV
                emitted.add(index)
                body.append(lines[index])
        body.append("END " + name)
    return ("\n".join(header + body) + "\n").encode("utf-8")


def refresh_source() -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {}
    for name, (digest, size) in UPSTREAM_FILES.items():
        data = download(UPSTREAM_BASE + name)
        if sha256_bytes(data) != digest or len(data) != size:
            raise SystemExit(f"{name}: sha256/size mismatch ({sha256_bytes(data)}, {len(data)} B); refusing to build")
        files[name] = data
        print(f"downloaded {name}: {len(data)} bytes sha256 {digest} OK")
    readme = download(UPSTREAM_BASE + "README.md")
    if sha256_bytes(readme) != UPSTREAM_README_SHA256:
        raise SystemExit("upstream README changed; re-read the provenance statement before rebuilding")
    extract = make_extract(files, utc_now())
    EXTRACT_PATH.write_bytes(extract)
    print(f"wrote {EXTRACT_PATH.relative_to(HERE.parents[2])} ({len(extract)} bytes) sha256={sha256_bytes(extract)}")


def parse_extract(text: str) -> tuple[dict[str, np.ndarray], str]:
    tables: dict[str, list[tuple[float, float]]] = {}
    retrieved = ""
    current: str | None = None
    for line in text.split("\n"):
        if line.startswith("Retrieved (UTC) :"):
            retrieved = line.partition(":")[2].strip()
        elif line.startswith("FILE "):
            current = line.split()[1]
            tables[current] = []
        elif line.startswith("END "):
            current = None
        elif current is not None and line.strip():
            a, b = line.split()
            tables[current].append((float(a), float(b)))
    return {name: np.asarray(rows) for name, rows in tables.items()}, retrieved


# ---------------------------------------------------------------------------------------------------------------
# cross sections
# ---------------------------------------------------------------------------------------------------------------
def miller_cex_m2(energy_ev: np.ndarray) -> np.ndarray:
    e = np.maximum(np.asarray(energy_ev, dtype=np.float64), MILLER_FLOOR_EV)
    return np.maximum(MILLER_A_A2 - MILLER_B_A2 * np.log10(e), 0.0) * A2_TO_M2


def power_law_extend(table: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, float]:
    """Linear interpolation inside the tabulated range; local power law (last decade) above it."""

    e_tab, s_tab = table[:, 0], table[:, 1]
    last = e_tab[-1]
    ref = e_tab[np.searchsorted(e_tab, last / 10.0)]
    p = math.log(s_tab[-1] / np.interp(ref, e_tab, s_tab)) / math.log(last / ref)
    out = np.interp(energies, e_tab, s_tab)
    above = energies > last
    out[above] = s_tab[-1] * (energies[above] / last) ** p
    below = energies < e_tab[0]
    out[below] = s_tab[0]
    return out, p


def build() -> dict:
    extract_bytes = EXTRACT_PATH.read_bytes()
    extract_sha = sha256_bytes(extract_bytes)
    if EXTRACT_SHA256 and extract_sha != EXTRACT_SHA256:
        raise SystemExit(f"extract sha256 {extract_sha} != pinned {EXTRACT_SHA256}")
    tables, retrieved = parse_extract(extract_bytes.decode("utf-8"))
    iso, back = tables["ion_scattering.dat"], tables["ion_back_scatter.dat"]
    grid = energy_grid()
    source_file = EXTRACT_PATH.relative_to(HERE).as_posix()

    # cex: Miller 2002 fit
    s_cex = miller_cex_m2(grid)
    # mex: Phelps isotropic component; check the Langevin form on the tabulated rows
    langevin_coefficient = float(np.median(iso[1:, 1] * np.sqrt(iso[1:, 0])))
    langevin_deviation = float(np.max(np.abs(iso[1:, 1] * np.sqrt(iso[1:, 0]) / langevin_coefficient - 1.0)))
    s_mex, p_mex = power_law_extend(iso, grid)
    s_back, p_back = power_law_extend(back, grid)
    probe = np.array([1.0, 10.0, 30.0, 100.0, 300.0])
    ratio = np.interp(probe, grid, s_back) / np.interp(probe, grid, s_cex)

    def table_dict(values: np.ndarray) -> tuple[list[float], list[float]]:
        return round_sig(grid, 10), round_sig(values, 6)

    energy_cex, sigma_cex = table_dict(s_cex)
    energy_mex, sigma_mex = table_dict(s_mex)
    energy_back, sigma_back = table_dict(s_back)
    processes = [
        {
            "id": "cex",
            "kind": "charge_exchange",
            "threshold_ev": 0.0,
            "energy_ev": energy_cex,
            "cross_section_m2": sigma_cex,
            "source": "Miller, Pullins, Levandier, Chiu, Dressler, J. Appl. Phys. 91, 984-991 (2002), doi:10.1063/1.1426246, Eq. (4)",
            "formula": (
                f"sigma_CEX(E) = ({MILLER_A_A2} - {MILLER_B_A2} log10(E/eV)) x 1e-20 m2 for E >= {MILLER_FLOOR_EV} eV; held at its "
                f"{MILLER_FLOOR_EV} eV value below; E = 1/2 M_Xe |v_ion - v_atom|^2 (ion energy for a stationary atom)."
            ),
            "validity_ev": list(MILLER_RANGE_EV),
            "notes": (
                "Guided-ion-beam measurement of symmetric Xe+ + Xe charge exchange, 1-300 eV, 'in good agreement with several earlier "
                "experimental studies and semiclassical calculations' and ~30 % above Rapp-Francis. Used outside 1-300 eV as the "
                "published log-linear fit (0.1-1 eV and 300-2000 eV). Kinematics: the ion continues with the atom's velocity, the atom "
                "with the ion's (velocity swap = pi backscatter in the CM for equal masses); no energy threshold (resonant)."
            ),
        },
        {
            "id": "mex",
            "kind": "momentum_transfer",
            "threshold_ev": 0.0,
            "energy_ev": energy_mex,
            "cross_section_m2": sigma_mex,
            "source": (
                "Phelps database (www.lxcat.net/Phelps), Xe+ / Xe ISOTROPIC elastic component, from Piscitelli, Phelps, de Urquijo, "
                "Basurto, Pitchford, Phys. Rev. E 68, 046408 (2003), doi:10.1103/PhysRevE.68.046408; bytes from BLAST-WarpX/warpx-data "
                f"MCC_cross_sections/Xe/ion_scattering.dat @ {UPSTREAM_COMMIT[:12]}"
            ),
            "source_bytes_sha256": extract_sha,
            "source_file": source_file,
            "upstream_file_sha256": UPSTREAM_FILES["ion_scattering.dat"][0],
            "formula": (
                "Linear interpolation of the tabulated rows (0.01 eV resample of the Phelps table, sub-sampled to this grid); the rows "
                f"follow sigma_iso(E) = {langevin_coefficient:.4g} E^-1/2 m2 to {langevin_deviation:.1e} relative (polarisation form), "
                f"which is the power law (exponent {p_mex:+.3f}) used from 750 to 2000 eV; held at the first row below 1e-4 eV."
            ),
            "notes": (
                "Isotropic (centre-of-mass) part of Phelps' two-component elastic model for symmetric rare-gas ion-atom pairs; the "
                "backscatter part is the charge-transfer half-sphere (recorded under cross_check, not a process). Kinematics: isotropic "
                "redirection of the relative velocity in the CM frame, both partners keep their CM speeds; the atom's recoil is handed to "
                "the (0-D) neutral gas. Energy argument: ion energy for a stationary atom (Phelps' laboratory frame)."
            ),
        },
    ]
    cross_check = {
        "id": "phelps_backscatter",
        "role": "cross-check only (not a process): Phelps' backscatter component ~ the charge-transfer cross section in his model",
        "energy_ev": energy_back,
        "cross_section_m2": sigma_back,
        "source": (
            "Phelps database (www.lxcat.net/Phelps), Xe+ / Xe BACKSCATTER component (Piscitelli et al. 2003); bytes from "
            f"BLAST-WarpX/warpx-data MCC_cross_sections/Xe/ion_back_scatter.dat @ {UPSTREAM_COMMIT[:12]}"
        ),
        "upstream_file_sha256": UPSTREAM_FILES["ion_back_scatter.dat"][0],
        "tail_exponent_above_750_ev": float(f"{p_back:+.4f}"),
        "ratio_to_miller_cex": {f"{e:g}": float(f"{r:.3f}") for e, r in zip(probe, ratio, strict=True)},
        "note": (
            "Phelps' backscatter lies above the Miller 2002 fit by the ratios recorded here (10-300 eV); the two agree within ~20 % below "
            "10 eV. The production 'cex' process uses the direct beam measurement (Miller); the spread is the data uncertainty of the "
            "CEX rate (audit section 4.e / 9)."
        ),
    }
    provenance = {
        "status": "published-fit+lxcat-tabulated",
        "retrieved_utc": retrieved,
        "cex": {"database": "Miller et al. 2002 (J. Appl. Phys. 91, 984) analytic fit", "doi": "10.1063/1.1426246",
                "companion": "Pullins, Chiu, Levandier, Dressler, AIAA 2000-0603, doi:10.2514/6.2000-603"},
        "mex": {"database": "Phelps database (LXCat) - Piscitelli et al. 2003", "doi": "10.1103/PhysRevE.68.046408",
                "mirror": UPSTREAM_BASE + "ion_scattering.dat", "mirror_commit": UPSTREAM_COMMIT, "mirror_readme_sha256": UPSTREAM_README_SHA256},
        "energy_convention": (
            "energy_ev = 1/2 M_Xe |v_ion - v_atom|^2: the ion's kinetic energy in the frame where the atom is at rest (laboratory energy "
            "of a beam on a stationary target). The centre-of-mass energy of the symmetric pair is half of it. The MCC evaluates the "
            "tables at this energy from the ion velocity and a Maxwellian atom velocity sampled at the gas temperature."
        ),
        "network_attempts": [
            {"url": "https://jila.colorado.edu/~avp/collision_data/ionneutral/IONATOM.TXT", "utc": "2026-09-04T16:05:00Z",
             "outcome": "HTTP 200 but a generic JILA site page (34158 B HTML), the Phelps ftp/http data tree is gone"},
            {"url": "https://us.lxcat.net/notes/index.php?download=phelps3", "utc": "2026-09-04T16:05:00Z",
             "outcome": "HTTP 200: Phelps' SYMMCOLL note (PDF, 198976 B) describing the isotropic + backscatter construction; section X (Xe+ + Xe) is a stub"},
            {"url": "https://raw.githubusercontent.com/lanl/ThunderBoltz/main/lxcat/assortment.txt", "utc": "2026-09-04T16:14:27Z",
             "outcome": "HTTP 200, sha256 7019dd0b... unchanged; the export contains e/Xe sets only (no Xe+ / Xe ion-neutral data)"},
            {"url": UPSTREAM_BASE + "README.md", "utc": "2026-09-04T16:16:00Z",
             "outcome": "HTTP 200; origin statement for ion_scattering.dat (Piscitelli 2003) and ion_back_scatter.dat (Phelps database on LXCat) - USED"},
        ],
        "citations": [
            "J. S. Miller, S. H. Pullins, D. J. Levandier, Y. Chiu, R. A. Dressler, 'Xenon charge exchange cross sections for electrostatic "
            "thruster models', J. Appl. Phys. 91, 984-991 (2002), doi:10.1063/1.1426246.",
            "S. H. Pullins, Y. Chiu, D. J. Levandier, R. A. Dressler, 'Ion dynamics in Hall effect and ion thrusters - Xe+ + Xe symmetric "
            "charge transfer', AIAA 2000-0603 (2000), doi:10.2514/6.2000-603.",
            "D. Piscitelli, A. V. Phelps, J. de Urquijo, E. Basurto, L. C. Pitchford, 'Ion mobilities in Xe/Ne and other rare-gas mixtures', "
            "Phys. Rev. E 68, 046408 (2003), doi:10.1103/PhysRevE.68.046408.",
            "Phelps database, www.lxcat.net/Phelps (ion-neutral scattering: isotropic + backscatter components), via BLAST-WarpX/warpx-data "
            f"commit {UPSTREAM_COMMIT}.",
            "L. C. Pitchford et al., 'LXCat: an Open-Access, Web-Based Platform for Data Needed for Modeling Low Temperature Plasmas', "
            "Plasma Processes and Polymers 14, 1600098 (2017), doi:10.1002/ppap.201600098.",
            "R. Hegerberg, M. T. Elford, H. R. Skullerud, 'The mobilities of xenon ions in xenon and the derived charge transfer cross section "
            "for Xe+(2P3/2) ions in xenon', J. Phys. B 19, 613 (1986), doi:10.1088/0022-3700/19/4/012 (thermal-energy context of the 0.1 eV floor).",
            "I. D. Boyd, R. A. Dressler, 'Far field modeling of the plasma plume of a Hall thruster', J. Appl. Phys. 92, 1764 (2002), "
            "doi:10.1063/1.1492014 (CEX + MEX practice in thruster plume codes).",
        ],
        "notes": (
            "Two processes for the Xe+ ion population against the 0-D neutral inventory (per-node density when spatial neutrals arrive): "
            "cex (velocity swap; the atom becomes a fast neutral, the ion a thermal one) and mex (isotropic CM scattering). The Phelps "
            "backscatter table is kept as a cross-check of the CEX magnitude. The 300 kB source-file policy is met by storing verbatim "
            "rows of the WarpX resample at the grid energies instead of the 1.9 MB files; upstream sha256s are pinned."
        ),
    }
    payload = {
        "schema": "cft.pic2d.xenon-ion-neutral-cross-sections.v1",
        "target": "Xe",
        "projectile": "Xe+",
        "units": {"energy": "eV", "cross_section": "m2"},
        "provenance": provenance,
        "processes": processes,
        "cross_check": cross_check,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8",
        "payload_sha256": sha256_bytes(canonical.encode("utf-8")),
    }
    return payload


def write_outputs(payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    OUT_JSON.write_bytes(data)
    file_sha = sha256_bytes(data)
    OUT_SHA.write_bytes(f"{file_sha}  {OUT_JSON.name}\n".encode("utf-8"))
    print(f"wrote {OUT_JSON.name}: {len(data)} bytes")
    print(f"file sha256    : {file_sha}")
    print(f"payload sha256 : {payload['integrity']['payload_sha256']}")


def sanity_report(payload: dict) -> None:
    probe = [0.05, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 2000.0]
    print("\nsigma(E) [A^2]      " + "".join(f"{e:>9.2f} eV" for e in probe))
    for item in payload["processes"] + [payload["cross_check"]]:
        e, s = np.asarray(item["energy_ev"]), np.asarray(item["cross_section_m2"])
        print(f"{item['id']:20s}" + "".join(f"{np.interp(p, e, s) / A2_TO_M2:12.2f}" for p in probe))
    cex = next(p for p in payload["processes"] if p["id"] == "cex")
    at300 = np.interp(300.0, cex["energy_ev"], cex["cross_section_m2"])
    assert abs(at300 - 5.361e-19) < 2e-22, at300      # 87.3 - 13.6 log10(300) = 53.61 A^2 (audit value 5.4e-19)
    mex = next(p for p in payload["processes"] if p["id"] == "mex")
    at10 = np.interp(10.0, mex["energy_ev"], mex["cross_section_m2"])
    assert 1.0e-19 < at10 < 1.2e-19, at10             # 10.7 A^2
    print("\nlambda_CEX at n_g = 3e19 m^-3, 300 eV:", f"{1.0 / (3e19 * at300) * 1e3:.1f} mm (audit: ~60 mm)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-source", action="store_true", help="download the pinned WarpX tables, verify sha256, rewrite the extract")
    args = parser.parse_args(argv)
    if args.refresh_source:
        refresh_source()
    if not EXTRACT_PATH.exists():
        raise SystemExit(f"missing {EXTRACT_PATH}; run with --refresh-source")
    payload = build()
    write_outputs(payload)
    sanity_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
