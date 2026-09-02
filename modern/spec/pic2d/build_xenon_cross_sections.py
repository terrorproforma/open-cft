"""Build ``xenon-cross-sections-v1.json`` (e-Xe cross sections for PIC-MCC).

Deterministic pipeline
----------------------
1. ``sources/lxcat_biagi-v7.1_xe_extract.txt`` is a byte-range extract (CRLF -> LF
   normalised) of the LXCat export ``lxcat/assortment.txt`` shipped in the
   ``lanl/ThunderBoltz`` repository (LXCat export generated 21 May 2023).  It
   contains the complete Biagi-v7.1 (Magboltz 7.1) e/Xe set: elastic momentum
   transfer, four lumped excitation levels and single ionisation.
   ``--refresh-source`` re-downloads the pinned upstream file, verifies its
   sha256 and rewrites the extract; the default build only reads the extract.
2. Each LXCat block is parsed and linearly interpolated in (E, sigma) (the
   LXCat/BOLSIG+ convention) onto a log-spaced grid.  Above the last tabulated
   energy (965-977 eV) a local power law through the last two tabulated points
   extends the table to exactly 1000 eV.
3. The four Biagi-v7.1 excitation levels are summed into one lumped channel with
   the 8.32 eV threshold used by Hall-thruster PIC codes.
4. Cross sections are rounded to 6 significant figures and energies to 10 so the
   JSON is reproducible across numpy builds.  The ``integrity.payload_sha256`` is
   the sha256 of the sort-keys/compact canonical JSON of everything except
   ``integrity``; the ``.sha256`` sidecar hashes the written file bytes.

Usage (from the repository root, PowerShell)::

    python modern/spec/pic2d/build_xenon_cross_sections.py            # build
    python modern/spec/pic2d/build_xenon_cross_sections.py --refresh-source

Requires Python 3.12 and numpy; no other dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES_DIR = HERE / "sources"
EXTRACT_PATH = SOURCES_DIR / "lxcat_biagi-v7.1_xe_extract.txt"
OUT_JSON = HERE / "xenon-cross-sections-v1.json"
OUT_SHA = HERE / "xenon-cross-sections-v1.json.sha256"

# --- pinned upstream ---------------------------------------------------------
UPSTREAM_COMMIT = "bdd3013da1954440ed68eec30611d6dad479b6b3"
UPSTREAM_URL_PINNED = (
    "https://raw.githubusercontent.com/lanl/ThunderBoltz/"
    f"{UPSTREAM_COMMIT}/lxcat/assortment.txt"
)
UPSTREAM_URL_BRANCH = "https://raw.githubusercontent.com/lanl/ThunderBoltz/main/lxcat/assortment.txt"
UPSTREAM_SHA256 = "7019dd0bdffe85c782684bddbc8433babd5e0a567166a05e3e1a6713006d1698"
UPSTREAM_BYTES = 1967511
UPSTREAM_RETRIEVED_UTC = "2026-09-02T15:35:58Z"
# sha256 of the LF-normalised extract written by --refresh-source (pinned after
# first generation; the build refuses to run on a modified extract).
EXTRACT_SHA256: str | None = "af8c3c1fb1003abda458ebbdd567c9030e9041089fb902db76f03caedb79f7a3"

# --- physics / grid constants ----------------------------------------------
E_MAX = 1000.0
THRESHOLD_EXCITATION = 8.32  # lumped Hall-thruster excitation threshold (eV)
THRESHOLD_IONIZATION = 12.13  # Xe -> Xe+ (Biagi-v7.1 uses 12.13 eV)
ELASTIC_GRID = (-2.0, 3.0, 121)  # log10 range and count: 0.01 .. 1000 eV, 24/decade
INELASTIC_EXCESS_GRID = (-2.0, 81)  # log10(E - E_th) from 0.01 eV, 16/decade, to E_MAX

DATABASE_HEADER = "DATABASE:         Biagi-v7.1"


# ---------------------------------------------------------------------------
# upstream handling
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cft-pic2d cross-section builder"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (pinned https URL)
        return resp.read()


def make_extract(upstream: bytes) -> bytes:
    """Cut the Biagi-v7.1 database section (header + all e/Xe blocks) out of the export."""
    if sha256_bytes(upstream) != UPSTREAM_SHA256:
        raise SystemExit("upstream sha256 mismatch; refusing to build extract from unexpected bytes")
    lines = upstream.decode("utf-8").split("\r\n")
    db_idx = next(i for i, l in enumerate(lines) if l.startswith(DATABASE_HEADER))
    start = db_idx - 1  # the 'xxxx' rule immediately above the DATABASE line
    assert set(lines[start]) == {"x"}, "unexpected LXCat section framing"
    nxt = next(i for i in range(db_idx + 1, len(lines)) if lines[i].startswith("DATABASE:"))
    end = nxt - 1  # 'xxxx' rule above the next DATABASE line (exclusive)
    assert set(lines[end]) == {"x"}
    header = [
        "EXTRACT of an LXCat export (not a verbatim upstream file).",
        f"Upstream file : {UPSTREAM_URL_PINNED}",
        f"  (same bytes served by {UPSTREAM_URL_BRANCH} on {UPSTREAM_RETRIEVED_UTC})",
        f"Upstream sha256: {UPSTREAM_SHA256}  ({UPSTREAM_BYTES} bytes, CRLF)",
        f"Retrieved (UTC): {UPSTREAM_RETRIEVED_UTC}",
        f"Extract        : upstream lines {start + 1}-{end} (1-based, inclusive), CRLF -> LF normalised, unchanged otherwise",
        "Upstream export header: '" + lines[0] + "' / '" + lines[1] + "'",
        "Reference (LXCat recommended format): Biagi-v7.1 database, www.lxcat.net, retrieved on May 21, 2023.",
        "LXCat data are redistributed with attribution as required by www.lxcat.net ('Redistribution of data').",
        "",
    ]
    body = lines[start:end]
    return ("\n".join(header + body) + "\n").encode("utf-8")


def refresh_source() -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    data = download(UPSTREAM_URL_PINNED)
    print(f"downloaded {len(data)} bytes sha256={sha256_bytes(data)}")
    extract = make_extract(data)
    EXTRACT_PATH.write_bytes(extract)
    print(f"wrote {EXTRACT_PATH.relative_to(HERE.parents[2])} ({len(extract)} bytes) sha256={sha256_bytes(extract)}")


# ---------------------------------------------------------------------------
# LXCat parsing
# ---------------------------------------------------------------------------
def parse_lxcat_blocks(text: str) -> list[dict]:
    """Return every ELASTIC/EXCITATION/IONIZATION block as dict(kind, name, param, comment, E, sigma)."""
    lines = text.split("\n")
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        kind = lines[i].strip()
        if kind in {"ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION"}:
            name = lines[i + 1].strip()
            j = i + 2
            param: float | None = None
            try:  # 3rd line = m/M (elastic) or threshold (inelastic); some databases omit it
                param = float(lines[j].split()[0])
                j += 1
            except ValueError:
                pass
            meta: dict[str, str] = {}
            while not lines[j].startswith("-----"):
                key, _, val = lines[j].partition(":")
                meta[key.strip()] = val.strip()
                j += 1
            j += 1
            e_vals: list[float] = []
            s_vals: list[float] = []
            while not lines[j].startswith("-----"):
                a, b = lines[j].split()
                e_vals.append(float(a))
                s_vals.append(float(b))
                j += 1
            blocks.append(
                dict(kind=kind, name=name, param=param, meta=meta,
                     E=np.asarray(e_vals), sigma=np.asarray(s_vals))
            )
            i = j
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------
def tail_exponent(E: np.ndarray, s: np.ndarray) -> float:
    """Local power-law exponent through the last two tabulated points."""
    return math.log(s[-1] / s[-2]) / math.log(E[-1] / E[-2])


def evaluate(block: dict, energies: np.ndarray) -> np.ndarray:
    """LXCat convention: linear interpolation in (E, sigma); 0 below the first
    tabulated energy; local power law above the last tabulated energy."""
    E, s = block["E"], block["sigma"]
    out = np.interp(energies, E, s, left=0.0, right=np.nan)
    above = energies > E[-1]
    if np.any(above):
        p = tail_exponent(E, s)
        out[above] = s[-1] * (energies[above] / E[-1]) ** p
    return out


def round_sig(values: np.ndarray, sig: int) -> list[float]:
    return [0.0 if v == 0 else float(f"{v:.{sig - 1}e}") for v in values]


def elastic_grid() -> np.ndarray:
    lo, hi, n = ELASTIC_GRID
    grid = np.logspace(lo, hi, n)
    grid[-1] = E_MAX
    return np.concatenate([[0.0], grid])


def inelastic_grid(threshold: float) -> np.ndarray:
    lo, n = INELASTIC_EXCESS_GRID
    excess = np.logspace(lo, math.log10(E_MAX - threshold), n)
    grid = threshold + excess
    grid[-1] = E_MAX
    return np.concatenate([[threshold], grid])


def check_table(energy: list[float], sigma: list[float]) -> None:
    e = np.asarray(energy)
    s = np.asarray(sigma)
    assert np.all(np.diff(e) > 0), "energies must be strictly increasing"
    assert np.all(np.isfinite(s)) and np.all(s >= 0), "cross sections must be finite and >= 0"
    assert len(e) >= 60


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build() -> dict:
    extract_bytes = EXTRACT_PATH.read_bytes()
    extract_sha = sha256_bytes(extract_bytes)
    if EXTRACT_SHA256 and extract_sha != EXTRACT_SHA256:
        raise SystemExit(f"extract sha256 {extract_sha} != pinned {EXTRACT_SHA256}")
    blocks = parse_lxcat_blocks(extract_bytes.decode("utf-8"))
    elastic = [b for b in blocks if b["kind"] == "ELASTIC"]
    excit = [b for b in blocks if b["kind"] == "EXCITATION"]
    ioniz = [b for b in blocks if b["kind"] == "IONIZATION"]
    assert len(elastic) == 1 and len(excit) == 4 and len(ioniz) == 1, "unexpected Biagi-v7.1 Xe block set"
    assert all(b["meta"].get("SPECIES") == "e / Xe" for b in blocks)
    assert abs(ioniz[0]["param"] - THRESHOLD_IONIZATION) < 1e-9
    assert abs(min(b["param"] for b in excit) - 8.315) < 1e-9
    source_file = EXTRACT_PATH.relative_to(HERE).as_posix()
    source_label = "LXCat Biagi-v7.1 (Magboltz 7.1) e/Xe set; LXCat export of 21 May 2023 mirrored in lanl/ThunderBoltz"

    def tail_note(bs: list[dict]) -> str:
        parts = [f"{b['name']}: last point {b['E'][-1]:.2f} eV, p={tail_exponent(b['E'], b['sigma']):+.3f}" for b in bs]
        return "; ".join(parts)

    # elastic -----------------------------------------------------------------
    e_el = elastic_grid()
    s_el = evaluate(elastic[0], e_el)
    energy_el = round_sig(e_el, 10)
    sigma_el = round_sig(s_el, 6)
    check_table(energy_el, sigma_el)

    # lumped excitation -------------------------------------------------------
    e_ex = inelastic_grid(THRESHOLD_EXCITATION)
    s_ex = np.zeros_like(e_ex)
    for b in excit:
        s_ex += evaluate(b, e_ex)
    s_ex[0] = 0.0
    energy_ex = round_sig(e_ex, 10)
    sigma_ex = round_sig(s_ex, 6)
    check_table(energy_ex, sigma_ex)

    # ionisation --------------------------------------------------------------
    e_io = inelastic_grid(THRESHOLD_IONIZATION)
    s_io = evaluate(ioniz[0], e_io)
    s_io[0] = 0.0
    energy_io = round_sig(e_io, 10)
    sigma_io = round_sig(s_io, 6)
    check_table(energy_io, sigma_io)

    tail_formula = (
        "Tabulated LXCat data, linear interpolation in (E, sigma). For E above the last tabulated "
        "energy E_N (only 965-977 eV < E <= 1000 eV): sigma(E) = sigma_N * (E/E_N)^p with "
        "p = ln(sigma_N/sigma_{N-1}) / ln(E_N/E_{N-1})."
    )

    processes = [
        {
            "id": "elastic",
            "kind": "elastic",
            "threshold_ev": 0.0,
            "energy_ev": energy_el,
            "cross_section_m2": sigma_el,
            "source": source_label,
            "source_bytes_sha256": extract_sha,
            "source_file": source_file,
            "formula": tail_formula,
            "notes": (
                "Elastic MOMENTUM-TRANSFER cross section (LXCat ELASTIC block, 'Elastic Momentum Transfer. "
                "Improved resolution at low energy, Oct 2011', m/M = 4.2e-6). Appropriate for isotropic-scattering "
                "MCC; it lies below the integral elastic cross section above ~10 eV because forward scattering is "
                "discounted. Grid: E=0 (tabulated value) plus 121 log-spaced points 0.01-1000 eV (24/decade). "
                f"Biagi-v7.1 table ends at {elastic[0]['E'][-1]:.2f} eV; tail extension {tail_note(elastic)}. "
                "Ramsauer-Townsend minimum of the momentum-transfer cross section: 2.75e-21 m2 at 0.62 eV (other "
                "LXCat momentum-transfer sets: 2.9-7.5e-21 m2 at 0.58-0.65 eV; the integral elastic minimum is "
                "~1.6e-20 m2 at 0.7 eV)."
            ),
        },
        {
            "id": "excitation",
            "kind": "excitation",
            "threshold_ev": THRESHOLD_EXCITATION,
            "energy_ev": energy_ex,
            "cross_section_m2": sigma_ex,
            "source": source_label,
            "source_bytes_sha256": extract_sha,
            "source_file": source_file,
            "formula": tail_formula + " Lumped channel: sigma_exc(E) = sum over the four Biagi-v7.1 excitation blocks "
            + ", ".join(b["name"] for b in excit) + " (each zero below its own threshold).",
            "notes": (
                "Single lumped excitation channel with the 8.32 eV energy loss used by Hall-thruster PIC codes "
                "(Szabo 2001 convention). It is the SUM of all Biagi-v7.1 effective excitation levels "
                "(thresholds 8.315, 9.447, 9.917, 11.7 eV), so the total excitation frequency is right while the "
                "per-event energy loss is under-estimated by up to 3.4 eV for the upper levels. Grid: exact "
                "threshold point (sigma=0) plus 81 points log-spaced in (E - 8.32 eV) from 0.01 eV to 1000 eV "
                f"(16/decade). Tail extension per level: {tail_note(excit)}."
            ),
        },
        {
            "id": "ionization",
            "kind": "ionization",
            "threshold_ev": THRESHOLD_IONIZATION,
            "energy_ev": energy_io,
            "cross_section_m2": sigma_io,
            "source": source_label,
            "source_bytes_sha256": extract_sha,
            "source_file": source_file,
            "formula": tail_formula,
            "notes": (
                "Single ionisation e + Xe -> Xe+ + 2e (LXCat IONIZATION block 'Xe -> Xe^+', threshold 12.13 eV; "
                "Magboltz 7.1 carries one lumped ionisation channel, so this is effectively the gross ionisation "
                "cross section: its 5.6e-20 m2 peak near 110 eV is ~10-15% above counting single-ionisation "
                "measurements such as Wetzel et al. 1987). Grid: exact threshold point (sigma=0) plus 81 points "
                "log-spaced in (E - 12.13 eV) from 0.01 eV to 1000 eV (16/decade). "
                f"Tail extension {tail_note(ioniz)}."
            ),
        },
    ]

    provenance = {
        "status": "lxcat-tabulated",
        "retrieved_utc": UPSTREAM_RETRIEVED_UTC,
        "network_attempts": NETWORK_ATTEMPTS,
        "citations": CITATIONS,
        "notes": (
            "All three processes come from one self-consistent, swarm-validated LXCat set (Biagi-v7.1, "
            "transcribed from Magboltz 7.1). LXCat itself only serves data through an interactive browser "
            "session, so the bytes were obtained from the LXCat export (generated 21 May 2023) committed to the "
            "open-source ThunderBoltz repository; the upstream file is 1.97 MB (> 300 kB policy limit), so only the "
            "Biagi-v7.1 e/Xe section is stored under sources/ with the upstream sha256 recorded here and in the "
            "extract header. The 3.5% power-law tail extension from 965-977 eV to 1000 eV is the only non-tabulated "
            "content. Cross-check against independent sets in the same export (values in m2): elastic momentum "
            "transfer at 10 / 100 eV Biagi-v7.1 1.86e-19 / 1.80e-20, Hayashi 1.70e-19 / 1.50e-20, SIGLO 2.00e-19 / "
            "2.40e-20; ionisation peak Biagi-v7.1 5.61e-20, Hayashi 5.88e-20, SIGLO and Morgan (Rapp & "
            "Englander-Golden 1965) 5.45e-20, Puech 4.96e-20; summed excitation peak Biagi-v7.1 2.95e-20 at 30 eV, "
            "SIGLO 3.00e-20 at 31 eV, Puech 1.97e-20 at 26 eV, Morgan 'Excitation total' 3.73e-20 at 20 eV, Biagi "
            "(v8.9, 33 levels) 1.35e-20 at 19 eV, Hayashi (6 levels, partial) 8.7e-21 at 35 eV - this factor-4 "
            "spread between databases is the dominant uncertainty of the lumped excitation channel. The WarpX MCC Xe "
            "tables ionization.dat and excitation_1.dat reproduce the Biagi-v7.1 ionisation and 8.315 eV blocks to 5 "
            "significant figures (independent confirmation of the transcription) but extend only to 750 eV."
        ),
    }

    payload = {
        "schema": "cft.pic2d.xenon-cross-sections.v1",
        "target": "Xe",
        "projectile": "e-",
        "units": {"energy": "eV", "cross_section": "m2"},
        "provenance": provenance,
        "processes": processes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8",
        "payload_sha256": sha256_bytes(canonical.encode("utf-8")),
    }
    return payload


NETWORK_ATTEMPTS = [
    {"url": "https://nl.lxcat.net/data/set_type.php", "utc": "2026-09-02T15:32:34Z",
     "outcome": "HTTP 200, interactive HTML session page only; LXCat text export requires a browser session, not scriptable"},
    {"url": "https://nl.lxcat.net/download/", "utc": "2026-09-02T15:34:59Z",
     "outcome": "HTTP 200, 'Download for offline use' landing page; links to /data/download.php (interactive form)"},
    {"url": "https://nl.lxcat.net/data/download.php", "utc": "2026-09-02T15:35:45Z",
     "outcome": "HTTP 200, HTML form requiring interactive database/species selection; no direct text URL"},
    {"url": "https://physics.nist.gov/cgi-bin/Ionization/table.pl?ionization=Xe", "utc": "2026-09-02T15:32:32Z",
     "outcome": "HTTP 200 'Please enter a molecule.'; NIST electron-impact ionisation (BEB) database has no Xe entry"},
    {"url": "https://physics.nist.gov/PhysRefData/Ionization/atom_index.html", "utc": "2026-09-02T15:35:00Z",
     "outcome": "HTTP 200; periodic-table index has no link for Xe (not covered)"},
    {"url": "https://api.github.com/repos/ECP-WarpX/warpx-data/contents/MCC_cross_sections/Xe", "utc": "2026-09-02T15:32:30Z",
     "outcome": "HTTP 200 (repo now BLAST-WarpX/warpx-data, branch master); lists electron_scattering.dat, excitation_1.dat, ionization.dat, ion files, README.md"},
    {"url": "https://raw.githubusercontent.com/BLAST-WarpX/warpx-data/master/MCC_cross_sections/Xe/README.md", "utc": "2026-09-02T15:32:56Z",
     "sha256": "1022835e125b39e150dc9cf70d6ee73e753c44af1938b920b0c35bb33e57ea80", "bytes": 642,
     "outcome": "HTTP 200; states excitation_1 and ionization derive from Magboltz 7.1, elastic from Zatsarinny & Bartschat (Allan et al. 2006)"},
    {"url": "https://raw.githubusercontent.com/BLAST-WarpX/warpx-data/master/MCC_cross_sections/Xe/electron_scattering.dat", "utc": "2026-09-02T15:32:56Z",
     "sha256": "10fd988ed96ef3e40f61a29b19d7ed6d02c0ee72b09a6cbb5d6ce30b34d8a6a7", "bytes": 1875000,
     "outcome": "HTTP 200; 75000-point 0.01 eV resample 0-750 eV (integral elastic, isotropic-scattering model); not used: > 300 kB, ends at 750 eV, constant above ~500 eV; cross-check only"},
    {"url": "https://raw.githubusercontent.com/BLAST-WarpX/warpx-data/master/MCC_cross_sections/Xe/excitation_1.dat", "utc": "2026-09-02T15:32:57Z",
     "sha256": "fb0ec3edf145d955145aba9366949dc56d4f15e32e1371f31278ad5c11a8362e", "bytes": 1854225,
     "outcome": "HTTP 200; single 8.315 eV level only (peak 6.34e-21 m2), not the lumped total; > 300 kB; cross-check only"},
    {"url": "https://raw.githubusercontent.com/BLAST-WarpX/warpx-data/master/MCC_cross_sections/Xe/ionization.dat", "utc": "2026-09-02T15:32:57Z",
     "sha256": "709855b8fa2c8915f5d0444e4db2a25be37188c973d43e5f4e4718e9ecb3af38", "bytes": 2213670,
     "outcome": "HTTP 200; Magboltz 7.1 ionisation resampled to 750 eV; > 300 kB; cross-check only (agrees with Biagi-v7.1 block used)"},
    {"url": "https://api.github.com/repos/IST-Lisbon/LoKI/contents/Code/Input", "utc": "2026-09-02T15:34:56Z",
     "outcome": "HTTP 200; LoKI-B ships Argon, CO, CO2, Helium, Nitrogen, Oxygen inputs only - no Xe"},
    {"url": "https://api.github.com/repos/PrincetonUniversity/EDIPIC-2D/contents", "utc": "2026-09-02T15:34:57Z",
     "outcome": "HTTP 200; no cross-section data directory exposed at repository root"},
    {"url": "https://api.github.com/repos/particleincell/Starfish/contents/dat", "utc": "2026-09-02T15:35:44Z",
     "outcome": "HTTP 200; no e-Xe cross-section tables (FILES.txt, examples, papers)"},
    {"url": "https://api.github.com/search/repositories?q=lxcat+xenon", "utc": "2026-09-02T15:35:00Z",
     "outcome": "HTTP 200; 0 repositories"},
    {"url": "https://api.github.com/repos/lanl/ThunderBoltz/contents/lxcat", "utc": "2026-09-02T15:35:43Z",
     "outcome": "HTTP 200; lists assortment.txt (1967511 B) and trinity_N2.txt"},
    {"url": UPSTREAM_URL_BRANCH, "utc": UPSTREAM_RETRIEVED_UTC, "sha256": UPSTREAM_SHA256, "bytes": UPSTREAM_BYTES,
     "outcome": "HTTP 200; LXCat export 'Generated on 21 May 2023' with e/Xe sets from Biagi, Biagi-v7.1, BSR, COP, Hayashi, Morgan, Puech, SIGLO, TRINITI. USED: Biagi-v7.1 e/Xe section extracted to sources/"},
    {"url": "https://api.github.com/repos/lanl/ThunderBoltz/commits?path=lxcat/assortment.txt&per_page=3", "utc": "2026-09-02T15:39:14Z",
     "outcome": f"HTTP 200; file last changed in commit {UPSTREAM_COMMIT} (2023-10-26); pinned URL {UPSTREAM_URL_PINNED} serves identical bytes"},
]

CITATIONS = [
    "Biagi-v7.1 database, www.lxcat.net, retrieved on May 21, 2023 (LXCat export shipped as lxcat/assortment.txt in "
    "github.com/lanl/ThunderBoltz, commit bdd3013da1954440ed68eec30611d6dad479b6b3).",
    "S. F. Biagi, Magboltz (Monte Carlo electron transport in gases), version 7.1, June 2004; "
    "http://consult.cern.ch/writeup/magboltz/ ('Cross sections extracted from PROGRAM MAGBOLTZ, VERSION 7.1 JUNE 2004').",
    "L. C. Pitchford et al., 'LXCat: an Open-Access, Web-Based Platform for Data Needed for Modeling Low Temperature "
    "Plasmas', Plasma Processes and Polymers 14, 1600098 (2017), doi:10.1002/ppap.201600098.",
    "M. C. Bordage, S. F. Biagi, L. L. Alves, K. Bartschat, S. Chowdhury, L. C. Pitchford, G. J. M. Hagelaar, "
    "W. L. Morgan, V. Puech, O. Zatsarinny, 'Comparisons of sets of electron-neutral scattering cross sections and "
    "swarm parameters in noble gases: III. Krypton and xenon', J. Phys. D: Appl. Phys. 46, 334003 (2013).",
    "ThunderBoltz (Los Alamos National Laboratory), https://github.com/lanl/ThunderBoltz - GPL-3.0 code whose "
    "lxcat/assortment.txt is a verbatim LXCat export; the LXCat data are redistributed under LXCat attribution terms.",
    "BLAST-WarpX/warpx-data, MCC_cross_sections/Xe (cross-check only): README cites Magboltz 7.1 for excitation/ionisation "
    "and O. Zatsarinny & K. Bartschat (M. Allan et al., Phys. Rev. A 74, 030701 (2006)) for elastic scattering.",
    "J. J. Szabo, 'Fully kinetic numerical modeling of a plasma thruster', PhD thesis, MIT (2001) - origin of the "
    "single 8.32 eV lumped Xe excitation channel convention used here.",
]


# ---------------------------------------------------------------------------
# output + sanity report
# ---------------------------------------------------------------------------
def write_outputs(payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    OUT_JSON.write_bytes(data)  # bytes => LF line endings on every platform
    file_sha = sha256_bytes(data)
    OUT_SHA.write_bytes(f"{file_sha}  {OUT_JSON.name}\n".encode("utf-8"))
    print(f"wrote {OUT_JSON.name}: {len(data)} bytes")
    print(f"file sha256    : {file_sha}")
    print(f"payload sha256 : {payload['integrity']['payload_sha256']}")


def sanity_report(payload: dict) -> None:
    probe = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0]
    tables = {p["id"]: (np.asarray(p["energy_ev"]), np.asarray(p["cross_section_m2"])) for p in payload["processes"]}
    print("\nsigma(E) [m2]     " + "".join(f"{e:>11.0f} eV" for e in probe))
    for pid, (e, s) in tables.items():
        vals = np.interp(probe, e, s, left=0.0)
        print(f"{pid:18s}" + "".join(f"{v:14.3e}" for v in vals))

    e, s = tables["elastic"]
    win = (e >= 0.3) & (e <= 1.5)
    ramsauer_e = e[win][np.argmin(s[win])]
    ramsauer_s = s[win].min()
    e10 = np.interp(10.0, e, s)
    # Momentum-transfer minimum: 2.7-7.5e-21 m2 across LXCat sets (Biagi, BSR, COP, Hayashi, SIGLO, TRINITI);
    # the integral elastic minimum is ~1.6e-20 m2 (BSR, COP totals), hence the wider band.
    ok_el = 0.4 <= ramsauer_e <= 1.0 and 1.5e-21 <= ramsauer_s <= 3e-20 and 1e-19 <= e10 <= 5e-19
    print(f"\nelastic   : Ramsauer minimum {ramsauer_s:.3e} m2 at {ramsauer_e:.3f} eV; sigma(10 eV)={e10:.3e}  {'OK' if ok_el else 'CHECK'}")
    e, s = tables["excitation"]
    ok_ex = 15 <= e[np.argmax(s)] <= 50 and 8e-21 <= s.max() <= 4e-20
    print(f"excitation: peak {s.max():.3e} m2 at {e[np.argmax(s)]:.2f} eV  {'OK' if ok_ex else 'CHECK'}")
    e, s = tables["ionization"]
    ok_io = 60 <= e[np.argmax(s)] <= 150 and 3.5e-20 <= s.max() <= 6.5e-20
    print(f"ionization: peak {s.max():.3e} m2 at {e[np.argmax(s)]:.2f} eV  {'OK' if ok_io else 'CHECK'}")
    if not (ok_el and ok_ex and ok_io):
        raise SystemExit("physics sanity check failed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-source", action="store_true",
                    help="re-download the pinned upstream LXCat export, verify sha256 and rewrite the extract")
    args = ap.parse_args(argv)
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
