"""Tests for the offline roadmap status dashboard (`modern/visualization/roadmap-status.html`).

The dashboard is the Cursor canvas `open-cft-roadmap-status.canvas.tsx` bundled with React into one
self-contained HTML file by `visualization/roadmap-status/build.mjs`. These checks read the canvas
copy and the checked-in HTML independently of the build and assert that

* the copy is the canvas (verbatim modulo line endings): one import, from ``cursor/canvas`` only,
  default export, LF, no BOM;
* every one of the 44 ladder-row ids and every row name reached the HTML, and a Python recount of
  the ladder (same rule as the canvas: highest rung without a gap, merged state) gives the pinned
  header chips;
* the HTML is self-contained (one inline script, no external script / stylesheet / network call;
  the only URLs are the canvas's github.com anchors and React DOM's namespace / error-decoder
  constants), UTF-8 without BOM, LF only, within a size band, and its sidecar pins its sha256;
* with Node available: the inline script parses (``node --check``); with the pinned toolchain
  installed (``npm ci`` in ``visualization/roadmap-status``, git-ignored): a rebuild is byte-exact
  and the jsdom run (`verify.mjs`) renders the same chips, all nine tabs and the 44 rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
VISUALIZATION = MODERN / "visualization"
BUILD_DIR = VISUALIZATION / "roadmap-status"
CANVAS_COPY = BUILD_DIR / "roadmap-status.canvas.tsx"
SHIM = BUILD_DIR / "cursor-canvas.tsx"
CHECKED_HTML = VISUALIZATION / "roadmap-status.html"
SIDECAR = VISUALIZATION / "roadmap-status.anchor-platform.json"
NODE_MODULES = BUILD_DIR / "node_modules"

ROW_COUNT = 44
# Flipped to True by the commit that records origin/main's fast-forward onto feat/sota-foundation
# (mergeTruth.mainMergedAt = mergeTruth.mainHead in the canvas); the fourth chip follows it.
MAIN_FAST_FORWARDED = True
EXPECTED_CHIPS = (
    "0/44 externally validated",
    "17 in the paper",
    "36/44 merged",
    "36/44 on main" if MAIN_FAST_FORWARDED else "0/44 on main",
)
TABS = (
    "Overview",
    "Phases",
    "Stage ladder",
    "Experiments",
    "Critical path",
    "Literature roadmap",
    "Actions",
    "Evidence",
    "Details",
)
ALLOWED_URL_PREFIXES = ("https://github.com/", "http://www.w3.org/", "https://react.dev/errors/")
MIN_HTML_BYTES = 1_500_000
MAX_HTML_BYTES = 6_000_000


@pytest.fixture(scope="module")
def canvas_text() -> str:
    return CANVAS_COPY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html_bytes() -> bytes:
    return CHECKED_HTML.read_bytes()


@pytest.fixture(scope="module")
def html(html_bytes: bytes) -> str:
    return html_bytes.decode("utf-8")


def _ladder_block(canvas_text: str) -> str:
    start = canvas_text.index("const ladderRows: LadderRow[] = [")
    end = canvas_text.index("const mergeTruth = {", start)
    return canvas_text[start:end]


def _ladder_rows(canvas_text: str) -> list[dict[str, object]]:
    """Rows as (id, name, cell states, merged state), parsed from the canvas source layout.

    Each row is a `{ ... }` literal at four-space indentation; the eight cells are the
    `ok(` / `cav(` / `no(` calls at six spaces inside its `cells: [` array.
    """
    block = _ladder_block(canvas_text)
    rows: list[dict[str, object]] = []
    for chunk in re.split(r"\n  \{\n", block)[1:]:
        row_id = re.search(r'^    id: "([^"]+)",$', chunk, re.MULTILINE)
        name = re.search(r'^    name: "((?:[^"\\]|\\.)*)",$', chunk, re.MULTILINE)
        merged = re.search(r'^    merged: \{ state: "([a-z-]+)"', chunk, re.MULTILINE)
        cells_start = chunk.index("    cells: [\n")
        cells_end = chunk.index("\n    ],\n", cells_start)
        cells = re.findall(r"^      (ok|cav|no)\(", chunk[cells_start:cells_end], re.MULTILINE)
        assert row_id and name and merged, chunk[:200]
        assert len(cells) == 8, (row_id.group(1), cells)
        rows.append(
            {
                "id": row_id.group(1),
                "name": name.group(1),
                "cells": cells,
                "merged": merged.group(1),
            }
        )
    return rows


def _highest_stage(cells: list[str]) -> int:
    highest = 0
    for state in cells:
        if state == "no":
            break
        highest += 1
    return highest


def _merge_truth(canvas_text: str) -> dict[str, str]:
    start = canvas_text.index("const mergeTruth = {")
    end = canvas_text.index("\n};\n", start)
    body = canvas_text[start:end]
    fields = {}
    for key in ("featureHead", "mainHead", "mainMergedAt"):
        match = re.search(rf'^  {key}: "([0-9a-f]*)",$', body, re.MULTILINE)
        assert match is not None, key
        fields[key] = match.group(1)
    return fields


def _inline_script(html: str) -> str:
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", html, re.DOTALL)
    assert len(scripts) == 1
    return scripts[0]


def test_canvas_copy_is_the_canvas(canvas_text: str) -> None:
    raw = CANVAS_COPY.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    imports = re.findall(
        r'^import (?:type )?\{[^}]*\} from "([^"]+)";$', canvas_text, re.MULTILINE | re.DOTALL
    )
    assert imports and set(imports) == {"cursor/canvas"}
    assert re.search(r"^import ", canvas_text, re.MULTILINE) is not None
    assert all(
        source == "cursor/canvas"
        for source in re.findall(
            r'^import [^;]*from "([^"]+)";', canvas_text, re.MULTILINE | re.DOTALL
        )
    )
    assert "export default function OpenCftRoadmapStatus()" in canvas_text
    for tab in TABS:
        assert f'"{tab}"' in canvas_text
    assert "fetch(" not in canvas_text


def test_ladder_rows_recount_gives_the_pinned_chips(canvas_text: str) -> None:
    rows = _ladder_rows(canvas_text)
    ids = [row["id"] for row in rows]
    assert len(ids) == ROW_COUNT and len(set(ids)) == ROW_COUNT
    external = sum(1 for row in rows if _highest_stage(row["cells"]) == 8)
    in_paper = sum(1 for row in rows if _highest_stage(row["cells"]) == 7)
    merged_feature = sum(1 for row in rows if row["merged"] == "feature")
    truth = _merge_truth(canvas_text)
    on_main = (
        merged_feature
        if truth["mainMergedAt"] and truth["mainMergedAt"] == truth["mainHead"]
        else 0
    )
    chips = (
        f"{external}/{len(rows)} externally validated",
        f"{in_paper} in the paper",
        f"{merged_feature}/{len(rows)} merged",
        f"{on_main}/{len(rows)} on main",
    )
    assert chips == EXPECTED_CHIPS
    if MAIN_FAST_FORWARDED:
        assert len(truth["mainMergedAt"]) == 8 and truth["mainMergedAt"] == truth["mainHead"]
    else:
        assert truth["mainMergedAt"] == ""


def _decoded(literal: str) -> str:
    """A TSX double-quoted string body -> its value (the escapes used in the canvas are JSON's)."""
    return json.loads(f'"{literal}"')


def _plain_prefix(value: str, length: int = 48) -> str:
    """The longest prefix (<= length) free of characters esbuild might re-escape."""
    cut = len(value)
    for char in ('"', "'", "\\", "`", "$"):
        index = value.find(char)
        if index != -1:
            cut = min(cut, index)
    return value[: min(cut, length)]


def test_every_ladder_row_reached_the_html(canvas_text: str, html: str) -> None:
    script = _inline_script(html)
    for row in _ladder_rows(canvas_text):
        row_id = row["id"]
        assert re.search(rf'\bid:\s*"{re.escape(row_id)}"', script) is not None, row_id
        prefix = _plain_prefix(_decoded(row["name"]))
        assert len(prefix) >= 8 and prefix in script, (row_id, prefix)
    assert "externally validated" in script and "in the paper" in script and "on main" in script


def test_html_is_self_contained_and_clean(html_bytes: bytes, html: str) -> None:
    assert not html_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in html_bytes
    assert MIN_HTML_BYTES <= len(html_bytes) <= MAX_HTML_BYTES, len(html_bytes)
    assert html.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in html
    assert "background: #181818" in html and "color-scheme: dark" in html
    assert re.search(r"<script[^>]*\ssrc=", html, re.IGNORECASE) is None
    assert re.search(r"<link[^>]*\shref=", html, re.IGNORECASE) is None
    assert "@import" not in html.split("<script>")[0]
    assert re.search(r"\bfetch\(|XMLHttpRequest|navigator\.sendBeacon", html) is None
    urls = re.findall(r"https?://[^\s\"'`<>)]+", html)
    foreign = sorted({url for url in urls if not url.startswith(ALLOWED_URL_PREFIXES)})
    assert foreign == [], foreign
    assert any(url.startswith("https://github.com/") for url in urls), (
        "the canvas's commit anchors are part of the content"
    )
    script = _inline_script(html)
    assert "</script" not in script
    assert "<!--" not in script
    assert 'getElementById("root")' in script


def test_sidecar_pins_the_checked_in_html(html_bytes: bytes) -> None:
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    assert sidecar["schema"] == "cft-roadmap-status-dashboard-anchor/1.0.0"
    assert sidecar["html_file"] == CHECKED_HTML.name
    assert sidecar["html_sha256"] == hashlib.sha256(html_bytes).hexdigest()
    assert sidecar["html_bytes"] == len(html_bytes)
    assert sidecar["canvas_sha256"] == hashlib.sha256(CANVAS_COPY.read_bytes()).hexdigest()
    assert sidecar["canvas_bytes"] == CANVAS_COPY.stat().st_size
    assert sidecar["shim_sha256"] == hashlib.sha256(SHIM.read_bytes()).hexdigest()
    package = json.loads((BUILD_DIR / "package.json").read_text(encoding="utf-8"))
    for name in ("esbuild", "react", "react-dom"):
        assert sidecar["toolchain"][name] == package["devDependencies"][name]
    assert "byte-deterministic" in sidecar["policy"]
    lock = json.loads((BUILD_DIR / "package-lock.json").read_text(encoding="utf-8"))
    assert (
        lock["packages"]["node_modules/esbuild"]["version"] == package["devDependencies"]["esbuild"]
    )


def test_head_comment_carries_the_input_hashes(html: str) -> None:
    head = html.split("<script>")[0]
    assert f"canvas sha256 {hashlib.sha256(CANVAS_COPY.read_bytes()).hexdigest()}" in head
    assert f"shim sha256 {hashlib.sha256(SHIM.read_bytes()).hexdigest()}" in head
    assert "Generated by modern/visualization/roadmap-status/build.mjs" in head


def test_inline_script_parses_when_node_is_available(html: str, tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable for JavaScript syntax checking")
    path = tmp_path / "roadmap-status.js"
    path.write_text(_inline_script(html), encoding="utf-8", newline="\n")
    completed = subprocess.run(
        [node, "--check", str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def _toolchain_available(*packages: str) -> str | None:
    node = shutil.which("node")
    if node is None:
        return None
    if not all((NODE_MODULES / package / "package.json").is_file() for package in packages):
        return None
    return node


def test_rebuild_is_byte_exact_when_the_toolchain_is_installed(
    html_bytes: bytes, tmp_path: Path
) -> None:
    node = _toolchain_available("esbuild", "react", "react-dom")
    if node is None:
        pytest.skip("pinned toolchain not installed (npm ci in visualization/roadmap-status)")
    out = tmp_path / "roadmap-status.html"
    completed = subprocess.run(
        [node, str(BUILD_DIR / "build.mjs"), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
        cwd=BUILD_DIR,
    )
    assert completed.returncode == 0, completed.stderr
    assert out.read_bytes() == html_bytes
    rebuilt_sidecar = json.loads(
        (tmp_path / "roadmap-status.anchor-platform.json").read_text(encoding="utf-8")
    )
    checked_sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    assert rebuilt_sidecar == checked_sidecar


def test_headless_dom_renders_chips_tabs_and_rows(canvas_text: str, tmp_path: Path) -> None:
    node = _toolchain_available("jsdom")
    if node is None:
        pytest.skip("jsdom not installed (npm ci in visualization/roadmap-status)")
    report_path = tmp_path / "verify.json"
    completed = subprocess.run(
        [
            node,
            str(BUILD_DIR / "verify.mjs"),
            "--html",
            str(CHECKED_HTML),
            "--json",
            str(report_path),
            "--expect-rows",
            str(ROW_COUNT),
            "--expect-chips",
            "|".join(EXPECTED_CHIPS),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=BUILD_DIR,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True and report["failures"] == []
    assert tuple(report["chips"]) == EXPECTED_CHIPS
    assert set(report["tabs"]) == set(TABS)
    assert report["ladder"]["rows"] == ROW_COUNT and report["ladder"]["firstRowExpands"] is True
    rendered = report["ladder"]["rowNames"]
    expected_names = [_decoded(row["name"]).split(" · ")[0] for row in _ladder_rows(canvas_text)]
    assert rendered == expected_names
    assert report["external"] == {"scripts_with_src": 0, "stylesheets": 0, "remote_images": 0}
    assert report["errors"] == []
