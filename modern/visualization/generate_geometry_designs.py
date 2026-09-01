"""Generate the standalone accepted-geometry v1.1 design viewer.

The accepted geometry bundle loader is the only artifact ingestion path. It
verifies the manifest, every sidecar, canonical geometry JSON, viewer
projection, generated SVG, descriptors, schemas, and directory closure before
any data is embedded. No timestamps are emitted, so generation is byte stable.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from typing import Any, Mapping

from cft_revival.geometry import (
    ARTIFACT_CLAIM_LIMIT,
    compute_descriptors,
    load_artifact_bundle,
    viewer_data,
)

MODERN = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIRECTORY = MODERN / "examples" / "geometry" / "artifacts"
DEFAULT_OUTPUT = Path(__file__).with_name("geometry-designs.html")

EXPECTED_MANIFEST_FILE_SHA256 = (
    "a327b56fcdb0dfcf4e0043fe2287692514423e66af001eceab9e805f8ce799f8"
)
EXPECTED_CONFIGURATIONS = (
    (
        "historical-envelope-baseline-v1",
        "Historical envelope",
        "2524848ff176d7a42b93286c3c6d27c99ce3ccdcf0084d30c0c9ffbfd38a4227",
        "c50eb903b27bf940f50364d1d9eaec93096193821714dadc2db06938d3916445",
        "d4a325d82f30ee04ee50b88c94d2a96586f97d7fac63543ffabd6e8e8be58673",
    ),
    (
        "compact-high-gradient-stack-v1",
        "Compact high-gradient",
        "66e033e98aa2dc98af96d4d7072218c4b298839c784a44ee66621e29a58a4f6f",
        "663c2210b93a39e0b164347637233b20a38c4f622c34f486ae88c56f8400ced1",
        "45311a3b669cbc3ac493502440995585f87c84048f3718ffe3a6cc8b76b15578",
    ),
    (
        "divergent-exit-stack-v1",
        "Divergent exit",
        "9b7ec5608d02754dcf7605d0389915efddfe4202f5bc466feb4652a46d004217",
        "ca58824cf6ba235f515a791b179e3d3c5c5e285a113a1e1e55492cb80d52a4e3",
        "4de4410c36db9dec317258a63d450373117cf582c0155b81c0b76da0b43afb38",
    ),
)

ROLE_COLORS = {
    "anode": "#f59e0b",
    "injector_plasma": "#7dd3fc",
    "channel_plasma": "#38bdf8",
    "dielectric_wall": "#e8d7a2",
    "permanent_magnet": "#e05263",
    "pole_piece": "#64748b",
    "shield": "#a7b0bc",
    "yoke": "#374151",
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _artifact_filename(config_id: str, suffix: str) -> str:
    return f"{config_id.removesuffix('-v1')}{suffix}"


def _build_design(
    geometry: Any,
    manifest_entry: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Project a verified geometry object into display data."""

    projected = viewer_data(geometry)
    descriptors = compute_descriptors(geometry).to_dict()
    model = geometry.to_dict()
    region_models = {region.region_id: region for region in geometry.regions}
    regions: list[dict[str, Any]] = []
    for projected_region in projected["regions"]:
        region_id = projected_region["region_id"]
        region = region_models[region_id]
        regions.append(
            {
                **projected_region,
                "owner_id": region.owner_id,
                "z_min_m": region.z_min_m,
                "z_max_m": region.z_max_m,
                "r_inner_start_m": region.r_inner_start_m,
                "r_inner_end_m": region.r_inner_end_m,
                "r_outer_start_m": region.r_outer_start_m,
                "r_outer_end_m": region.r_outer_end_m,
                "volume_m3": region.volume_m3,
            }
        )
    magnet_regions = [
        region_models[stage.magnet_region_id] for stage in geometry.stages
    ]
    magnet_volume_m3 = fsum(region.volume_m3 for region in magnet_regions)
    file_hashes = manifest_entry["artifact_file_sha256"]
    geometry_filename = _artifact_filename(geometry.config_id, ".json")
    viewer_filename = _artifact_filename(geometry.config_id, ".viewer.json")
    svg_filename = _artifact_filename(geometry.config_id, ".svg")
    return {
        "id": geometry.config_id,
        "label": label,
        "title": geometry.title,
        "classification": geometry.classification,
        "coordinate_system": geometry.coordinate_system,
        "length_unit": geometry.length_unit,
        "identity": {
            "geometry_payload_sha256": geometry.canonical_sha256,
            "geometry_file": geometry_filename,
            "geometry_file_sha256": file_hashes[geometry_filename],
            "viewer_file": viewer_filename,
            "viewer_file_sha256": file_hashes[viewer_filename],
            "svg_file": svg_filename,
            "svg_file_sha256": file_hashes[svg_filename],
            "viewer_schema_version": projected["schema_version"],
            "geometry_schema_version": model["schema_version"],
        },
        "chamber": model["chamber"],
        "electrodes": model["electrodes"],
        "manufacturing": model["manufacturing"],
        "materials": model["materials"],
        "regions": regions,
        "interfaces": projected["interfaces"],
        "stages": [
            {
                **annotation,
                "magnetization": model_stage["magnetization"],
                "magnet_region_id": model_stage["magnet_region_id"],
                "pole_after_region_id": model_stage["pole_after_region_id"],
            }
            for annotation, model_stage in zip(
                projected["stage_annotations"], model["stages"], strict=True
            )
        ],
        "external_components": projected["external_components"],
        "permanent_magnet_plan": projected["permanent_magnet_plan"],
        "evidence": model["evidence"],
        "descriptors": {
            **descriptors,
            "magnet_volume_m3": magnet_volume_m3,
        },
        "number_sources": {
            "chamber_and_regions": (
                f"{geometry_filename}: chamber, regions, stages, materials, manufacturing"
            ),
            "interfaces": f"{viewer_filename}: interfaces",
            "descriptors": (
                "manifest.json: configurations[].descriptors "
                "(recomputed and equality-checked by load_artifact_bundle)"
            ),
            "magnet_volume_m3": (
                f"{geometry_filename}: sum(volume_m3 of stages[].magnet_region_id)"
            ),
        },
        "claim_limit": projected["claim_limit"],
    }


def build_payload(
    artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> dict[str, Any]:
    """Strictly load the accepted bundle and build deterministic display data."""

    directory = artifact_directory.resolve()
    bundle = load_artifact_bundle(directory)
    manifest_digest = _sha256(directory / "manifest.json")
    if manifest_digest != EXPECTED_MANIFEST_FILE_SHA256:
        raise ValueError("geometry manifest identity is invalid or superseded")
    entries = bundle.manifest["configurations"]
    actual_ids = [geometry.config_id for geometry in bundle.geometries]
    expected_ids = [item[0] for item in EXPECTED_CONFIGURATIONS]
    if actual_ids != expected_ids or [entry["config_id"] for entry in entries] != expected_ids:
        raise ValueError("accepted geometry configuration identity/order changed")

    designs: list[dict[str, Any]] = []
    for geometry, entry, expected in zip(
        bundle.geometries, entries, EXPECTED_CONFIGURATIONS, strict=True
    ):
        config_id, label, payload_hash, geometry_hash, viewer_hash = expected
        hashes = entry["artifact_file_sha256"]
        if (
            geometry.config_id != config_id
            or geometry.canonical_sha256 != payload_hash
            or hashes[_artifact_filename(config_id, ".json")] != geometry_hash
            or hashes[_artifact_filename(config_id, ".viewer.json")] != viewer_hash
        ):
            raise ValueError(f"{config_id} reviewed artifact identity is invalid")
        designs.append(_build_design(geometry, entry, label))

    payload = {
        "schema": "cft-geometry-design-viewer/1.1.0",
        "manifest": {
            "file": "manifest.json",
            "file_sha256": manifest_digest,
            "schema_version": bundle.manifest["schema_version"],
            "generator": bundle.manifest["generator"],
            "generator_version": bundle.manifest["generator_version"],
            "claim_limit": bundle.manifest["claim_limit"],
        },
        "warning": ARTIFACT_CLAIM_LIMIT,
        "physics_boundary": (
            "The hardware inspiration is TWT periodic permanent-magnet focusing: "
            "annular magnets alternate axial polarity and pole pieces shape a periodic "
            "magnetic stack. CFT operation is distinct plasma-discharge and charged-"
            "particle transport physics; this geometry does not import TWT RF slow-wave "
            "electron-beam amplification physics and predicts no CFT performance."
        ),
        "designs": designs,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Reject incomplete or identity-substituted visualization payloads."""

    if set(payload) != {
        "schema",
        "manifest",
        "warning",
        "physics_boundary",
        "designs",
    }:
        raise ValueError("visualization payload keys differ from the closed schema")
    if payload["schema"] != "cft-geometry-design-viewer/1.1.0":
        raise ValueError("visualization payload schema is unsupported")
    manifest = payload["manifest"]
    if (
        not isinstance(manifest, Mapping)
        or set(manifest)
        != {
            "file",
            "file_sha256",
            "schema_version",
            "generator",
            "generator_version",
            "claim_limit",
        }
        or manifest["file_sha256"] != EXPECTED_MANIFEST_FILE_SHA256
        or manifest["schema_version"]
        != "cft_revival.geometry.artifact_manifest/1.1.0"
        or manifest["claim_limit"] != ARTIFACT_CLAIM_LIMIT
    ):
        raise ValueError("embedded geometry manifest identity is invalid")
    designs = payload["designs"]
    if not isinstance(designs, list) or len(designs) != 3:
        raise ValueError("visualization requires exactly three accepted designs")
    for design, expected in zip(designs, EXPECTED_CONFIGURATIONS, strict=True):
        if not isinstance(design, Mapping) or design.get("id") != expected[0]:
            raise ValueError("embedded design identity/order is invalid")
        identity = design.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("geometry_payload_sha256") != expected[2]
            or identity.get("geometry_file_sha256") != expected[3]
            or identity.get("viewer_file_sha256") != expected[4]
            or identity.get("viewer_schema_version")
            != "cft_revival.geometry.viewer_data/1.1.0"
            or identity.get("geometry_schema_version")
            != "cft_revival.geometry.axisymmetric_cft/1.1.0"
        ):
            raise ValueError(f"{expected[0]} embedded artifact identity is invalid")
        regions = design.get("regions")
        stages = design.get("stages")
        interfaces = design.get("interfaces")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"{expected[0]} must include regions")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{expected[0]} must include stages")
        if not isinstance(interfaces, list) or not interfaces:
            raise ValueError(f"{expected[0]} must include interfaces")
        polarities = [stage.get("polarity") for stage in stages]
        if polarities != [1 if index % 2 == 0 else -1 for index in range(len(stages))]:
            raise ValueError(f"{expected[0]} stage polarity is not alternating")
        descriptor = design.get("descriptors")
        if not isinstance(descriptor, Mapping) or any(
            descriptor.get(key, 0) <= 0
            for key in (
                "active_volume_m3",
                "channel_volume_m3",
                "magnet_volume_m3",
                "magnet_mass_estimate_kg",
                "minimum_radial_gap_m",
                "minimum_axial_gap_m",
            )
        ):
            raise ValueError(f"{expected[0]} descriptors are incomplete")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Accepted Geometry v1.1 Viewer</title>
<style>
:root{color-scheme:dark;--bg:#071018;--panel:#101b26;--panel2:#142332;--text:#edf6fc;--muted:#9fb3c2;--line:#304657;--accent:#4fd1c5;--warn:#ffd166;--shadow:#0008;--axis:#bed0dc}
[data-theme=light]{color-scheme:light;--bg:#eef4f7;--panel:#fff;--panel2:#f4f8fa;--text:#10212c;--muted:#536a78;--line:#bdcfd8;--accent:#087f72;--warn:#715000;--shadow:#3452;--axis:#314854}
@media(prefers-color-scheme:light){:root:not([data-theme]){color-scheme:light;--bg:#eef4f7;--panel:#fff;--panel2:#f4f8fa;--text:#10212c;--muted:#536a78;--line:#bdcfd8;--accent:#087f72;--warn:#715000;--shadow:#3452;--axis:#314854}}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#15344b 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header,main,footer{width:min(1540px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:800;letter-spacing:.13em;text-transform:uppercase}h1{font-size:clamp(2rem,5vw,4.3rem);line-height:.98;margin:.15rem 0 .8rem}h2{font-size:1.1rem;margin:.1rem 0 .7rem}h3{font-size:.93rem;margin:.7rem 0 .35rem}.warning{padding:.75rem 1rem;border:1px solid #8c6b1e;border-radius:.7rem;background:#8c6b1e22;color:var(--warn);font-weight:700}.physics{border-left:4px solid var(--accent);padding:.65rem .85rem;background:#4fd1c511}
button,select,input{font:inherit}button,select{color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible,canvas:focus-visible{outline:3px solid var(--accent);outline-offset:2px}.controls{display:flex;gap:.65rem;align-items:end;flex-wrap:wrap;margin:1rem 0}.control{display:grid;gap:.2rem}.control label,.small{font-size:.81rem;color:var(--muted)}.check{display:flex;align-items:center;gap:.35rem;padding:.45rem}
.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 28px var(--shadow);min-width:0}.viewer-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(310px,.78fr);gap:1rem}.canvases{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}.canvases.overlay{grid-template-columns:1fr}.canvases.overlay .secondary{display:none}.drawing{min-width:0}.drawing canvas{display:block;width:100%;height:clamp(390px,48vw,650px);border:1px solid var(--line);border-radius:.55rem;background:var(--panel)}.drawing h3{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:1rem 0}.metric{padding:.8rem;border:1px solid var(--line);border-radius:.7rem;background:var(--panel)}.metric.active{border-color:var(--accent)}.kv{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:.3rem .7rem}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}.source{grid-column:1/-1;font-size:.72rem;color:var(--muted);overflow-wrap:anywhere}
.legend{display:flex;gap:.4rem;flex-wrap:wrap;margin:.8rem 0}.legend button{padding:.25rem .5rem}.swatch{display:inline-block;width:.75rem;height:.75rem;border-radius:.15rem;margin-right:.3rem;vertical-align:-.05rem;border:1px solid #0005}.details{max-height:650px;overflow:auto;padding-right:.25rem}.details code,.identity code{font-size:.76rem;overflow-wrap:anywhere}.detail-block{border-top:1px solid var(--line);padding-top:.55rem;margin-top:.55rem}.interface{padding:.45rem;border-left:3px solid var(--accent);margin:.45rem 0;background:#4fd1c50c}.warning-list{color:var(--warn)}.identity{margin:1rem 0}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.08rem .45rem;color:var(--muted);margin-right:.3rem}footer{padding:1.3rem 0 2.5rem;color:var(--muted)}
@media(max-width:980px){.viewer-grid{grid-template-columns:1fr}.details{max-height:none}.drawing canvas{height:520px}}@media(max-width:720px){.metrics,.canvases{grid-template-columns:1fr}.drawing canvas{height:440px}.canvases.overlay .secondary{display:none}}@media(max-width:480px){header,main,footer{width:calc(100% - 1rem)}.panel{padding:.65rem}.drawing canvas{height:390px}}
</style>
</head>
<body>
<header>
<div class="eyebrow">Verified artifact viewer · geometry v1.1</div>
<h1>Periodic PM geometry, inspected</h1>
<p id="warning" class="warning"></p>
<p id="physics" class="physics"></p>
<div class="controls" aria-label="Viewer controls">
<div class="control"><label for="primary">Primary configuration</label><select id="primary"></select></div>
<div class="control"><label for="secondary">Comparison configuration</label><select id="secondary"></select></div>
<div class="control"><label for="mode">Compare mode</label><select id="mode"><option value="side">Side by side</option><option value="overlay">Overlay</option></select></div>
<label class="check"><input id="dimensions" type="checkbox" checked> Dimensions</label>
<button id="reset" type="button" aria-keyshortcuts="Escape">Reset view</button>
<button id="theme" type="button" aria-pressed="false">Light theme</button>
</div>
<p class="small">Keyboard: 1–3 choose the primary configuration; arrow keys move region selection; Enter switches between primary and comparison; Escape resets the view. Mouse or touch selects a material region.</p>
</header>
<main>
<section id="metrics" class="metrics" aria-label="Configuration metrics"></section>
<section class="viewer-grid">
<div class="panel">
<h2>Actual axisymmetric r–z material cross-sections</h2>
<div id="canvases" class="canvases">
<div class="drawing"><h3 id="primaryTitle"></h3><canvas id="primaryCanvas" tabindex="0" role="img" aria-label="Primary geometry material cross-section"></canvas></div>
<div class="drawing secondary"><h3 id="secondaryTitle"></h3><canvas id="secondaryCanvas" tabindex="0" role="img" aria-label="Comparison geometry material cross-section"></canvas></div>
</div>
<div id="legend" class="legend" aria-label="Material role legend"></div>
<p class="small">Both ±r meridional halves are drawn from each region’s accepted polygon. Overlay uses shared physical axes: primary is solid; comparison is hatched. Dashed bounds are the accepted envelope. Polarity labels are +z/−z.</p>
</div>
<aside class="panel"><h2>Selection details</h2><div id="details" class="details" aria-live="polite"></div></aside>
</section>
<section class="panel identity"><h2>Artifact identity and numeric traceability</h2><div id="identity"></div></section>
</main>
<footer>Self-contained, zero-network artifact viewer. Geometry only—no field or propulsion-performance result.</footer>
<script id="geometry-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("geometry-data").textContent);
const $=id=>document.getElementById(id);
const COLORS=__COLORS__;
const INITIAL_VIEW_STATE=Object.freeze({primary:0,secondary:1,mode:"side",showDimensions:true,selectionDesign:0,selectionRegion:0,hover:null,cursor:null,zoom:1,panX:0,panY:0});
let primary=INITIAL_VIEW_STATE.primary,secondary=INITIAL_VIEW_STATE.secondary,mode=INITIAL_VIEW_STATE.mode,showDimensions=INITIAL_VIEW_STATE.showDimensions,selection={design:INITIAL_VIEW_STATE.selectionDesign,region:INITIAL_VIEW_STATE.selectionRegion},hover=INITIAL_VIEW_STATE.hover,cursor=INITIAL_VIEW_STATE.cursor,viewport={zoom:INITIAL_VIEW_STATE.zoom,panX:INITIAL_VIEW_STATE.panX,panY:INITIAL_VIEW_STATE.panY},raf=0,layouts={};
const osTheme=window.matchMedia("(prefers-color-scheme: light)");let themePreference="system";
const esc=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(value,digits=4)=>Number(value).toLocaleString(undefined,{maximumSignificantDigits:digits});
const mm=value=>`${fmt(value*1000)} mm`;
const cm3=value=>`${fmt(value*1e6)} cm³`;
function options(select){DATA.designs.forEach((d,i)=>{const option=document.createElement("option");option.value=i;option.textContent=d.label;select.append(option)})}
options($("primary"));options($("secondary"));$("secondary").value=1;
$("warning").textContent=DATA.warning;$("physics").textContent=DATA.physics_boundary;
function sourceRow(name,value,source){return `<span>${esc(name)}</span><span>${esc(value)}</span><span class="source">source: ${esc(source)}</span>`}
function renderMetrics(){const root=$("metrics");root.textContent="";DATA.designs.forEach((d,i)=>{const x=d.descriptors,card=document.createElement("article");card.className="metric"+(i===primary||i===secondary?" active":"");card.innerHTML=`<h2>${esc(d.label)}</h2><div class="kv">${sourceRow("Envelope",`${mm(x.envelope.radius_m)} × ${mm(x.envelope.z_max_m-x.envelope.z_min_m)}`,"manifest.configurations[].descriptors.envelope")}${sourceRow("Stages / pitch",`${d.stages.length} / ${mm(x.stage_pitch_m)}`,"geometry stages[] / manifest descriptors.stage_pitch_m")}${sourceRow("Active volume",cm3(x.active_volume_m3),"manifest descriptors.active_volume_m3")}${sourceRow("Channel volume",cm3(x.channel_volume_m3),"manifest descriptors.channel_volume_m3")}${sourceRow("Magnet volume",cm3(x.magnet_volume_m3),"geometry magnet regions[].volume_m3")}${sourceRow("Magnet mass estimate",`${fmt(x.magnet_mass_estimate_kg*1000)} g`,"manifest descriptors.magnet_mass_estimate_kg")}${sourceRow("Minimum clearances",`${mm(x.minimum_radial_gap_m)} radial / ${mm(x.minimum_axial_gap_m)} axial`,"manifest descriptors.minimum_*_gap_m")}</div>`;root.append(card)})}
function setup(canvas){const box=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),width=Math.max(1,Math.round(box.width*dpr)),height=Math.max(1,Math.round(box.height*dpr));if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,width:box.width,height:box.height}}
const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function domain(indices){const designs=indices.map(i=>DATA.designs[i]);return {z0:Math.min(...designs.map(d=>d.descriptors.envelope.z_min_m)),z1:Math.max(...designs.map(d=>d.descriptors.envelope.z_max_m)),r:Math.max(...designs.map(d=>d.descriptors.envelope.radius_m))}}
function transform(width,height,dom){const pad={l:56,r:20,t:25,b:48},sx=(width-pad.l-pad.r)/(dom.z1-dom.z0),sy=(height-pad.t-pad.b)/(2*dom.r),scale=Math.min(sx,sy),plotW=(dom.z1-dom.z0)*scale,plotH=2*dom.r*scale,left=pad.l+(width-pad.l-pad.r-plotW)/2,top=pad.t+(height-pad.t-pad.b-plotH)/2;return {left,top,right:left+plotW,bottom:top+plotH,scale,x:z=>left+(z-dom.z0)*scale,y:r=>top+(dom.r-r)*scale,rz:(x,y)=>({z:dom.z0+(x-left)/scale,r:dom.r-(y-top)/scale})}}
function polygon(region,sign,t){return region.polygon_rz_m.map(([r,z])=>[t.x(z),t.y(sign*r)])}
function path(ctx,points){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath()}
function pointIn(points,x,y){let inside=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const [xi,yi]=points[i],[xj,yj]=points[j];if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))inside=!inside}return inside}
function drawDesign(ctx,design,index,t,overlay){design.regions.forEach((region,regionIndex)=>{for(const sign of [-1,1]){const points=polygon(region,sign,t);path(ctx,points);ctx.globalAlpha=overlay?.comparison?.7:.94;ctx.fillStyle=COLORS[region.role]||"#94a3b8";ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=selection.design===index&&selection.region===regionIndex?css("--accent"):css("--bg");ctx.lineWidth=selection.design===index&&selection.region===regionIndex?3:1;ctx.stroke();if(overlay?.comparison){ctx.save();path(ctx,points);ctx.clip();ctx.strokeStyle="#ffffff80";ctx.lineWidth=1;for(let x=t.left-t.bottom;x<t.right+t.bottom;x+=8){ctx.beginPath();ctx.moveTo(x,t.bottom);ctx.lineTo(x+t.bottom-t.top,t.top);ctx.stroke()}ctx.restore()}}});
design.stages.forEach(stage=>{const region=design.regions.find(r=>r.region_id===stage.magnet_region_id),x=t.x(stage.center_z_m),r=(region.r_inner_start_m+region.r_outer_start_m)/2;ctx.fillStyle="#fff";ctx.font="bold 11px system-ui";ctx.textAlign="center";ctx.fillText(stage.polarity>0?"+z":"−z",x,t.y(r)+4)});
ctx.strokeStyle=overlay?.comparison?"#ffffff99":css("--axis");ctx.setLineDash([5,4]);ctx.strokeRect(t.x(design.descriptors.envelope.z_min_m),t.y(design.descriptors.envelope.radius_m),(design.descriptors.envelope.z_max_m-design.descriptors.envelope.z_min_m)*t.scale,2*design.descriptors.envelope.radius_m*t.scale);ctx.setLineDash([])}
function axes(ctx,t,dom,width,height){ctx.strokeStyle=css("--axis");ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(t.left,t.y(0));ctx.lineTo(t.right,t.y(0));ctx.stroke();ctx.fillStyle=css("--muted");ctx.font="11px system-ui";ctx.textAlign="center";for(let i=0;i<=4;i++){const z=dom.z0+(dom.z1-dom.z0)*i/4,x=t.x(z);ctx.fillText(fmt(z*1000,3),x,t.bottom+18)}ctx.fillText("z (mm)",(t.left+t.right)/2,height-9);ctx.textAlign="right";for(let i=0;i<=4;i++){const r=dom.r-(2*dom.r)*i/4;ctx.fillText(fmt(r*1000,3),t.left-7,t.y(r)+4)}ctx.save();ctx.translate(14,(t.top+t.bottom)/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("r (mm)",0,0);ctx.restore()}
function dimensions(ctx,design,t){if(!showDimensions)return;const c=design.chamber,e=design.descriptors.envelope;ctx.strokeStyle=css("--accent");ctx.fillStyle=css("--text");ctx.font="11px system-ui";ctx.textAlign="center";ctx.setLineDash([3,2]);const y=t.bottom+31;ctx.beginPath();ctx.moveTo(t.x(0),y);ctx.lineTo(t.x(c.length_m),y);ctx.stroke();ctx.fillText(`chamber L ${mm(c.length_m)}`,(t.x(0)+t.x(c.length_m))/2,y-3);const x=t.right-8;ctx.beginPath();ctx.moveTo(x,t.y(0));ctx.lineTo(x,t.y(e.radius_m));ctx.stroke();ctx.textAlign="right";ctx.fillText(`envelope R ${mm(e.radius_m)}`,x-3,(t.y(0)+t.y(e.radius_m))/2);if(design.stages.length>1){const a=design.stages[0].center_z_m,b=design.stages[1].center_z_m,py=t.top+13;ctx.beginPath();ctx.moveTo(t.x(a),py);ctx.lineTo(t.x(b),py);ctx.stroke();ctx.textAlign="center";ctx.fillText(`pitch ${mm(design.stages[0].pitch_m)}`,(t.x(a)+t.x(b))/2,py-3)}ctx.setLineDash([])}
function drawCanvas(canvasId,indices){const canvas=$(canvasId),s=setup(canvas),ctx=s.ctx,dom=domain(indices),t=transform(s.width,s.height,dom);ctx.clearRect(0,0,s.width,s.height);ctx.fillStyle=css("--panel");ctx.fillRect(0,0,s.width,s.height);axes(ctx,t,dom,s.width,s.height);indices.forEach((index,k)=>drawDesign(ctx,DATA.designs[index],index,t,{comparison:k>0}));dimensions(ctx,DATA.designs[indices[0]],t);layouts[canvasId]={indices,t};}
function renderCanvases(){const overlay=mode==="overlay";$("canvases").classList.toggle("overlay",overlay);$("primaryTitle").textContent=overlay?`${DATA.designs[primary].label} + ${DATA.designs[secondary].label}`:DATA.designs[primary].label;$("secondaryTitle").textContent=DATA.designs[secondary].label;drawCanvas("primaryCanvas",overlay?[primary,secondary]:[primary]);if(!overlay)drawCanvas("secondaryCanvas",[secondary])}
function renderLegend(){const root=$("legend");root.textContent="";Object.entries(COLORS).forEach(([role,color])=>{const button=document.createElement("button");button.type="button";button.innerHTML=`<span class="swatch" style="background:${color}"></span>${esc(role.replaceAll("_"," "))}`;button.onclick=()=>{const d=DATA.designs[selection.design],found=d.regions.findIndex(r=>r.role===role);if(found>=0){selection.region=found;schedule()}};root.append(button)})}
function objectRows(object){return Object.entries(object).map(([key,value])=>`<div class="kv"><span>${esc(key)}</span><span>${esc(typeof value==="object"?JSON.stringify(value):value)}</span></div>`).join("")}
function renderDetails(){const design=DATA.designs[selection.design],region=design.regions[selection.region]||design.regions[0],material=design.materials.find(m=>m.material_id===region.material_id),interfaces=design.interfaces.filter(i=>i.region_id===region.region_id),stage=design.stages.find(s=>s.magnet_region_id===region.region_id||s.pole_after_region_id===region.region_id),warnings=design.descriptors.manufacturability_warnings;let html=`<h3>${esc(design.label)} · ${esc(region.region_id)}</h3><p><span class="pill">${esc(region.role)}</span><span class="pill">${esc(region.shape)}</span></p><div class="detail-block"><h3>Complete region record</h3>${objectRows(region)}</div><div class="detail-block"><h3>Material</h3>${objectRows(material)}</div>`;if(stage)html+=`<div class="detail-block"><h3>Stage / polarity</h3>${objectRows(stage)}</div>`;html+=`<div class="detail-block"><h3>Interfaces (${interfaces.length})</h3>${interfaces.map(item=>`<div class="interface">${objectRows(item)}</div>`).join("")}</div><div class="detail-block"><h3>Manufacturability warnings</h3><ul class="warning-list">${warnings.map(w=>`<li>${esc(w)}</li>`).join("")}</ul><p class="small">Rules: radial tolerance ${mm(design.manufacturing.radial_tolerance_m)}; axial tolerance ${mm(design.manufacturing.axial_tolerance_m)}; minimum clearance ${mm(design.manufacturing.minimum_clearance_m)}; thermal clearance ${mm(design.manufacturing.thermal_clearance_m)}; minimum thickness ${mm(design.manufacturing.minimum_thickness_m)}. Source: geometry manufacturing + manifest descriptors.</p></div><div class="detail-block"><h3>External components</h3>${design.external_components.map(objectRows).join("<br>")}</div>`;$("details").innerHTML=html}
function renderIdentity(){const m=DATA.manifest;let html=`<p><span class="pill">manifest SHA-256</span><code>${esc(m.file_sha256)}</code></p><p><span class="pill">schema</span>${esc(m.schema_version)} · generator ${esc(m.generator_version)}</p>`;DATA.designs.forEach(d=>{const i=d.identity;html+=`<div class="detail-block"><h3>${esc(d.label)}</h3><p><span class="pill">geometry payload</span><code>${esc(i.geometry_payload_sha256)}</code></p><p><span class="pill">${esc(i.geometry_file)}</span><code>${esc(i.geometry_file_sha256)}</code></p><p><span class="pill">${esc(i.viewer_file)}</span><code>${esc(i.viewer_file_sha256)}</code></p><p><span class="pill">${esc(i.svg_file)}</span><code>${esc(i.svg_file_sha256)}</code></p><p class="small">Numeric map: ${esc(Object.values(d.number_sources).join(" · "))}</p></div>`});$("identity").innerHTML=html}
function hit(canvasId,event){const layout=layouts[canvasId];if(!layout)return;const rect=$(canvasId).getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;for(const designIndex of [...layout.indices].reverse()){const regions=DATA.designs[designIndex].regions;for(let index=regions.length-1;index>=0;index--)for(const sign of [-1,1])if(pointIn(polygon(regions[index],sign,layout.t),x,y)){selection={design:designIndex,region:index};schedule();return}}}
function keySelect(canvasId,event){const layout=layouts[canvasId];if(!layout)return;const designs=layout.indices;if(event.key==="Enter"&&designs.length>1){selection.design=selection.design===designs[0]?designs[1]:designs[0];selection.region=0}else if(["ArrowRight","ArrowDown"].includes(event.key))selection.region=(selection.region+1)%DATA.designs[selection.design].regions.length;else if(["ArrowLeft","ArrowUp"].includes(event.key))selection.region=(selection.region-1+DATA.designs[selection.design].regions.length)%DATA.designs[selection.design].regions.length;else if(event.key==="Home")selection.region=0;else return;event.preventDefault();schedule()}
function drawAll(){renderMetrics();renderCanvases();renderDetails()}
function schedule(){cancelAnimationFrame(raf);raf=requestAnimationFrame(drawAll)}
function syncControls(){$("primary").value=String(primary);$("secondary").value=String(secondary);$("mode").value=mode;$("dimensions").checked=showDimensions}
function resetView(){cancelAnimationFrame(raf);primary=INITIAL_VIEW_STATE.primary;secondary=INITIAL_VIEW_STATE.secondary;mode=INITIAL_VIEW_STATE.mode;showDimensions=INITIAL_VIEW_STATE.showDimensions;selection={design:INITIAL_VIEW_STATE.selectionDesign,region:INITIAL_VIEW_STATE.selectionRegion};hover=INITIAL_VIEW_STATE.hover;cursor=INITIAL_VIEW_STATE.cursor;viewport={zoom:INITIAL_VIEW_STATE.zoom,panX:INITIAL_VIEW_STATE.panX,panY:INITIAL_VIEW_STATE.panY};layouts={};syncControls();$("details").scrollTop=0;schedule()}
function applyTheme(theme){document.documentElement.dataset.theme=theme;const light=theme==="light";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",String(light));schedule()}
$("primary").onchange=e=>{primary=Number(e.target.value);selection={design:primary,region:0};schedule()};$("secondary").onchange=e=>{secondary=Number(e.target.value);schedule()};$("mode").onchange=e=>{mode=e.target.value;schedule()};$("dimensions").onchange=e=>{showDimensions=e.target.checked;schedule()};
$("reset").onclick=resetView;
$("theme").onclick=()=>{themePreference=document.documentElement.dataset.theme==="light"?"dark":"light";applyTheme(themePreference)};
osTheme.addEventListener("change",()=>{if(themePreference==="system")applyTheme(osTheme.matches?"light":"dark")});
for(const id of ["primaryCanvas","secondaryCanvas"]){$(id).addEventListener("pointerdown",e=>hit(id,e));$(id).addEventListener("keydown",e=>keySelect(id,e))}
window.addEventListener("keydown",e=>{if(e.key==="Escape"){e.preventDefault();resetView();return}if(["INPUT","SELECT","BUTTON"].includes(e.target.tagName))return;if(["1","2","3"].includes(e.key)){primary=Number(e.key)-1;$("primary").value=primary;selection={design:primary,region:0};schedule()}});
new ResizeObserver(schedule).observe($("canvases"));window.addEventListener("pageshow",schedule);renderLegend();renderIdentity();syncControls();applyTheme(osTheme.matches?"light":"dark");
</script>
</body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    colors = json.dumps(ROLE_COLORS, sort_keys=True, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA__", encoded).replace("__COLORS__", colors)


def generate(
    output_path: Path = DEFAULT_OUTPUT,
    artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(build_payload(artifact_directory)),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(generate(arguments.output, arguments.artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
