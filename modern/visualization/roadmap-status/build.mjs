#!/usr/bin/env node
/**
 * Build modern/visualization/roadmap-status.html from the canvas source.
 *
 *   node build.mjs                 bundle roadmap-status.canvas.tsx -> ../roadmap-status.html (+ sidecar)
 *   node build.mjs --sync [path]   first copy the live Cursor canvas into roadmap-status.canvas.tsx
 *                                  (CRLF -> LF; content otherwise byte-identical), then build
 *   node build.mjs --out <file>    write the HTML somewhere else (the sidecar goes beside it)
 *   node build.mjs --no-sidecar    skip the anchor sidecar
 *
 * The output is deterministic: same canvas + same pinned toolchain (package-lock.json) -> same
 * bytes, on any platform. No timestamps are embedded; provenance is expressed as content hashes.
 * The page is self-contained: React, ReactDOM and the compiled canvas are inlined as one <script>,
 * nothing is fetched at runtime (no CDN), so it opens from file://.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const HERE = dirname(fileURLToPath(import.meta.url));
const CANVAS_COPY = resolve(HERE, "roadmap-status.canvas.tsx");
const SHIM = resolve(HERE, "cursor-canvas.tsx");
const ENTRY = resolve(HERE, "entry.tsx");
const TEMPLATE = resolve(HERE, "template.html");
const DEFAULT_OUT = resolve(HERE, "..", "roadmap-status.html");
const CANVAS_BASENAME = "open-cft-roadmap-status.canvas.tsx";
const SIDECAR_SCHEMA = "cft-roadmap-status-dashboard-anchor/1.0.0";

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function pkgVersion(name) {
  const file = resolve(HERE, "node_modules", name, "package.json");
  return JSON.parse(readFileSync(file, "utf8")).version;
}

/**
 * Where Cursor keeps the live canvas for this workspace:
 * ~/.cursor/projects/<slug>/canvases/<name>.canvas.tsx, where <slug> is the workspace path with
 * the drive colon dropped and every separator replaced by "-" (C:\a\b -> c-a-b).
 */
function defaultLiveCanvasPath() {
  const repoRoot = resolve(HERE, "..", "..", "..");
  const slug = repoRoot.replace(/^([A-Za-z]):/, (_, d) => d.toLowerCase()).replace(/[\\/]+/g, "-");
  const home = process.env.USERPROFILE || process.env.HOME || "";
  return resolve(home, ".cursor", "projects", slug, "canvases", CANVAS_BASENAME);
}

function parseArgs(argv) {
  const args = { sync: false, syncPath: null, out: DEFAULT_OUT, sidecar: true };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--sync") {
      args.sync = true;
      if (argv[i + 1] && !argv[i + 1].startsWith("--")) {
        args.syncPath = resolve(argv[i + 1]);
        i += 1;
      }
    } else if (arg === "--out") {
      args.out = resolve(argv[i + 1]);
      i += 1;
    } else if (arg === "--no-sidecar") {
      args.sidecar = false;
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return args;
}

/** Copy the live canvas into the repo: bytes unchanged except CRLF -> LF (the repo pins eol=lf). */
function syncCanvas(sourcePath) {
  if (!existsSync(sourcePath)) throw new Error(`live canvas not found: ${sourcePath}`);
  const raw = readFileSync(sourcePath);
  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    throw new Error("live canvas carries a UTF-8 BOM; refusing to sync");
  }
  const text = raw.toString("utf8").replace(/\r\n/g, "\n");
  if (text.includes("\r")) throw new Error("live canvas has bare CR characters; refusing to sync");
  writeFileSync(CANVAS_COPY, text, { encoding: "utf8" });
  return { sourcePath, bytes: Buffer.byteLength(text, "utf8"), sha256: sha256(Buffer.from(text, "utf8")) };
}

async function bundle() {
  const result = await esbuild.build({
    entryPoints: [ENTRY],
    bundle: true,
    write: false,
    format: "iife",
    platform: "browser",
    target: ["es2020"],
    jsx: "automatic",
    minify: true,
    legalComments: "none",
    sourcemap: false,
    charset: "utf8",
    treeShaking: true,
    define: { "process.env.NODE_ENV": '"production"' },
    alias: { "cursor/canvas": SHIM },
    logLevel: "warning",
  });
  if (result.outputFiles.length !== 1) throw new Error(`expected one output file, got ${result.outputFiles.length}`);
  return result.outputFiles[0].text;
}

/** Make the bundle safe to inline: no "</script" or "<!--" sequences may survive inside the script element. */
function inlineSafe(js) {
  return js.replace(/<\/script/gi, "<\\/script").replace(/<!--/g, "<\\!--");
}

/**
 * URL prefixes that may legitimately appear in the page without anything being fetched:
 * github.com commit anchors from the canvas text, the W3C XML namespace identifiers React DOM
 * carries as constants (svg / xlink / MathML), and the react.dev error-decoder link embedded in
 * production React's minified error messages.
 */
export const ALLOWED_URL_PREFIXES = ["https://github.com/", "http://www.w3.org/", "https://react.dev/errors/"];

function assertOffline(html) {
  const head = html.split("<script>")[0];
  if (/<script[^>]*\ssrc=/i.test(html)) throw new Error("built HTML references an external script");
  if (/<link[^>]*\shref=/i.test(head)) throw new Error("built HTML references an external stylesheet");
  if (/@import|url\(\s*["']?https?:/i.test(head)) throw new Error("built HTML head imports a remote resource");
  if (/\bfetch\(|XMLHttpRequest|navigator\.sendBeacon|import\(\s*["'`]https?:/.test(html)) throw new Error("built HTML contains network calls");
  const urls = [...html.matchAll(/https?:\/\/[^\s"'`<>)]+/g)].map((m) => m[0]);
  const foreign = urls.filter((u) => !ALLOWED_URL_PREFIXES.some((prefix) => u.startsWith(prefix)));
  if (foreign.length) throw new Error(`unexpected URLs in the built HTML (allowed prefixes: ${ALLOWED_URL_PREFIXES.join(" ")}): ${foreign.slice(0, 5).join(", ")}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const report = { canvas_copy: CANVAS_COPY };
  if (args.sync) {
    const synced = syncCanvas(args.syncPath ?? defaultLiveCanvasPath());
    report.synced_from = synced.sourcePath;
    console.log(`synced ${synced.sourcePath} -> ${CANVAS_COPY} (${synced.bytes} bytes LF, sha256 ${synced.sha256.slice(0, 16)}...)`);
  }
  if (!existsSync(CANVAS_COPY)) throw new Error(`missing ${CANVAS_COPY}; run with --sync`);

  const canvasBytes = readFileSync(CANVAS_COPY);
  const shimBytes = readFileSync(SHIM);
  const template = readFileSync(TEMPLATE, "utf8");
  const versions = { esbuild: pkgVersion("esbuild"), react: pkgVersion("react"), "react-dom": pkgVersion("react-dom") };

  const js = inlineSafe(await bundle());
  const provenance = [
    `canvas sha256 ${sha256(canvasBytes)}`,
    `shim sha256 ${sha256(shimBytes)}`,
    `toolchain esbuild ${versions.esbuild}, react ${versions.react}, react-dom ${versions["react-dom"]}`,
  ].join("\n  ");
  let html = template.replace("{{PROVENANCE}}", provenance).replace("{{BUNDLE}}", `<script>\n${js}\n</script>`);
  html = html.replace(/\r\n/g, "\n");
  if (!html.endsWith("\n")) html += "\n";
  assertOffline(html);

  const htmlBytes = Buffer.from(html, "utf8");
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, htmlBytes);
  report.html = args.out;
  report.html_bytes = htmlBytes.length;
  report.html_sha256 = sha256(htmlBytes);

  if (args.sidecar) {
    const sidecar = {
      schema: SIDECAR_SCHEMA,
      html_file: "roadmap-status.html",
      html_sha256: report.html_sha256,
      html_bytes: htmlBytes.length,
      canvas_file: "roadmap-status/roadmap-status.canvas.tsx",
      canvas_sha256: sha256(canvasBytes),
      canvas_bytes: canvasBytes.length,
      shim_file: "roadmap-status/cursor-canvas.tsx",
      shim_sha256: sha256(shimBytes),
      toolchain: versions,
      policy:
        "the build is byte-deterministic given the canvas copy, the shim, the template and the pinned toolchain " +
        "(package-lock.json); a byte-exact rebuild is asserted on every platform. The Node.js version is not part " +
        "of the identity (esbuild output does not depend on it).",
    };
    const sidecarPath = args.out.replace(/\.html$/, ".anchor-platform.json");
    writeFileSync(sidecarPath, `${JSON.stringify(sidecar, null, 1)}\n`, { encoding: "utf8" });
    report.sidecar = sidecarPath;
  }
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
