#!/usr/bin/env node
/**
 * Headless-DOM check of the built dashboard (jsdom, offline).
 *
 *   node verify.mjs [--html <file>] [--json <file>] [--expect-rows 44]
 *                   [--expect-chips "0/44 externally validated|17 in the paper|36/44 merged|0/44 on main"]
 *
 * Loads roadmap-status.html into jsdom with scripts enabled, waits for React to mount, then:
 *   - reads the four header status chips (derived at runtime from `ladderRows`);
 *   - clicks every tab pill and checks each view renders;
 *   - on the Stage ladder tab counts the matrix rows (one per `ladderRows` entry) and records the
 *     row names, then expands the first row and checks its citation table appears;
 *   - asserts nothing in the DOM points at an external resource.
 * Prints a JSON report and exits non-zero when an expectation fails. jsdom never loads
 * subresources here (none are declared), and the page performs no network calls.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const HERE = dirname(fileURLToPath(import.meta.url));
const TABS = ["Overview", "Phases", "Stage ladder", "Experiments", "Critical path", "Literature roadmap", "Actions", "Evidence", "Details"];

function parseArgs(argv) {
  const args = { html: resolve(HERE, "..", "roadmap-status.html"), json: null, expectRows: null, expectChips: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      return argv[i];
    };
    if (arg === "--html") args.html = resolve(next());
    else if (arg === "--json") args.json = resolve(next());
    else if (arg === "--expect-rows") args.expectRows = Number(next());
    else if (arg === "--expect-chips") args.expectChips = next().split("|");
    else throw new Error(`unknown argument: ${arg}`);
  }
  return args;
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

async function waitFor(predicate, timeoutMs, what) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await sleep(20);
  }
  throw new Error(`timed out waiting for ${what}`);
}

const text = (node) => (node?.textContent ?? "").replace(/\s+/g, " ").trim();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const html = readFileSync(args.html, "utf8");
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", (error) => errors.push(`jsdom: ${error.message}`));
  virtualConsole.on("error", (...parts) => errors.push(`console.error: ${parts.map(String).join(" ")}`));
  const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, url: "file:///roadmap-status.html", virtualConsole });
  const { window } = dom;
  const { document } = window;

  const root = await waitFor(() => document.querySelector("#root h1"), 15000, "the React root to mount");
  const header = root.parentElement; // Stack: H1, Text, Row(meta chips), Row(status chips), Row(legend)
  const chipRow = header.children[3];
  const chips = Array.from(chipRow.querySelectorAll(":scope > span")).map(text);

  const external = {
    scripts_with_src: document.querySelectorAll("script[src]").length,
    stylesheets: document.querySelectorAll("link[rel~='stylesheet'], link[href]").length,
    remote_images: Array.from(document.querySelectorAll("img[src], iframe[src]")).filter((el) => /^https?:/i.test(el.getAttribute("src") || "")).length,
  };

  const findPill = (label) => Array.from(document.querySelectorAll("#root button")).find((button) => text(button) === label);
  const tabs = {};
  let ladder = null;
  for (const tab of TABS) {
    const pill = findPill(tab);
    if (!pill) {
      errors.push(`tab pill not found: ${tab}`);
      continue;
    }
    pill.click();
    await sleep(30);
    await waitFor(() => text(document.querySelector("#root")).length > 200, 5000, `${tab} to render`);
    const body = text(document.querySelector("#root"));
    tabs[tab] = { chars: body.length };
    if (tab === "Stage ladder") {
      // Desktop matrix: each row's name cell carries the disclosure glyph and the row's lineage as title.
      const rowCells = Array.from(document.querySelectorAll("#root div[title]")).filter((div) => /^[▸▾]/.test(text(div)));
      const rowNames = rowCells.map((div) => text(div).replace(/^[▸▾]\s*/, ""));
      let expanded = false;
      if (rowCells[0]) {
        rowCells[0].click();
        await sleep(30);
        expanded = text(document.querySelector("#root")).includes("Citation (path · SHA · line)");
        rowCells[0].click();
        await sleep(10);
      }
      ladder = { rows: rowCells.length, rowNames, firstRowExpands: expanded };
    }
  }
  findPill("Overview")?.click();

  const report = { html: args.html, bytes: Buffer.byteLength(html, "utf8"), chips, tabs, ladder, external, errors };
  const failures = [];
  if (chips.length !== 4) failures.push(`expected 4 header chips, found ${chips.length}: ${JSON.stringify(chips)}`);
  if (args.expectChips && JSON.stringify(chips) !== JSON.stringify(args.expectChips)) failures.push(`chips ${JSON.stringify(chips)} != expected ${JSON.stringify(args.expectChips)}`);
  if (!ladder) failures.push("Stage ladder tab did not render a matrix");
  else {
    if (args.expectRows !== null && ladder.rows !== args.expectRows) failures.push(`ladder rows ${ladder.rows} != expected ${args.expectRows}`);
    if (!ladder.firstRowExpands) failures.push("expanding the first ladder row did not show its citation table");
    if (new Set(ladder.rowNames).size !== ladder.rowNames.length) failures.push("ladder row names are not unique");
  }
  for (const tab of TABS) if (!tabs[tab]) failures.push(`tab did not render: ${tab}`);
  if (external.scripts_with_src || external.stylesheets || external.remote_images) failures.push(`external resources in the DOM: ${JSON.stringify(external)}`);
  if (errors.length) failures.push(`runtime errors: ${errors.slice(0, 3).join(" | ")}`);
  report.ok = failures.length === 0;
  report.failures = failures;

  const serialised = `${JSON.stringify(report, null, 2)}\n`;
  if (args.json) writeFileSync(args.json, serialised, { encoding: "utf8" });
  process.stdout.write(serialised);
  window.close();
  process.exit(report.ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(2);
});
