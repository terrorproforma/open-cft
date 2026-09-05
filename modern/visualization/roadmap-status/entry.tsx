/**
 * Browser entry point for the offline roadmap dashboard.
 *
 * Mounts the canvas component (default export of `roadmap-status.canvas.tsx`) into `#root`
 * with plain ReactDOM. The IDE host does the same through its canvas runtime; here there
 * is no host, so `cursor/canvas` resolves to `./cursor-canvas.tsx` (see build.mjs).
 */
import { createRoot } from "react-dom/client";
import OpenCftRoadmapStatus from "./roadmap-status.canvas";

const root = document.getElementById("root");
if (!(root instanceof HTMLElement)) {
  throw new Error("roadmap-status.html: #root element not found");
}
createRoot(root).render(<OpenCftRoadmapStatus />);
