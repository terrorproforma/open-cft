/**
 * Minimal stand-in for the Cursor canvas runtime module `cursor/canvas`.
 *
 * The roadmap canvas (`roadmap-status.canvas.tsx`, copied verbatim from the Cursor
 * canvases directory) imports only from `cursor/canvas`. Inside the IDE that module is
 * provided by the canvas host (React 19 + the SDK primitives, themed from the editor).
 * This file re-implements the subset the canvas uses so that esbuild can bundle the
 * canvas into a single offline HTML page with plain React:
 *
 *   layout       Stack, Row, Grid, Divider, Spacer
 *   typography   H1, H2, H3, Text, Code, Link
 *   surfaces     Card, CardHeader, CardBody, CollapsibleSection, Callout, Table, Stat
 *   controls     Pill, Button
 *   hooks        useHostTheme (fixed dark theme), useEffect/useState/useMemo/useRef (React)
 *
 * Styling follows the SDK's dark theme tokens (`canvasPaletteDark`, `categoryPaletteDark`,
 * typography, spacing, radius) so the page looks like the canvas does beside the chat.
 * Host-only features that make no sense in a static file are NOT provided:
 * `useCanvasState` (persisted state sidecar) and `useCanvasAction` (IDE actions) - the
 * roadmap canvas uses neither. Charts, diff views, forms and the DAG layout are omitted for
 * the same reason (unused). `Text` renders its children as-is, exactly like the runtime.
 */
import {
  createContext,
  isValidElement,
  cloneElement,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from "react";

export { useEffect, useMemo, useRef, useState };
export type { CSSProperties };
export type { RefObject } from "react";

// ---------------------------------------------------------------------------------------------
// Tokens (pinned copies of the canvas SDK dark theme)
// ---------------------------------------------------------------------------------------------

export type Color = "gray" | "purple" | "green" | "yellow" | "cyan" | "pink" | "blue" | "orange" | "red";
export type CategoryPalette = Readonly<Record<Color, string>>;

export interface CanvasPalette {
  readonly foreground: string;
  readonly foregroundSecondary: string;
  readonly foregroundTertiary: string;
  readonly foregroundQuaternary: string;
  readonly editor: string;
  readonly chrome: string;
  readonly sidebar: string;
  readonly elevated: string;
  readonly fillPrimary: string;
  readonly fillSecondary: string;
  readonly fillTertiary: string;
  readonly fillQuaternary: string;
  readonly strokePrimary: string;
  readonly strokeSecondary: string;
  readonly strokeTertiary: string;
  readonly strokeFocused: string;
  readonly accent: string;
  readonly buttonBackground: string;
  readonly buttonForeground: string;
  readonly buttonHoverBackground: string;
  readonly link: string;
  readonly diffInsertedLine: string;
  readonly diffRemovedLine: string;
  readonly diffStripAdded: string;
  readonly diffStripRemoved: string;
}

export interface CanvasTokens {
  bg: { editor: string; chrome: string; elevated: string };
  text: { primary: string; secondary: string; tertiary: string; quaternary: string; link: string; onAccent: string };
  stroke: { primary: string; secondary: string; tertiary: string; focused: string };
  fill: { primary: string; secondary: string; tertiary: string; quaternary: string };
  accent: { primary: string; control: string; controlHover: string };
  diff: { insertedLine: string; removedLine: string; stripAdded: string; stripRemoved: string };
  category: CategoryPalette;
}

export interface CanvasHostTheme extends CanvasTokens {
  readonly kind: string;
  readonly tokens: CanvasTokens;
  readonly palette: CanvasPalette;
}

export const canvasPaletteDark: CanvasPalette = {
  foreground: "#F0F0F0",
  foregroundSecondary: "#F0F0F0BD",
  foregroundTertiary: "#F0F0F099",
  foregroundQuaternary: "#F0F0F05C",
  editor: "#181818",
  chrome: "#141414",
  sidebar: "#181818",
  elevated: "#181818",
  fillPrimary: "#F0F0F033",
  fillSecondary: "#F0F0F024",
  fillTertiary: "#F0F0F014",
  fillQuaternary: "#F0F0F00F",
  strokePrimary: "#F0F0F033",
  strokeSecondary: "#F0F0F01F",
  strokeTertiary: "#F0F0F014",
  strokeFocused: "#F0F0F0",
  accent: "#599CE7",
  buttonBackground: "#599CE7",
  buttonForeground: "#191C22",
  buttonHoverBackground: "#68A4E8",
  link: "#7BAFE9",
  diffInsertedLine: "#3FA26633",
  diffRemovedLine: "#B8004933",
  diffStripAdded: "#3FA2668F",
  diffStripRemoved: "#FC6B838F",
};

export const categoryPaletteDark: CategoryPalette = {
  gray: canvasPaletteDark.foregroundTertiary,
  purple: "#9386F2",
  green: "#3FA266",
  yellow: "#F1B467",
  cyan: "#81A1C1",
  pink: "#B48EAD",
  blue: "#7BAFE9",
  orange: "#DD7F76",
  red: "#FC6B83",
};

export const colorPalette = categoryPaletteDark;

function buildTokens(palette: CanvasPalette, category: CategoryPalette): CanvasTokens {
  return {
    bg: { editor: palette.editor, chrome: palette.chrome, elevated: palette.elevated },
    text: {
      primary: palette.foreground,
      secondary: palette.foregroundSecondary,
      tertiary: palette.foregroundTertiary,
      quaternary: palette.foregroundQuaternary,
      link: palette.link,
      onAccent: palette.buttonForeground,
    },
    stroke: {
      primary: palette.strokePrimary,
      secondary: palette.strokeSecondary,
      tertiary: palette.strokeTertiary,
      focused: palette.strokeFocused,
    },
    fill: {
      primary: palette.fillPrimary,
      secondary: palette.fillSecondary,
      tertiary: palette.fillTertiary,
      quaternary: palette.fillQuaternary,
    },
    accent: { primary: palette.accent, control: palette.buttonBackground, controlHover: palette.buttonHoverBackground },
    diff: {
      insertedLine: palette.diffInsertedLine,
      removedLine: palette.diffRemovedLine,
      stripAdded: palette.diffStripAdded,
      stripRemoved: palette.diffStripRemoved,
    },
    category,
  };
}

export const canvasTokens: CanvasTokens = buildTokens(canvasPaletteDark, categoryPaletteDark);

const DARK_THEME: CanvasHostTheme = { ...canvasTokens, kind: "dark", tokens: canvasTokens, palette: canvasPaletteDark };

/** The page is always rendered with the Cursor dark theme (no host to read the editor theme from). */
export function useHostTheme(): CanvasHostTheme {
  return DARK_THEME;
}

export const canvasTypography = {
  h1: { fontSize: "24px", lineHeight: "30px", fontWeight: 590 },
  h2: { fontSize: "18px", lineHeight: "24px", fontWeight: 590 },
  h3: { fontSize: "16px", lineHeight: "22px", fontWeight: 590 },
  body: { fontSize: "14px", lineHeight: "20px", fontWeight: 400 },
  small: { fontSize: "12px", lineHeight: "16px", fontWeight: 400 },
} as const;

export const canvasSpacing = {
  "0.5": 2, "1": 4, "1.5": 6, "2": 8, "2.5": 10, "3": 12, "3.5": 14, "4": 16, "4.5": 18, "5": 20,
  "6": 24, "7": 28, "8": 32, "9": 36, "10": 40,
} as const;

export const canvasRadius = { none: 0, xs: 2, sm: 4, md: 6, lg: 8, xl: 12, full: 9999 } as const;

const chartPalette = {
  lightGreen: "#52B896E0",
  lightBlue: "#70B0D8E0",
  brightOrange: "#F0A040E0",
  darkAmber: "#C04848E0",
  muted: "#8888A8E0",
} as const;

const MONO = 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';

export function mergeStyle(base: CSSProperties, override?: CSSProperties): CSSProperties {
  return override ? { ...base, ...override } : base;
}

// ---------------------------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------------------------

export type StackProps = { children?: ReactNode; gap?: number; style?: CSSProperties };
export function Stack({ children, gap = 12, style }: StackProps) {
  return <div style={mergeStyle({ display: "flex", flexDirection: "column", gap: `${gap}px`, width: "100%" }, style)}>{children}</div>;
}

export type RowProps = {
  children?: ReactNode;
  gap?: number;
  align?: "start" | "center" | "end" | "stretch";
  justify?: "start" | "center" | "end" | "space-between";
  wrap?: boolean;
  style?: CSSProperties;
};
export function Row({ children, gap = 8, align = "center", justify = "start", wrap = false, style }: RowProps) {
  return (
    <div
      style={mergeStyle(
        { display: "flex", flexDirection: "row", flexWrap: wrap ? "wrap" : "nowrap", alignItems: align, justifyContent: justify, gap: `${gap}px`, width: "100%" },
        style,
      )}
    >
      {children}
    </div>
  );
}

export type GridProps = {
  children?: ReactNode;
  columns: number | string;
  gap?: number;
  align?: "start" | "center" | "end" | "stretch";
  style?: CSSProperties;
};
export function Grid({ children, columns, gap = 12, align = "stretch", style }: GridProps) {
  const template = typeof columns === "number" ? `repeat(${columns}, minmax(0, 1fr))` : columns;
  return <div style={mergeStyle({ display: "grid", gridTemplateColumns: template, gap: `${gap}px`, alignItems: align, width: "100%" }, style)}>{children}</div>;
}

export type DividerProps = { style?: CSSProperties };
export function Divider({ style }: DividerProps) {
  const { tokens } = useHostTheme();
  return <hr style={mergeStyle({ width: "100%", border: "none", borderTop: `1px solid ${tokens.stroke.tertiary}`, margin: 0 }, style)} />;
}

export function Spacer() {
  return <div style={{ flex: 1 }} />;
}

// ---------------------------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------------------------

/** True inside a heading or Text so nested Text renders a <span>, not a <p>. */
const typographyInlineContext = createContext<boolean>(false);

function heading(tag: "h1" | "h2" | "h3", preset: { fontSize: string; lineHeight: string; fontWeight: number }) {
  return function Heading({ children, style }: { children?: ReactNode; style?: CSSProperties }) {
    const { tokens } = useHostTheme();
    const Tag = tag;
    return (
      <typographyInlineContext.Provider value={true}>
        <Tag style={mergeStyle({ margin: 0, color: tokens.text.primary, fontSize: preset.fontSize, lineHeight: preset.lineHeight, fontWeight: preset.fontWeight }, style)}>{children}</Tag>
      </typographyInlineContext.Provider>
    );
  };
}
export type H1Props = { children?: ReactNode; style?: CSSProperties };
export type H2Props = H1Props;
export type H3Props = H1Props;
export const H1 = heading("h1", canvasTypography.h1);
export const H2 = heading("h2", canvasTypography.h2);
export const H3 = heading("h3", canvasTypography.h3);

export type TextWeight = "normal" | "medium" | "semibold" | "bold";
const textWeightMap: Record<TextWeight, number> = { normal: 400, medium: 500, semibold: 590, bold: 650 };

export type TextProps = {
  children?: ReactNode;
  tone?: "primary" | "secondary" | "tertiary" | "quaternary";
  size?: "body" | "small";
  as?: "p" | "span";
  weight?: TextWeight;
  italic?: boolean;
  truncate?: boolean | "start" | "end";
  style?: CSSProperties;
};
export function Text({ children, tone = "primary", size = "body", as, weight = "normal", italic = false, truncate = false, style }: TextProps) {
  const { tokens } = useHostTheme();
  const inline = useContext(typographyInlineContext);
  const tag = as ?? (inline ? "span" : "p");
  const preset = size === "small" ? canvasTypography.small : canvasTypography.body;
  const mode = truncate === true ? "end" : truncate === false ? null : truncate;
  const truncation: CSSProperties | undefined =
    mode !== null
      ? {
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          ...(tag === "span" ? { display: "inline-block", maxWidth: "100%" } : {}),
          ...(mode === "start" ? { direction: "rtl", textAlign: "left" } : {}),
        }
      : undefined;
  const body = mode === "start" ? <bdi>{children}</bdi> : children;
  const css = mergeStyle(
    { margin: 0, color: tokens.text[tone], fontSize: preset.fontSize, lineHeight: preset.lineHeight, fontWeight: textWeightMap[weight], fontStyle: italic ? "italic" : undefined, ...truncation },
    style,
  );
  return <typographyInlineContext.Provider value={true}>{tag === "span" ? <span style={css}>{body}</span> : <p style={css}>{body}</p>}</typographyInlineContext.Provider>;
}

export type CodeProps = { children?: ReactNode; style?: CSSProperties };
export function Code({ children, style }: CodeProps) {
  const { tokens } = useHostTheme();
  return (
    <code style={mergeStyle({ fontFamily: MONO, fontSize: "0.92em", padding: "2px 5px", borderRadius: `${canvasRadius.sm}px`, background: tokens.fill.quaternary, color: tokens.text.primary }, style)}>
      {children}
    </code>
  );
}

export type LinkProps = { children?: ReactNode; href: string; style?: CSSProperties };
export function Link({ children, href, style }: LinkProps) {
  const { tokens } = useHostTheme();
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" style={mergeStyle({ color: tokens.text.link, textDecoration: "underline", textUnderlineOffset: "2px", textDecorationColor: `${tokens.text.link}80` }, style)}>
      {children}
    </a>
  );
}

// ---------------------------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------------------------

export type TableColumnAlign = "left" | "center" | "right";
export type TableRowTone = "success" | "danger" | "warning" | "info" | "neutral";
export type TableProps = {
  headers: ReactNode[];
  rows: ReactNode[][];
  columnAlign?: Array<TableColumnAlign | undefined>;
  rowTone?: Array<TableRowTone | undefined>;
  framed?: boolean;
  striped?: boolean;
  stickyHeader?: boolean;
  style?: CSSProperties;
  emptyMessage?: ReactNode;
};

const tableCellPadding: CSSProperties = { padding: `${canvasSpacing[2]}px ${canvasSpacing[3]}px` };
const rowToneMarkerColors: Record<TableRowTone, string> = {
  success: chartPalette.lightGreen,
  danger: chartPalette.darkAmber,
  warning: chartPalette.brightOrange,
  info: chartPalette.lightBlue,
  neutral: chartPalette.muted,
};
const rowToneMarkerSizePx = canvasSpacing["1.5"];
const rowToneMarkerOffsetTopPx = (parseInt(canvasTypography.body.lineHeight, 10) - rowToneMarkerSizePx) / 2;

function RowToneMarker({ tone }: { tone: TableRowTone }) {
  return <span aria-hidden style={{ width: rowToneMarkerSizePx, height: rowToneMarkerSizePx, marginTop: rowToneMarkerOffsetTopPx, borderRadius: "50%", background: rowToneMarkerColors[tone], flexShrink: 0 }} />;
}

export function Table({ headers = [], rows = [], columnAlign, rowTone, framed = true, striped = false, stickyHeader = false, style, emptyMessage = "No rows." }: TableProps) {
  const { tokens } = useHostTheme();
  if (headers.length === 0) {
    return <div style={{ padding: `${canvasSpacing[3]}px`, color: tokens.text.secondary, fontSize: canvasTypography.body.fontSize }}>Add at least one header.</div>;
  }
  const columns = headers.length;
  const align = (index: number): TableColumnAlign => columnAlign?.[index] ?? "left";
  const tableStyle: CSSProperties = {
    minWidth: "100%",
    borderCollapse: stickyHeader ? "separate" : "collapse",
    ...(stickyHeader ? { borderSpacing: 0 } : undefined),
    tableLayout: "auto",
    fontSize: canvasTypography.body.fontSize,
    lineHeight: canvasTypography.body.lineHeight,
    color: tokens.text.primary,
  };
  const headerCell = (index: number): CSSProperties => ({
    ...tableCellPadding,
    textAlign: align(index),
    fontWeight: 600,
    color: tokens.text.primary,
    borderBottom: `1px solid ${tokens.stroke.secondary}`,
    ...(stickyHeader ? { position: "sticky", top: 0, zIndex: 2, backgroundColor: tokens.bg.editor } : undefined),
  });
  const bodyCell = (index: number): CSSProperties => ({ ...tableCellPadding, textAlign: align(index), verticalAlign: "top" });
  const table = (
    <table style={mergeStyle(tableStyle, style)}>
      <thead style={{ background: stickyHeader ? undefined : tokens.fill.quaternary }}>
        <tr>
          {headers.map((header, index) => (
            <th key={index} scope="col" style={headerCell(index)}>
              {header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 ? (
          <tr>
            <td colSpan={columns} style={mergeStyle(bodyCell(0), { color: tokens.text.secondary, borderBottom: `1px solid ${tokens.stroke.tertiary}` })}>
              {emptyMessage}
            </td>
          </tr>
        ) : (
          rows.map((row, rowIndex) => {
            const tone = rowTone?.[rowIndex];
            return (
              <tr
                key={rowIndex}
                style={{
                  ...(rowIndex < rows.length - 1 ? { borderBottom: `1px solid ${tokens.stroke.tertiary}` } : undefined),
                  ...(striped && rowIndex % 2 === 1 ? { background: tokens.fill.quaternary } : undefined),
                }}
              >
                {Array.from({ length: columns }, (_, cellIndex) => {
                  const cell = row[cellIndex] ?? null;
                  return (
                    <td key={cellIndex} style={bodyCell(cellIndex)}>
                      {tone && cellIndex === 0 ? (
                        <span style={{ display: "inline-flex", alignItems: "flex-start", gap: `${canvasSpacing["1.5"]}px`, minWidth: 0, maxWidth: "100%" }}>
                          <RowToneMarker tone={tone} />
                          <span style={{ minWidth: 0, flex: 1 }}>{cell}</span>
                        </span>
                      ) : (
                        cell
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })
        )}
      </tbody>
    </table>
  );
  if (!framed) return table;
  return (
    <div
      style={{
        width: "100%",
        minWidth: 0,
        boxSizing: "border-box",
        ...(stickyHeader ? { height: "100%", maxHeight: "100%", minHeight: 0 } : undefined),
        border: `1px solid ${tokens.stroke.tertiary}`,
        borderRadius: `${canvasRadius.lg}px`,
        background: tokens.bg.editor,
        overflowX: "auto",
        overflowY: stickyHeader ? "auto" : "clip",
      }}
    >
      {table}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Cards and collapsibles
// ---------------------------------------------------------------------------------------------

export function CanvasChevron({ expanded }: { expanded: boolean }) {
  return (
    <svg width={12} height={12} viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden style={{ display: "block", flexShrink: 0, color: "inherit", transform: expanded ? undefined : "rotate(-90deg)" }}>
      <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export type CardSize = "base" | "lg";
export type CardVariant = "default" | "borderless";
type CardChrome = { size: CardSize; stickyHeader: boolean; collapsible: boolean; isOpen: boolean; toggle: () => void };
const CardChromeContext = createContext<CardChrome>({ size: "base", stickyHeader: false, collapsible: false, isOpen: true, toggle: () => {} });

export type CardProps = {
  children?: ReactNode;
  variant?: CardVariant;
  size?: CardSize;
  stickyHeader?: boolean;
  collapsible?: boolean;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  style?: CSSProperties;
};
export function Card({ children, variant = "default", size = "base", stickyHeader = false, collapsible = false, defaultOpen = true, open: openProp, onOpenChange, style }: CardProps) {
  const { tokens } = useHostTheme();
  const controlled = openProp !== undefined;
  const [openState, setOpenState] = useState(defaultOpen);
  const isOpen = !collapsible || (controlled ? Boolean(openProp) : openState);
  const toggle = useCallback(() => {
    if (!collapsible) return;
    const next = !isOpen;
    if (!controlled) setOpenState(next);
    onOpenChange?.(next);
  }, [collapsible, isOpen, controlled, onOpenChange]);
  const shell: CSSProperties = {
    boxSizing: "border-box",
    background: tokens.bg.editor,
    overflow: "clip",
    ...(variant === "default" ? { border: `1px solid ${tokens.stroke.tertiary}`, borderRadius: `${canvasRadius.lg}px` } : { border: "none", borderRadius: 0 }),
  };
  return (
    <CardChromeContext.Provider value={{ size, stickyHeader, collapsible, isOpen, toggle }}>
      <div style={mergeStyle(shell, style)}>
        <div style={{ boxSizing: "border-box", position: "relative" }}>{children}</div>
      </div>
    </CardChromeContext.Provider>
  );
}

function compactTrailingPills(node: ReactNode): ReactNode {
  if (isValidElement(node) && node.type === Pill && (node as ReactElement<PillProps>).props.size !== "sm") {
    return cloneElement(node as ReactElement<PillProps>, { size: "sm" });
  }
  return node;
}

export type CardHeaderProps = { children?: ReactNode; trailing?: ReactNode; style?: CSSProperties };
export function CardHeader({ children, trailing, style }: CardHeaderProps) {
  const { tokens } = useHostTheme();
  const { size, stickyHeader, collapsible, isOpen, toggle } = useContext(CardChromeContext);
  const height = size === "lg" ? 32 : 28;
  const padX = size === "lg" ? canvasSpacing["2.5"] : canvasSpacing[2];
  const gap = size === "lg" ? canvasSpacing[2] : canvasSpacing["1.5"];
  const shell: CSSProperties = {
    boxSizing: "border-box",
    ...(stickyHeader ? { position: "sticky", top: 0, zIndex: 5, background: tokens.bg.editor } : { position: "relative" }),
    display: "flex",
    alignItems: "center",
    height: `${height}px`,
    fontSize: "12px",
    color: tokens.text.primary,
    borderBottom: collapsible && !isOpen ? "none" : `1px solid ${tokens.stroke.tertiary}`,
  };
  const title: CSSProperties = { boxSizing: "border-box", flex: 1, minWidth: 0, display: "flex", alignItems: "center", height: "100%", padding: `0 ${padX}px`, gap: `${gap}px`, overflow: "hidden" };
  const trail: CSSProperties = { display: "flex", alignItems: "center", gap: `${canvasSpacing["1.5"]}px`, paddingRight: `${padX}px`, flexShrink: 0, fontSize: "11px", color: tokens.text.secondary };
  const trailingNode = trailing != null ? compactTrailingPills(trailing) : null;
  if (collapsible) {
    return (
      <button type="button" onClick={toggle} aria-expanded={isOpen} style={mergeStyle({ all: "unset", ...shell, cursor: "pointer", width: "100%", font: "inherit", color: "inherit" }, style)}>
        <div style={title}>
          <CanvasChevron expanded={isOpen} />
          {children}
        </div>
        {trailingNode ? <div style={trail}>{trailingNode}</div> : null}
      </button>
    );
  }
  return (
    <div style={mergeStyle(shell, style)}>
      <div style={title}>{children}</div>
      {trailingNode ? <div style={trail}>{trailingNode}</div> : null}
    </div>
  );
}

export type CardBodyProps = { children?: ReactNode; style?: CSSProperties };
export function CardBody({ children, style }: CardBodyProps) {
  const { tokens } = useHostTheme();
  const { collapsible, isOpen } = useContext(CardChromeContext);
  if (collapsible && !isOpen) return null;
  return (
    <div style={mergeStyle({ boxSizing: "border-box", padding: `${canvasSpacing[3]}px`, fontSize: canvasTypography.small.fontSize, lineHeight: canvasTypography.small.lineHeight, color: tokens.text.secondary }, style)}>
      {children}
    </div>
  );
}

export type CollapsibleSectionProps = {
  title: string;
  leading?: ReactNode;
  count?: number;
  trailing?: ReactNode;
  children?: ReactNode;
  defaultOpen?: boolean;
  style?: CSSProperties;
};
export function CollapsibleSection({ title, leading, count, trailing, children, defaultOpen = false, style }: CollapsibleSectionProps) {
  const { tokens } = useHostTheme();
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const toggle = useCallback(() => setOpen((value) => !value), []);
  const button: CSSProperties = {
    boxSizing: "border-box",
    appearance: "none",
    border: "none",
    background: "transparent",
    margin: 0,
    padding: `${canvasSpacing[2]}px 0`,
    width: "100%",
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: `${canvasSpacing[2]}px`,
    cursor: "pointer",
    textAlign: "left",
    color: tokens.text.primary,
    font: "inherit",
    transition: "color 100ms ease",
  };
  const body = canvasTypography.body;
  return (
    <div style={mergeStyle({ boxSizing: "border-box", display: "flex", flexDirection: "column", minWidth: 0, width: "100%" }, style)}>
      <button type="button" onClick={toggle} aria-expanded={open} aria-controls={id} style={button}>
        <span style={{ color: tokens.text.quaternary, display: "inline-flex", flexShrink: 0 }}>
          <CanvasChevron expanded={open} />
        </span>
        {leading != null ? <span style={{ flexShrink: 0 }}>{leading}</span> : null}
        <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: tokens.text.primary, fontSize: body.fontSize, lineHeight: body.lineHeight, fontWeight: body.fontWeight }}>{title}</span>
        {count != null ? <span style={{ flexShrink: 0, color: tokens.text.secondary, fontVariantNumeric: "tabular-nums", fontSize: body.fontSize, lineHeight: body.lineHeight }}>{count}</span> : null}
        {trailing != null ? <span style={{ flexShrink: 0, display: "inline-flex", alignItems: "center", color: tokens.text.tertiary, fontVariantNumeric: "tabular-nums", fontSize: body.fontSize, lineHeight: body.lineHeight }}>{trailing}</span> : null}
      </button>
      {open ? (
        <div id={id} style={{ paddingLeft: `${canvasSpacing[5]}px`, paddingBottom: `${canvasSpacing[2]}px`, minWidth: 0 }}>
          {children}
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------------------------

export type CalloutTone = "info" | "success" | "warning" | "danger" | "neutral";
const calloutToneColors: Record<CalloutTone, string> = {
  info: chartPalette.lightBlue,
  success: chartPalette.lightGreen,
  warning: chartPalette.brightOrange,
  danger: chartPalette.darkAmber,
  neutral: chartPalette.muted,
};

/** Same glyphs as the SDK (300-unit viewBox, y-flipped); `neutral` is a filled dot. */
const CALLOUT_ICON_PATHS: Record<Exclude<CalloutTone, "neutral">, string> = {
  info: "M150 293Q189 293 222.0 274.0Q255 255 274.0 222.0Q293 189 293.0 150.0Q293 111 274.0 78.0Q255 45 222.0 26.0Q189 7 150.0 7.0Q111 7 78.0 26.0Q45 45 26.0 78.0Q7 111 7.0 150.0Q7 189 26.0 222.0Q45 255 78.0 274.0Q111 293 150 293ZM150 270Q118 270 90.5 253.5Q63 237 46.5 209.5Q30 182 30.0 150.0Q30 118 46.5 90.5Q63 63 90.5 46.5Q118 30 150.0 30.0Q182 30 209.5 46.5Q237 63 253.5 90.5Q270 118 270.0 150.0Q270 182 253.5 209.5Q237 237 209.5 253.5Q182 270 150 270ZM150 152Q155 152 158.5 148.5Q162 145 162 141V94Q162 89 158.5 85.5Q155 82 150.0 82.0Q145 82 141.5 85.5Q138 89 138 94V141Q138 145 141.5 148.5Q145 152 150 152ZM150 216Q158 216 163.5 210.5Q169 205 169.0 197.0Q169 189 163.5 183.5Q158 178 150.0 178.0Q142 178 136.5 183.5Q131 189 131.0 197.0Q131 205 136.5 210.5Q142 216 150 216Z",
  warning: "M116 268Q123 281 136.5 286.0Q150 291 163.5 286.0Q177 281 185 268L290 86Q298 73 295.0 59.0Q292 45 281.0 35.5Q270 26 256 26H44Q30 26 19.0 35.5Q8 45 5.0 59.0Q2 73 10 86ZM164 257Q159 265 150.0 265.0Q141 265 136 257L30 74Q26 66 30.5 57.5Q35 49 44 49H256Q265 49 269.5 57.5Q274 66 270 74ZM150 113Q158 113 163.5 107.5Q169 102 169.0 94.0Q169 86 163.5 80.5Q158 75 150.0 75.0Q142 75 136.5 80.5Q131 86 131.0 94.0Q131 102 136.5 107.5Q142 113 150 113ZM150 209Q155 209 158.5 205.5Q162 202 162 197V150Q162 145 158.5 141.5Q155 138 150.0 138.0Q145 138 141.5 141.5Q138 145 138 150V197Q138 202 141.5 205.5Q145 209 150 209Z",
  success: "M117 227Q147 227 172.5 212.5Q198 198 212.5 172.5Q227 147 227.0 117.0Q227 87 212.5 62.0Q198 37 172.5 22.0Q147 7 117.0 7.0Q87 7 62.0 22.0Q37 37 22.0 62.0Q7 87 7.0 117.0Q7 147 22.0 172.5Q37 198 62.0 212.5Q87 227 117 227ZM117 204Q94 204 74.0 192.5Q54 181 42.0 161.0Q30 141 30.0 117.5Q30 94 42.0 74.0Q54 54 74.0 42.0Q94 30 117.5 30.0Q141 30 161.0 42.0Q181 54 192.5 74.0Q204 94 204.0 117.5Q204 141 192.5 161.0Q181 181 161.0 192.5Q141 204 117 204ZM146 144Q150 148 155.0 148.0Q160 148 163.0 144.5Q166 141 166.0 136.0Q166 131 163 128L123 87Q117 81 108.0 81.0Q99 81 93 87L71 109Q68 112 68.0 117.0Q68 122 71.5 125.5Q75 129 80.0 129.0Q85 129 88 125L108 106ZM128 278Q160 297 197.0 292.0Q234 287 260.5 260.5Q287 234 292.0 197.0Q297 160 278 128Q276 124 271.0 122.5Q266 121 262.0 123.5Q258 126 256.5 130.5Q255 135 258 139Q273 165 269.0 194.0Q265 223 244.0 244.0Q223 265 194.0 269.0Q165 273 139 258Q135 255 130.5 256.5Q126 258 123.5 262.0Q121 266 122.5 271.0Q124 276 128 278Z",
  danger: "M150 293Q189 293 222.0 274.0Q255 255 274.0 222.0Q293 189 293.0 150.0Q293 111 274.0 78.0Q255 45 222.0 26.0Q189 7 150.0 7.0Q111 7 78.0 26.0Q45 45 26.0 78.0Q7 111 7.0 150.0Q7 189 26.0 222.0Q45 255 78.0 274.0Q111 293 150 293ZM150 270Q118 270 90.5 253.5Q63 237 46.5 209.5Q30 182 30.0 150.0Q30 118 46.5 90.5Q63 63 90.5 46.5Q118 30 150.0 30.0Q182 30 209.5 46.5Q237 63 253.5 90.5Q270 118 270.0 150.0Q270 182 253.5 209.5Q237 237 209.5 253.5Q182 270 150 270ZM150 113Q158 113 163.5 107.5Q169 102 169.0 94.0Q169 86 163.5 80.5Q158 75 150.0 75.0Q142 75 136.5 80.5Q131 86 131.0 94.0Q131 102 136.5 107.5Q142 113 150 113ZM150 227Q155 227 158.5 223.5Q162 220 162 216V150Q162 145 158.5 141.5Q155 138 150.0 138.0Q145 138 141.5 141.5Q138 145 138 150V216Q138 220 141.5 223.5Q145 227 150 227Z",
};

function CalloutToneIcon({ tone, color }: { tone: CalloutTone; color: string }) {
  return (
    <svg width={12} height={12} viewBox="0 0 300 300" aria-hidden="true" style={{ display: "block", flexShrink: 0, color }}>
      {tone === "neutral" ? <circle cx={150} cy={150} r={132} fill="currentColor" /> : <g transform="matrix(1 0 0 -1 0 300)"><path fill="currentColor" d={CALLOUT_ICON_PATHS[tone]} /></g>}
    </svg>
  );
}

export type CalloutProps = { children?: ReactNode; tone?: CalloutTone; title?: ReactNode; icon?: ReactNode; style?: CSSProperties };
export function Callout({ children, tone = "info", title, icon, style }: CalloutProps) {
  const { tokens } = useHostTheme();
  const color = calloutToneColors[tone];
  const lineHeight = canvasTypography.body.lineHeight;
  const iconBox: CSSProperties = { display: "inline-flex", alignItems: "center", justifyContent: "center", flexShrink: 0, height: lineHeight, color };
  return (
    <div role="note" style={mergeStyle({ boxSizing: "border-box", display: "flex", alignItems: "flex-start", gap: `${canvasSpacing["2.5"]}px`, width: "100%", color: tokens.text.primary, fontSize: canvasTypography.body.fontSize, fontWeight: canvasTypography.body.fontWeight, lineHeight }, style)}>
      {icon != null ? (
        <span style={{ ...iconBox, minWidth: canvasSpacing[3] }}>{icon}</span>
      ) : (
        <span style={{ ...iconBox, width: canvasSpacing[3] }} aria-hidden>
          <CalloutToneIcon tone={tone} color={color} />
        </span>
      )}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: `${canvasSpacing[1]}px` }}>
        {title != null ? <div style={{ color: tokens.text.primary, fontWeight: 500, fontSize: canvasTypography.body.fontSize, lineHeight }}>{title}</div> : null}
        {children != null ? <div style={{ color: tokens.text.primary, fontSize: canvasTypography.body.fontSize, fontWeight: canvasTypography.body.fontWeight, lineHeight }}>{children}</div> : null}
      </div>
    </div>
  );
}

export type StatTone = "success" | "danger" | "warning" | "info";
const statToneColors: Record<StatTone, string> = {
  success: chartPalette.lightGreen,
  danger: chartPalette.darkAmber,
  warning: chartPalette.brightOrange,
  info: chartPalette.lightBlue,
};
export type StatProps = { value: ReactNode; label: string; tone?: StatTone; style?: CSSProperties };
export function Stat({ value, label, tone, style }: StatProps) {
  const { tokens } = useHostTheme();
  return (
    <div style={mergeStyle({ display: "flex", flexDirection: "column", alignItems: "center", gap: `${canvasSpacing["0.5"]}px`, padding: `${canvasSpacing[3]}px ${canvasSpacing[2]}px` }, style)}>
      <div style={{ fontSize: "24px", lineHeight: "28px", fontWeight: 600, fontVariantNumeric: "tabular-nums", color: tone ? statToneColors[tone] : tokens.text.primary }}>{value}</div>
      <div style={{ fontSize: canvasTypography.small.fontSize, lineHeight: canvasTypography.small.lineHeight, color: tokens.text.secondary }}>{label}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------------------------

export type PillTone = "neutral" | "added" | "deleted" | "renamed" | "success" | "warning" | "info";
export type PillSize = "sm" | "md";
export type PillProps = {
  children?: ReactNode;
  active?: boolean;
  tone?: PillTone;
  size?: PillSize;
  leadingContent?: ReactNode;
  keyboardHint?: string;
  disabled?: boolean;
  title?: string;
  style?: CSSProperties;
  onClick?: () => void;
};
export function Pill({ children, active = false, size = "md", leadingContent, keyboardHint, disabled = false, title, style, onClick }: PillProps) {
  const { tokens } = useHostTheme();
  const interactive = Boolean(onClick);
  const small = size === "sm";
  const color = active ? tokens.text.primary : tokens.text.secondary;
  const stroke = active ? "transparent" : tokens.stroke.secondary;
  const fill = active ? tokens.fill.secondary : "transparent";
  const base: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    boxSizing: "border-box",
    borderRadius: `${canvasRadius.full}px`,
    whiteSpace: "nowrap",
    userSelect: "none",
    fontFamily: "inherit",
    fontWeight: active ? 500 : 400,
    fontSize: small ? "10px" : "12px",
    lineHeight: small ? "12px" : "14px",
    background: small ? tokens.fill.quaternary : fill,
    color,
    border: small ? "none" : `1px solid ${stroke}`,
    padding: small ? `${canvasSpacing["0.5"]}px ${canvasSpacing["1.5"]}px` : `${canvasSpacing["1.5"]}px ${canvasSpacing["2.5"]}px`,
    gap: small ? `${canvasSpacing[1]}px` : `${canvasSpacing["1.5"]}px`,
    cursor: interactive ? (disabled ? "not-allowed" : "pointer") : "default",
    opacity: disabled ? 0.3 : 1,
    transition: interactive ? "color 120ms ease, background 120ms ease" : undefined,
  };
  const content = (
    <>
      {leadingContent ? <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "inherit" }}>{leadingContent}</span> : null}
      <span style={{ flexShrink: 0, color: "inherit" }}>{children}</span>
      {keyboardHint ? <span style={{ flexShrink: 0, color: tokens.text.primary, opacity: 0.3 }}>{keyboardHint}</span> : null}
    </>
  );
  if (interactive) {
    return (
      <button type="button" disabled={disabled} title={title} onClick={disabled ? undefined : onClick} style={mergeStyle({ ...base, margin: 0 }, style)}>
        {content}
      </button>
    );
  }
  return (
    <span title={title} style={mergeStyle(base, style)}>
      {content}
    </span>
  );
}

export type ButtonProps = {
  children?: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
  style?: CSSProperties;
  onClick?: () => void;
};
export function Button({ children, variant = "secondary", disabled = false, type = "button", style, onClick }: ButtonProps) {
  const { tokens } = useHostTheme();
  const palette: CSSProperties =
    variant === "primary"
      ? { background: tokens.accent.control, color: tokens.text.onAccent, border: "1px solid transparent" }
      : variant === "ghost"
        ? { background: "transparent", color: tokens.text.secondary, border: "1px solid transparent" }
        : { background: tokens.fill.tertiary, color: tokens.text.primary, border: `1px solid ${tokens.stroke.secondary}` };
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={disabled ? undefined : onClick}
      style={mergeStyle(
        { ...palette, display: "inline-flex", alignItems: "center", height: 24, padding: "0 10px", borderRadius: `${canvasRadius.md}px`, fontSize: "12px", lineHeight: "16px", fontFamily: "inherit", cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.5 : 1 },
        style,
      )}
    >
      {children}
    </button>
  );
}
