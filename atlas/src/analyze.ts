// Copyright 2026 The Sphota Authors. SPDX-License-Identifier: Apache-2.0
// "Analyze a file": parse a CSV/TSV/TXT/JSONL of messages, pick the text column, and summarise
// the encoded rows by theme (first code block, L1) or by region. Everything runs in the browser.

export interface FileRow {
  text: string;
  lang?: string;
  script: string;
  cell: number;
  cellP: number;
  group: number; // per-file refined group (D-041), -1 before refinement
  region: number;
  regionP: number;
  code: string;
}

export type GroupBy = "group" | "cell" | "region";

export interface GroupSummary {
  k: number;
  count: number;
  share: number;
  uncertain: number;
  samples: string[];
  scripts: [string, number][];
  langs: [string, number][];
}

const TEXT_COLUMNS = ["text", "message", "body", "content", "comment", "feedback", "review", "description",
  "question", "query", "ticket", "title", "sentence"];
const LANG_COLUMNS = ["lang", "language", "locale"];

/** RFC 4180-style CSV parser (quotes, escaped quotes, CR/LF). Delimiter auto-detected. */
export function parseDelimited(src: string, delim?: string): string[][] {
  const text = src.replace(/^﻿/, "");
  if (!delim) {
    const head = text.slice(0, 2000).split(/\r?\n/)[0] ?? "";
    const counts: [string, number][] = [",", "\t", ";", "|"].map((d) => [d, head.split(d).length - 1]);
    counts.sort((a, b) => b[1] - a[1]);
    delim = counts[0][1] > 0 ? counts[0][0] : ",";
  }
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQ = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQ) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false;
      } else field += ch;
    } else if (ch === '"' && field === "") inQ = true;
    else if (ch === delim) { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((c) => c.trim() !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((c) => c.trim() !== "")) rows.push(row);
  return rows;
}

export interface ParsedFile { texts: string[]; langs: (string | undefined)[]; column: string; columns: string[]; total: number }

/** Turn a file's text into messages. `column` overrides the auto-detected text column. */
export function readMessages(name: string, src: string, column?: string): ParsedFile {
  const lower = name.toLowerCase();
  if (lower.endsWith(".txt") || lower.endsWith(".md")) {
    const texts = src.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    return { texts, langs: texts.map(() => undefined), column: "(one message per line)", columns: [], total: texts.length };
  }
  if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) {
    const objs = src.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l) as Record<string, unknown>);
    const columns = Object.keys(objs[0] ?? {});
    const col = column ?? pickColumn(columns, objs.map((o) => columns.map((c) => String(o[c] ?? ""))));
    const lc = columns.find((c) => LANG_COLUMNS.includes(c.toLowerCase()));
    const texts = objs.map((o) => String(o[col] ?? "").trim());
    const langs = objs.map((o) => (lc ? String(o[lc] ?? "") || undefined : undefined));
    return filterEmpty({ texts, langs, column: col, columns, total: texts.length });
  }
  const rows = parseDelimited(src, lower.endsWith(".tsv") ? "\t" : undefined);
  if (rows.length < 2) throw new Error("The file needs a header row and at least one message.");
  const columns = rows[0].map((c) => c.trim());
  const body = rows.slice(1);
  const col = column ?? pickColumn(columns, body);
  const ci = columns.indexOf(col);
  const li = columns.findIndex((c) => LANG_COLUMNS.includes(c.toLowerCase()));
  const texts = body.map((r) => (r[ci] ?? "").trim());
  const langs = body.map((r) => (li >= 0 ? (r[li] ?? "").trim() || undefined : undefined));
  return filterEmpty({ texts, langs, column: col, columns, total: texts.length });
}

function filterEmpty(p: ParsedFile): ParsedFile {
  const keep = p.texts.map((t, i) => (t.length > 1 ? i : -1)).filter((i) => i >= 0);
  return { ...p, texts: keep.map((i) => p.texts[i]), langs: keep.map((i) => p.langs[i]), total: keep.length };
}

/** A column named like text, else the column with the longest average values. */
export function pickColumn(columns: string[], rows: string[][]): string {
  const named = columns.find((c) => TEXT_COLUMNS.includes(c.toLowerCase()));
  if (named) return named;
  let best = 0, bestLen = -1;
  columns.forEach((_, i) => {
    const sample = rows.slice(0, 200);
    const avg = sample.reduce((s, r) => s + (r[i] ?? "").length, 0) / Math.max(1, sample.length);
    if (avg > bestLen) { bestLen = avg; best = i; }
  });
  return columns[best];
}

const SCRIPTS: [string, RegExp][] = [
  ["Devanagari", /[ऀ-ॿ]/], ["Bengali", /[ঀ-৿]/], ["Gurmukhi", /[਀-੿]/],
  ["Gujarati", /[઀-૿]/], ["Odia", /[଀-୿]/], ["Tamil", /[஀-௿]/],
  ["Telugu", /[ఀ-౿]/], ["Kannada", /[ಀ-೿]/], ["Malayalam", /[ഀ-ൿ]/],
  ["Arabic", /[؀-ۿ]/], ["Cyrillic", /[Ѐ-ӿ]/], ["Han/Kana", /[぀-ヿ一-鿿]/],
  ["Hangul", /[가-힯]/], ["Thai", /[฀-๿]/], ["Latin", /[A-Za-zÀ-ž]/],
];

/** The dominant writing system of a message (cheap, no model). */
export function scriptOf(text: string): string {
  let best = "Other", bestN = 0;
  for (const [name, re] of SCRIPTS) {
    const n = (text.match(new RegExp(re.source, "g")) ?? []).length;
    if (n > bestN) { bestN = n; best = name; }
  }
  return best;
}

function top<T>(xs: T[], n: number): [T, number][] {
  const m = new Map<T, number>();
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, n);
}

/**
 * Group rows by theme (L1 cell: the first code block) or by region. Samples are the group's most
 * confident messages, deduplicated; the first one names the group in the report.
 */
export function summarize(rows: FileRow[], by: GroupBy = "cell", minConfident = 0.5): GroupSummary[] {
  const key = (r: FileRow) => (by === "group" ? r.group : by === "cell" ? r.cell : r.region);
  const conf = (r: FileRow) => (by === "region" ? r.regionP : r.cellP);
  const groups = new Map<number, FileRow[]>();
  for (const r of rows) {
    const k = key(r);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(r);
  }
  const out: GroupSummary[] = [];
  for (const [k, rs] of groups) {
    const pool = rs.slice().sort((a, b) => conf(b) - conf(a));
    const seen = new Set<string>();
    const samples: string[] = [];
    for (const r of pool) {
      const s = r.text.toLowerCase().slice(0, 40);
      if (seen.has(s)) continue;
      seen.add(s);
      samples.push(r.text);
      if (samples.length === 3) break;
    }
    out.push({
      k, count: rs.length, share: rs.length / rows.length,
      uncertain: rs.filter((r) => conf(r) < minConfident).length,
      samples, scripts: top(rs.map((r) => r.script), 3),
      langs: top(rs.map((r) => r.lang).filter((l): l is string => !!l), 4),
    });
  }
  return out.sort((a, b) => b.count - a.count);
}

// Spreadsheet formula injection (OWASP): a message like "=HYPERLINK(...)" must not run when the CSV
// is opened in Excel/Sheets, so cells starting with = + - @ tab or CR get a leading apostrophe.
const defuse = (s: string) => (/^[=+\-@\t\r]/.test(s) ? "'" + s : s);
const q = (raw: string) => {
  const s = defuse(raw);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Results as CSV: the original text plus theme (first code block), region and full code. */
export function toCsv(rows: FileRow[], regionName: (k: number) => string, groupIds: string[] = [],
                      groupNames: string[] = []): string {
  const head = ["text", "group", "group_name", "theme", "theme_confidence", "region", "region_name",
    "region_confidence", "sphota_code", "script"];
  const lines = rows.map((r) => [r.text, groupIds[r.group] ?? "", groupNames[r.group] ?? "", r.code.slice(0, 4),
    r.cellP.toFixed(3), `R${String(r.region).padStart(2, "0")}`, regionName(r.region), r.regionP.toFixed(3),
    r.code, r.script].map(q).join(","));
  return "﻿" + [head.join(","), ...lines].join("\r\n");
}
