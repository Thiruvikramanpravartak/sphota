// Copyright 2026 The Sphota Authors. SPDX-License-Identifier: Apache-2.0
// Per-file refinement in the browser: a direct port of sphota.refine (kmeans, k = ceil(2*sqrt n),
// language centring) and sphota.langid (anchor-space language identification), D-041.
// Same algorithm, same mulberry32 seeding and canonical row order as the Python library.

export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// writing systems, named like Python's unicodedata (first word of the letter's name)
const SCRIPTS: [string, RegExp][] = [
  ["LATIN", /\p{Script=Latin}/u], ["CYRILLIC", /\p{Script=Cyrillic}/u], ["ARABIC", /\p{Script=Arabic}/u],
  ["DEVANAGARI", /\p{Script=Devanagari}/u], ["BENGALI", /\p{Script=Bengali}/u],
  ["CJK", /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u], ["HANGUL", /\p{Script=Hangul}/u],
  ["GREEK", /\p{Script=Greek}/u], ["HEBREW", /\p{Script=Hebrew}/u], ["TAMIL", /\p{Script=Tamil}/u],
  ["TELUGU", /\p{Script=Telugu}/u], ["KANNADA", /\p{Script=Kannada}/u], ["MALAYALAM", /\p{Script=Malayalam}/u],
  ["GUJARATI", /\p{Script=Gujarati}/u], ["GURMUKHI", /\p{Script=Gurmukhi}/u], ["ORIYA", /\p{Script=Oriya}/u],
  ["THAI", /\p{Script=Thai}/u], ["LAO", /\p{Script=Lao}/u], ["MYANMAR", /\p{Script=Myanmar}/u],
  ["KHMER", /\p{Script=Khmer}/u], ["GEORGIAN", /\p{Script=Georgian}/u], ["ARMENIAN", /\p{Script=Armenian}/u],
  ["ETHIOPIC", /\p{Script=Ethiopic}/u], ["SINHALA", /\p{Script=Sinhala}/u], ["TIBETAN", /\p{Script=Tibetan}/u],
  ["THAANA", /\p{Script=Thaana}/u], ["OL", /\p{Script=Ol_Chiki}/u], ["MEETEI", /\p{Script=Meetei_Mayek}/u],
];
const SHARED = new Set(["LATIN", "CYRILLIC", "ARABIC", "DEVANAGARI", "BENGALI", "CJK"]);
const LETTER = /\p{L}/u;

export function scriptName(text: string): string {
  const counts = new Map<string, number>();
  for (const ch of text) {
    if (!LETTER.test(ch)) continue;
    let name = "OTHER";
    for (const [n, re] of SCRIPTS) if (re.test(ch)) { name = n; break; }
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  let best = "NONE", bc = 0;
  for (const [n, c] of counts) if (c > bc) { best = n; bc = c; }
  return best;
}

export class LangId {
  constructor(public W: Float32Array, public b: Float32Array, public classes: string[],
              public classScripts: Record<string, string[]>, public dim: number) {}

  static async load(base: string): Promise<LangId> {
    const [meta, buf] = await Promise.all([
      fetch(`${base}/langid.json`).then((r) => r.json()),
      fetch(`${base}/langid.bin`).then((r) => r.arrayBuffer()),
    ]);
    const C = meta.classes.length;
    const W = new Float32Array(buf, 0, meta.dim * C);
    const b = new Float32Array(buf, meta.dim * C * 4, C);
    return new LangId(W, b, meta.classes, meta.class_scripts, meta.dim);
  }

  /** Most likely language for a unit-norm raw anchor vector, restricted to the text's script. */
  predict(v: Float32Array, script: string): string {
    const C = this.classes.length;
    let best = -Infinity, arg = 0;
    for (let c = 0; c < C; c++) {
      const ok = this.classScripts[this.classes[c]];
      if (ok && ok.length && !ok.includes(script)) continue;
      let s = this.b[c];
      for (let i = 0; i < this.dim; i++) s += v[i] * this.W[i * C + c];
      if (s > best) { best = s; arg = c; }
    }
    return this.classes[arg];
  }
}

export function languageKey(text: string, v: Float32Array, lid: LangId | null): string {
  const s = scriptName(text);
  return SHARED.has(s) && lid ? `${s}:${lid.predict(v, s)}` : s;
}

export interface Refined { labels: Int32Array; sizes: number[]; representative: number[]; parent: number[] }

function centre(V: Float32Array[], keys: string[], minRows = 30): Float32Array[] {
  const by = new Map<string, number[]>();
  keys.forEach((k, i) => { if (!by.has(k)) by.set(k, []); by.get(k)!.push(i); });
  const big = [...by.values()].filter((ix) => ix.length >= minRows);
  if (big.length < 2) return V;
  const d = V[0].length, mu = new Float64Array(d);
  for (const v of V) for (let j = 0; j < d; j++) mu[j] += v[j] / V.length;
  const out = V.map((v) => Float32Array.from(v));
  for (const ix of big) {
    const m = new Float64Array(d);
    for (const i of ix) for (let j = 0; j < d; j++) m[j] += V[i][j] / ix.length;
    for (const i of ix) for (let j = 0; j < d; j++) out[i][j] += mu[j] - m[j];
  }
  return out;
}

function canonicalOrder(V: Float32Array[]): number[] {
  const r = (x: number) => Math.round(x * 1e4) / 1e4;
  const m = Math.min(8, V[0].length);
  return V.map((_, i) => i).sort((a, b) => {
    for (let j = 0; j < m; j++) { const d = r(V[a][j]) - r(V[b][j]); if (d) return d; }
    return 0;
  });
}

function kmeans(X: Float32Array[], k: number, seed = 0, iters = 15): Int32Array {
  const n = X.length, d = X[0].length;
  k = Math.max(1, Math.min(k, n));
  const rng = mulberry32(seed);
  const xn = X.map((x) => { let s = 0; for (let j = 0; j < d; j++) s += x[j] * x[j]; return s; });
  const dot = (a: Float32Array, b: Float32Array) => { let s = 0; for (let j = 0; j < d; j++) s += a[j] * b[j]; return s; };
  const centres = [Math.min(Math.floor(rng() * n), n - 1)];
  const d2 = X.map((x, i) => Math.max(xn[i] + xn[centres[0]] - 2 * dot(x, X[centres[0]]), 0));
  for (let c = 1; c < k; c++) {
    let total = 0; for (const v of d2) total += v;
    let pick: number;
    if (total > 0) {
      const u = rng() * total; let acc = 0; pick = n - 1;
      for (let i = 0; i < n; i++) { acc += d2[i]; if (acc > u) { pick = i; break; } }
    } else pick = Math.min(Math.floor(rng() * n), n - 1);
    centres.push(pick);
    for (let i = 0; i < n; i++) d2[i] = Math.min(d2[i], Math.max(xn[i] + xn[pick] - 2 * dot(X[i], X[pick]), 0));
  }
  const C = centres.map((i) => Float32Array.from(X[i]));
  let lab = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    const cn = C.map((c) => { let s = 0; for (let j = 0; j < d; j++) s += c[j] * c[j]; return s; });
    const nw = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let best = Infinity, arg = 0;
      for (let c = 0; c < k; c++) { const s = cn[c] - 2 * dot(X[i], C[c]); if (s < best) { best = s; arg = c; } }
      nw[i] = arg;
    }
    if (it && nw.every((v, i) => v === lab[i])) break;
    lab = nw;
    const sums = C.map(() => new Float64Array(d)), cnt = new Int32Array(k);
    for (let i = 0; i < n; i++) { cnt[lab[i]]++; const s = sums[lab[i]]; for (let j = 0; j < d; j++) s[j] += X[i][j]; }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let j = 0; j < d; j++) C[c][j] = sums[c][j] / cnt[c];
  }
  return lab;
}

/** Refine one file: raw anchor vectors (unit norm), language keys, first-level cells. */
export function refineFile(raw: Float32Array[], keys: string[], l1: number[]): Refined {
  const n = raw.length;
  if (!n) return { labels: new Int32Array(0), sizes: [], representative: [], parent: [] };
  const V = centre(raw, keys);
  const order = canonicalOrder(V);
  const Vo = order.map((i) => V[i]);
  const lab = kmeans(Vo, Math.ceil(2 * Math.sqrt(n)));
  const groups = new Map<number, number[]>();
  lab.forEach((g, i) => { if (!groups.has(g)) groups.set(g, []); groups.get(g)!.push(i); });
  const list = [...groups.values()].sort((a, b) => b.length - a.length || a[0] - b[0]);
  const labels = new Int32Array(n), sizes: number[] = [], representative: number[] = [], parent: number[] = [];
  const d = Vo[0].length;
  list.forEach((g, gi) => {
    for (const i of g) labels[order[i]] = gi;
    const c = new Float64Array(d);
    for (const i of g) for (let j = 0; j < d; j++) c[j] += Vo[i][j] / g.length;
    let best = Infinity, rep = g[0];
    for (const i of g) { let s = 0; for (let j = 0; j < d; j++) { const t = Vo[i][j] - c[j]; s += t * t; } if (s < best) { best = s; rep = i; } }
    representative.push(order[rep]);
    const cnt = new Map<number, number>();
    for (const i of g) cnt.set(l1[order[i]], (cnt.get(l1[order[i]]) ?? 0) + 1);
    let pv = Infinity, pc = -1;
    for (const [v, c2] of cnt) if (c2 > pc || (c2 === pc && v < pv)) { pv = v; pc = c2; }
    parent.push(pv);
    sizes.push(g.length);
  });
  return { labels, sizes, representative, parent };
}
