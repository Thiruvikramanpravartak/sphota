// Copyright 2026 The Sphota Authors. SPDX-License-Identifier: Apache-2.0
// In-browser Sphota encoder: transformers.js (multilingual-e5-small, q8 ONNX) + the frozen
// s1 projection and codebooks. Mirrors sphota.space.quantize (default mode) exactly.

export interface SpaceMeta {
  space: string;
  dim_in: number;
  dim: number;
  sizes: number[];
  offsets: { mu: number; W: number; codebooks: number[] };
  temperatures: number[];
  top_m: number;
  widths: number[];
  perm_a: number[];
  perm_b: number[];
  browser_model: string;
}

export interface Encoded {
  text: string;
  levels: number[];
  levelConf: number[];
  conf: number[]; // cumulative
  l1Candidates: { index: number; p: number }[];
  code: string;
}

const ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

export function formatLevels(levels: number[], meta: SpaceMeta): string {
  return levels
    .map((i, lvl) => {
      const bits = 5 * meta.widths[lvl];
      const m = 2 ** bits;
      let s = Number((BigInt(meta.perm_a[lvl]) * BigInt(i) + BigInt(meta.perm_b[lvl])) % BigInt(m));
      let out = "";
      for (let k = 0; k < meta.widths[lvl]; k++) {
        out = ALPHABET[s & 31] + out;
        s = Math.floor(s / 32);
      }
      return out;
    })
    .join("-");
}

function modInverse(a: bigint, m: bigint): bigint {
  let [oldR, r] = [a, m];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return ((oldS % m) + m) % m;
}

/** Parse "7F3A-C2" style codes into level indices (null if invalid). */
export function parseCode(text: string, meta: SpaceMeta): number[] | null {
  const raw = text.trim().toUpperCase().replace(/^S1:/, "").split(".")[0].replace(/O/g, "0").replace(/[IL]/g, "1");
  const segs = raw.split("-").filter(Boolean);
  if (!segs.length || segs.length > 4) return null;
  const out: number[] = [];
  for (let lvl = 0; lvl < segs.length; lvl++) {
    const seg = segs[lvl];
    if (seg.length !== meta.widths[lvl]) return null;
    let s = 0n;
    for (const ch of seg) {
      const v = ALPHABET.indexOf(ch);
      if (v < 0) return null;
      s = s * 32n + BigInt(v);
    }
    const m = 2n ** BigInt(5 * meta.widths[lvl]);
    const inv = modInverse(BigInt(meta.perm_a[lvl]), m);
    const i = Number((((s - BigInt(meta.perm_b[lvl])) % m + m) % m * inv) % m);
    if (i >= meta.sizes[lvl]) return null;
    out.push(i);
  }
  return out;
}

export class SphotaSpace {
  meta: SpaceMeta;
  mu: Float32Array;
  W: Float32Array;
  C: Float32Array[];
  cnorm: Float64Array[];

  constructor(meta: SpaceMeta, buf: ArrayBuffer) {
    this.meta = meta;
    const { dim_in, dim, sizes, offsets } = meta;
    this.mu = new Float32Array(buf, offsets.mu, dim_in);
    this.W = new Float32Array(buf, offsets.W, dim_in * dim);
    this.C = sizes.map((k, l) => new Float32Array(buf, offsets.codebooks[l], k * dim));
    this.cnorm = this.C.map((c, l) => {
      const out = new Float64Array(sizes[l]);
      for (let k = 0; k < sizes[l]; k++) {
        let s = 0;
        for (let d = 0; d < dim; d++) s += c[k * dim + d] * c[k * dim + d];
        out[k] = s;
      }
      return out;
    });
  }

  static async load(base: string): Promise<SphotaSpace> {
    const [meta, buf] = await Promise.all([
      fetch(`${base}/space.json`).then((r) => r.json()),
      fetch(`${base}/space.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new SphotaSpace(meta, buf);
  }

  project(x: Float32Array): Float64Array {
    const { dim_in, dim } = this.meta;
    let n = 0;
    for (let i = 0; i < dim_in; i++) n += x[i] * x[i];
    n = Math.sqrt(n) || 1;
    const y = new Float64Array(dim);
    for (let i = 0; i < dim_in; i++) {
      const xi = x[i] / n - this.mu[i];
      const row = i * dim;
      for (let d = 0; d < dim; d++) y[d] += xi * this.W[row + d];
    }
    return y;
  }

  quantize(y: Float64Array, text = ""): Encoded {
    const { dim, sizes, temperatures, top_m } = this.meta;
    const r = Float64Array.from(y);
    const levels: number[] = [];
    const levelConf: number[] = [];
    let l1Candidates: { index: number; p: number }[] = [];
    for (let lvl = 0; lvl < sizes.length; lvl++) {
      const C = this.C[lvl];
      let rn = 0;
      for (let d = 0; d < dim; d++) rn += r[d] * r[d];
      const d2 = new Float64Array(sizes[lvl]);
      for (let k = 0; k < sizes[lvl]; k++) {
        let dot = 0;
        const off = k * dim;
        for (let d = 0; d < dim; d++) dot += r[d] * C[off + d];
        d2[k] = this.cnorm[lvl][k] - 2 * dot + rn;
      }
      const idx = Array.from(d2.keys()).sort((a, b) => d2[a] - d2[b] || a - b).slice(0, top_m);
      const T = temperatures[lvl];
      const w = idx.map((k) => Math.exp(-(d2[k] - d2[idx[0]]) / T));
      const z = w.reduce((a, b) => a + b, 0);
      const p = w.map((v) => v / z);
      levels.push(idx[0]);
      levelConf.push(p[0]);
      if (lvl === 0) l1Candidates = idx.slice(0, 5).map((k, j) => ({ index: k, p: p[j] }));
      const off = idx[0] * dim;
      for (let d = 0; d < dim; d++) r[d] -= C[off + d];
    }
    const conf: number[] = [];
    levelConf.reduce((acc, v) => {
      const c = acc * v;
      conf.push(c);
      return c;
    }, 1);
    return { text, levels, levelConf, conf, l1Candidates, code: formatLevels(levels, this.meta) };
  }
}

type Extractor = (text: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>;
let extractorPromise: Promise<Extractor> | null = null;

export function loadModel(model: string, onProgress?: (msg: string) => void): Promise<Extractor> {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      env.allowLocalModels = false;
      const files = new Map<string, number>();
      const ex = await pipeline("feature-extraction", model, {
        dtype: "q8",
        progress_callback: (p: { status: string; file?: string; progress?: number }) => {
          if (p.file && typeof p.progress === "number") {
            files.set(p.file, p.progress);
            const pct = Math.round([...files.values()].reduce((a, b) => a + b, 0) / files.size);
            onProgress?.(`Loading the encoder in your browser… ${pct}%`);
          }
        },
      });
      return ex as unknown as Extractor;
    })();
  }
  return extractorPromise;
}

export async function embed(model: string, text: string): Promise<Float32Array> {
  const ex = await loadModel(model);
  const out = await ex("query: " + text, { pooling: "mean", normalize: true });
  return out.data;
}
