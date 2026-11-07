// Copyright 2026 The Sphota Authors. SPDX-License-Identifier: Apache-2.0
// Meaning Atlas: a zoomable map of the Sphota s1 space, coloured by its 32 regions.
import "./style.css";
import { SphotaSpace, embed, embedBatch, formatLevels, loadModel, parseCode, precision, type Encoded } from "./encoder";
import { readMessages, scriptOf, summarize, toCsv, type FileRow, type GroupSummary, type GroupBy } from "./analyze";

interface Cell { i: number; code: string; x: number; y: number; n: number; r: number; l: Record<string, string>; ex: string[]; nb: number[] }
interface Region { k: number; id: string; x: number; y: number; color: string; name: string; full: string; members: number; spread: number }
interface L2 { code: string; p: number; x: number; y: number; l: Record<string, string> }
interface MapData { space: string; cells: Cell[]; regions: Region[]; l2: L2[] }
interface RegionModel { temperature: number; dim: number; centroids: number[]; labels: Record<string, string>; labels_i18n?: Record<string, Record<string, string>> }
interface Marker { enc: Encoded; region: { k: number; p: number }; x: number; y: number; radius: number; color: string; born: number }

const DATA = `${import.meta.env.BASE_URL}data`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("map");
const ctx = canvas.getContext("2d")!;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const FONT = "Inter, system-ui, sans-serif";
const START_ZOOM = 0.92; // the 32-region overview
const MARKER_ANIM_MS = 8000; // markers pulse this long, then go static (no idle redraws)

let data: MapData;
let space: SphotaSpace;
let regionModel: RegionModel;
let lang = "en";
let W = 0, H = 0, DPR = 1;
const cam = { x: 0, y: 0, k: 1 };
let baseK = 1;
let camReady = false;
let hover: Cell | null = null;
let selected: Cell | null = null;
let markers: Marker[] = [];
let anim: number | null = null;
let lastShared = -1;
let stars: [number, number, number][] = [];
// derived once after load
let bySize: Cell[] = [];
let regionsBySize: Region[] = [];
let cellsByRegion: Cell[][] = [];
// per-frame screen positions (one projection per cell per frame)
let SX = new Float32Array(0), SY = new Float32Array(0);

const label = (l: Record<string, string> | undefined) => (l ? l[lang] || l.en || "" : "");
const stripTilde = (s: string) => s.replace(/^~\s*/, "");
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

// ---------------------------------------------------------------- camera & geometry
function toScreen(x: number, y: number): [number, number] {
  return [W / 2 + (x - cam.x) * cam.k, H / 2 - (y - cam.y) * cam.k];
}
function toWorld(sx: number, sy: number): [number, number] {
  return [cam.x + (sx - W / 2) / cam.k, cam.y - (sy - H / 2) / cam.k];
}
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.clientWidth;
  H = canvas.clientHeight;
  canvas.width = W * DPR;
  canvas.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const z = camReady ? cam.k / baseK : START_ZOOM;
  baseK = Math.min(W, H) / 2.25;
  cam.k = baseK * z;
  camReady = true;
  buildBackground();
  requestDraw();
}
function zoomLevel() { return cam.k / baseK; }
function home() { return { x: 0, y: 0, k: baseK * START_ZOOM }; }

// ---------------------------------------------------------------- cached layers & sprites
let bg: HTMLCanvasElement | null = null;
function buildBackground() {
  bg = document.createElement("canvas");
  bg.width = Math.max(1, W * DPR);
  bg.height = Math.max(1, H * DPR);
  const b = bg.getContext("2d")!;
  const g = b.createRadialGradient(bg.width / 2, bg.height / 2, 0, bg.width / 2, bg.height / 2, Math.max(bg.width, bg.height) * 0.75);
  g.addColorStop(0, "#0d1b2a");
  g.addColorStop(1, "#05080f");
  b.fillStyle = g;
  b.fillRect(0, 0, bg.width, bg.height);
}

// One soft radial sprite per colour, drawn with drawImage instead of per-frame gradients.
const sprites = new Map<string, HTMLCanvasElement>();
function sprite(color: string, peak: number): HTMLCanvasElement {
  const key = `${color}|${peak}`;
  let s = sprites.get(key);
  if (s) return s;
  s = document.createElement("canvas");
  s.width = s.height = 64;
  const c = s.getContext("2d")!;
  const g = c.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, hexA(color, peak));
  g.addColorStop(0.55, hexA(color, peak * 0.4));
  g.addColorStop(1, hexA(color, 0));
  c.fillStyle = g;
  c.fillRect(0, 0, 64, 64);
  sprites.set(key, s);
  return s;
}

// ---------------------------------------------------------------- redraw scheduling
let dirty = true;
let lastFrame = 0;
function requestDraw() { dirty = true; }
function markersAnimating(t: number) {
  return !reduceMotion && markers.some((m) => t - m.born < MARKER_ANIM_MS);
}
function loop(t: number) {
  // draw only when something changed; marker pulses run at ~30 fps and stop after a few seconds
  if (dirty || (markersAnimating(t) && t - lastFrame > 33)) {
    dirty = false;
    lastFrame = t;
    draw(t);
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- drawing
function draw(t = performance.now()) {
  if (!data) return;
  const z = zoomLevel();
  if (bg) ctx.drawImage(bg, 0, 0, W, H);
  else { ctx.fillStyle = "#05080f"; ctx.fillRect(0, 0, W, H); }
  drawStars();
  drawGrid();

  const cells = data.cells;
  for (let i = 0; i < cells.length; i++) {
    SX[i] = W / 2 + (cells[i].x - cam.x) * cam.k;
    SY[i] = H / 2 - (cells[i].y - cam.y) * cam.k;
  }
  const on = (i: number, pad: number) => SX[i] > -pad && SY[i] > -pad && SX[i] < W + pad && SY[i] < H + pad;

  drawRegions(z);

  // land masses: per region, two batched layers of discs (outer faint, inner brighter) give a
  // soft glow for 64 fills a frame instead of one blended image per cell. Zoomed out, only big cells.
  const glowA = clamp01((z - 0.9) / 0.6);
  if (glowA > 0.02) {
    const minN = z < 1.5 ? 20 : z < 2.5 ? 8 : 0;
    const glow = Math.max(10, 0.035 * cam.k);
    ctx.globalCompositeOperation = "lighter";
    for (const reg of data.regions) {
      for (const [scale, alpha] of [[1, 0.05], [0.5, 0.06]]) {
        ctx.fillStyle = hexA(reg.color, alpha * glowA);
        ctx.beginPath();
        for (const c of cellsByRegion[reg.k]) {
          if (c.n < minN) break;
          const rad = glow * scale * (0.6 + 0.12 * Math.sqrt(c.n));
          if (!on(c.i, rad)) continue;
          ctx.moveTo(SX[c.i] + rad, SY[c.i]);
          ctx.arc(SX[c.i], SY[c.i], rad, 0, Math.PI * 2);
        }
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = "source-over";
  }

  // constellation edges when zoomed in
  if (z > 2.2) {
    ctx.strokeStyle = `rgba(160,190,255,${Math.min(0.18, (z - 2.2) * 0.08)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const c of cells) {
      if (!on(c.i, 0)) continue;
      for (let k = 0; k < 3 && k < c.nb.length; k++) {
        const j = c.nb[k];
        ctx.moveTo(SX[c.i], SY[c.i]);
        ctx.lineTo(SX[j], SY[j]);
      }
    }
    ctx.stroke();
  }

  // cell dots, batched into one path per region; level-of-detail by cell size
  const minDotN = z < 1.2 ? 9 : z < 2 ? 3 : 0;
  const dotA = z < 1.2 ? "99" : "cc";
  const sz = Math.sqrt(z);
  for (const reg of data.regions) {
    ctx.fillStyle = reg.color + dotA;
    ctx.beginPath();
    for (const c of cellsByRegion[reg.k]) {
      if (c.n < minDotN) break;
      if (!on(c.i, 5)) continue;
      const r = Math.max(1.2, Math.min(5, (0.7 + 0.25 * Math.sqrt(c.n)) * sz));
      ctx.moveTo(SX[c.i] + r, SY[c.i]);
      ctx.arc(SX[c.i], SY[c.i], r, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  if (fileRows.length) drawFileLayer(sz);
  for (const c of [hover, selected]) {
    if (!c) continue;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(SX[c.i], SY[c.i], Math.max(2.5, Math.min(5, (0.7 + 0.25 * Math.sqrt(c.n)) * sz)), 0, Math.PI * 2);
    ctx.fill();
  }

  // L2 sub-regions at high zoom (one path)
  if (z > 7) {
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.beginPath();
    for (const s of data.l2) {
      const [sx, sy] = toScreen(s.x, s.y);
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      ctx.moveTo(sx + 1.6, sy);
      ctx.arc(sx, sy, 1.6, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  drawLabels(z);
  drawMarkers(t);
  if (selected) ring(selected.x, selected.y, "#ffffff", 9, 1.5);
}

function drawStars() {
  // parallax starfield: moves slower than the map (static, so idle frames cost nothing)
  ctx.fillStyle = "rgba(200,215,255,0.35)";
  for (const [sx0, sy0, r] of stars) {
    const sx = ((sx0 - cam.x * 18) % W + W) % W;
    const sy = ((sy0 + cam.y * 18) % H + H) % H;
    ctx.fillRect(sx, sy, r, r);
  }
}

// Region layer: one soft territory per region, strongest when zoomed out.
function drawRegions(z: number) {
  const a = clamp01((2.6 - z) / 1.4);
  if (a < 0.02) return;
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = a;
  for (const c of data.regions) {
    const [sx, sy] = toScreen(c.x, c.y);
    const r = Math.max(40, c.spread * cam.k * 2.2);
    if (sx < -r || sy < -r || sx > W + r || sy > H + r) continue;
    ctx.drawImage(sprite(c.color, 0.2), sx - r, sy - r, r * 2, r * 2);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

function drawGrid() {
  ctx.strokeStyle = "rgba(120,150,200,0.06)";
  ctx.lineWidth = 1;
  const step = 0.25;
  const [x0, y1] = toWorld(0, 0);
  const [x1, y0] = toWorld(W, H);
  if ((x1 - x0) / step > 400) return;
  ctx.beginPath();
  for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) {
    const [sx] = toScreen(x, 0);
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, H);
  }
  for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) {
    const [, sy] = toScreen(0, y);
    ctx.moveTo(0, sy);
    ctx.lineTo(W, sy);
  }
  ctx.stroke();
}

// ---------------------------------------------------------------- labels (canvas, collision-culled)
type Box = [number, number, number, number];
// Uniform-grid index of placed label boxes: a label that would overlap one already placed is hidden.
class Placed {
  private grid = new Map<number, Box[]>();
  private readonly s = 96;
  private keys(b: Box, f: (k: number) => void) {
    const i0 = Math.floor(b[0] / this.s), i1 = Math.floor(b[2] / this.s);
    const j0 = Math.floor(b[1] / this.s), j1 = Math.floor(b[3] / this.s);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) f((i + 512) * 4096 + (j + 512));
  }
  fits(b: Box) {
    let ok = true;
    this.keys(b, (k) => {
      if (!ok) return;
      for (const o of this.grid.get(k) || []) {
        if (b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]) { ok = false; return; }
      }
    });
    return ok;
  }
  add(b: Box) {
    this.keys(b, (k) => {
      const l = this.grid.get(k);
      if (l) l.push(b); else this.grid.set(k, [b]);
    });
  }
  tryAdd(b: Box) {
    if (!this.fits(b)) return false;
    this.add(b);
    return true;
  }
}

// screen boxes of the header / panel / legend, refreshed on resize and when the panel changes
let overlayRects: Box[] = [];
function measureOverlays() {
  overlayRects = ["header", "#panel", "#legend"].flatMap((sel) => {
    const el = document.querySelector<HTMLElement>(sel);
    if (!el || getComputedStyle(el).display === "none") return [];
    const r = el.getBoundingClientRect();
    return r.width && r.height ? [[r.left, r.top, r.right, r.bottom] as Box] : [];
  });
  requestDraw();
}
new ResizeObserver(measureOverlays).observe(document.body);
["header", "#panel", "#legend"].forEach((sel) => {
  const el = document.querySelector(sel);
  if (el) new ResizeObserver(measureOverlays).observe(el);
});

const widths = new Map<string, number>();
function textW(font: string, txt: string) {
  const key = font + "\u0000" + txt;
  let w = widths.get(key);
  if (w === undefined) {
    ctx.font = font;
    w = ctx.measureText(txt).width;
    widths.set(key, w);
  }
  return w;
}

function drawLabels(z: number) {
  const placed = new Placed();
  // labels hidden under the floating panels are wasted: reserve those areas
  for (const r of overlayRects) placed.add(r);
  // marker code tags win: reserve their space first
  markers.forEach((m, i) => placed.add(markerTagBox(m, i)));
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // region labels (fade out as you zoom in); bigger regions get first claim on space
  const regA = clamp01(2.4 - z);
  if (regA > 0.02) {
    const small = W < 600;
    const fName = `600 ${small ? 11 : 12.5}px ${FONT}`;
    const fId = `700 ${small ? 9 : 10}px 'JetBrains Mono', ui-monospace, monospace`;
    for (const c of regionsBySize) {
      const [sx, sy] = toScreen(c.x, c.y);
      if (sx < -200 || sy < -20 || sx > W + 200 || sy > H + 20) continue;
      const name = shortName(c.name, small ? 22 : 30);
      const wi = textW(fId, c.id), wn = textW(fName, name);
      const w = wi + 6 + wn;
      const b: Box = [sx - w / 2 - 9, sy - 11, sx + w / 2 + 9, sy + 11];
      if (!placed.tryAdd(b)) continue;
      ctx.fillStyle = `rgba(5,8,15,${0.72 * regA})`;
      roundRect(b[0], b[1], b[2] - b[0], b[3] - b[1], 11);
      ctx.fill();
      ctx.strokeStyle = hexA(c.color, 0.5 * regA);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.textAlign = "left";
      ctx.font = fId;
      ctx.fillStyle = hexA(c.color, 0.8 * regA);
      ctx.fillText(c.id, sx - w / 2, sy + 0.5);
      ctx.font = fName;
      ctx.fillStyle = `rgba(230,238,255,${0.95 * regA})`;
      ctx.fillText(name, sx - w / 2 + wi + 6, sy + 0.5);
      ctx.textAlign = "center";
    }
  }
  // L1 cell labels: bigger cells first, more appear as you zoom
  if (z > 1.6) {
    const minN = z > 8 ? 0 : z > 5 ? 6 : z > 3 ? 15 : 30;
    const font = `${z > 4 ? 13 : 12}px ${FONT}`;
    ctx.font = font;
    ctx.fillStyle = `rgba(230,238,255,${0.92 * clamp01((z - 1.6) / 0.5)})`;
    let shown = 0;
    for (const c of bySize) {
      if (c.n < minN || shown > 260) break;
      const sx = SX[c.i], sy = SY[c.i];
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      const txt = stripTilde(label(c.l));
      if (!txt) continue;
      const w = textW(font, txt);
      if (!placed.tryAdd([sx - w / 2 - 3, sy - 18, sx + w / 2 + 3, sy - 4])) continue;
      ctx.fillText(txt, sx, sy - 11);
      shown++;
    }
  }
  if (z > 9) {
    const font = `11px ${FONT}`;
    ctx.font = font;
    ctx.fillStyle = "rgba(180,200,240,0.8)";
    for (const s of data.l2) {
      const [sx, sy] = toScreen(s.x, s.y);
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      const txt = label(s.l);
      const w = textW(font, txt);
      if (!placed.tryAdd([sx - w / 2, sy + 3, sx + w / 2, sy + 15])) continue;
      ctx.fillText(txt, sx, sy + 9);
    }
  }
}

function shortName(s: string, max: number) {
  return s.length <= max ? s : s.slice(0, max - 1).replace(/\s+\S*$/, "") + "…";
}

function roundRect(x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function hexA(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function ring(x: number, y: number, color: string, r: number, w: number) {
  const [sx, sy] = toScreen(x, y);
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.beginPath();
  ctx.arc(sx, sy, Math.max(0, r), 0, Math.PI * 2);
  ctx.stroke();
}

const TAG_FONT = "700 13px 'JetBrains Mono', ui-monospace, monospace";
function markerTagBox(m: Marker, i: number): Box {
  const [sx, sy] = toScreen(m.x, m.y);
  const ly = sy - 14 - i * 24;
  return [sx + 10, ly - 10, sx + 24 + textW(TAG_FONT, m.enc.code), ly + 10];
}

function drawMarkers(t: number) {
  if (markers.length === 2) {
    const [a, b] = markers;
    const [ax, ay] = toScreen(a.x, a.y);
    const [bx, by] = toScreen(b.x, b.y);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  markers.forEach((m, i) => {
    const [sx, sy] = toScreen(m.x, m.y);
    const age = Math.max(0, t - m.born); // rAF time can be slightly older than born
    const live = !reduceMotion && age < MARKER_ANIM_MS;
    // accuracy radius: world units → pixels, never smaller than 14px
    const rad = Math.max(14, m.radius * cam.k);
    const pulse = live ? 0.5 + 0.5 * Math.sin(age / 380) : 0.5;
    if (live) {
      const fade = 1 - age / MARKER_ANIM_MS;
      for (let k = 0; k < 3; k++) {
        const u = ((age / 2400 + k / 3) % 1);
        ctx.strokeStyle = hexA(m.color, 0.5 * (1 - u) * fade);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, 8 + u * rad * 1.6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    const drop = reduceMotion ? 1 : Math.min(1, age / 500);
    ctx.drawImage(sprite(m.color, 0.32), sx - rad, sy - rad, rad * 2, rad * 2);
    ctx.strokeStyle = hexA(m.color, 0.55 + 0.35 * pulse);
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // pin: a cheap halo instead of shadowBlur
    ctx.drawImage(sprite(m.color, 0.6), sx - 16, sy - 16, 32, 32);
    ctx.fillStyle = m.color;
    ctx.beginPath();
    ctx.arc(sx, sy, 6 * (0.4 + 0.6 * drop) + (1 - drop) * 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    const b = markerTagBox(m, i);
    ctx.fillStyle = "rgba(5,8,15,0.85)";
    roundRect(b[0], b[1], b[2] - b[0], b[3] - b[1], 6);
    ctx.fill();
    ctx.strokeStyle = hexA(m.color, 0.8);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = TAG_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#fff";
    ctx.globalAlpha = drop;
    ctx.fillText(m.enc.code, b[0] + 7, (b[1] + b[3]) / 2 + 0.5);
    ctx.globalAlpha = 1;
    ctx.textAlign = "center";
  });
}

// ---------------------------------------------------------------- regions (mirrors sphota.regions.region_of)
function regionOf(y: Float64Array): { k: number; p: number } {
  const { centroids: C, dim, temperature: T } = regionModel;
  const n = C.length / dim;
  const d2 = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let d = 0; d < dim; d++) { const v = C[k * dim + d] - y[d]; s += v * v; }
    d2[k] = s;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d2[a] - d2[b]).slice(0, 16);
  let z = 0;
  for (const k of order) z += Math.exp(-(d2[k] - d2[order[0]]) / T);
  return { k: order[0], p: 1 / z };
}
const regionTag = (k: number) => {
  const r = data.regions.find((x) => x.k === k);
  return r ? `<span style="color:${r.color}">${r.id}</span> · ${esc(r.name)}` : `R${String(k).padStart(2, "0")}`;
};

// ---------------------------------------------------------------- animation
function flyTo(x: number, y: number, k: number, ms = 1600): Promise<void> {
  return new Promise((done) => flyToCb(x, y, k, ms, done));
}
function flyToCb(x: number, y: number, k: number, ms: number, done: () => void) {
  if (anim) cancelAnimationFrame(anim);
  if (reduceMotion) {
    Object.assign(cam, { x, y, k });
    requestDraw();
    done();
    return;
  }
  const s = { ...cam };
  const t0 = performance.now();
  const dist = Math.hypot(x - s.x, y - s.y);
  const outK = Math.min(s.k, k, baseK / (0.6 + dist));
  const step = (now: number) => {
    const u = Math.min(1, (now - t0) / ms);
    const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
    cam.x = s.x + (x - s.x) * e;
    cam.y = s.y + (y - s.y) * e;
    // zoom out in the first half, in during the second (log-space)
    const lk = u < 0.5
      ? Math.log(s.k) + (Math.log(outK) - Math.log(s.k)) * (e * 2)
      : Math.log(outK) + (Math.log(k) - Math.log(outK)) * ((e - 0.5) * 2);
    cam.k = Math.exp(lk);
    requestDraw();
    anim = u < 1 ? requestAnimationFrame(step) : null;
    if (!anim) done();
  };
  anim = requestAnimationFrame(step);
}
function stopFlight() {
  if (anim) { cancelAnimationFrame(anim); anim = null; }
}

// ---------------------------------------------------------------- interaction
let drag: { x: number; y: number; cx: number; cy: number } | null = null;
let pinch: { d: number; k: number } | null = null;
let pendingHover: { x: number; y: number } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  stopFlight();
  drag = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (drag && !pinch) {
    cam.x = drag.cx - (e.clientX - drag.x) / cam.k;
    cam.y = drag.cy + (e.clientY - drag.y) / cam.k;
    requestDraw();
    return;
  }
  // hover hit-testing at most once per frame
  const r = canvas.getBoundingClientRect();
  if (!pendingHover) requestAnimationFrame(updateHover);
  pendingHover = { x: e.clientX - r.left, y: e.clientY - r.top };
});
function updateHover() {
  const p = pendingHover;
  pendingHover = null;
  if (!p) return;
  const c = nearestCell(p.x, p.y, 14);
  if (c !== hover) {
    hover = c;
    showTooltip(c, p.x, p.y);
    requestDraw();
  } else if (c) moveTooltip(p.x, p.y);
}
canvas.addEventListener("pointerleave", () => {
  if (hover) { hover = null; showTooltip(null, 0, 0); requestDraw(); }
});
canvas.addEventListener("pointerup", (e) => {
  const moved = drag && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4;
  drag = null;
  if (!moved) {
    const r = canvas.getBoundingClientRect();
    const c = nearestCell(e.clientX - r.left, e.clientY - r.top, 16);
    if (c) selectCell(c, true);
  }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  stopFlight();
  const r = canvas.getBoundingClientRect();
  const [wx, wy] = toWorld(e.clientX - r.left, e.clientY - r.top);
  const f = Math.exp(-e.deltaY * 0.0015);
  cam.k = Math.min(baseK * 60, Math.max(baseK * 0.5, cam.k * f));
  const [nx, ny] = toWorld(e.clientX - r.left, e.clientY - r.top);
  cam.x += wx - nx;
  cam.y += wy - ny;
  requestDraw();
}, { passive: false });
canvas.addEventListener("touchstart", (e) => {
  if (e.touches.length === 2) {
    pinch = { d: Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY), k: cam.k };
  }
});
canvas.addEventListener("touchmove", (e) => {
  if (pinch && e.touches.length === 2) {
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    cam.k = Math.min(baseK * 60, Math.max(baseK * 0.5, (pinch.k * d) / pinch.d));
    requestDraw();
  }
});
canvas.addEventListener("touchend", () => (pinch = null));
window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
  const step = 60 / cam.k;
  const keys: Record<string, () => void> = {
    ArrowLeft: () => (cam.x -= step), ArrowRight: () => (cam.x += step),
    ArrowUp: () => (cam.y += step), ArrowDown: () => (cam.y -= step),
    "+": () => (cam.k *= 1.25), "=": () => (cam.k *= 1.25), "-": () => (cam.k /= 1.25),
    "0": () => Object.assign(cam, home()),
    "/": () => $<HTMLInputElement>("q").focus(),
  };
  if (keys[e.key]) {
    e.preventDefault();
    stopFlight();
    keys[e.key]();
    requestDraw();
  }
});

function nearestCell(sx: number, sy: number, maxPx: number): Cell | null {
  if (!data) return null;
  let best: Cell | null = null;
  let bd = maxPx * maxPx;
  const z = zoomLevel();
  const minN = z < 1.2 ? 9 : z < 2 ? 3 : 0; // only cells that are drawn
  for (const c of data.cells) {
    if (c.n < minN) continue;
    const x = W / 2 + (c.x - cam.x) * cam.k, y = H / 2 - (c.y - cam.y) * cam.k;
    const d = (x - sx) ** 2 + (y - sy) ** 2;
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}

const tip = $<HTMLDivElement>("tooltip");
function showTooltip(c: Cell | null, x: number, y: number) {
  if (!c) { tip.hidden = true; return; }
  const reg = data.regions[c.r];
  tip.innerHTML = `<b>${c.code}</b> · ${esc(stripTilde(label(c.l)))}<br><span style="color:${reg.color}">${reg.id}</span> <span>${esc(reg.name)}</span><br><span>${esc((c.ex[0] || "").slice(0, 110))}</span>`;
  tip.hidden = false;
  moveTooltip(x, y);
}
function moveTooltip(x: number, y: number) {
  tip.style.left = `${Math.min(x + 14, W - 300)}px`;
  tip.style.top = `${y + 14}px`;
}
function esc(s: string) {
  return s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]!);
}

// ---------------------------------------------------------------- panel
const panel = $<HTMLDivElement>("panel");
const live = $<HTMLDivElement>("live");

function selectCell(c: Cell, fly: boolean, sub?: number[]) {
  selected = c;
  const code = sub && sub.length > 1 ? formatLevels(sub, space.meta) : c.code;
  setHash(code);
  const l2 = sub && sub.length > 1 ? data.l2.find((s) => s.code === formatLevels(sub.slice(0, 2), space.meta)) : undefined;
  panel.innerHTML = `
    <div class="code">${code}</div>
    <div class="crumbs">${regionTag(c.r)} › ${esc(stripTilde(label(c.l)))}${l2 ? ` › ${esc(label(l2.l))}` : ""}</div>
    ${c.ex.length ? `<h3>Example sentences in this cell</h3><ul>${c.ex.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}
    <h3>Nearby cells</h3>
    <div class="chips">${c.nb.map((j) => `<button class="chip" data-cell="${j}">${data.cells[j].code} · ${esc(stripTilde(label(data.cells[j].l)).split(",")[0])}</button>`).join("")}</div>
    <p class="muted">${c.n} training sentences landed here.</p>`;
  bindChips();
  live.textContent = `Cell ${code}: ${stripTilde(label(c.l))}`;
  if (fly) flyTo(c.x, c.y, Math.max(cam.k, baseK * 5));
  else requestDraw();
}

function selectRegion(r: Region) {
  selected = null;
  const top = cellsByRegion[r.k].slice(0, 8);
  panel.innerHTML = `
    <div class="code">${r.id}</div>
    <div class="crumbs" style="color:${r.color}">${esc(r.full)}</div>
    <p class="muted">One of 32 coarse regions: the layer that stays stable across languages and models. ${r.members.toLocaleString()} map cells belong to it.</p>
    <h3>Largest cells in this region</h3>
    <div class="chips">${top.map((c) => `<button class="chip" data-cell="${c.i}">${c.code} · ${esc(stripTilde(label(c.l)).split(",")[0])}</button>`).join("")}</div>`;
  bindChips();
  live.textContent = `Region ${r.id}: ${r.full}`;
  flyTo(r.x, r.y, baseK * 2.2);
}

function bindChips() {
  panel.querySelectorAll<HTMLButtonElement>("[data-cell]").forEach((b) =>
    b.addEventListener("click", () => selectCell(data.cells[+b.dataset.cell!], true)));
}

function setHash(code: string) {
  history.replaceState(null, "", `#${code}`);
}

function positionFor(enc: Encoded): { x: number; y: number; radius: number } {
  // position: probability-weighted mix of the top L1 candidates (mostly the winner)
  let x = 0, y = 0, z = 0;
  for (const c of enc.l1Candidates) {
    const w = c.p * c.p;
    x += w * data.cells[c.index].x;
    y += w * data.cells[c.index].y;
    z += w;
  }
  x /= z; y /= z;
  // accuracy radius grows as L1/L2 confidence drops
  const radius = 0.012 + 0.09 * (1 - enc.conf[0]) + 0.03 * (1 - enc.conf[1]);
  return { x, y, radius };
}

function confBars(enc: Encoded) {
  return `<div class="bars">${enc.conf.map((p, i) => `
    <div class="bar"><span>L${i + 1}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(0)}%;background:${p > 0.7 ? "#69db7c" : p > 0.4 ? "#ffd43b" : "#ff8787"}"></div></div><b>${p.toFixed(2)}</b></div>`).join("")}</div>`;
}

function truncated(enc: Encoded, tau = 0.5) {
  const d = Math.max(1, enc.conf.filter((p) => p >= tau).length);
  return formatLevels(enc.levels.slice(0, d), space.meta);
}

async function encodeAndShow() {
  const a = $<HTMLInputElement>("q").value.trim();
  const b = $<HTMLInputElement>("q2").value.trim();
  const compare = $<HTMLInputElement>("compare").checked && b;
  if (!a) return;
  const status = $<HTMLDivElement>("status");
  status.textContent = "Encoding in your browser…";
  try {
    await loadModel(space.meta.browser_model, (m) => (status.textContent = m));
    const texts = compare ? [a, b] : [a];
    const colors = ["#ff6b6b", "#4dabf7"];
    const now = performance.now();
    const next: Marker[] = [];
    for (const [i, t] of texts.entries()) {
      const y = space.project(await embed(space.meta.browser_model, t));
      const enc = space.quantize(y, t);
      next.push({ enc, region: regionOf(y), ...positionFor(enc), color: colors[i], born: now });
    }
    markers = next;
    const encs = markers.map((m) => m.enc);
    const shared = encs.length === 2 ? sharedLevels(encs[0].levels, encs[1].levels) : -1;
    lastShared = shared;
    selected = data.cells[encs[0].levels[0]];
    const sameRegion = markers.length === 2 && markers[0].region.k === markers[1].region.k;
    panel.innerHTML = markers.map((m) => `
      <div class="result" style="border-color:${m.color}">
        <div class="code">${m.enc.code}</div>
        <div class="q">“${esc(m.enc.text)}”</div>
        <div class="crumbs">${regionTag(m.region.k)} <span class="muted">(${(m.region.p * 100).toFixed(0)}%)</span></div>
        <div class="crumbs">${m.enc.conf[0] >= 0.7
          ? `${esc(stripTilde(label(data.cells[m.enc.levels[0]].l)))}${l2Label(m.enc)} <span class="muted">(nearest general topic)</span>`
          : `<span class="muted">No topic label: first-level confidence ${m.enc.conf[0].toFixed(2)} is below 0.7, so the cell's general topic would likely mislead.</span>`}</div>
        ${confBars(m.enc)}
        <div class="muted">At 50% confidence: <b>${truncated(m.enc)}</b> · alternates: ${m.enc.l1Candidates.slice(1, 3).map((c) => `<button class="chip" data-cell="${c.index}">${data.cells[c.index].code} ${(c.p * 100).toFixed(0)}%</button>`).join(" ")}</div>
      </div>`).join("") +
      (markers.length === 2 ? `<div class="shared ${sameRegion ? "ok" : ""}">${sameRegion ? `Both land in region <b>${data.regions[markers[0].region.k].id}</b>` : "Different regions"}${shared ? `, and share ${shared} of 4 code levels (<b>${formatLevels(encs[0].levels.slice(0, shared), space.meta)}</b>).` : sameRegion ? ", with different fine codes." : ": see the dashed route between them."}</div>` : "") +
      `<p class="muted">Encoded entirely in your browser (multilingual-e5-small, ${precision() === "fp32" ? "full precision" : "compressed int8"}). The ring is the accuracy radius: wider means less certain.</p>`;
    bindChips();
    setHash(encs[0].code);
    status.textContent = "";
    live.textContent = markers.map((m) => `${m.enc.text}: ${m.enc.code}, region ${data.regions[m.region.k].id}`).join("; ");
    const mx = markers.reduce((s, m) => s + m.x, 0) / markers.length;
    const my = markers.reduce((s, m) => s + m.y, 0) / markers.length;
    const span = markers.length === 2 ? Math.hypot(markers[0].x - markers[1].x, markers[0].y - markers[1].y) : 0;
    await flyTo(mx, my, span > 0.05 ? Math.min(baseK * 6, (Math.min(W, H) * 0.35) / span) : baseK * 7);
  } catch (err) {
    status.textContent = `Could not run the encoder here (${(err as Error).message}). Try another browser, or use: pip install sphota`;
  }
}

function l2Label(enc: Encoded) {
  const s = data.l2.find((x) => x.code === formatLevels(enc.levels.slice(0, 2), space.meta));
  return s ? ` › ${esc(label(s.l))}` : "";
}
function sharedLevels(a: number[], b: number[]) {
  let n = 0;
  while (n < a.length && a[n] === b[n]) n++;
  return n;
}

function fromHash() {
  if (!data || !space) return;
  const h = decodeURIComponent(location.hash.slice(1));
  if (!h || h === "demo") return;
  if (h.startsWith("q=")) {
    $<HTMLInputElement>("q").value = h.slice(2);
    void encodeAndShow();
    return;
  }
  const lv = parseCode(h, space.meta);
  if (lv) selectCell(data.cells[lv[0]], true, lv);
}

// ---------------------------------------------------------------- analyze a file (all in the browser)
const MAX_ROWS = 3000;
let fileRows: FileRow[] = [];
let fileCounts = new Map<number, number>();
let fileName = "";
let fileFocus: number | null = null; // group key (L1 cell or region) the report is focused on
let fileGroupBy: GroupBy = "cell";
const MAX_GROUPS = 25;
let fileJob: { cancelled: boolean } | null = null;

function regionName(k: number): string {
  const key = String(k);
  return regionModel.labels_i18n?.[lang]?.[key] || regionModel.labels[key] || `R${String(k).padStart(2, "0")}`;
}

function drawFileLayer(sz: number) {
  // the user's messages: amber discs on the cells they landed in, sized by count
  ctx.fillStyle = "rgba(255,190,80,0.28)";
  ctx.strokeStyle = "rgba(255,200,110,0.95)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (const [i, n] of fileCounts) {
    const c = data.cells[i];
    if (fileFocus !== null && (fileGroupBy === "cell" ? i !== fileFocus : c.r !== fileFocus)) continue;
    const r = Math.min(26, (2.5 + 2.2 * Math.sqrt(n)) * Math.max(0.8, sz * 0.7));
    if (SX[i] < -r || SY[i] < -r || SX[i] > W + r || SY[i] > H + r) continue;
    ctx.moveTo(SX[i] + r, SY[i]);
    ctx.arc(SX[i], SY[i], r, 0, Math.PI * 2);
  }
  ctx.fill();
  ctx.stroke();
}

async function analyzeFile(name: string, src: string, column?: string) {
  const status = $<HTMLDivElement>("status");
  let parsed;
  try {
    parsed = readMessages(name, src, column);
  } catch (err) {
    status.textContent = `Could not read ${name}: ${(err as Error).message}`;
    return;
  }
  if (!parsed.texts.length) { status.textContent = `No messages found in ${name}.`; return; }
  const texts = parsed.texts.slice(0, MAX_ROWS);
  const langs = parsed.langs.slice(0, MAX_ROWS);
  if (fileJob) fileJob.cancelled = true;
  const job = { cancelled: false };
  fileJob = job;
  fileName = name;
  markers = [];
  panel.innerHTML = `<div class="report"><h2>Analyzing ${esc(name)}</h2>
    <p class="muted">${texts.length.toLocaleString()} messages from column <b>${esc(parsed.column)}</b>${parsed.total > MAX_ROWS ? ` (first ${MAX_ROWS.toLocaleString()} of ${parsed.total.toLocaleString()})` : ""}. Nothing leaves your browser.</p>
    <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${texts.length}"><i style="width:0%"></i></div>
    <p class="muted" id="prog-text">Loading the encoder…</p>
    <button type="button" class="chip" id="cancel-file">Cancel</button></div>`;
  $<HTMLButtonElement>("cancel-file").onclick = () => { job.cancelled = true; };
  try {
    await loadModel(space.meta.browser_model, (m) => { const el = document.getElementById("prog-text"); if (el) el.textContent = m; });
    const t0 = performance.now();
    const vecs = await embedBatch(space.meta.browser_model, texts, (done, total) => {
      const bar = panel.querySelector<HTMLElement>(".progress i");
      const el = document.getElementById("prog-text");
      if (bar) bar.style.width = `${(100 * done) / total}%`;
      const rate = done / Math.max(0.001, (performance.now() - t0) / 1000);
      if (el) el.textContent = `Encoded ${done.toLocaleString()} / ${total.toLocaleString()} · about ${Math.max(0, Math.round((total - done) / rate))} s left`;
    }, 16, job);
    const rows: FileRow[] = vecs.map((v, i) => {
      const y = space.project(v);
      const enc = space.quantize(y, texts[i]);
      const reg = regionOf(y);
      return { text: texts[i], lang: langs[i], script: scriptOf(texts[i]), cell: enc.levels[0], cellP: enc.levelConf[0], region: reg.k, regionP: reg.p, code: enc.code };
    });
    fileRows = rows;
    fileCounts = new Map();
    for (const r of rows) fileCounts.set(r.cell, (fileCounts.get(r.cell) ?? 0) + 1);
    fileFocus = null;
    renderReport(parsed.columns, parsed.column);
    status.textContent = "";
    live.textContent = `Analyzed ${rows.length} messages from ${name}.`;
    const h = home();
    void flyTo(h.x, h.y, h.k, 1400);
  } catch (err) {
    if ((err as Error).message === "cancelled") { panel.innerHTML = `<p class="muted">Analysis cancelled.</p>`; return; }
    panel.innerHTML = `<p>Could not run the encoder here (${esc((err as Error).message)}). Try another browser, or use <code>sphota analyze file.csv</code>.</p>`;
  } finally {
    if (fileJob === job) fileJob = null;
    requestDraw();
  }
}

function renderReport(columns: string[] = [], column = "") {
  const byCell = fileGroupBy === "cell";
  const all: GroupSummary[] = summarize(fileRows, fileGroupBy);
  const sums = all.slice(0, MAX_GROUPS);
  const rest = all.slice(MAX_GROUPS).reduce((t, s) => t + s.count, 0);
  const n = fileRows.length;
  const max = sums[0]?.count ?? 1;
  const uncertain = fileRows.filter((r) => (byCell ? r.cellP : r.regionP) < 0.5).length;
  const scripts = summarize(fileRows.map((r) => ({ ...r, region: 0 })), "region")[0]?.scripts ?? [];
  const colPicker = columns.length > 1
    ? `<label class="muted">Text column <select id="file-col">${columns.map((c) => `<option${c === column ? " selected" : ""}>${esc(c)}</option>`).join("")}</select></label>` : "";
  const clip = (t: string, m: number) => (t.length > m ? t.slice(0, m - 1) + "…" : t);
  panel.innerHTML = `<div class="report">
    <h2>${esc(fileName)}</h2>
    <p class="muted">${n.toLocaleString()} messages in <b>${all.length}</b> ${byCell ? "themes" : "of 32 regions"} · scripts: ${scripts.map(([s, c]) => `${esc(s)} ${Math.round((100 * c) / n)}%`).join(", ")}${uncertain ? ` · ${Math.round((100 * uncertain) / n)}% low-confidence` : ""}</p>
    <div class="seg" role="group" aria-label="Group by">
      <button type="button" data-by="cell" aria-pressed="${byCell}">Themes (code)</button>
      <button type="button" data-by="region" aria-pressed="${!byCell}">32 regions</button>
    </div>
    ${colPicker}
    <ol class="regions">${sums.map((s) => {
      const cell = byCell ? data.cells[s.k] : null;
      const r = data.regions[cell ? cell.r : s.k];
      const id = cell ? cell.code.slice(0, 4) : r.id;
      const name = cell ? `“${clip(s.samples[0] ?? "", 70)}”` : regionName(s.k);
      const mix = s.langs.length ? s.langs.map(([l, c]) => `${esc(l)} ${c}`).join(" · ") : s.scripts.map(([l, c]) => `${esc(l)} ${c}`).join(" · ");
      return `<li class="${fileFocus === s.k ? "on" : ""}">
        <button type="button" class="rrow" data-k="${s.k}" aria-expanded="${fileFocus === s.k}">
          <span class="rid" style="color:${r.color}">${esc(id)}</span>
          <span class="rname">${esc(name)}</span>
          <span class="rnum">${s.count} · ${Math.round(100 * s.share)}%</span>
          <span class="rbar"><i style="width:${(100 * s.count) / max}%;background:${r.color}"></i></span>
        </button>
        <div class="rdetail"${fileFocus === s.k ? "" : " hidden"}>
          <p class="muted">${mix}${s.uncertain ? ` · ${s.uncertain} low-confidence` : ""}${cell ? ` · region ${r.id} ${esc(regionName(r.k))}` : ""}</p>
          <ul>${s.samples.map((t) => `<li>${esc(clip(t, 160))}</li>`).join("")}</ul>
        </div></li>`;
    }).join("")}</ol>
    ${rest ? `<p class="muted">+ ${rest.toLocaleString()} messages in ${all.length - sums.length} smaller groups (all in the CSV).</p>` : ""}
    <div class="row">
      <button type="button" class="chip" id="dl-file">Download results (CSV)</button>
      <button type="button" class="chip" id="clear-file">Clear</button>
    </div>
    <p class="muted">Encoded in your browser with multilingual-e5-small (${precision() === "fp32" ? "full precision: codes match the Python library" : "compressed model: the first block almost always matches the Python library, finer blocks can differ; tick “exact codes” for identical codes"}); nothing was uploaded. ${byCell
      ? "A theme is the first block of the code: translations of the same complaint usually share it. Themes are named by your own messages."
      : "Regions are 32 broad, general-knowledge areas; most messages from one domain (e.g. support tickets) fall into a few of them."}</p>
  </div>`;
  panel.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) => b.addEventListener("click", () => {
    const by = b.dataset.by as GroupBy;
    if (by === fileGroupBy) return;
    fileGroupBy = by;
    fileFocus = null;
    renderReport(columns, column);
    requestDraw();
  }));
  panel.querySelectorAll<HTMLButtonElement>(".rrow").forEach((b) => b.addEventListener("click", () => {
    const k = +b.dataset.k!;
    fileFocus = fileFocus === k ? null : k;
    renderReport(columns, column);
    if (fileFocus !== null) {
      const t = fileGroupBy === "cell" ? data.cells[k] : data.regions[k];
      void flyTo(t.x, t.y, baseK * (fileGroupBy === "cell" ? 4 : 2.2), 1200);
    }
    requestDraw();
  }));
  $<HTMLButtonElement>("dl-file").onclick = () => {
    const blob = new Blob([toCsv(fileRows, regionName)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileName.replace(/\.[^.]+$/, "") + "-sphota.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
  $<HTMLButtonElement>("clear-file").onclick = () => {
    fileRows = []; fileCounts = new Map(); fileFocus = null;
    panel.innerHTML = `<p class="muted">Cleared. Drop another file, or type a sentence.</p>`;
    requestDraw();
  };
  const sel = document.getElementById("file-col") as HTMLSelectElement | null;
  if (sel) sel.onchange = () => { if (lastFile) void analyzeFile(lastFile.name, lastFile.text, sel.value); };
}

let lastFile: { name: string; text: string } | null = null;
async function openFile(f: File) {
  if (f.size > 25 * 1024 * 1024) { $<HTMLDivElement>("status").textContent = "That file is over 25 MB; try a sample of it."; return; }
  const text = await f.text();
  lastFile = { name: f.name, text };
  await analyzeFile(f.name, text);
}

// ---------------------------------------------------------------- boot
async function main() {
  const [d, s, rm] = await Promise.all([
    fetch(`${DATA}/map.json`).then((r) => r.json()),
    SphotaSpace.load(DATA),
    fetch(`${DATA}/regions.json`).then((r) => r.json()),
  ]);
  data = d;
  space = s;
  regionModel = rm;
  bySize = data.cells.slice().sort((a, b) => b.n - a.n);
  regionsBySize = data.regions.slice().sort((a, b) => b.members - a.members);
  cellsByRegion = data.regions.map(() => []);
  for (const c of bySize) cellsByRegion[c.r].push(c);
  SX = new Float32Array(data.cells.length);
  SY = new Float32Array(data.cells.length);
  $<HTMLSpanElement>("count").textContent = `${data.regions.length} regions`;
  const legend = $<HTMLDivElement>("legend");
  const SHOW = 8;
  const chip = (c: Region, hidden: boolean) =>
    `<button class="leg${hidden ? " extra" : ""}" data-k="${c.k}"${hidden ? " hidden" : ""} title="${esc(c.full)}"><i style="background:${c.color}"></i><b>${c.id}</b> ${esc(shortName(c.name, 26))}</button>`;
  legend.innerHTML = regionsBySize.map((c, i) => chip(c, i >= SHOW)).join("") +
    `<button class="leg more" aria-expanded="false">+${regionsBySize.length - SHOW} more</button>`;
  legend.querySelectorAll<HTMLButtonElement>(".leg[data-k]").forEach((b) =>
    b.addEventListener("click", () => selectRegion(data.regions[+b.dataset.k!])));
  const more = legend.querySelector<HTMLButtonElement>(".more")!;
  more.addEventListener("click", () => {
    const open = more.getAttribute("aria-expanded") !== "true";
    legend.querySelectorAll<HTMLButtonElement>(".extra").forEach((b) => (b.hidden = !open));
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "fewer" : `+${regionsBySize.length - SHOW} more`;
  });
  stars = Array.from({ length: 220 }, () => [Math.random() * 4000, Math.random() * 3000, Math.random() < 0.85 ? 1 : 2]);
  resize();
  // intro: drift in from far away to the 32-region overview
  const h = home();
  if (!reduceMotion && !location.hash) { cam.k = h.k * 0.35; void flyTo(h.x, h.y, h.k, 1800); }
  requestAnimationFrame(loop);
  document.body.classList.add("ready");
  setTimeout(measureOverlays, 800); // after the panels finish sliding in
  fromHash();
  if (new URLSearchParams(location.search).has("demo") || location.hash === "#demo") setTimeout(() => void runDemo(), 1200);
  window.addEventListener("hashchange", fromHash);
}

window.addEventListener("resize", resize);
// analyze a file: button, drag-and-drop anywhere, and a bundled multilingual sample
const fileInput = $<HTMLInputElement>("file");
$<HTMLButtonElement>("open-file").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { const f = fileInput.files?.[0]; if (f) void openFile(f); fileInput.value = ""; });
$<HTMLButtonElement>("sample-file").addEventListener("click", async () => {
  const text = await fetch(`${DATA}/sample-feedback.csv`).then((r) => r.text());
  lastFile = { name: "sample-feedback.csv", text };
  await analyzeFile("sample-feedback.csv", text);
});
// exact codes: the fp32 model (~470 MB) gives codes identical to the Python reference encoder;
// the default q8 model (~120 MB) is faster, and its codes can differ in the finer blocks (D-023)
const exact = $<HTMLInputElement>("exact");
exact.checked = precision() === "fp32";
exact.addEventListener("change", () => {
  try { localStorage.setItem("sphota-precision", exact.checked ? "fp32" : "q8"); } catch { /* private mode */ }
  const u = new URL(location.href);
  u.searchParams.delete("precision");
  location.href = u.toString();
});
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; document.body.classList.add("dropping"); } });
window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) document.body.classList.remove("dropping"); });
window.addEventListener("dragover", (e) => { if (e.dataTransfer?.types.includes("Files")) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (!e.dataTransfer?.files.length) return;
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dropping");
  void openFile(e.dataTransfer.files[0]);
});
$<HTMLFormElement>("search").addEventListener("submit", (e) => {
  e.preventDefault();
  void encodeAndShow();
});
$<HTMLInputElement>("compare").addEventListener("change", (e) => {
  $<HTMLDivElement>("q2wrap").hidden = !(e.target as HTMLInputElement).checked;
});
$<HTMLSelectElement>("lang").addEventListener("change", (e) => {
  lang = (e.target as HTMLSelectElement).value;
  document.documentElement.lang = lang;
  if (fileRows.length) renderReport();
  requestDraw();
});
document.querySelectorAll<HTMLButtonElement>("[data-example]").forEach((b) =>
  b.addEventListener("click", () => {
    const [x, y] = b.dataset.example!.split("|");
    $<HTMLInputElement>("q").value = x;
    if (y) {
      $<HTMLInputElement>("q2").value = y;
      $<HTMLInputElement>("compare").checked = true;
      $<HTMLDivElement>("q2wrap").hidden = false;
    }
    void encodeAndShow();
  }));
void main();

// ---------------------------------------------------------------- guided demo (~30 s)
const DEMO = [
  // verified with the Python encoder: same L1 cell (691M, confidence 0.99 / 0.99) and same region (1.00)
  { lang: "English", text: "The team won the football match" },
  { lang: "हिन्दी · Hindi", text: "टीम ने फुटबॉल मैच जीत लिया" },
];
let demoRunning = false;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function caption(html: string) {
  const el = $<HTMLDivElement>("caption");
  el.innerHTML = html;
  el.classList.toggle("show", !!html);
}
async function typeInto(el: HTMLInputElement, text: string) {
  el.value = "";
  el.focus();
  for (const ch of Array.from(text)) {
    el.value += ch;
    await sleep(reduceMotion ? 0 : 55);
  }
}
async function runDemo() {
  if (demoRunning || !data || !space) return;
  demoRunning = true;
  document.body.classList.add("demo");
  const q = $<HTMLInputElement>("q"), q2 = $<HTMLInputElement>("q2"), cmp = $<HTMLInputElement>("compare");
  try {
    cmp.checked = false;
    $<HTMLDivElement>("q2wrap").hidden = true;
    markers = [];
    caption("<b>The Meaning Atlas</b> · readable meaning codes for text in any language");
    const h = home();
    await flyTo(h.x, h.y, h.k, 2200);
    caption("Zoomed out, you see the <b>32 regions</b>: the coarse layer that holds across languages");
    await sleep(2600);
    const status = $<HTMLDivElement>("status");
    status.textContent = "Loading the in-browser encoder…";
    await loadModel(space.meta.browser_model, (m) => (status.textContent = m));
    status.textContent = "";
    caption(`Type a sentence in <b>${DEMO[0].lang}</b>…`);
    await typeInto(q, DEMO[0].text);
    await sleep(400);
    caption("…it is encoded <b>in your browser</b> and lands on its Sphota code");
    await encodeAndShow();
    await sleep(3200);
    caption("Back out to the regions…");
    await flyTo(cam.x, cam.y, baseK * 1.1, 1600);
    cmp.checked = true;
    $<HTMLDivElement>("q2wrap").hidden = false;
    caption(`Now the same meaning in <b>${DEMO[1].lang}</b>…`);
    await typeInto(q2, DEMO[1].text);
    await sleep(400);
    caption("…different script, different words…");
    await encodeAndShow();
    const [a, b] = markers;
    const sameRegion = a && b && a.region.k === b.region.k;
    caption(sameRegion
      ? `…same meaning, <b>same region</b> (${data.regions[a.region.k].id})${lastShared > 0 ? ` and ${lastShared} of 4 code levels shared` : ""}`
      : "…different regions this time: fine codes are not guaranteed to match across languages");
    await sleep(4200);
    caption("Sphota · readable multilingual meaning codes · <span class=muted>press Demo to replay</span>");
    await sleep(3000);
  } catch (err) {
    caption(`Demo stopped: ${esc((err as Error).message)}`);
    await sleep(3000);
  } finally {
    caption("");
    demoRunning = false;
    document.body.classList.remove("demo");
  }
}
$<HTMLButtonElement>("demo").addEventListener("click", () => void runDemo());
(window as unknown as { sphotaDemo: () => Promise<void> }).sphotaDemo = runDemo;
// dev-only: lets tests time a frame and move the camera without waiting for rAF
if (import.meta.env.DEV) {
  (window as unknown as { __atlas: object }).__atlas = { draw, cam, zoom: (z: number) => { cam.k = baseK * z; } };
}
