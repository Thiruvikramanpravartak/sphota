// Copyright 2026 The Sphota Authors. SPDX-License-Identifier: Apache-2.0
// Meaning Atlas: a zoomable "Google Maps of meaning" for the Sphota s1 space.
import "./style.css";
import { SphotaSpace, embed, formatLevels, loadModel, parseCode, type Encoded } from "./encoder";

interface Cell { i: number; code: string; x: number; y: number; n: number; c: number; l: Record<string, string>; ex: string[]; nb: number[] }
interface Continent { k: number; x: number; y: number; color: string; name: string; members: number; top: string[] }
interface L2 { code: string; p: number; x: number; y: number; l: Record<string, string> }
interface MapData { space: string; cells: Cell[]; continents: Continent[]; l2: L2[] }
interface Marker { enc: Encoded; x: number; y: number; radius: number; color: string; born: number }

const DATA = `${import.meta.env.BASE_URL}data`;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("map");
const ctx = canvas.getContext("2d")!;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let data: MapData;
let space: SphotaSpace;
let lang = "en";
let W = 0, H = 0, DPR = 1;
const cam = { x: 0, y: 0, k: 300 };
let baseK = 300;
let hover: Cell | null = null;
let selected: Cell | null = null;
let markers: Marker[] = [];
let anim: number | null = null;

const label = (l: Record<string, string> | undefined) => (l ? l[lang] || l.en || "" : "");
const stripTilde = (s: string) => s.replace(/^~\s*/, "");

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
  baseK = Math.min(W, H) / 2.25;
  if (!anim && cam.k === 300) cam.k = baseK;
  draw();
}
function zoomLevel() { return cam.k / baseK; }

// ---------------------------------------------------------------- drawing
function draw() {
  if (!data) return;
  const t = performance.now();
  const z = zoomLevel();
  const g = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75);
  g.addColorStop(0, "#0d1b2a");
  g.addColorStop(1, "#05080f");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  drawGrid();

  // land masses: soft glows per cell, additive blending
  ctx.globalCompositeOperation = "lighter";
  const glow = Math.max(10, 0.035 * cam.k);
  for (const c of data.cells) {
    const [sx, sy] = toScreen(c.x, c.y);
    if (sx < -glow || sy < -glow || sx > W + glow || sy > H + glow) continue;
    const col = data.continents[c.c].color;
    ctx.fillStyle = col + "14";
    ctx.beginPath();
    ctx.arc(sx, sy, glow * (0.6 + 0.12 * Math.sqrt(c.n)), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";

  // constellation edges when zoomed in
  if (z > 2.2) {
    ctx.strokeStyle = `rgba(160,190,255,${Math.min(0.18, (z - 2.2) * 0.08)})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const c of data.cells) {
      const [ax, ay] = toScreen(c.x, c.y);
      if (ax < 0 || ay < 0 || ax > W || ay > H) continue;
      for (const j of c.nb.slice(0, 3)) {
        const d = data.cells[j];
        const [bx, by] = toScreen(d.x, d.y);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
    }
    ctx.stroke();
  }

  // cell dots
  for (const c of data.cells) {
    const [sx, sy] = toScreen(c.x, c.y);
    if (sx < -5 || sy < -5 || sx > W + 5 || sy > H + 5) continue;
    const r = Math.max(1.2, Math.min(5, (0.7 + 0.25 * Math.sqrt(c.n)) * Math.sqrt(z)));
    ctx.fillStyle = c === hover || c === selected ? "#ffffff" : data.continents[c.c].color + "cc";
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
  }
  // L2 sub-regions at high zoom
  if (z > 7) {
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    for (const s of data.l2) {
      const [sx, sy] = toScreen(s.x, s.y);
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      ctx.beginPath();
      ctx.arc(sx, sy, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  drawLabels(z);
  drawMarkers(t);
  if (selected) ring(selected.x, selected.y, "#ffffff", 9, 1.5);
}

function drawGrid() {
  ctx.strokeStyle = "rgba(120,150,200,0.06)";
  ctx.lineWidth = 1;
  const step = 0.25;
  const [x0, y1] = toWorld(0, 0);
  const [x1, y0] = toWorld(W, H);
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

type Box = [number, number, number, number];
function drawLabels(z: number) {
  const boxes: Box[] = [];
  const fits = (b: Box) => !boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // continents (fade out as you zoom in)
  const contAlpha = Math.max(0, Math.min(1, 2.4 - z));
  if (contAlpha > 0.02) {
    ctx.font = "600 15px Inter, system-ui, sans-serif";
    for (const c of data.continents) {
      const [sx, sy] = toScreen(c.x, c.y);
      const txt = continentName(c).toUpperCase();
      const w = ctx.measureText(txt).width;
      const b: Box = [sx - w / 2 - 6, sy - 11, sx + w / 2 + 6, sy + 11];
      if (!fits(b)) continue;
      boxes.push(b);
      ctx.fillStyle = `rgba(5,8,15,${0.55 * contAlpha})`;
      ctx.fillRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.fillStyle = hexA(c.color, contAlpha);
      ctx.fillText(txt, sx, sy);
    }
  }
  // L1 labels: bigger cells first, more appear as you zoom
  if (z > 1.3) {
    const minN = z > 8 ? 0 : z > 5 ? 6 : z > 3 ? 15 : 30;
    ctx.font = `${z > 4 ? 13 : 12}px Inter, system-ui, sans-serif`;
    const order = [...data.cells].sort((a, b) => b.n - a.n);
    for (const c of order) {
      if (c.n < minN) break;
      const [sx, sy] = toScreen(c.x, c.y);
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      const txt = stripTilde(label(c.l));
      if (!txt) continue;
      const w = ctx.measureText(txt).width;
      const b: Box = [sx - w / 2 - 3, sy - 18, sx + w / 2 + 3, sy - 4];
      if (!fits(b)) continue;
      boxes.push(b);
      ctx.fillStyle = "rgba(230,238,255,0.92)";
      ctx.fillText(txt, sx, sy - 11);
    }
  }
  if (z > 9) {
    ctx.font = "11px Inter, system-ui, sans-serif";
    for (const s of data.l2) {
      const [sx, sy] = toScreen(s.x, s.y);
      if (sx < 0 || sy < 0 || sx > W || sy > H) continue;
      const txt = label(s.l);
      const w = ctx.measureText(txt).width;
      const b: Box = [sx - w / 2, sy + 3, sx + w / 2, sy + 15];
      if (!fits(b)) continue;
      boxes.push(b);
      ctx.fillStyle = "rgba(180,200,240,0.8)";
      ctx.fillText(txt, sx, sy + 9);
    }
  }
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
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  ctx.stroke();
}

function drawMarkers(t: number) {
  if (markers.length === 2) {
    const [a, b] = markers;
    const [ax, ay] = toScreen(a.x, a.y);
    const [bx, by] = toScreen(b.x, b.y);
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const m of markers) {
    const [sx, sy] = toScreen(m.x, m.y);
    // accuracy radius: world units → pixels, never smaller than 14px
    const rad = Math.max(14, m.radius * cam.k);
    const pulse = reduceMotion ? 0 : 0.5 + 0.5 * Math.sin((t - m.born) / 380);
    const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, rad);
    g.addColorStop(0, hexA(m.color, 0.32));
    g.addColorStop(1, hexA(m.color, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = hexA(m.color, 0.55 + 0.35 * pulse);
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(sx, sy, rad, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // pin
    ctx.fillStyle = m.color;
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = "600 13px 'JetBrains Mono', ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "#fff";
    ctx.fillText(m.enc.code, sx + 12, sy - 12);
    ctx.textAlign = "center";
  }
}

function continentName(c: Continent) {
  return c.name.split(/[,›]/)[0].replace(/^~\s*/, "");
}

// ---------------------------------------------------------------- animation
function loop() {
  draw();
  if (markers.length && !reduceMotion) requestAnimationFrame(loop);
}

function flyTo(x: number, y: number, k: number, ms = 1600) {
  if (anim) cancelAnimationFrame(anim);
  if (reduceMotion) {
    Object.assign(cam, { x, y, k });
    draw();
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
    draw();
    anim = u < 1 ? requestAnimationFrame(step) : null;
    if (!anim) loop();
  };
  anim = requestAnimationFrame(step);
}

// ---------------------------------------------------------------- interaction
let drag: { x: number; y: number; cx: number; cy: number } | null = null;
let pinch: { d: number; k: number } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (drag && !pinch) {
    cam.x = drag.cx - (e.clientX - drag.x) / cam.k;
    cam.y = drag.cy + (e.clientY - drag.y) / cam.k;
    draw();
    return;
  }
  const r = canvas.getBoundingClientRect();
  const c = nearestCell(e.clientX - r.left, e.clientY - r.top, 14);
  if (c !== hover) {
    hover = c;
    showTooltip(c, e.clientX - r.left, e.clientY - r.top);
    draw();
  } else if (c) moveTooltip(e.clientX - r.left, e.clientY - r.top);
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
  const r = canvas.getBoundingClientRect();
  const [wx, wy] = toWorld(e.clientX - r.left, e.clientY - r.top);
  const f = Math.exp(-e.deltaY * 0.0015);
  cam.k = Math.min(baseK * 60, Math.max(baseK * 0.5, cam.k * f));
  const [nx, ny] = toWorld(e.clientX - r.left, e.clientY - r.top);
  cam.x += wx - nx;
  cam.y += wy - ny;
  draw();
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
    draw();
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
    "0": () => Object.assign(cam, { x: 0, y: 0, k: baseK }),
    "/": () => $<HTMLInputElement>("q").focus(),
  };
  if (keys[e.key]) {
    e.preventDefault();
    keys[e.key]();
    draw();
  }
});

function nearestCell(sx: number, sy: number, maxPx: number): Cell | null {
  if (!data) return null;
  let best: Cell | null = null;
  let bd = maxPx * maxPx;
  for (const c of data.cells) {
    const [x, y] = toScreen(c.x, c.y);
    const d = (x - sx) ** 2 + (y - sy) ** 2;
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}

const tip = $<HTMLDivElement>("tooltip");
function showTooltip(c: Cell | null, x: number, y: number) {
  if (!c) { tip.hidden = true; return; }
  tip.innerHTML = `<b>${c.code}</b> · ${esc(stripTilde(label(c.l)))}<br><span>${esc((c.ex[0] || "").slice(0, 110))}</span>`;
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
  const cont = data.continents[c.c];
  const l2 = sub && sub.length > 1 ? data.l2.find((s) => s.code === formatLevels(sub.slice(0, 2), space.meta)) : undefined;
  panel.innerHTML = `
    <div class="code">${code}</div>
    <div class="crumbs"><span style="color:${cont.color}">${esc(continentName(cont))}</span> › ${esc(stripTilde(label(c.l)))}${l2 ? ` › ${esc(label(l2.l))}` : ""}</div>
    ${c.ex.length ? `<h3>Example sentences in this region</h3><ul>${c.ex.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : ""}
    <h3>Nearby regions</h3>
    <div class="chips">${c.nb.map((j) => `<button class="chip" data-cell="${j}">${data.cells[j].code} · ${esc(stripTilde(label(data.cells[j].l)).split(",")[0])}</button>`).join("")}</div>
    <p class="muted">${c.n} training sentences landed here.</p>`;
  bindChips();
  live.textContent = `Region ${code}: ${stripTilde(label(c.l))}`;
  if (fly) flyTo(c.x, c.y, Math.max(cam.k, baseK * 5));
  else draw();
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
  // accuracy radius grows as L1/L2 confidence drops (GPS-style)
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
    const encs: Encoded[] = [];
    for (const t of texts) encs.push(space.quantize(space.project(await embed(space.meta.browser_model, t)), t));
    const colors = ["#ff6b6b", "#4dabf7"];
    const now = performance.now();
    markers = encs.map((enc, i) => ({ enc, ...positionFor(enc), color: colors[i], born: now }));
    const shared = encs.length === 2 ? sharedLevels(encs[0].levels, encs[1].levels) : -1;
    const cell = data.cells[encs[0].levels[0]];
    selected = cell;
    panel.innerHTML = encs.map((enc, i) => `
      <div class="result" style="border-color:${colors[i]}">
        <div class="code">${enc.code}</div>
        <div class="q">“${esc(enc.text)}”</div>
        <div class="crumbs">${esc(stripTilde(label(data.cells[enc.levels[0]].l)))}${l2Label(enc)}</div>
        ${confBars(enc)}
        <div class="muted">At 50% confidence: <b>${truncated(enc)}</b> · alternates: ${enc.l1Candidates.slice(1, 3).map((c) => `<button class="chip" data-cell="${c.index}">${data.cells[c.index].code} ${(c.p * 100).toFixed(0)}%</button>`).join(" ")}</div>
      </div>`).join("") +
      (shared >= 0 ? `<div class="shared ${shared ? "ok" : ""}">${shared ? `Both land in <b>${formatLevels(encs[0].levels.slice(0, shared), space.meta)}</b>: they share ${shared} of 4 levels.` : "Different L1 regions: see the dashed route between them."}</div>` : "") +
      `<p class="muted">Encoded entirely in your browser (multilingual-e5-small, int8). The ring is the accuracy radius: wider means less certain.</p>`;
    bindChips();
    setHash(encs[0].code + (compare ? "" : ""));
    status.textContent = "";
    live.textContent = encs.map((e) => `${e.text}: ${e.code}`).join("; ");
    const mx = markers.reduce((s, m) => s + m.x, 0) / markers.length;
    const my = markers.reduce((s, m) => s + m.y, 0) / markers.length;
    const span = markers.length === 2 ? Math.hypot(markers[0].x - markers[1].x, markers[0].y - markers[1].y) : 0;
    flyTo(mx, my, span > 0.05 ? Math.min(baseK * 6, (Math.min(W, H) * 0.35) / span) : baseK * 7);
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
  if (!h) return;
  if (h.startsWith("q=")) {
    $<HTMLInputElement>("q").value = h.slice(2);
    void encodeAndShow();
    return;
  }
  const lv = parseCode(h, space.meta);
  if (lv) selectCell(data.cells[lv[0]], true, lv);
}

// ---------------------------------------------------------------- boot
async function main() {
  const [d, s] = await Promise.all([fetch(`${DATA}/map.json`).then((r) => r.json()), SphotaSpace.load(DATA)]);
  data = d;
  space = s;
  $<HTMLSpanElement>("count").textContent = `${data.cells.length.toLocaleString()} regions · ${data.l2.length.toLocaleString()} labelled sub-regions`;
  const legend = $<HTMLDivElement>("legend");
  const sorted = data.continents.slice().sort((a, b) => b.members - a.members);
  const chip = (c: Continent, hidden: boolean) =>
    `<button class="leg${hidden ? " extra" : ""}" data-k="${c.k}"${hidden ? " hidden" : ""}><i style="background:${c.color}"></i>${esc(continentName(c))}</button>`;
  legend.innerHTML = sorted.map((c, i) => chip(c, i >= 8)).join("") +
    `<button class="leg more" aria-expanded="false">+${sorted.length - 8} more</button>`;
  legend.querySelectorAll<HTMLButtonElement>(".leg[data-k]").forEach((b) =>
    b.addEventListener("click", () => {
      const c = data.continents[+b.dataset.k!];
      flyTo(c.x, c.y, baseK * 3);
    }));
  const more = legend.querySelector<HTMLButtonElement>(".more")!;
  more.addEventListener("click", () => {
    const open = more.getAttribute("aria-expanded") !== "true";
    legend.querySelectorAll<HTMLButtonElement>(".extra").forEach((b) => (b.hidden = !open));
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "fewer" : `+${sorted.length - 8} more`;
  });
  resize();
  fromHash();
  window.addEventListener("hashchange", fromHash);
}

window.addEventListener("resize", resize);
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
  draw();
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
