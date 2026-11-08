#!/usr/bin/env python3
# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the paper: Markdown → standalone HTML → PDF (headless Chromium), plus LaTeX for arXiv.

    python scripts/build_paper.py            # writes docs/paper/build/sphota.{html,pdf,tex}

Needs pandoc and Node Playwright with Chromium (no TeX installation needed: arXiv compiles the
.tex itself).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper" / "sphota.md"
OUT = ROOT / "docs" / "paper" / "build"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Times New Roman", "Noto Serif", Georgia, serif; font-size: 10.5pt;
       line-height: 1.38; color: #111; max-width: 170mm; margin: 0 auto; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; } h3 { font-size: 11pt; margin: 10pt 0 3pt; }
p { margin: 4pt 0; text-align: justify; hyphens: auto; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 8.6pt; }
th, td { border-top: 0.5pt solid #999; border-bottom: 0.5pt solid #999; padding: 2pt 5pt; }
th { background: #f0f0f0; } code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; }
pre { font-size: 8.4pt; background: #f6f6f6; padding: 5pt; overflow-wrap: anywhere; white-space: pre-wrap; }
blockquote { margin: 4pt 14pt; color: #333; } li { margin: 1pt 0; }
"""


UNICODE_TEX = {
    "→": r"$\rightarrow$",
    "↔": r"$\leftrightarrow$",
    "·": r"$\cdot$",
    "§": r"\S{}",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "×": r"$\times$",
    "−": "-",
    "μ": r"$\mu$",
    "₁": r"$_1$",
    "₂": r"$_2$",
    "⁻⁸": r"$^{-8}$",
    "²": r"$^2$",
    "–": "--",
    "—": "---",
    "’": "'",
    "“": "``",
    "”": "''",
    "…": r"\ldots{}",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    css = OUT / "paper.css"
    css.write_text(CSS, encoding="utf-8")
    html = OUT / "sphota.html"
    subprocess.run(
        [
            "pandoc",
            str(SRC),
            "-s",
            "--embed-resources",
            "--css",
            str(css),
            "--metadata",
            "pagetitle=Sphota",
            "-o",
            str(html),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pandoc",
            str(SRC),
            "-s",
            "-o",
            str(OUT / "sphota.tex"),
            "-V",
            "geometry:margin=2cm",
            "-V",
            "fontsize=10pt",
        ],
        check=True,
    )
    # arXiv compiles with pdflatex: map the few Unicode symbols we use to LaTeX
    tex = OUT / "sphota.tex"
    t = tex.read_text(encoding="utf-8")
    for k, v in UNICODE_TEX.items():
        t = t.replace(k, v)
    left = sorted({ch for ch in t if ord(ch) > 127})
    if left:
        print(f"warning: non-ASCII characters left in the LaTeX: {left}")
    tex.write_text(t, encoding="utf-8")
    js = """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const page = await b.newPage();
  await page.goto(process.argv[1], { waitUntil: 'load' });
  await page.pdf({ path: process.argv[2], format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center"><span class="pageNumber"></span></div>',
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' } });
  await b.close();
})();
"""
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ, "NODE_PATH": npm_root}
    subprocess.run(["node", "-e", js, html.as_uri(), str(OUT / "sphota.pdf")], check=True, env=env)
    print(f"wrote {OUT}/sphota.html, sphota.pdf, sphota.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
