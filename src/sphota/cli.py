# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""``sphota`` command-line interface. Every API call has a one-line command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - depends on extras

    def main() -> None:
        sys.stderr.write("The sphota CLI needs extras: pip install 'sphota[cli,encode]'\n")
        raise SystemExit(2)
else:
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
        help="Sphota: readable multilingual meaning codes. Encode text in any language, group files by theme.",
    )
    con = Console()
    err = Console(stderr=True)

    def _fail(e: Exception) -> None:
        err.print(f"[bold red]error:[/] {e}")
        raise typer.Exit(1)

    def _conf_bar(conf: list[float], depth: int) -> str:
        cells = []
        for i, p in enumerate(conf):
            style = "green" if p >= 0.8 else "yellow" if p >= 0.5 else "red"
            dim = "" if i < depth else " dim"
            cells.append(f"[{style}{dim}]L{i + 1} {p:.2f}[/]")
        return "  ".join(cells)

    @app.command()
    def encode(
        text: list[str] = typer.Argument(
            ..., help="Text(s) to encode. Use '-' to read lines from stdin."
        ),
        min_confidence: float | None = typer.Option(
            None, "--min-confidence", "-c", help="Truncate unsure levels."
        ),
        max_level: int = typer.Option(
            4, "--max-level", "-l", min=1, max=4, help="Privacy by depth."
        ),
        stable: bool = typer.Option(False, "--stable", help="Stable mode (fewer boundary flips)."),
        model: str | None = typer.Option(
            None, "--model", "-m", help="Encode via another model's adapter."
        ),
        adapters: str | None = typer.Option(
            None,
            "--adapters",
            help="Adapter set for --model: v1.1 (default, nonlinear) or v1 (linear).",
        ),
        lang: str = typer.Option("en", "--lang", help="Label language."),
        as_json: bool = typer.Option(False, "--json", help="JSON lines output."),
        alternates: bool = typer.Option(False, "--alternates", "-a", help="Show alternates."),
        top_k: int = typer.Option(0, "--top-k", help="Show the k most probable full codes."),
        labels: bool = typer.Option(
            False,
            "--labels",
            help="Also show the cell's general topic name (when confident). Off by default: names "
            "come from Wikipedia topics and often do not describe everyday text.",
        ),
    ) -> None:
        """Encode text into Sphota codes.  [dim]sphota encode "The CEO resigned"[/]"""
        from sphota.api import encode_batch

        texts = [line.rstrip("\n") for line in sys.stdin] if text == ["-"] else text
        try:
            codes = encode_batch(
                texts,
                min_confidence=min_confidence,
                max_level=max_level,
                stable=stable,
                model=model,
                adapters=adapters,
            )
        except Exception as e:
            _fail(e)
        for c in codes:
            if as_json:
                d = c.to_dict()
                if top_k:
                    d["top_k"] = [
                        {"code": str(x), "p": round(p, 4)} for x, p in c.candidates(top_k)
                    ]
                print(json.dumps(d, ensure_ascii=False))
                continue
            con.print(f"[bold cyan]{c}[/]")
            con.print(
                f"   region [bold]{c.region}[/] {c.region_label_in(lang)}  (conf {c.region_confidence:.2f})"
            )
            if labels:
                topic = c.confident_label(lang)
                con.print(
                    f"   topic  {topic}  [dim](general topic of this cell, from Wikipedia)[/]"
                    if topic
                    else f"   topic  [dim]none: first-level confidence {c.confidence[0]:.2f} < 0.7[/]"
                )
            con.print(f"   {_conf_bar(c.confidence, c.depth)}")
            if alternates:
                for a, p in c.alternates:
                    con.print(f"   [dim]alt[/] {a}  p={p:.2f}  {a.label_in(lang)}")
            if top_k:
                for a, p in c.candidates(top_k):
                    con.print(f"   [dim]top[/] {a}  p={p:.2f}")

    @app.command()
    def explain(code: str, lang: str = typer.Option("en", "--lang")) -> None:
        """Decode a code (or text): labels per level and neighbouring regions."""
        from sphota.api import explain as _explain

        try:
            info = _explain(code, lang=lang)
        except Exception as e:
            _fail(e)
        con.print(f"[bold cyan]{info['code']}[/]  {info['label']}")
        for lv in info["levels"]:
            extra = f"  conf={lv['confidence']:.2f}" if "confidence" in lv else ""
            con.print(f"  L{lv['level']} {lv['prefix']:<14} {lv.get('label') or ''}{extra}")
        con.print("[dim]nearby regions:[/]")
        for nb in info["neighbors"]:
            con.print(f"  {nb['code']}  {nb['label']}")

    @app.command()
    def similarity(a: str, b: str) -> None:
        """Similarity (0..1) of two codes or texts."""
        from sphota.api import similarity as _sim

        try:
            con.print(f"{_sim(a, b):.4f}")
        except Exception as e:
            _fail(e)

    @app.command()
    def neighbors(code: str, k: int = typer.Option(10, "-k")) -> None:
        """Nearest codes at the same depth."""
        from sphota.api import neighbors as _nb

        try:
            for c in _nb(code, k):
                con.print(f"{c}  {c.label}")
        except Exception as e:
            _fail(e)

    @app.command()
    def path(a: str, b: str) -> None:
        """Semantic route between two codes or texts."""
        from sphota.api import path as _path

        try:
            for i, c in enumerate(_path(a, b)):
                con.print(f"{'→ ' if i else '  '}{c}  {c.label}")
        except Exception as e:
            _fail(e)

    @app.command()
    def validate(code: str) -> None:
        """Check a code string (exit status 1 if invalid)."""
        from sphota.codes import parse

        try:
            c = parse(code)
        except Exception as e:
            _fail(e)
        con.print(f"[green]valid[/] {c.format(version=True, check=True)} depth={c.depth}")

    def _read_doc(p: str) -> str:
        if p == "-":
            return sys.stdin.read()
        path_ = Path(p)
        if path_.exists():
            return path_.read_text(encoding="utf-8")
        return p

    @app.command()
    def fingerprint(
        doc: str = typer.Argument(..., help="File path, '-' for stdin, or text."),
        level: int = typer.Option(2, "--level"),
        top: int = typer.Option(8, "--top"),
        as_json: bool = typer.Option(False, "--json"),
    ) -> None:
        """Meaning fingerprint of a document (bag of codes + top topics)."""
        from sphota.api import fingerprint as _fp

        fp = _fp(_read_doc(doc), level=level)
        if as_json:
            print(
                json.dumps(
                    {"level": level, "n_sentences": fp.n_sentences, "weights": fp.weights},
                    ensure_ascii=False,
                )
            )
            return
        t = Table(title=f"Fingerprint ({fp.n_sentences} sentences)")
        t.add_column("prefix")
        t.add_column("weight", justify="right")
        t.add_column("label")
        for k, w, lab in fp.top(top):
            t.add_row(k, f"{w:.1%}", lab)
        con.print(t)

    @app.command()
    def compare(doc_a: str, doc_b: str, level: int = typer.Option(2, "--level")) -> None:
        """Similarity (0..1) of two documents' fingerprints."""
        from sphota.api import compare as _cmp

        con.print(f"{_cmp(_read_doc(doc_a), _read_doc(doc_b), level=level):.4f}")

    @app.command()
    def diff(
        doc_a: str,
        doc_b: str,
        level: int = typer.Option(2, "--level"),
        threshold: float = typer.Option(0.05, "--threshold"),
    ) -> None:
        """Topics added / removed between two documents."""
        from sphota.api import diff as _diff

        d = _diff(_read_doc(doc_a), _read_doc(doc_b), level=level, threshold=threshold)
        con.print(f"similarity {d.similarity:.3f}")
        for k, w, lab in d.added:
            con.print(f"[green]+ {k}[/] {w:+.1%}  {lab}")
        for k, w, lab in d.removed:
            con.print(f"[red]- {k}[/] {w:+.1%}  {lab}")

    @app.command()
    def analyze(
        file: Path = typer.Argument(..., exists=True, help="CSV/TSV/JSONL/TXT file."),
        column: str | None = typer.Option(None, "--column", help="Text column (auto-detected)."),
        out: Path = typer.Option(Path("sphota-report.html"), "--out", "-o"),
        min_count: int = typer.Option(
            1, "--min-count", help="Suppress prefixes with fewer items (privacy)."
        ),
        lang: str = typer.Option("en", "--lang"),
        limit: int | None = typer.Option(None, "--limit"),
        by: str = typer.Option(
            "refined",
            "--by",
            help="Grouping: refined (default: groups fitted to this file, each tied to a stable "
            "theme code), theme (first code block only), region (32 broad areas) or l1.",
        ),
        csv_out: Path | None = typer.Option(
            None, "--csv", help="Also write one row per message (group, theme, code) to this CSV."
        ),
    ) -> None:
        """Topic analytics → a standalone HTML report (prefix tree, counts, samples)."""
        from sphota.usecases.analytics import analyze_file

        try:
            res = analyze_file(
                file,
                column=column,
                out=out,
                min_count=min_count,
                lang=lang,
                limit=limit,
                by=by,
                csv_out=csv_out,
            )
        except Exception as e:
            _fail(e)
        con.print(
            f"[green]wrote[/] {res['out']}  ({res['n']} rows, {res['n_l1']} groups)"
            + (f", {csv_out}" if csv_out else "")
        )

    @app.command("train-adapter")
    def train_adapter(
        model: str = typer.Option(..., "--model", help="sentence-transformers model id."),
        out: Path | None = typer.Option(None, "--out"),
        texts: Path | None = typer.Option(
            None, "--texts", help="Text file (one per line) to fit on; default: bundled sample."
        ),
        prefix: str = typer.Option("", "--prefix", help="Input prefix the model expects."),
        n: int = typer.Option(20000, "--n", help="Max training texts."),
    ) -> None:
        """Fit an adapter from MODEL into the s1 space (paired texts, ridge regression)."""
        from sphota.certify import train_adapter as _train

        try:
            p = _train(model, out=out, texts_path=texts, prefix=prefix, n=n)
        except Exception as e:
            _fail(e)
        con.print(
            f"[green]adapter written[/] {p}\nnext: sphota certify --model {model} --adapter {p}"
        )

    @app.command()
    def certify(
        model: str = typer.Option(..., "--model"),
        adapter: Path | None = typer.Option(
            None, "--adapter", help="Adapter .npz (default: registry or train one)."
        ),
        prefix: str = typer.Option("", "--prefix"),
        out: Path | None = typer.Option(None, "--out", help="Write the certificate JSON here."),
        quick: bool = typer.Option(
            False, "--quick", help="Smaller test set (faster, less precise)."
        ),
    ) -> None:
        """Evaluate a model+adapter against the s1 conformance thresholds; print a certificate."""
        from sphota.certify import certify as _certify
        from sphota.certify import render_certificate

        try:
            cert = _certify(model, adapter=adapter, prefix=prefix, quick=quick)
        except Exception as e:
            _fail(e)
        con.print(render_certificate(cert))
        if out:
            out.write_text(json.dumps(cert, indent=2), encoding="utf-8")
            con.print(f"[dim]certificate written to {out}[/]")
        raise typer.Exit(0 if cert["pass"] else 3)

    @app.command()
    def adapters() -> None:
        """List certified adapters in the registry."""
        from sphota.adapters import registry

        reg = registry()["adapters"]
        t = Table(title="Certified adapters (space s1)")
        for col in ["model", "short", "multilingual", "L1 exact", "L1 nbhd", "status"]:
            t.add_column(col)
        for k, v in sorted(reg.items()):
            c = v.get("certificate", {})
            ag = [c.get("agreement_same_text_en", [None])[0], c.get("l1_neighborhood_en")]
            t.add_row(
                k,
                v.get("short", ""),
                str(v.get("multilingual", "")),
                f"{ag[0]:.2f}" if ag[0] is not None else "-",
                f"{ag[1]:.2f}" if ag[1] is not None else "-",
                c.get("grade", "-"),
            )
        con.print(t)

    @app.command()
    def migrate(
        codes_file: Path | None = typer.Argument(None, help="CSV with a 'code' column to migrate."),
        table: Path | None = typer.Option(None, "--table", help="Migration table JSON."),
        build: bool = typer.Option(
            False, "--build", help="Build a table between two release dirs."
        ),
        src: Path | None = typer.Option(None, "--from"),
        dst: Path | None = typer.Option(None, "--to"),
        texts: Path | None = typer.Option(None, "--texts", help="Reference texts for --build."),
        out: Path = typer.Option(Path("migration.json"), "--out", "-o"),
    ) -> None:
        """Build or apply a vN → vN+1 migration table."""
        from sphota import migrate as m

        try:
            if build:
                if not (src and dst):
                    raise ValueError("--build needs --from <release dir> and --to <release dir>")
                res = m.build_table(src, dst, texts_path=texts, out=out)
                con.print(
                    f"[green]wrote[/] {out}: {res['n_prefixes']} prefixes, "
                    f"top-1 mass {res['mean_top1']:.2f}"
                )
            else:
                if not (codes_file and table):
                    raise ValueError("give a codes CSV and --table, or use --build")
                n = m.apply_table(codes_file, table, out)
                con.print(f"[green]migrated[/] {n} codes → {out}")
        except Exception as e:
            _fail(e)

    @app.command()
    def bundle(
        out: Path = typer.Option(Path("sphota-bundle"), "--out", "-o"),
        int8: bool = typer.Option(False, "--int8", help="Bundle the 4x smaller int8 encoder too."),
    ) -> None:
        """Create an offline (air-gapped) bundle: release + encoder model."""
        from sphota.bundle import make_bundle

        try:
            p = make_bundle(out, int8=int8)
        except Exception as e:
            _fail(e)
        con.print(
            f'[green]bundle ready[/] {p}\nuse it with: SPHOTA_OFFLINE=1 SPHOTA_BUNDLE={p} sphota encode "hello"'
        )

    @app.command()
    def info() -> None:
        """Show the installed space version, files and settings."""
        import os

        from sphota import __version__
        from sphota.release import load_release

        rel = load_release()
        con.print(
            f"sphota {__version__}  space [bold]{rel.version}[/] ({rel.manifest.get('status')})"
        )
        con.print(f"anchor: {rel.anchor['model']}  dim={rel.dim}  sizes={rel.sizes}")
        con.print(f"release dir: {rel.root}")
        con.print(f"labels: {len(rel.labels)} prefixes")
        for k in ["SPHOTA_OFFLINE", "SPHOTA_BUNDLE", "SPHOTA_ENCODER", "SPHOTA_RELEASE_DIR"]:
            if os.environ.get(k):
                con.print(f"{k}={os.environ[k]}")

    def main() -> None:
        app()


if __name__ == "__main__":  # pragma: no cover
    main()
