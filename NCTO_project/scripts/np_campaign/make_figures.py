#!/usr/bin/env python3
"""Driver: generate manuscript figures from campaign outputs.

Usage:
    python make_figures.py --list
    python make_figures.py --all
    python make_figures.py --only F4 F7
    python make_figures.py --all --format pdf       # PDF only, no PNG
"""
from __future__ import annotations

import argparse
import sys
import traceback

import figures
from figures import ALL_FIGURES, FIG_DIR, set_paper_style


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="list available figure IDs and exit")
    p.add_argument("--all", action="store_true", help="render every figure")
    p.add_argument("--only", nargs="+", default=[],
                   help="render only these figure IDs (e.g. F4 F7)")
    p.add_argument("--format", default="pdf,png",
                   help="comma-separated list of output formats")
    p.add_argument("--combined", action="store_true",
                   help="also produce figures_all.pdf if pypdf is available")
    args = p.parse_args()

    if args.list:
        for k in sorted(ALL_FIGURES, key=lambda s: int(s[1:])):
            print(k)
        return 0

    targets = list(ALL_FIGURES) if args.all else list(args.only)
    if not targets:
        p.error("specify --all or --only FXX [FYY ...]")

    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    set_paper_style()

    failures = []
    produced = []
    for tag in sorted(targets, key=lambda s: int(s[1:])):
        if tag not in ALL_FIGURES:
            print(f"  ! unknown figure id: {tag}")
            failures.append(tag)
            continue
        print(f"==> {tag}")
        try:
            outs = ALL_FIGURES[tag](formats=formats)
            for o in outs:
                produced.append(o)
                print(f"    {o}")
        except Exception:
            traceback.print_exc()
            failures.append(tag)

    if args.combined:
        try:
            from pypdf import PdfWriter
            w = PdfWriter()
            pdfs = sorted([p for p in produced if str(p).endswith(".pdf")])
            for pdf in pdfs:
                w.append(str(pdf))
            out = FIG_DIR / "figures_all.pdf"
            with out.open("wb") as fh:
                w.write(fh)
            print(f"==> combined: {out}")
        except Exception as exc:
            print(f"  combined PDF skipped: {exc}")

    if failures:
        print(f"failed: {failures}")
        return 1
    print(f"OK: {len(produced)} files in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
