#!/usr/bin/env python3
"""Real-space spin configurations used across the campaign -- visual sanity check.

Panels (all L=36):
  1. 3Q reference   (relaxed L18 seed tiled to L36 -- seeds every study)
  2. ZZ reference   (relaxed zz_relax state tiled to L36)
  3. width-sweep strip endpoint  (pinned ZZ strip in 3Q, hw defect marked)
  4. width-sweep saddle image
  5. width-sweep 3Q endpoint     (relaxed with the defect)

Rendering: arrows = in-plane (Sx, Sy), colour = Sz.  Each panel is annotated
with r3 (M-point balance; ~1 = 3Q, ~0 = single zigzag) and the ZZ-aligned
site fraction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ncto_common as nc

WS = nc.CAMPAIGN / "kinetic_barrier" / "width_sweep"


def draw(ax, ctx, spins, title, hw=None):
    xy = ctx.xy
    sz = spins[:, 2]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=sz, cmap="coolwarm", vmin=-1, vmax=1,
                    s=14, linewidths=0)
    ax.quiver(xy[:, 0], xy[:, 1], spins[:, 0], spins[:, 1],
              angles="xy", scale_units="xy", scale=1.6, width=0.0022, color="black",
              alpha=0.75)
    if hw is not None:
        for x in (ctx.xc - hw, ctx.xc + hw):
            ax.axvline(x, color="limegreen", lw=1.6, ls="--")
    r3 = nc.r3_of(spins, ctx)
    zzf = nc.zz_fraction(spins, ctx)
    ax.set_title(f"{title}\n$r_3$={r3:.2f}   ZZ-frac={zzf:.2f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=float, default=2.0, help="width-sweep hw to show")
    ap.add_argument("--out", type=Path, default=WS / "spin_configs.png")
    args = ap.parse_args()

    ctx = nc.LatticeContext(36)
    q3_ref = nc.tile_l18_reference(nc.REF_3Q_L18, 36)
    zz_ref = nc.tile_l18_reference(nc.REF_ZZ_L18, 36)

    tag = f"hw{args.hw:.2f}".replace(".", "p")
    dat = np.load(WS / tag / "final_band.npz")
    band = dat["band"]
    rep = json.loads((WS / tag / "report.json").read_text())
    isad = rep["saddle_image"]

    panels = [(q3_ref, "3Q reference (tiled L18)", None),
              (zz_ref, "ZZ reference (tiled L18)", None),
              (band[0], f"strip endpoint (hw={args.hw:g})", args.hw),
              (band[isad], f"saddle (image {isad})", args.hw),
              (band[-1], "3Q endpoint (with defect)", args.hw)]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    for ax, (sp, title, hw) in zip(axes, panels):
        sc = draw(ax, ctx, nc.rn(sp) if hasattr(nc, "rn") else sp / np.linalg.norm(sp, axis=1, keepdims=True),
                  title, hw)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("$S_z$")
    fig.suptitle("Spin configurations (arrows: in-plane $S_x,S_y$; colour: $S_z$; "
                 "green dashes: defect band)", fontsize=12)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out.relative_to(nc.ROOT)}")


if __name__ == "__main__":
    main()
