#!/usr/bin/env python3
"""
Three cube plots on the (J7, lambda_K2, E0) phase-diagram axes, each showing
ONLY the floor and the two far walls (open-corner view) for clean reading:

  1. switching phase diagram   (3Q survives = blue, switched to ZZ = red)
  2. forward  kinetic barrier  E_f(3Q->ZZ)
  3. backward kinetic barrier  E_b(ZZ->3Q)

The static barrier depends only on J7 (verified: eps=0 along the MEP, and
lambda_K2 enters only as |eps|^2, while E0 is a pure drive parameter), so the
barrier cubes vary along J7 and are constant along lambda and E0.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm, colors
import argparse

ROOT = "/home/pc_linux/ClassicalSpin_Cpp"

def set_dirs(all_channels: bool):
    global PD, ANA, KB
    subdir = "phase_diagram_allchan" if all_channels else "phase_diagram"
    PD = f"{ROOT}/NCTO_project/tuned_kruger_campaign/{subdir}"
    ANA = f"{PD}/analysis"
    KB = f"{ROOT}/NCTO_project/tuned_kruger_campaign/kinetic_barrier"
    import os; os.makedirs(ANA, exist_ok=True)

PD = f"{ROOT}/NCTO_project/tuned_kruger_campaign/phase_diagram"
ANA = f"{PD}/analysis"
KB = f"{ROOT}/NCTO_project/tuned_kruger_campaign/kinetic_barrier"

# ----- view: floor (Z=E0 min) + two FAR walls (X=J7 far, Y=lambda far) -----
ELEV = 24
AZIM = -123       # places the far corner (X max, Y max) at the back
NG = 48           # face mesh resolution


def load_phase():
    d = np.loadtxt(f"{ANA}/phase_summary_full.csv", delimiter=",", skiprows=1)
    return d  # J7, lambda, E0, r3, switched


def load_barrier():
    d = np.loadtxt(f"{KB}/barrier_vs_j7.csv", delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    return d  # J7, E_f, E_b, Df, valid


# ---------------------------------------------------------------------------
# generic open-corner cube renderer
# ---------------------------------------------------------------------------
def draw_cube(ax, j7s, lams, e0s, value_on_face, cmap, norm,
              xlabel, ylabel, zlabel, black_axis=True):
    """value_on_face(face, A, B) -> 2D array of scalar values, where
    face in {'floor','wallX','wallY'} and A,B are real-coordinate meshgrids
    for the two in-plane axes of that face."""
    Xr = (j7s.min(), j7s.max())
    Yr = (lams.min(), lams.max())
    Zr = (e0s.min(), e0s.max())

    gj = np.linspace(*Xr, NG)
    gl = np.linspace(*Yr, NG)
    ge = np.linspace(*Zr, NG)

    # floor: Z = E0 min, spans (J7, lambda)
    JJ, LL = np.meshgrid(gj, gl, indexing="ij")
    Cf = value_on_face("floor", JJ, LL)
    ax.plot_surface(JJ, LL, np.full_like(JJ, Zr[0]),
                    facecolors=cmap(norm(Cf)), rstride=1, cstride=1,
                    shade=False, antialiased=False, alpha=1.0)

    # far wall X: J7 = J7 max (back), spans (lambda, E0)
    LL2, EE2 = np.meshgrid(gl, ge, indexing="ij")
    Cx = value_on_face("wallX", LL2, EE2)
    ax.plot_surface(np.full_like(LL2, Xr[1]), LL2, EE2,
                    facecolors=cmap(norm(Cx)), rstride=1, cstride=1,
                    shade=False, antialiased=False, alpha=1.0)

    # far wall Y: lambda = lambda max (back), spans (J7, E0)
    JJ3, EE3 = np.meshgrid(gj, ge, indexing="ij")
    Cy = value_on_face("wallY", JJ3, EE3)
    ax.plot_surface(JJ3, np.full_like(JJ3, Yr[1]), EE3,
                    facecolors=cmap(norm(Cy)), rstride=1, cstride=1,
                    shade=False, antialiased=False, alpha=1.0)

    if black_axis:
        # outline the three visible faces (floor + two far walls) in crisp black
        x0, x1 = Xr; y0, y1 = Yr; z0, z1 = Zr
        lw = 1.1
        col = "black"
        seg = lambda p, q: ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                                   color=col, lw=lw, zorder=10)
        # floor rectangle (z = z0)
        seg((x0, y0, z0), (x1, y0, z0))
        seg((x1, y0, z0), (x1, y1, z0))
        seg((x1, y1, z0), (x0, y1, z0))
        seg((x0, y1, z0), (x0, y0, z0))
        # back vertical edges (the open corner) and top edges of the two walls
        seg((x1, y1, z0), (x1, y1, z1))   # shared back vertical edge
        seg((x1, y0, z0), (x1, y0, z1))   # wallX outer vertical
        seg((x0, y1, z0), (x0, y1, z1))   # wallY outer vertical
        seg((x1, y0, z1), (x1, y1, z1))   # wallX top
        seg((x0, y1, z1), (x1, y1, z1))   # wallY top

    ax.set_xlabel(xlabel, labelpad=16)
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_zlabel(zlabel, labelpad=12)
    ax.set_xlim(*Xr); ax.set_ylim(*Yr); ax.set_zlim(*Zr)
    ax.view_init(elev=ELEV, azim=AZIM)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)


# ---------------------------------------------------------------------------
# 1. switching phase diagram
# ---------------------------------------------------------------------------
def plot_phase():
    d = load_phase()
    j7s = np.unique(d[:, 0]); lams = np.unique(d[:, 1]); e0s = np.unique(d[:, 2])
    cj = {v: i for i, v in enumerate(j7s)}
    cl = {v: i for i, v in enumerate(lams)}
    ce = {v: i for i, v in enumerate(e0s)}
    P = np.array([[cj[a], cl[b], ce[c]] for a, b, c in d[:, :3]], float)
    V = d[:, 4]

    def idx(vals, grid):
        return np.interp(vals, grid, np.arange(len(grid)))

    def field(jr, lr, er):
        q = np.column_stack([idx(jr.ravel(), j7s),
                             idx(lr.ravel(), lams),
                             idx(er.ravel(), e0s)])
        out = np.zeros(len(q))
        for i, qq in enumerate(q):
            dd = np.linalg.norm(P - qq, axis=1)
            near = np.argsort(dd)[:6]
            w = 1.0 / (dd[near] ** 8 + 1e-9)
            out[i] = (w * V[near]).sum() / w.sum()
        out = 1.0 / (1.0 + np.exp(-15.0 * (out - 0.5)))
        return out.reshape(jr.shape)

    def vof(face, A, B):
        if face == "floor":   # (J7, lambda) at E0 min
            return field(A, B, np.full_like(A, e0s.min()))
        if face == "wallX":   # (lambda, E0) at J7 max
            return field(np.full_like(A, j7s.max()), A, B)
        if face == "wallY":   # (J7, E0) at lambda max
            return field(A, np.full_like(A, lams.max()), B)

    # low-saturation, modern diverging map: muted slate-blue (3Q) -> soft
    # off-white -> muted clay-red (ZZ). Desaturated for a clean publication look.
    phase_cmap = colors.LinearSegmentedColormap.from_list(
        "muted_phase",
        ["#6E89AE", "#AFC0D2", "#EAEAE6", "#D8B6AE", "#BE8A80"])
    cmap = phase_cmap
    norm = colors.Normalize(0, 1)
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_cube(ax, j7s, lams, e0s, vof, cmap, norm,
              "$J_7$ (code)", "$\\lambda_{K2}$", "$E_0$ (pump fluence)")
    ax.set_title("Optical switching phase diagram", fontsize=12, pad=10)
    ax.legend(handles=[Patch(facecolor="#6E89AE", edgecolor="black",
                             linewidth=0.6, label="3Q (no switch)"),
                       Patch(facecolor="#BE8A80", edgecolor="black",
                             linewidth=0.6, label="ZZ (switched)")],
              loc="upper left", framealpha=0.95, edgecolor="black", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.06, top=0.92)
    out = f"{ANA}/cube_phase_diagram.png"
    fig.savefig(out, dpi=180); print("saved", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2/3. barrier cubes (value depends on J7 only)
# ---------------------------------------------------------------------------
def plot_barrier(which):
    pd = load_phase()
    j7s = np.unique(pd[:, 0]); lams = np.unique(pd[:, 1]); e0s = np.unique(pd[:, 2])
    b = load_barrier()
    order = np.argsort(b[:, 0])
    bj = b[order, 0]
    col = 1 if which == "forward" else 2
    bv = b[order, col]

    # linear interpolation of barrier vs J7 (constant in lambda, E0)
    def bar(j7):
        return np.interp(j7, bj, bv)

    def vof(face, A, B):
        if face == "floor":   # (J7, lambda)
            return bar(A)
        if face == "wallX":   # (lambda, E0) at J7 max -> single value
            return np.full_like(A, bar(j7s.max()))
        if face == "wallY":   # (J7, E0)
            return bar(A)

    vmin, vmax = bv.min(), bv.max()
    pad = 0.08 * (vmax - vmin + 1e-9)
    cmap = matplotlib.colormaps["magma"]
    norm = colors.Normalize(vmin - pad, vmax + pad)
    fig = plt.figure(figsize=(7.8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_cube(ax, j7s, lams, e0s, vof, cmap, norm,
              "$J_7$ (code)", "$\\lambda_{K2}$", "$E_0$ (pump fluence)")
    label = ("Forward barrier $E_f$ (3Q $\\rightarrow$ ZZ)" if which == "forward"
             else "Backward barrier $E_b$ (ZZ $\\rightarrow$ 3Q)")
    ax.set_title(label + "\nstrain-relaxed climbing-image GNEB", fontsize=11)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cb = fig.colorbar(m, ax=ax, shrink=0.55, pad=0.12)
    cb.set_label("barrier  ($\\mu$eV / site)")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.06, top=0.92)
    out = f"{ANA}/cube_barrier_{which}.png"
    fig.savefig(out, dpi=180); print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-channels", action="store_true",
                    help="Read from phase_diagram_allchan/ (Grüneisen-scaled coupling).")
    args = ap.parse_args()
    set_dirs(args.all_channels)
    plot_phase()
    if os.path.exists(f"{KB}/barrier_vs_j7.csv"):
        plot_barrier("forward")
        plot_barrier("backward")
    else:
        print("barrier_vs_j7.csv not ready yet -- run gneb_barrier_sweep.py first")
