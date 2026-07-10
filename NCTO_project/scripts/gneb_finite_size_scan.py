#!/usr/bin/env python3
"""B1 -- finite-size scaling of the (y-uniform) 3Q<->ZZ depinning barrier.

Recomputes the enhanced-|K| line-defect GNEB barrier at fixed physical defect
(half-width hw=2.0, dK=-0.5|K|) for a sequence of lattice sizes L.  The endpoints
and every image are built from the x-coordinate only (strip_init depends on
dx=X-XC), so the whole path is translationally invariant along the defect line:
the barrier it returns is the *slab* (whole-line flip) barrier.

If that barrier grows ~linearly with L, it is extensive -> the box contains no
localized critical nucleus and the reported single-L barrier is a slab artifact
(see gneb_localized_nucleus.py for the localized test).  If it plateaus, the
localized nucleus fits inside L.

Machinery (slerp / climbing-image string / field eval) is the proven L36
width-sweep code, with the lattice size lifted to a parameter.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ncto_common as nc
from ncto_common import ROOT, EXE, FIELD_EVAL as FE, J, K, GAMMA, GAMMAP, J2_A, J2_B, J3, J7_DEFAULT as J7

NN_CUTOFF = nc.NN_CUTOFF
OUT_ROOT = nc.CAMPAIGN / "kinetic_barrier" / "finite_size_scaling"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def rel(p: Path) -> Path:
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def rn(s: np.ndarray) -> np.ndarray:
    return s / np.linalg.norm(s, axis=-1, keepdims=True)


class Geom:
    """Per-lattice geometry + tiled 3Q/ZZ references."""

    def __init__(self, lattice: int):
        self.L = lattice
        self.N = 2 * lattice * lattice
        pos = nc.honeycomb_positions(self.N)
        self.XY = pos[:, :2]
        self.X = pos[:, 0]
        self.Y = pos[:, 1]
        self.XC = 0.5 * (float(self.X.min()) + float(self.X.max()))
        self.YC = 0.5 * (float(self.Y.min()) + float(self.Y.max()))
        bonds = []
        for i in range(len(self.XY)):
            d = self.XY[i + 1:] - self.XY[i]
            for off in np.where(np.sqrt(np.sum(d * d, axis=1)) < NN_CUTOFF)[0]:
                bonds.append((i, i + 1 + int(off)))
        self.BONDS = bonds
        self.BMID = np.array([0.5 * (self.XY[i] + self.XY[j]) for i, j in bonds])
        self.Q3 = nc.tile_l18_reference(nc.REF_3Q_L18, lattice)
        self.ZZ = nc.tile_l18_reference(nc.REF_ZZ_L18, lattice)


def defect_file(g: Geom, work_dir: Path, strength: float, half_width: float) -> tuple[Path, int]:
    selected = [k for k in range(len(g.BONDS)) if abs(g.BMID[k, 0] - g.XC) <= half_width]
    path = work_dir / "defect.txt"
    lines = ["# site partner dJ dK dGamma dGammap",
             f"# L={g.L} enhanced-K line: dK=-{strength:g}|K|, hw={half_width:g}"]
    for k in selected:
        i, j = g.BONDS[k]
        lines.append(f"{i} {j} 0.0 {-strength * abs(K):.8e} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path, len(selected)


def _ham_block(g: Geom) -> str:
    return (f"lattice_size = {g.L},{g.L},1\nJ = {J}\nK = {K}\nGamma = {GAMMA}\n"
            f"Gammap = {GAMMAP}\nJ2_A = {J2_A}\nJ2_B = {J2_B}\nJ3 = {J3}\nJ7 = {J7}\n")


def static_config(g: Geom, out: Path, seed: Path, deffile: Path, ndet: int) -> str:
    return (f"system = NCTO\n{_ham_block(g)}simulation_mode = SA\noutput_dir = {rel(out)}\n"
            f"num_trials = 1\nfield_strength = 0.0\nfield_direction = 0,0,1\n"
            f"omega_E1 = {nc.OMEGA_E1}\ngamma_E1 = {nc.GAMMA_E1}\nlambda_E1_K_2 = 0.0\n"
            f"T_start = 1e-9\nT_end = 1e-9\nannealing_steps = 1\ncooling_rate = 0.9\n"
            f"overrelaxation_rate = 0\nT_zero = true\nn_deterministics = {ndet}\n"
            f"initial_spin_config = {rel(seed)}\n"
            f"nn_exchange_channel_disorder_config = {rel(deffile)}\n")


def sa_relax(g: Geom, work_dir: Path, spins: np.ndarray, deffile: Path,
             tag: str, ndet: int, force: bool) -> np.ndarray:
    out = work_dir / tag
    spin_out = out / "sample_0" / "spins_T=0.txt"
    if spin_out.exists() and not force:
        return np.loadtxt(spin_out)
    out.mkdir(parents=True, exist_ok=True)
    seed = work_dir / f"{tag}_in.txt"
    cfg = work_dir / f"{tag}.param"
    np.savetxt(seed, spins, fmt="%.10e")
    cfg.write_text(static_config(g, out, seed, deffile, ndet))
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT),
                          text=True, env=nc._solver_env())
    if proc.returncode != 0 or not spin_out.exists():
        raise RuntimeError(f"SA relax {tag} L={g.L} failed: {proc.stderr[-400:]}")
    return np.loadtxt(spin_out)


def make_cfg(g: Geom, work_dir: Path, deffile: Path) -> Path:
    cfg = work_dir / "cfg.param"
    cfg.write_text(f"system = NCTO\n{_ham_block(g)}simulation_mode = SA\n"
                   f"output_dir = {rel(work_dir)}\nnum_trials = 1\nfield_strength = 0.0\n"
                   f"field_direction = 0,0,1\nomega_E1 = {nc.OMEGA_E1}\n"
                   f"gamma_E1 = {nc.GAMMA_E1}\nlambda_E1_K_2 = 0.0\n"
                   f"nn_exchange_channel_disorder_config = {rel(deffile)}\n")
    return cfg


def band_eval(g: Geom, work_dir: Path, cfg: Path, band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    manifest = work_dir / "band_manifest.txt"
    lines = []
    for idx, image in enumerate(band):
        sf = work_dir / f"img_{idx}.txt"
        ff = work_dir / f"img_{idx}_H.txt"
        np.savetxt(sf, image, fmt="%.10e")
        lines.append(f"{sf} 0.0 0.0 0 {ff}")
    manifest.write_text("\n".join(lines) + "\n")
    subprocess.run([str(FE), str(cfg), str(manifest), str(work_dir / "band_e.csv")],
                   capture_output=True, cwd=str(ROOT), check=True, env=nc._solver_env())
    rows = np.loadtxt(work_dir / "band_e.csv", delimiter=",", skiprows=1)
    energies = np.atleast_2d(rows)[:, 1] * g.N
    fields = np.array([np.loadtxt(work_dir / f"img_{idx}_H.txt") for idx in range(len(band))])
    return energies, fields


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    omega = np.arccos(dot)
    so = np.sin(omega)
    out = np.empty_like(a)
    for i in range(len(a)):
        if so[i] < 1e-7:
            out[i] = (1.0 - t) * a[i] + t * b[i]
        else:
            out[i] = (np.sin((1.0 - t) * omega[i]) * a[i] + np.sin(t * omega[i]) * b[i]) / so[i]
    return rn(out)


def strip_init(g: Geom, width: float, wall: float = 1.2) -> np.ndarray:
    dx = g.X - g.XC
    dx -= g.L * np.round(dx / g.L)
    blend = np.clip(0.5 - 0.5 * (np.abs(dx) - width) / wall, 0.0, 1.0)
    out = np.empty_like(g.ZZ)
    for i, val in enumerate(blend):
        out[i] = slerp(g.Q3[i:i + 1], g.ZZ[i:i + 1], float(val))[0]
    return rn(out)


def geodesic_len(band: np.ndarray) -> np.ndarray:
    d = [0.0]
    for i in range(1, len(band)):
        d.append(d[-1] + np.linalg.norm((band[i] - band[i - 1]).ravel()))
    return np.array(d)


def reparametrize(band: np.ndarray) -> np.ndarray:
    coord = geodesic_len(band)
    total = coord[-1]
    if total < 1e-9:
        return band
    target = np.linspace(0.0, total, len(band))
    out = [band[0].copy()]
    for i in range(1, len(band) - 1):
        v = target[i]
        si = max(0, min(int(np.searchsorted(coord, v) - 1), len(band) - 2))
        seg = coord[si + 1] - coord[si]
        frac = 0.0 if seg < 1e-12 else (v - coord[si]) / seg
        out.append(slerp(band[si], band[si + 1], frac))
    out.append(band[-1].copy())
    return np.array(out)


def mep_barrier(g: Geom, work_dir: Path, cfg: Path, start: np.ndarray, end: np.ndarray, *,
                images: int, n_iter: int, dt: float, climb_start: int, plateau_win: int,
                init_band: np.ndarray | None = None) -> dict:
    if init_band is not None and len(init_band) == images:
        # warm start (continuation): reuse a converged band from a neighbouring
        # parameter point; endpoints are replaced and the path re-relaxed.
        band = rn(init_band.copy())
    else:
        band = np.array([slerp(start, end, i / (images - 1)) for i in range(images)])
    band[0] = start
    band[-1] = end
    band = reparametrize(band)
    band[0] = start
    band[-1] = end
    history = []
    last_e = last_f = None
    for it in range(n_iter):
        energy, fields = band_eval(g, work_dir, cfg, band)
        imax = 1 + int(np.argmax(energy[1:-1]))
        climbing = it >= climb_start
        forces = np.zeros_like(band)
        for idx in range(1, images - 1):
            tangent = (band[idx + 1] - band[idx - 1]).reshape(g.N, 3)
            tangent = tangent - np.sum(tangent * band[idx], axis=1, keepdims=True) * band[idx]
            norm = np.linalg.norm(tangent.ravel())
            tau = tangent / norm if norm > 1e-12 else tangent
            grad = fields[idx]
            if climbing and idx == imax:
                force = grad - 2.0 * np.sum(grad * tau) * tau
            else:
                force = grad - np.sum(grad * tau) * tau
            force = force - np.sum(force * band[idx], axis=1, keepdims=True) * band[idx]
            forces[idx] = force
        band = rn(band + dt * forces)
        band[0] = start
        band[-1] = end
        if not climbing:
            band = reparametrize(band)
            band[0] = start
            band[-1] = end
        barrier = float(np.max(energy) - energy[0])
        history.append({"iteration": it, "barrier_meV": barrier})
        last_e, last_f = energy.copy(), forces.copy()
        if (climbing and len(history) >= plateau_win
                and np.std([h["barrier_meV"] for h in history[-plateau_win:]]) < 0.05):
            break
    imax = int(np.argmax(last_e))
    coord = geodesic_len(band)
    coord = coord / coord[-1] if coord[-1] > 0 else coord
    np.savez_compressed(work_dir / "final_band.npz", band=band, energies=last_e, path_coordinate=coord)
    return {"barrier_meV": float(last_e[imax] - last_e[0]), "saddle_image": imax,
            "iterations_completed": len(history),
            "relative_energy_meV": (last_e - last_e[0]).tolist()}


def run_one_L(lattice: int, *, strength: float, half_width: float, images: int,
              n_iter: int, dt: float, climb_start: int, plateau_win: int, force: bool) -> dict:
    tag = f"L{lattice}_s{strength:.3f}_hw{half_width:.2f}".replace(".", "p")
    work_dir = OUT_ROOT / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report.json"
    if report_path.exists() and not force:
        print(f"  L={lattice}: cached", flush=True)
        return json.loads(report_path.read_text())
    print(f"  L={lattice}: geom + defect ...", flush=True)
    g = Geom(lattice)
    deffile, n_bonds = defect_file(g, work_dir, strength, half_width)
    cfg = make_cfg(g, work_dir, deffile)
    print(f"  L={lattice}: relaxing endpoints ({n_bonds} defect bonds, N={g.N}) ...", flush=True)
    q3 = sa_relax(g, work_dir, g.Q3, deffile, "r3q", 1200, force)
    strip = sa_relax(g, work_dir, strip_init(g, 2.5), deffile, "rstrip", 1200, force)
    q3 = sa_relax(g, work_dir, q3, deffile, "r3q2", 1600, force)
    strip = sa_relax(g, work_dir, strip, deffile, "rstrip2", 1600, force)
    print(f"  L={lattice}: GNEB ...", flush=True)
    gneb = mep_barrier(g, work_dir, cfg, strip, q3, images=images, n_iter=n_iter,
                       dt=dt, climb_start=climb_start, plateau_win=plateau_win)
    report = {"lattice": lattice, "N": g.N, "n_defect_bonds": n_bonds,
              "strength": strength, "half_width": half_width,
              "barrier_meV": gneb["barrier_meV"], "saddle_image": gneb["saddle_image"],
              "iterations": gneb["iterations_completed"]}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  L={lattice}: barrier={gneb['barrier_meV']:.2f} meV", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description="B1 finite-size scaling of the slab barrier")
    ap.add_argument("--lattices", nargs="+", type=int, default=[18, 36, 54])
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--half-width", type=float, default=2.0)
    ap.add_argument("--images", type=int, default=21)
    ap.add_argument("--n-iter", type=int, default=1600)
    ap.add_argument("--dt", type=float, default=0.006)
    ap.add_argument("--climb-start", type=int, default=900)
    ap.add_argument("--plateau-win", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    results = []
    for L in sorted(args.lattices):
        print(f"\n--- L = {L} ---", flush=True)
        results.append(run_one_L(L, strength=args.strength, half_width=args.half_width,
                                 images=args.images, n_iter=args.n_iter, dt=args.dt,
                                 climb_start=args.climb_start, plateau_win=args.plateau_win,
                                 force=args.force))
    Ls = [r["lattice"] for r in results]
    bars = [r["barrier_meV"] for r in results]
    summary = OUT_ROOT / "finite_size_summary.json"
    summary.write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(Ls, bars, "o-", color="C0", ms=8)
    # linear fit through the data; slope>0 with ~0 intercept => extensive slab
    if len(Ls) >= 2:
        m, b = np.polyfit(Ls, bars, 1)
        xs = np.linspace(min(Ls), max(Ls), 50)
        ax.plot(xs, m * xs + b, "--", color="C3",
                label=f"fit: {m:.3f}·L {'+' if b >= 0 else '-'} {abs(b):.1f}")
        ax.legend()
    ax.set_xlabel("lattice L"); ax.set_ylabel("3Q→ZZ barrier (meV)")
    ax.set_title("B1: slab-barrier finite-size scaling (hw=2, s=0.5)")
    fig.tight_layout()
    png = OUT_ROOT / "finite_size_barrier.png"
    fig.savefig(png, dpi=160); plt.close(fig)

    print("\n=== B1 finite-size summary ===")
    print(f"{'L':>5} {'N':>7} {'barrier(meV)':>13}")
    for r in results:
        print(f"{r['lattice']:5d} {r['N']:7d} {r['barrier_meV']:13.2f}")
    print(f"\nwrote {rel(summary)}\nwrote {rel(png)}")
    if len(Ls) >= 2:
        m, b = np.polyfit(Ls, bars, 1)
        print(f"linear fit: barrier ≈ {m:.3f}·L + {b:.1f} meV  "
              f"(slope>0, small intercept => extensive slab)")


if __name__ == "__main__":
    main()
