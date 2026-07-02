#!/usr/bin/env python3
"""Kinetic-barrier (GNEB MEP) sweep over half-width for the L36 kred defect.

Runs the climbing-image string update for a sequence of half-width values
(default 0.5 → 1.0 → 1.5 → 2.0) so we can see how the 3Q→ZZ barrier
evolves as the defect band widens toward the 'current point' (hw=2.0).

Each width gets its own subdirectory so runs never collide and are fully
cached.  A summary JSON and two-panel PNG are written at the end.
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

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "spin_solver"
FE = ROOT / "build" / "gneb_field_eval"
sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from reader_strain_lattice import generate_honeycomb_positions  # noqa: E402

# Tuned-Kruger Hamiltonian.
J, K, GAMMA, GAMMAP = 0.68, -7.89, 3.07, -2.94
J2_A, J2_B, J3, J7 = -0.06, -0.70, 0.52, -0.40
NN_CUTOFF = 0.65
LATTICE = 36
N = 2 * LATTICE * LATTICE

REF_3Q_L18 = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "phase_diagram"
               / "analysis" / "seed3Q_J70400.txt")
REF_ZZ_L18 = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "kinetic_barrier"
               / "zz_relax" / "sample_0" / "spins_T=0.txt")

OUT_ROOT = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
            / "defect_catalogue_L36" / "selected_kred_mep_width_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

POS = generate_honeycomb_positions(N)
XY = POS[:, :2]
X = POS[:, 0]
XC = 0.5 * (float(X.min()) + float(X.max()))


def rel(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def rn(spins: np.ndarray) -> np.ndarray:
    return spins / np.linalg.norm(spins, axis=-1, keepdims=True)


def tile_l18(path: Path) -> np.ndarray:
    reference = np.loadtxt(path).reshape(18, 18, 2, 3)
    tiled = np.tile(reference, (2, 2, 1, 1)).reshape(N, 3)
    return rn(tiled)


Q3 = tile_l18(REF_3Q_L18)
ZZ = tile_l18(REF_ZZ_L18)

BONDS: list[tuple[int, int]] = []
for _i in range(len(XY)):
    _delta = XY[_i + 1:] - XY[_i]
    _dist = np.sqrt(np.sum(_delta * _delta, axis=1))
    for _off in np.where(_dist < NN_CUTOFF)[0]:
        BONDS.append((_i, _i + 1 + int(_off)))
BMID = np.array([0.5 * (XY[i] + XY[j]) for i, j in BONDS])


# ---------------------------------------------------------------------------
# Per-width helpers (all take work_dir so they stay isolated)
# ---------------------------------------------------------------------------

def defect_file(work_dir: Path, strength: float, half_width: float) -> tuple[Path, int]:
    selected = [k for k in range(len(BONDS)) if abs(BMID[k, 0] - XC) <= half_width]
    path = work_dir / "defect.txt"
    lines = ["# site partner dJ dK dGamma dGammap",
             f"# L=36 selected enhanced-K line: dK=-{strength:g}|K|, hw={half_width:g}"]
    for k in selected:
        i, j = BONDS[k]
        lines.append(f"{i} {j} 0.0 {-strength * abs(K):.8e} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path, len(selected)


def static_config(work_dir: Path, out: Path, seed: Path, deffile: Path, ndet: int) -> str:
    return f"""\
system = NCTO
lattice_size = {LATTICE},{LATTICE},1
simulation_mode = SA
output_dir = {rel(out)}
num_trials = 1
J = {J}
K = {K}
Gamma = {GAMMA}
Gammap = {GAMMAP}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {J7}
field_strength = 0.0
field_direction = 0,0,1
omega_E1 = 4.0
gamma_E1 = 0.0849
lambda_E1_K_2 = 0.0
T_start = 1e-9
T_end = 1e-9
annealing_steps = 1
cooling_rate = 0.9
overrelaxation_rate = 0
T_zero = true
n_deterministics = {ndet}
initial_spin_config = {rel(seed)}
nn_exchange_channel_disorder_config = {rel(deffile)}
"""


def sa_relax(work_dir: Path, spins: np.ndarray, deffile: Path,
             tag: str, ndet: int, force: bool) -> np.ndarray:
    out = work_dir / tag
    spin_out = out / "sample_0" / "spins_T=0.txt"
    if spin_out.exists() and not force:
        return np.loadtxt(spin_out)
    out.mkdir(parents=True, exist_ok=True)
    seed = work_dir / f"{tag}_in.txt"
    cfg = work_dir / f"{tag}.param"
    np.savetxt(seed, spins, fmt="%.10e")
    cfg.write_text(static_config(work_dir, out, seed, deffile, ndet))
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True,
                          cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not spin_out.exists():
        raise RuntimeError(f"SA relax {tag} failed: {proc.stderr[-500:]}")
    return np.loadtxt(spin_out)


def make_cfg(work_dir: Path, deffile: Path) -> Path:
    cfg = work_dir / "cfg.param"
    cfg.write_text(f"""\
system = NCTO
lattice_size = {LATTICE},{LATTICE},1
simulation_mode = SA
output_dir = {rel(work_dir)}
num_trials = 1
J = {J}
K = {K}
Gamma = {GAMMA}
Gammap = {GAMMAP}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {J7}
field_strength = 0.0
field_direction = 0,0,1
omega_E1 = 4.0
gamma_E1 = 0.0849
lambda_E1_K_2 = 0.0
nn_exchange_channel_disorder_config = {rel(deffile)}
""")
    return cfg


def band_eval(work_dir: Path, cfg: Path,
              band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    manifest = work_dir / "band_manifest.txt"
    lines = []
    for idx, image in enumerate(band):
        spin_file = work_dir / f"img_{idx}.txt"
        field_file = work_dir / f"img_{idx}_H.txt"
        np.savetxt(spin_file, image, fmt="%.10e")
        lines.append(f"{spin_file} 0.0 0.0 0 {field_file}")
    manifest.write_text("\n".join(lines) + "\n")
    subprocess.run([str(FE), str(cfg), str(manifest), str(work_dir / "band_e.csv")],
                   capture_output=True, cwd=str(ROOT), check=True)
    rows = np.loadtxt(work_dir / "band_e.csv", delimiter=",", skiprows=1)
    energies = np.atleast_2d(rows)[:, 1] * N
    fields = np.array([np.loadtxt(work_dir / f"img_{idx}_H.txt")
                       for idx in range(len(band))])
    return energies, fields


def energy_single(work_dir: Path, cfg: Path, spins: np.ndarray) -> float:
    energies, _ = band_eval(work_dir, cfg, np.array([spins]))
    return float(energies[0])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    out = np.empty_like(a)
    for idx in range(len(a)):
        if sin_omega[idx] < 1e-7:
            out[idx] = (1.0 - t) * a[idx] + t * b[idx]
        else:
            out[idx] = (np.sin((1.0 - t) * omega[idx]) * a[idx]
                        + np.sin(t * omega[idx]) * b[idx]) / sin_omega[idx]
    return rn(out)


def strip_init(width: float, wall: float = 1.2) -> np.ndarray:
    """ZZ-strip initial guess; interpolates between Q3 and ZZ inside the band."""
    dx = X - XC
    dx -= LATTICE * np.round(dx / LATTICE)
    blend = np.clip(0.5 - 0.5 * (np.abs(dx) - width) / wall, 0.0, 1.0)
    out = np.empty_like(ZZ)
    for idx, val in enumerate(blend):
        out[idx] = slerp(Q3[idx:idx + 1], ZZ[idx:idx + 1], float(val))[0]
    return rn(out)


def geodesic_len(band: np.ndarray) -> np.ndarray:
    dist = [0.0]
    for idx in range(1, len(band)):
        dist.append(dist[-1] + np.linalg.norm((band[idx] - band[idx - 1]).ravel()))
    return np.array(dist)


def reparametrize(band: np.ndarray) -> np.ndarray:
    coord = geodesic_len(band)
    total = coord[-1]
    if total < 1e-9:
        return band
    target = np.linspace(0.0, total, len(band))
    out = [band[0].copy()]
    for idx in range(1, len(band) - 1):
        value = target[idx]
        seg_idx = max(0, min(int(np.searchsorted(coord, value) - 1), len(band) - 2))
        seg = coord[seg_idx + 1] - coord[seg_idx]
        frac = 0.0 if seg < 1e-12 else (value - coord[seg_idx]) / seg
        out.append(slerp(band[seg_idx], band[seg_idx + 1], frac))
    out.append(band[-1].copy())
    return np.array(out)


def mep_barrier(work_dir: Path, cfg: Path,
                start: np.ndarray, end: np.ndarray, *,
                images: int, n_iter: int, dt: float,
                climb_start: int, plateau_win: int) -> dict:
    band = np.array([slerp(start, end, idx / (images - 1)) for idx in range(images)])
    band[0] = start
    band[-1] = end
    history = []
    last_energy = None
    last_force = None
    for it in range(n_iter):
        energy_values, fields = band_eval(work_dir, cfg, band)
        imax = 1 + int(np.argmax(energy_values[1:-1]))
        climbing = it >= climb_start
        forces = np.zeros_like(band)
        for idx in range(1, images - 1):
            tangent = (band[idx + 1] - band[idx - 1]).reshape(N, 3)
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
        barrier = float(np.max(energy_values) - energy_values[0])
        history.append({"iteration": it, "barrier_meV": barrier,
                        "saddle_image": int(np.argmax(energy_values))})
        last_energy = energy_values.copy()
        last_force = forces.copy()
        if (climbing and len(history) >= plateau_win
                and np.std([row["barrier_meV"] for row in history[-plateau_win:]]) < 0.05):
            print(f"    converged at iteration {it}", flush=True)
            break
    if last_energy is None or last_force is None:
        raise RuntimeError("MEP loop did not run")
    imax = int(np.argmax(last_energy))
    force = float(np.sqrt(np.max(np.sum(last_force[imax] ** 2, axis=1))))
    coord = geodesic_len(band)
    coord = coord / coord[-1] if coord[-1] > 0 else coord
    np.savez_compressed(work_dir / "final_band.npz", band=band, energies=last_energy,
                        path_coordinate=coord)
    return {
        "barrier_meV": float(last_energy[imax] - last_energy[0]),
        "saddle_image": imax,
        "saddle_force": force,
        "barrier_std_last_window_meV": float(
            np.std([row["barrier_meV"] for row in history[-plateau_win:]])),
        "relative_energy_meV": (last_energy - last_energy[0]).tolist(),
        "path_coordinate": coord.tolist(),
        "history_tail": history[-min(len(history), 300):],
        "iterations_completed": len(history),
    }


def zfrac(spins: np.ndarray) -> float:
    return float((np.einsum("ij,ij->i", spins, ZZ) > 0.8).mean())


# ---------------------------------------------------------------------------
# Per-width runner
# ---------------------------------------------------------------------------

def run_one_width(strength: float, half_width: float, *,
                  images: int, n_iter: int, dt: float,
                  climb_start: int, plateau_win: int, force: bool) -> dict:
    tag = f"s{strength:.3f}_hw{half_width:.2f}".replace(".", "p")
    work_dir = OUT_ROOT / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report.json"

    if report_path.exists() and not force:
        print(f"  hw={half_width:.2f}: cached — loading {rel(report_path)}", flush=True)
        return json.loads(report_path.read_text())

    print(f"  hw={half_width:.2f}: building defect file ...", flush=True)
    deffile, n_bonds = defect_file(work_dir, strength, half_width)
    cfg = make_cfg(work_dir, deffile)

    print(f"  hw={half_width:.2f}: relaxing endpoints ({n_bonds} defect bonds) ...", flush=True)
    q3_relaxed = sa_relax(work_dir, Q3, deffile, "r3q", 1200, force)
    strip_relaxed = sa_relax(work_dir, strip_init(2.5), deffile, "rstrip", 1200, force)
    q3_relaxed = sa_relax(work_dir, q3_relaxed, deffile, "r3q2", 1600, force)
    strip_relaxed = sa_relax(work_dir, strip_relaxed, deffile, "rstrip2", 1600, force)

    e3q = energy_single(work_dir, cfg, q3_relaxed)
    estrip = energy_single(work_dir, cfg, strip_relaxed)

    print(f"  hw={half_width:.2f}: running GNEB ({n_iter} iter max) ...", flush=True)
    gneb = mep_barrier(work_dir, cfg, strip_relaxed, q3_relaxed,
                       images=images, n_iter=n_iter, dt=dt,
                       climb_start=climb_start, plateau_win=plateau_win)

    report = {
        "half_width": half_width,
        "strength": strength,
        "n_bonds": n_bonds,
        "endpoints": {
            "E3Q_meV": e3q,
            "Estrip_meV": estrip,
            "E3Q_minus_Estrip_meV": e3q - estrip,
            "zfrac_3Q": zfrac(q3_relaxed),
            "zfrac_strip": zfrac(strip_relaxed),
        },
        "gneb": gneb,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  hw={half_width:.2f}: barrier={gneb['barrier_meV']:.2f} meV  "
          f"({gneb['iterations_completed']} iter)  → {rel(report_path)}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def plot_summary(results: list[dict]) -> Path:
    widths = [r["half_width"] for r in results]
    n_bonds = [r["n_bonds"] for r in results]
    barriers = [r["gneb"]["barrier_meV"] for r in results]
    e_diff = [r["endpoints"]["E3Q_minus_Estrip_meV"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(widths, barriers, "o-", color="C0", ms=7)
    # mark the 'current point'
    ax.axvline(x=2.0, color="C3", ls="--", lw=1.2, label="current (hw=2.0)")
    ax.set_xlabel("half-width (lattice units)")
    ax.set_ylabel("barrier (meV)")
    ax.set_title("3Q→ZZ kinetic barrier vs defect width")
    ax.legend(fontsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(widths)
    ax2.set_xticklabels([str(nb) for nb in n_bonds], fontsize=8)
    ax2.set_xlabel("defect bonds")

    ax = axes[1]
    ax.plot(widths, barriers, "o-", color="C0", ms=7, label="barrier")
    ax.plot(widths, [-d for d in e_diff], "s--", color="C1", ms=6,
            label="–ΔE (3Q–ZZ)")
    ax.axvline(x=2.0, color="C3", ls="--", lw=1.2)
    ax.set_xlabel("half-width (lattice units)")
    ax.set_ylabel("energy (meV)")
    ax.set_title("Barrier vs endpoint energy difference")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = OUT_ROOT / "width_sweep_barrier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNEB MEP barrier sweep over kred defect half-width")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="dK/|K| strength (default 0.5)")
    parser.add_argument("--half-width-values", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0],
                        help="half-widths to sweep (default 0.5 1.0 1.5 2.0)")
    parser.add_argument("--images", type=int, default=21)
    parser.add_argument("--n-iter", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--climb-start", type=int, default=900)
    parser.add_argument("--plateau-win", type=int, default=120)
    parser.add_argument("--force", action="store_true",
                        help="recompute even if cached output exists")
    args = parser.parse_args()

    widths = sorted(args.half_width_values)
    print(f"Width sweep: strength={args.strength}  half-widths={widths}", flush=True)

    results = []
    for hw in widths:
        print(f"\n--- half_width = {hw:.2f} ---", flush=True)
        report = run_one_width(
            args.strength, hw,
            images=args.images, n_iter=args.n_iter, dt=args.dt,
            climb_start=args.climb_start, plateau_win=args.plateau_win,
            force=args.force)
        results.append(report)

    # Summary JSON (lightweight: strip large arrays)
    summary = []
    for r in results:
        summary.append({
            "half_width": r["half_width"],
            "strength": r["strength"],
            "n_bonds": r["n_bonds"],
            "barrier_meV": r["gneb"]["barrier_meV"],
            "saddle_image": r["gneb"]["saddle_image"],
            "iterations_completed": r["gneb"]["iterations_completed"],
            "E3Q_meV": r["endpoints"]["E3Q_meV"],
            "Estrip_meV": r["endpoints"]["Estrip_meV"],
            "E3Q_minus_Estrip_meV": r["endpoints"]["E3Q_minus_Estrip_meV"],
        })
    summary_path = OUT_ROOT / "width_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    png = plot_summary(results)

    print("\n=== Width sweep summary ===")
    print(f"{'hw':>6}  {'bonds':>6}  {'barrier(meV)':>13}  {'ΔE(meV)':>10}")
    for row in summary:
        print(f"{row['half_width']:6.2f}  {row['n_bonds']:6d}  "
              f"{row['barrier_meV']:13.2f}  {row['E3Q_minus_Estrip_meV']:10.2f}")
    print(f"\nwrote {rel(summary_path)}")
    print(f"wrote {rel(png)}")


if __name__ == "__main__":
    main()
