#!/usr/bin/env python3
"""Selected 36x36 hard-fault depinning MEP.

This is the L36 counterpart of the selected L18 enhanced-K line/band check.  It
uses the solver lattice convention, tiles the trusted tuned-Kruger L18 3Q/ZZ
references to 36x36, relaxes both endpoints with the L36 defect present, and
then runs the same climbing-image string update used by the L18 catalogue.
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
OUT = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
       / "defect_catalogue_L36" / "selected_kred_mep")
WORK = OUT / "work"
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

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


def infer_nn_bonds() -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(XY)):
        delta = XY[i + 1:] - XY[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        for off in np.where(dist < NN_CUTOFF)[0]:
            bonds.append((i, i + 1 + int(off)))
    return bonds


BONDS = infer_nn_bonds()
BMID = np.array([0.5 * (XY[i] + XY[j]) for i, j in BONDS])


def defect_file(strength: float, half_width: float) -> tuple[Path, int]:
    selected = [k for k in range(len(BONDS)) if abs(BMID[k, 0] - XC) <= half_width]
    path = WORK / "defect.txt"
    lines = ["# site partner dJ dK dGamma dGammap",
             f"# L=36 selected enhanced-K line: dK=-{strength:g}|K|, hw={half_width:g}"]
    for k in selected:
        i, j = BONDS[k]
        lines.append(f"{i} {j} 0.0 {-strength * abs(K):.8e} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n")
    return path, len(selected)


def static_config(out: Path, seed: Path, deffile: Path, ndet: int) -> str:
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


def sa_relax(spins: np.ndarray, deffile: Path, tag: str, ndet: int, force: bool) -> np.ndarray:
    out = WORK / tag
    spin_out = out / "sample_0" / "spins_T=0.txt"
    if spin_out.exists() and not force:
        return np.loadtxt(spin_out)
    out.mkdir(parents=True, exist_ok=True)
    seed = WORK / f"{tag}_in.txt"
    cfg = WORK / f"{tag}.param"
    np.savetxt(seed, spins, fmt="%.10e")
    cfg.write_text(static_config(out, seed, deffile, ndet))
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not spin_out.exists():
        raise RuntimeError(f"SA relax {tag} failed: {proc.stderr[-500:]}")
    return np.loadtxt(spin_out)


def cfg_path(deffile: Path) -> Path:
    cfg = WORK / "cfg.param"
    cfg.write_text(f"""\
system = NCTO
lattice_size = {LATTICE},{LATTICE},1
simulation_mode = SA
output_dir = {rel(WORK)}
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


def band_eval(cfg: Path, band: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    manifest = WORK / "band_manifest.txt"
    lines = []
    for idx, image in enumerate(band):
        spin_file = WORK / f"img_{idx}.txt"
        field_file = WORK / f"img_{idx}_H.txt"
        np.savetxt(spin_file, image, fmt="%.10e")
        lines.append(f"{spin_file} 0.0 0.0 0 {field_file}")
    manifest.write_text("\n".join(lines) + "\n")
    subprocess.run([str(FE), str(cfg), str(manifest), str(WORK / "band_e.csv")],
                   capture_output=True, cwd=str(ROOT), check=True)
    rows = np.loadtxt(WORK / "band_e.csv", delimiter=",", skiprows=1)
    energies = np.atleast_2d(rows)[:, 1] * N
    fields = np.array([np.loadtxt(WORK / f"img_{idx}_H.txt") for idx in range(len(band))])
    return energies, fields


def energy(cfg: Path, spins: np.ndarray) -> float:
    energies, _fields = band_eval(cfg, np.array([spins]))
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


def strip(width: float, wall: float = 1.2) -> np.ndarray:
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


def mep_barrier(cfg: Path, start: np.ndarray, end: np.ndarray, *, images: int,
                n_iter: int, dt: float, climb_start: int,
                plateau_win: int) -> dict:
    band = np.array([slerp(start, end, idx / (images - 1)) for idx in range(images)])
    band[0] = start
    band[-1] = end
    history = []
    last_energy = None
    last_force = None
    for it in range(n_iter):
        energy_values, fields = band_eval(cfg, band)
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
            break
    if last_energy is None or last_force is None:
        raise RuntimeError("MEP loop did not run")
    imax = int(np.argmax(last_energy))
    force = float(np.sqrt(np.max(np.sum(last_force[imax] ** 2, axis=1))))
    coord = geodesic_len(band)
    coord = coord / coord[-1] if coord[-1] > 0 else coord
    np.savez_compressed(WORK / "final_band.npz", band=band, energies=last_energy,
                        path_coordinate=coord)
    return {
        "barrier_meV": float(last_energy[imax] - last_energy[0]),
        "saddle_image": imax,
        "saddle_force": force,
        "barrier_std_last_window_meV": float(np.std([row["barrier_meV"] for row in history[-plateau_win:]])),
        "relative_energy_meV": (last_energy - last_energy[0]).tolist(),
        "path_coordinate": coord.tolist(),
        "history_tail": history[-min(len(history), 300):],
        "iterations_completed": len(history),
    }


def zfrac(spins: np.ndarray) -> float:
    return float((np.einsum("ij,ij->i", spins, ZZ) > 0.8).mean())


def plot_report(report: dict) -> None:
    path = np.array(report["gneb"]["path_coordinate"])
    energy_values = np.array(report["gneb"]["relative_energy_meV"])
    saddle = int(report["gneb"]["saddle_image"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(path, energy_values, "o-", lw=1.7, ms=4.2, color="#2171b5")
    ax.plot(path[saddle], energy_values[saddle], "*", ms=13, color="k")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axhline(energy_values[-1], color="#238b45", ls="--", lw=1.0)
    ax.set_xlabel("normalized GNEB arc length")
    ax.set_ylabel(r"$E-E_{\rm pinned\ ZZ}$ (meV)")
    ax.set_title(r"$36\times36$ selected enhanced-$K$ MEP")
    ax.grid(alpha=0.25, lw=0.5)
    ax.annotate(rf"$U_{{\rm pin}}={report['gneb']['barrier_meV']:.2f}$ meV",
                xy=(path[saddle], energy_values[saddle]), xytext=(0.35, 0.80),
                textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "selected_kred_mep_L36.png", dpi=220)
    fig.savefig(OUT / "selected_kred_mep_L36.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--half-width", type=float, default=2.0)
    parser.add_argument("--images", type=int, default=21)
    parser.add_argument("--n-iter", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.006)
    parser.add_argument("--climb-start", type=int, default=900)
    parser.add_argument("--plateau-win", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    deffile, n_bonds = defect_file(args.strength, args.half_width)
    cfg = cfg_path(deffile)
    q3_relaxed = sa_relax(Q3, deffile, "r3q", 1200, args.force)
    strip_relaxed = sa_relax(strip(2.5), deffile, "rstrip", 1200, args.force)
    q3_relaxed = sa_relax(q3_relaxed, deffile, "r3q2", 1600, args.force)
    strip_relaxed = sa_relax(strip_relaxed, deffile, "rstrip2", 1600, args.force)
    e3q = energy(cfg, q3_relaxed)
    estrip = energy(cfg, strip_relaxed)
    report = {
        "geometry": {"lattice": LATTICE, "n_sites": N, "dtype": "kred",
                     "strength": args.strength, "half_width": args.half_width,
                     "n_bonds": n_bonds},
        "endpoints": {"E3Q_meV": e3q, "Estrip_meV": estrip,
                      "E3Q_minus_Estrip_meV": e3q - estrip,
                      "zfrac_3Q": zfrac(q3_relaxed),
                      "zfrac_strip": zfrac(strip_relaxed)},
    }
    report["gneb"] = mep_barrier(cfg, strip_relaxed, q3_relaxed,
                                  images=args.images, n_iter=args.n_iter,
                                  dt=args.dt, climb_start=args.climb_start,
                                  plateau_win=args.plateau_win)
    (OUT / "selected_kred_mep_L36_report.json").write_text(json.dumps(report, indent=2))
    plot_report(report)
    print(json.dumps({"barrier_meV": report["gneb"]["barrier_meV"],
                      "saddle_image": report["gneb"]["saddle_image"],
                      "E3Q_minus_Estrip_meV": report["endpoints"]["E3Q_minus_Estrip_meV"],
                      "n_bonds": n_bonds,
                      "iterations": report["gneb"]["iterations_completed"]}, indent=2))
    print(f"wrote {rel(OUT / 'selected_kred_mep_L36_report.json')}")
    print(f"wrote {rel(OUT / 'selected_kred_mep_L36.png')}")


if __name__ == "__main__":
    main()