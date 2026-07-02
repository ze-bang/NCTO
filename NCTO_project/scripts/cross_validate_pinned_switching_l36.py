#!/usr/bin/env python3
"""L36 drive/disorder switching cross-check for enhanced-|K| line disorder.

This is deliberately separate from cross_validate_pinned_switching.py, whose
defect catalogue helpers are tied to cached L18 GNEB endpoints.  Here all
geometry, analytic 3Q/ZZ references, disorder files, relaxations, MD runs, and
quenches are generated for a 36x36 honeycomb lattice.  The default final-story
protocol uses the same enhanced-|K| line as the lifetime calculation and samples
selected-bond dK values from a Gaussian centred on the chosen offset.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "spin_solver"
NP = ROOT / "NCTO_project" / "scripts" / "np_campaign"
sys.path.insert(0, str(NP))
from analysis_utils import M_VECTORS  # noqa: E402
from initial_states import triple_q_state, single_zigzag_state  # noqa: E402

sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from reader_strain_lattice import generate_honeycomb_positions as honeycomb_positions  # noqa: E402

# Tuned-Kruger Hamiltonian and E1 drive point.
J, K, GAMMA, GAMMAP = 0.68, -7.89, 3.07, -2.94
J2_A, J2_B, J3, J7 = -0.06, -0.70, 0.52, -0.40
# Isotropic Grüneisen magnetoelastic coupling for the quadratic E1 striction model
# dX_gamma(eps) = lambda_{X,2} * [(eps_x^2-eps_y^2)cos2theta + 2 eps_x eps_y sin2theta].
# Uniform Grueneisen means a common fractional modulation dX/X, i.e. the deformation
# potential scales linearly (and *with sign*) with the bare exchange constant:
#   lambda_{X,2} = (X/K) * lambda_{K,2}.
# (The earlier (|X|/|K|) choice used the wrong sign on the J and Gamma channels.)
LAMBDA_K2 = 0.02
LAMBDA_J2   = LAMBDA_K2 * 0.68  / -7.89   # = -0.001724
LAMBDA_G2   = LAMBDA_K2 * 3.07  / -7.89   # = -0.007782
LAMBDA_GP2  = LAMBDA_K2 * -2.94 / -7.89   # = +0.007451
# Tag appended to every run/config/quench directory so all-channel trajectories
# never collide with cached K-only directories that share the same E0/sigma/seed.
COUPLING_TAG = "allJKGG"
OMEGA_E1 = 4.0
GAMMA_E1 = 0.08488263631567752
T0_PUMP = 50.0
SIGMA_PUMP = 15.0
ALPHA = 0.05
MD_END = 150.0
DT = 0.005
SAVE_INTERVAL = 100
NN_CUTOFF = 0.65

OUTROOT = ROOT / "NCTO_project" / "tuned_kruger_campaign" / "pinning_switching_crosscheck_L36"
CFG_DIR = OUTROOT / "configs"
RUN_DIR = OUTROOT / "runs"
DEF_DIR = OUTROOT / "disorder_files"
INIT_DIR = OUTROOT / "initial_states"
Q_DIR = OUTROOT / "quenches"
ANALYSIS = OUTROOT / "analysis"
for directory in (CFG_DIR, RUN_DIR, DEF_DIR, INIT_DIR, Q_DIR, ANALYSIS):
    directory.mkdir(parents=True, exist_ok=True)

REF_3Q_L18 = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "phase_diagram"
              / "analysis" / "seed3Q_J70400.txt")
REF_ZZ_L18 = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "kinetic_barrier"
              / "zz_relax" / "sample_0" / "spins_T=0.txt")


def rel(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def infer_nn_bonds(positions: np.ndarray) -> list[tuple[int, int]]:
    xy = positions[:, :2]
    bonds: list[tuple[int, int]] = []
    for i in range(len(xy)):
        delta = xy[i + 1:] - xy[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        for off in np.where(dist < NN_CUTOFF)[0]:
            bonds.append((i, i + 1 + int(off)))
    return bonds


def bond_class(xy: np.ndarray, i: int, j: int) -> int:
    vec = xy[j] - xy[i]
    angle = np.degrees(np.arctan2(vec[1], vec[0])) % 180.0
    return int(round((angle - 30.0) / 60.0)) % 3


def tile_l18_reference(path: Path, lattice: int) -> np.ndarray:
    if lattice % 18 != 0:
        raise ValueError("L36 workflow expects a lattice that tiles the trusted L18 references.")
    reference = np.loadtxt(path).reshape(18, 18, 2, 3)
    reps = lattice // 18
    tiled = np.tile(reference, (reps, reps, 1, 1)).reshape(2 * lattice * lattice, 3)
    norm = np.linalg.norm(tiled, axis=1)
    return tiled / norm[:, None]


class LatticeContext:
    def __init__(self, lattice: int):
        self.lattice = lattice
        self.n_sites = 2 * lattice * lattice
        self.positions = honeycomb_positions(self.n_sites)
        self.xy = self.positions[:, :2]
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]
        self.xc = 0.5 * (float(self.x.min()) + float(self.x.max()))
        self.bonds = infer_nn_bonds(self.positions)
        self.classes = np.array([bond_class(self.xy, i, j) for i, j in self.bonds])
        self.bmid = np.array([0.5 * (self.xy[i] + self.xy[j]) for i, j in self.bonds])
        self.q3 = triple_q_state(lattice, lattice)
        self.zz = tile_l18_reference(REF_ZZ_L18, lattice)  # physical relaxed ZZ, not analytic ±z


def r3_of(spins: np.ndarray, ctx: LatticeContext) -> float:
    values = []
    for qvec in M_VECTORS:
        phase = ctx.xy[:, 0] * qvec[0] + ctx.xy[:, 1] * qvec[1]
        sq = np.sum(spins * np.exp(1j * phase[:, None]), axis=0)
        values.append(float(np.real(sq @ np.conj(sq))) / ctx.n_sites)
    return min(values) / max(values) if max(values) > 0 else 1.0


def selected_bond_indices(ctx: LatticeContext, dtype: str, half_width: float) -> list[int]:
    selected = [k for k in range(len(ctx.bonds)) if abs(ctx.bmid[k, 0] - ctx.xc) <= half_width]
    if dtype == "nematic":
        selected = [k for k in selected if ctx.classes[k] == 0]
    return selected


def defect_increment(dtype: str, strength: float) -> tuple[float, float, float, float]:
    if dtype == "kred":
        return 0.0, -strength * abs(K), 0.0, 0.0
    if dtype == "kflip":
        return 0.0, +2.0 * strength * abs(K), 0.0, 0.0
    if dtype == "vacancy":
        return -strength * J, -strength * K, -strength * GAMMA, -strength * GAMMAP
    if dtype == "nematic":
        return 0.0, +strength * abs(K), 0.0, 0.0
    raise ValueError(dtype)


def build_disorder_file(ctx: LatticeContext, dtype: str, strength: float, half_width: float,
                        sigma_k: float, seed: int, disorder_mode: str,
                        force: bool = False) -> tuple[Path, np.ndarray]:
    mode_tag = "" if disorder_mode == "selected-centered" else f"_{disorder_mode}"
    tag = (f"L{ctx.lattice}_{dtype}_s{strength:.3f}_hw{half_width:.2f}"
           f"{mode_tag}_sig{sigma_k:.3f}_seed{seed:03d}").replace(".", "p")
    path = DEF_DIR / f"{tag}.txt"
    selected = selected_bond_indices(ctx, dtype, half_width)
    selected_set = set(selected)
    if path.exists() and not force:
        return path, np.array(selected, dtype=int)
    rng = np.random.default_rng(seed)
    dJ0, dK0, dG0, dGp0 = defect_increment(dtype, strength)
    with path.open("w") as handle:
        handle.write("# site partner dJ dK dGamma dGammap\n")
        if disorder_mode == "selected-centered":
            mode_text = "selected-bond dK Gaussian centered on the fixed offset"
        elif disorder_mode == "global-zero-k":
            mode_text = "fixed selected-bond offset plus zero-mean dK Gaussian on all NN bonds"
        else:
            raise ValueError(disorder_mode)
        handle.write(f"# L={ctx.lattice}; fixed defect dtype={dtype} strength={strength:g} "
                     f"half_width={half_width:g}; {mode_text}; sigma_K={sigma_k:g}; seed={seed}\n")
        for k, (i, j) in enumerate(ctx.bonds):
            dJ = dG = dGp = 0.0
            dK = 0.0
            if disorder_mode == "global-zero-k" and sigma_k > 0:
                dK += rng.normal(0.0, sigma_k)
            if k in selected_set:
                dJ += dJ0
                dK += dK0
                if disorder_mode == "selected-centered" and sigma_k > 0:
                    dK += rng.normal(0.0, sigma_k)
                dG += dG0
                dGp += dGp0
            if any(abs(v) > 1e-14 for v in (dJ, dK, dG, dGp)):
                handle.write(f"{i} {j} {dJ:.12e} {dK:.12e} {dG:.12e} {dGp:.12e}\n")
    return path, np.array(selected, dtype=int)


def write_static_config(ctx: LatticeContext, out: Path, initial: Path,
                        disorder_file: Path, n_deterministics: int) -> str:
    return f"""\
system = NCTO
lattice_size = {ctx.lattice},{ctx.lattice},1
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
omega_E1 = {OMEGA_E1}
gamma_E1 = {GAMMA_E1}
lambda_E1_K_2 = 0.0
T_start = 1e-9
T_end = 1e-9
annealing_steps = 1
cooling_rate = 0.9
overrelaxation_rate = 0
T_zero = true
n_deterministics = {n_deterministics}
initial_spin_config = {rel(initial)}
nn_exchange_channel_disorder_config = {rel(disorder_file)}
"""


def relax_initial_3q(ctx: LatticeContext, disorder_file: Path, tag: str, force: bool) -> Path:
    out = INIT_DIR / tag
    spin_out = out / "sample_0" / "spins_T=0.txt"
    if spin_out.exists() and not force:
        return spin_out
    out.mkdir(parents=True, exist_ok=True)
    seed_path = out / "seed_3q.txt"
    np.savetxt(seed_path, ctx.q3, fmt="%.10e")
    cfg = out / "relax_3q.param"
    cfg.write_text(write_static_config(ctx, out, seed_path, disorder_file, 1200))
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not spin_out.exists():
        raise RuntimeError(f"failed to relax L{ctx.lattice} 3Q for {tag}: {proc.stderr[-500:]}")
    return spin_out


def band_site_mask(ctx: LatticeContext, half_width: float, shell: float = 1.5) -> np.ndarray:
    return np.abs(ctx.x - ctx.xc) <= half_width + shell


@dataclass(frozen=True)
class DriveRun:
    lattice: int
    dtype: str
    strength: float
    half_width: float
    e0: float
    sigma_k: float
    seed: int
    disorder_mode: str = "selected-centered"

    @property
    def label(self) -> str:
        mode_tag = "" if self.disorder_mode == "selected-centered" else f"_{self.disorder_mode}"
        return (f"L{self.lattice}_{self.dtype}_s{self.strength:.3f}_hw{self.half_width:.2f}"
            f"{mode_tag}_{COUPLING_TAG}_E{self.e0:.2f}_sig{self.sigma_k:.3f}_seed{self.seed:03d}").replace(".", "p")


def write_md_config(ctx: LatticeContext, run: DriveRun, disorder_file: Path, init_spin: Path) -> Path:
    cfg = CFG_DIR / f"{run.label}.param"
    out = RUN_DIR / run.label
    cfg.write_text(f"""\
system = NCTO
lattice_size = {ctx.lattice},{ctx.lattice},1
simulation_mode = MD
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
omega_E1 = {OMEGA_E1}
gamma_E1 = {GAMMA_E1}
lambda_E1_quartic = 0.0
Z_star = 1.0
lambda_E1_K_2 = {LAMBDA_K2:.12g}
lambda_E1_J_2 = {LAMBDA_J2:.12g}
lambda_E1_Gamma_2 = {LAMBDA_G2:.12g}
lambda_E1_Gammap_2 = {LAMBDA_GP2:.12g}
lambda_E1_K_0 = 0.0
lambda_E1_J_0 = 0.0
lambda_E1_Gamma_0 = 0.0
lambda_E1_Gammap_0 = 0.0
lambda_E1_J7_0 = 0.0
alpha_gilbert = {ALPHA}
langevin_temperature = 0.0
pump_amplitude = {run.e0:.12g}
pump_frequency = {OMEGA_E1:.12g}
pump_time = {T0_PUMP}
pump_width = {SIGMA_PUMP}
pump_phase = 0.0
pump_polarization = 0.0
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = {MD_END}
md_timestep = {DT}
md_save_interval = {SAVE_INTERVAL}
md_integrator = rk4
md_abs_tol = 1e-8
md_rel_tol = 1e-8
initial_spin_config = {rel(init_spin)}
nn_exchange_channel_disorder_config = {rel(disorder_file)}
relax_phonons = false
adiabatic_phonons = false
phonon_only_relax = false
""")
    return cfg


def run_solver(cfg: Path, output_h5: Path, force: bool) -> bool:
    if output_h5.exists() and not force:
        try:
            import h5py
            with h5py.File(output_h5, "r") as h5:
                _ = h5["trajectory/spins"][-1]
            return True
        except Exception:
            pass
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT), text=True)
    if proc.returncode != 0:
        print(f"[FAIL] {cfg.stem}: {proc.stderr[-500:]}", flush=True)
        return False
    return True


def quench_final(ctx: LatticeContext, spins: np.ndarray, run: DriveRun,
                 disorder_file: Path, force: bool) -> Path | None:
    qdir = Q_DIR / run.label
    qspin = qdir / "sample_0" / "spins_T=0.txt"
    if qspin.exists() and not force:
        return qspin
    qdir.mkdir(parents=True, exist_ok=True)
    seed = qdir / "final_in.txt"
    np.savetxt(seed, spins, fmt="%.10e")
    cfg = qdir / "quench.param"
    cfg.write_text(write_static_config(ctx, qdir, seed, disorder_file, 800))
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not qspin.exists():
        print(f"[QFAIL] {run.label}: {proc.stderr[-500:]}", flush=True)
        return None
    return qspin


def texture_metrics(ctx: LatticeContext, spins: np.ndarray, band_mask: np.ndarray) -> dict[str, float]:
    zz = np.einsum("ij,ij->i", spins, ctx.zz) > 0.8
    off_mask = ~band_mask
    on_density = float(zz[band_mask].mean()) if np.any(band_mask) else 0.0
    off_density = float(zz[off_mask].mean()) if np.any(off_mask) else 0.0
    zz_total = int(zz.sum())
    zz_on_band_pct = 100.0 * int((zz & band_mask).sum()) / zz_total if zz_total else 0.0
    enrichment = on_density / max(off_density, 1e-9)
    return dict(r3=r3_of(spins, ctx), zz_frac=float(zz.mean()),
                zz_on_band_density=on_density,
                zz_off_band_density=off_density,
                zz_on_band_pct=zz_on_band_pct,
                pinning_enrichment=enrichment)


def analyze_one(ctx: LatticeContext, run: DriveRun, disorder_file: Path,
                init_spin: Path, band_mask: np.ndarray, force: bool,
                no_quench: bool) -> dict | None:
    cfg = write_md_config(ctx, run, disorder_file, init_spin)
    h5_path = RUN_DIR / run.label / "sample_0" / "trajectory.h5"
    if not run_solver(cfg, h5_path, force):
        return None
    import h5py
    with h5py.File(h5_path, "r") as h5:
        final = h5["trajectory/spins"][-1]
    rec = dict(lattice=ctx.lattice, dtype=run.dtype, strength=run.strength,
               half_width=run.half_width, e0=run.e0, sigma_k=run.sigma_k,
               seed=run.seed, label=run.label)
    for key, value in texture_metrics(ctx, final, band_mask).items():
        rec[f"final_{key}"] = value
    if not no_quench:
        qspin = quench_final(ctx, final, run, disorder_file, force)
        if qspin is not None:
            quenched = np.loadtxt(qspin)
            for key, value in texture_metrics(ctx, quenched, band_mask).items():
                rec[f"quench_{key}"] = value
    qzz = rec.get("quench_zz_frac", rec["final_zz_frac"])
    qon = rec.get("quench_zz_on_band_density", rec["final_zz_on_band_density"])
    qoff = rec.get("quench_zz_off_band_density", rec["final_zz_off_band_density"])
    rec["written"] = int(qzz > 0.06)
    rec["pinned_written"] = int((qzz > 0.06) and (qon > max(0.10, 2.0 * qoff)))
    return rec


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["e0"], row["sigma_k"]), []).append(row)
    baseline_by_seed: dict[tuple[float, int], dict[str, float]] = {}
    for (e0, sigma_k), rs in groups.items():
        if abs(e0) < 1e-12:
            for r in rs:
                baseline_by_seed[(sigma_k, int(r["seed"]))] = dict(
                    qzz=r.get("quench_zz_frac", r["final_zz_frac"]),
                    on=r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]),
                    off=r.get("quench_zz_off_band_density", r["final_zz_off_band_density"]),
                )
    baselines: dict[float, dict[str, float]] = {}
    for (e0, sigma_k), rs in groups.items():
        if abs(e0) < 1e-12:
            baselines[sigma_k] = dict(
                mean_quench_zz=float(np.mean([r.get("quench_zz_frac", r["final_zz_frac"]) for r in rs])),
                mean_on_band_density=float(np.mean([
                    r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]) for r in rs])),
            )
    out = []
    for (e0, sigma_k), rs in sorted(groups.items()):
        drive_written = []
        drive_pinned = []
        for r in rs:
            qzz = r.get("quench_zz_frac", r["final_zz_frac"])
            on = r.get("quench_zz_on_band_density", r["final_zz_on_band_density"])
            off = r.get("quench_zz_off_band_density", r["final_zz_off_band_density"])
            base = baseline_by_seed.get((sigma_k, int(r["seed"])), dict(qzz=0.0, on=0.0, off=0.0))
            excess_qzz = qzz - base["qzz"]
            excess_on = on - base["on"]
            is_written = excess_qzz > 0.05
            is_pinned = is_written and excess_on > 0.05 and on > max(0.10, 2.0 * off)
            drive_written.append(float(is_written))
            drive_pinned.append(float(is_pinned))
        mean_quench_zz = float(np.mean([r.get("quench_zz_frac", r["final_zz_frac"]) for r in rs]))
        mean_on_density = float(np.mean([r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]) for r in rs]))
        baseline = baselines.get(sigma_k, dict(mean_quench_zz=0.0, mean_on_band_density=0.0))
        out.append(dict(
            lattice=int(rs[0].get("lattice", 36)),
            e0=e0,
            sigma_k=sigma_k,
            n=len(rs),
            write_fraction=float(np.mean([r["written"] for r in rs])),
            pinned_write_fraction=float(np.mean([r["pinned_written"] for r in rs])),
            drive_write_fraction=float(np.mean(drive_written)),
            drive_pinned_fraction=float(np.mean(drive_pinned)),
            mean_quench_zz=mean_quench_zz,
            excess_quench_zz=mean_quench_zz - baseline["mean_quench_zz"],
            mean_on_band_density=mean_on_density,
            excess_on_band_density=mean_on_density - baseline["mean_on_band_density"],
            mean_on_band_pct=float(np.mean([r.get("quench_zz_on_band_pct", r["final_zz_on_band_pct"]) for r in rs])),
            mean_enrichment=float(np.mean([
                r.get("quench_pinning_enrichment", r["final_pinning_enrichment"]) for r in rs])),
        ))
    return out


def output_paths(ctx: LatticeContext, dtype: str, strength: float, half_width: float,
                 suffix: str = "") -> tuple[Path, Path]:
    tag = (f"L{ctx.lattice}_{dtype}_s{strength:.3f}_hw{half_width:.2f}").replace(".", "p")
    if suffix:
        clean_suffix = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in suffix)
        tag = f"{tag}_{clean_suffix}"
    return (ANALYSIS / f"{tag}_drive_crosscheck.csv",
            ANALYSIS / f"{tag}_drive_crosscheck_summary.json")


def write_outputs(ctx: LatticeContext, dtype: str, strength: float, half_width: float,
                  rows: list[dict], suffix: str = "") -> tuple[Path, Path]:
    csv_path, summary_path = output_paths(ctx, dtype, strength, half_width, suffix)
    ordered = sorted(rows, key=lambda row: (float(row["sigma_k"]), int(row["seed"]), float(row["e0"])))
    fieldnames: list[str] = []
    for row in ordered:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    summary_path.write_text(json.dumps(aggregate(ordered), indent=2))
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", type=int, default=36)
    parser.add_argument("--dtype", choices=["kred", "kflip", "vacancy", "nematic"], default="nematic")
    parser.add_argument("--strength", type=float, default=0.10)
    parser.add_argument("--half-width", type=float, default=4.5)
    parser.add_argument("--e0-values", nargs="+", type=float,
                        default=[round(0.1 * i, 1) for i in range(21)])
    parser.add_argument("--sigma-k-values", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean-initial", action="store_true")
    parser.add_argument("--no-quench", action="store_true")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--disorder-mode", choices=["selected-centered", "global-zero-k"],
                        default="selected-centered",
                        help="selected-centered: Gaussian only on selected line bonds around the fixed offset; "
                             "global-zero-k: fixed line offset plus zero-mean dK Gaussian on every NN bond")
    args = parser.parse_args()

    ctx = LatticeContext(args.lattice)
    band_mask = band_site_mask(ctx, args.half_width)

    # Phase 1: build disorder files serially (fast – just file writes / cache hits)
    seed_sigma_items: list[tuple[float, int, Path, str]] = []
    for sigma_k in args.sigma_k_values:
        seed_range = range(args.seeds if sigma_k > 0 else 1)
        for seed in seed_range:
            disorder_file, _selected = build_disorder_file(
                ctx, args.dtype, args.strength, args.half_width, sigma_k, seed,
                args.disorder_mode, args.force)
            tag = (f"L{ctx.lattice}_{args.dtype}_s{args.strength:.3f}_hw{args.half_width:.2f}"
                   f"_sig{sigma_k:.3f}_seed{seed:03d}").replace(".", "p")
            seed_sigma_items.append((sigma_k, seed, disorder_file, tag))

    # Phase 2: relax initial 3Q states in parallel (bottleneck for large --seeds)
    def _relax_item(item: tuple[float, int, Path, str]) -> tuple[float, int, Path, Path]:
        sigma_k, seed, disorder_file, tag = item
        if args.clean_initial:
            init_dir = INIT_DIR / f"L{ctx.lattice}_clean_seed"
            init_dir.mkdir(parents=True, exist_ok=True)
            init_spin = init_dir / "seed_3q.txt"
            if args.force or not init_spin.exists():
                np.savetxt(init_spin, ctx.q3, fmt="%.10e")
        else:
            init_spin = relax_initial_3q(ctx, disorder_file, tag, args.force)
        return sigma_k, seed, disorder_file, init_spin

    n_relax = len(seed_sigma_items)
    n_relax_workers = min(args.workers, n_relax)
    print(f"relaxing {n_relax} initial states with {n_relax_workers} workers ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_relax_workers) as rpool:
        relax_results = list(rpool.map(_relax_item, seed_sigma_items))
    init_map: dict[tuple[float, int], tuple[Path, Path]] = {
        (r[0], r[1]): (r[2], r[3]) for r in relax_results}

    # Phase 3: build jobs list
    jobs: list[tuple[DriveRun, Path, Path]] = []
    for sigma_k in args.sigma_k_values:
        seed_range = range(args.seeds if sigma_k > 0 else 1)
        for seed in seed_range:
            disorder_file, init_spin = init_map[(sigma_k, seed)]
            for e0 in args.e0_values:
                jobs.append((DriveRun(ctx.lattice, args.dtype, args.strength, args.half_width,
                                      e0, sigma_k, seed, args.disorder_mode),
                             disorder_file, init_spin))

    print(f"candidate: L{ctx.lattice} {args.dtype} strength={args.strength:g} "
          f"half_width={args.half_width:g}; band sites={int(band_mask.sum())}; "
          f"bonds={len(ctx.bonds)}; jobs={len(jobs)}", flush=True)
    rows: list[dict] = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(analyze_one, ctx, run, disorder_file, init_spin,
                                   band_mask, args.force, args.no_quench)
                       for run, disorder_file, init_spin in jobs]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                rec = future.result()
                if rec is not None:
                    rows.append(rec)
                    write_outputs(ctx, args.dtype, args.strength, args.half_width,
                                  rows, args.output_suffix)
                    print(f"[{idx}/{len(jobs)}] E0={rec['e0']:g} sig={rec['sigma_k']:g} "
                          f"seed={rec['seed']} qZZ={rec.get('quench_zz_frac', rec['final_zz_frac']):.3f} "
                          f"on%={rec.get('quench_zz_on_band_pct', rec['final_zz_on_band_pct']):.1f} "
                          f"pin={rec['pinned_written']}", flush=True)
    else:
        for idx, (run, disorder_file, init_spin) in enumerate(jobs, 1):
            rec = analyze_one(ctx, run, disorder_file, init_spin, band_mask,
                              args.force, args.no_quench)
            if rec is not None:
                rows.append(rec)
                write_outputs(ctx, args.dtype, args.strength, args.half_width,
                              rows, args.output_suffix)
                print(f"[{idx}/{len(jobs)}] E0={rec['e0']:g} sig={rec['sigma_k']:g} "
                      f"seed={rec['seed']} qZZ={rec.get('quench_zz_frac', rec['final_zz_frac']):.3f} "
                      f"on%={rec.get('quench_zz_on_band_pct', rec['final_zz_on_band_pct']):.1f} "
                      f"pin={rec['pinned_written']}", flush=True)
    if not rows:
        print("no successful rows", flush=True)
        return
    csv_path, summary_path = output_paths(ctx, args.dtype, args.strength, args.half_width, args.output_suffix)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== L36 cross-check summary ===")
    print("   E0  sigK     n  drive  dpin  <ZZ>  d<ZZ>  <on%>  d<on>  <enrich>")
    for row in summary:
        print(f"{row['e0']:5.2f} {row['sigma_k']:5.2f} {row['n']:5d} "
              f"{row['drive_write_fraction']:6.2f} {row['drive_pinned_fraction']:5.2f} "
              f"{row['mean_quench_zz']:5.2f} {row['excess_quench_zz']:6.2f} "
              f"{row['mean_on_band_pct']:6.1f} {row['excess_on_band_density']:6.2f} "
              f"{row['mean_enrichment']:8.1f}")
    print(f"\nwrote {rel(csv_path)}")
    print(f"wrote {rel(summary_path)}")


if __name__ == "__main__":
    main()