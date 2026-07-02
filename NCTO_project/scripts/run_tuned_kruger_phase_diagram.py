"""Tuned-Kruger 3-face switching phase diagram.

Kruger (2023) bilinear fit + retuned ring exchange J7 at the switchable
near-SU(2) degeneracy:
  J=0.68, K=-7.89, Gamma=3.07, Gamma'=-2.94, J2A=-0.06, J2B=-0.70, J3=0.52 meV
  J7 (code units) scanned around -0.40 (J7_paper = J7_code/4 = -0.10).
Axes: J7 (3Q robustness), lambda_K2 (E2 magnetoelastic), E0 (fluence).
Each J7 uses a VERIFIED 3Q ground state. Long-time r3<0.2 => switched.
"""
from __future__ import annotations
import argparse, subprocess, csv, sys, concurrent.futures
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "spin_solver"
CAMP = ROOT / "NCTO_project" / "tuned_kruger_campaign" / "phase_diagram"
CFG = CAMP / "configs"; RUN = CAMP / "runs"; ANA = CAMP / "analysis"
# Seeds (pure-spin SA relaxations) do not depend on the magnetoelastic coupling,
# so the all-channel rerun reuses the original K-only seeds in this directory.
SEED_ANA = CAMP / "analysis"
SEED_3Q_TEMPLATE = Path("/tmp/3q_L18_init.txt")


def set_campaign(all_channels: bool) -> None:
    """Point the campaign at a dedicated directory when all channels are on so the
    original K-only data is preserved."""
    global CAMP, CFG, RUN, ANA, ALL_CHANNELS, INTEGRATOR
    ALL_CHANNELS = all_channels
    sub = "phase_diagram_allchan" if all_channels else "phase_diagram"
    CAMP = ROOT / "NCTO_project" / "tuned_kruger_campaign" / sub
    CFG = CAMP / "configs"; RUN = CAMP / "runs"; ANA = CAMP / "analysis"
    # All-channel coupling renormalizes the phonon strongly; the fixed-step rk4
    # integrator is verified stable, whereas adaptive dopri5 can collapse.
    INTEGRATOR = "rk4" if all_channels else "dopri5"

BILINEAR = dict(J=0.68, K=-7.89, Gamma=3.07, Gammap=-2.94,
                J2A=-0.06, J2B=-0.70, J3=0.52)
# Isotropic Grueneisen magnetoelastic coupling for the quadratic E1 striction model
# dX_gamma(eps) = lambda2_X * [(eps_x^2-eps_y^2)cos2theta + 2 eps_x eps_y sin2theta].
# Uniform Grueneisen means a common fractional modulation dX/X, i.e. the deformation
# potential scales linearly (and *with sign*) with the bare exchange constant:
#   lambda2_X = (X/K) * lambda2_K.  The lambda axis below is lam = lambda2_K.
# (The earlier (|X|/|K|) choice used the wrong sign on the J and Gamma channels.)
ME_RATIOS = {"K": 1.0,
             "J": BILINEAR["J"] / BILINEAR["K"],
             "Gamma": BILINEAR["Gamma"] / BILINEAR["K"],
             "Gammap": BILINEAR["Gammap"] / BILINEAR["K"]}
# Switched on by --all-channels (otherwise only the K channel is modulated).
ALL_CHANNELS = False
INTEGRATOR = "dopri5"
OMEGA = 4.0; GAMMA_E1 = 0.0849; ALPHA = 0.05; MD_END = 150.0
J7_GRID = [-0.40, -0.45, -0.50, -0.55, -0.60, -0.70]
LAMBDA_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
E0_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0]


def me_lambda_lines(lam: float) -> str:
    """Magnetoelastic lambda block for the .param config at K-channel value lam."""
    if ALL_CHANNELS:
        return (f"lambda_E1_K_2 = {ME_RATIOS['K'] * lam:.12g}\n"
                f"lambda_E1_J_2 = {ME_RATIOS['J'] * lam:.12g}\n"
                f"lambda_E1_Gamma_2 = {ME_RATIOS['Gamma'] * lam:.12g}\n"
                f"lambda_E1_Gammap_2 = {ME_RATIOS['Gammap'] * lam:.12g}")
    return (f"lambda_E1_K_2 = {lam:.12g}\n"
            f"lambda_E1_J_2 = 0.0\n"
            f"lambda_E1_Gamma_2 = 0.0\n"
            f"lambda_E1_Gammap_2 = 0.0")

sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from reader_strain_lattice import generate_honeycomb_positions  # noqa
POS = generate_honeycomb_positions(648)[:, :2]
B1 = 2*np.pi*np.array([1.0, -1/np.sqrt(3)]); B2 = 2*np.pi*np.array([0.0, 2/np.sqrt(3)])
MV = [B1/2, B2/2, (B1+B2)/2]


def r3_of(sp):
    sm = [float(np.real(np.dot(
        np.sum(sp*np.exp(1j*(POS[:, 0]*q[0]+POS[:, 1]*q[1])[:, None]), axis=0),
        np.conj(np.sum(sp*np.exp(1j*(POS[:, 0]*q[0]+POS[:, 1]*q[1])[:, None]), axis=0))))) / 648 for q in MV]
    return min(sm)/max(sm)


def seed_for(j7):
    # Seeds are pure-spin SA relaxations, independent of the magnetoelastic
    # coupling, so they always live in (and are reused from) the K-only campaign.
    SEED_ANA.mkdir(parents=True, exist_ok=True)
    seed = SEED_ANA / f"seed3Q_J7{int(abs(j7)*1000):04d}.txt"
    if seed.exists():
        return seed, r3_of(np.loadtxt(seed))
    odir = SEED_ANA / f"relax_J7{int(abs(j7)*1000):04d}"
    cfg = SEED_ANA / f"relax_J7{int(abs(j7)*1000):04d}.param"
    b = BILINEAR
    cfg.write_text(f"""system = NCTO
simulation_mode = SA
lattice_size = 18,18,1
output_dir = {odir}
num_trials = 1
J = {b['J']}
K = {b['K']}
Gamma = {b['Gamma']}
Gammap = {b['Gammap']}
J2_A = {b['J2A']}
J2_B = {b['J2B']}
J3 = {b['J3']}
J7 = {j7}
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = 0.0
omega_E1 = 4.0
gamma_E1 = 0.0849
Z_star = 1.0
pump_amplitude = 0.0
probe_amplitude = 0.0
adiabatic_phonons = false
relax_phonons = false
phonon_only_relax = false
T_start = 1e-9
T_end = 1e-9
annealing_steps = 1
cooling_rate = 0.9
overrelaxation_rate = 0
T_zero = true
n_deterministics = 4000
initial_spin_config = {SEED_3Q_TEMPLATE}
""")
    subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT))
    sp = np.loadtxt(odir / "sample_0" / "spins_T=0.txt")
    np.savetxt(seed, sp, fmt="%.12e")
    return seed, r3_of(sp)


def label_of(j7, lam, e0):
    return f"J7{int(abs(j7)*1000):04d}_lK{int(lam*1000):03d}_E{int(e0*100):05d}"


def write_md(j7, lam, e0, seed):
    CFG.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    b = BILINEAR; lab = label_of(j7, lam, e0)
    cfg = CFG / f"{lab}.param"
    cfg.write_text(f"""system = NCTO
lattice_size = 18,18,1
simulation_mode = MD
output_dir = {(RUN / lab).relative_to(ROOT)}
num_trials = 1
J = {b['J']}
K = {b['K']}
Gamma = {b['Gamma']}
Gammap = {b['Gammap']}
J2_A = {b['J2A']}
J2_B = {b['J2B']}
J3 = {b['J3']}
J7 = {j7}
field_strength = 0.0
field_direction = 0,0,1
omega_E1 = {OMEGA}
gamma_E1 = {GAMMA_E1}
lambda_E1_quartic = 0.0
Z_star = 1.0
{me_lambda_lines(lam)}
lambda_E1_K_0 = 0.0
lambda_E1_J_0 = 0.0
lambda_E1_Gamma_0 = 0.0
lambda_E1_Gammap_0 = 0.0
lambda_E1_J7_0 = 0.0
alpha_gilbert = {ALPHA}
langevin_temperature = 0.0
pump_amplitude = {e0}
pump_frequency = {OMEGA}
pump_time = 50.0
pump_width = 15.0
pump_phase = 0.0
pump_polarization = 0.0
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = {MD_END}
md_timestep = 0.005
md_save_interval = 100
md_integrator = {INTEGRATOR}
md_abs_tol = 1e-8
md_rel_tol = 1e-8
initial_spin_config = {seed.relative_to(ROOT)}
relax_phonons = false
adiabatic_phonons = false
phonon_only_relax = false
""")
    return cfg


def run_md(lab):
    out = RUN / lab / "sample_0" / "trajectory.h5"
    if out.exists():
        return True
    r = subprocess.run([str(EXE), str(CFG / f"{lab}.param")], capture_output=True, text=True, cwd=str(ROOT))
    return r.returncode == 0


def analyze(j7, lam, e0):
    import h5py
    lab = label_of(j7, lam, e0)
    h5 = RUN / lab / "sample_0" / "trajectory.h5"
    if not h5.exists():
        return None
    try:
        with h5py.File(h5, "r") as f:
            sp = f["trajectory/spins"][-1]
    except Exception:
        return None
    r3 = r3_of(sp)
    return dict(J7=j7, lambda_K2=lam, E0=e0, r3=r3, switched=int(r3 < 0.2))


def make_plots(rows):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    j7s = sorted(set(r["J7"] for r in rows))
    lams = sorted(set(r["lambda_K2"] for r in rows))
    e0s = sorted(set(r["E0"] for r in rows))
    sw = {(r["J7"], r["lambda_K2"], r["E0"]): r["switched"] for r in rows}
    ncol = 3; nrow = (len(j7s)+ncol-1)//ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.4*nrow), squeeze=False)
    for k, j7 in enumerate(j7s):
        ax = axes[k//ncol][k % ncol]
        Z = np.full((len(lams), len(e0s)), np.nan)
        for i, lam in enumerate(lams):
            for j, e0 in enumerate(e0s):
                v = sw.get((j7, lam, e0))
                if v is not None:
                    Z[i, j] = v
        im = ax.pcolormesh(e0s, lams, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="nearest")
        ax.set_title(f"J7={j7:.2f} (paper {j7/4:.3f})", fontsize=9)
        ax.set_xlabel("E0"); ax.set_ylabel("lambda_K2")
    for k in range(len(j7s), nrow*ncol):
        axes[k//ncol][k % ncol].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="switched (r3<0.2)")
    fig.suptitle("Tuned-Kruger switching: (E0 x lambda_K2) per J7", fontsize=10)
    out = ANA / "phase_2d_slices.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}", flush=True)
    pts = np.array([[r["E0"], r["lambda_K2"], r["J7"]] for r in rows], float)
    val = np.array([r["switched"] for r in rows], float)

    def idw(q):
        d = np.linalg.norm(pts[None, :, :] - q[:, None, :], axis=2) + 1e-9
        w = 1.0/d**2
        return (w*val[None, :]).sum(1)/w.sum(1)

    eg = np.linspace(min(e0s), max(e0s), 40)
    lg = np.linspace(min(lams), max(lams), 40)
    jg = np.linspace(min(j7s), max(j7s), 40)
    fig = plt.figure(figsize=(13, 4.2))
    specs = [("E0", "lambda_K2", eg, lg, 2, jg.min(), "J7=%.2f" % jg.min()),
             ("E0", "J7", eg, jg, 1, lg.max(), "lambda=%.2f" % lg.max()),
             ("lambda_K2", "J7", lg, jg, 0, eg.max(), "E0=%.0f" % eg.max())]
    for i, (xl, yl, xg, yg, fix_ax, fix_v, ttl) in enumerate(specs):
        ax = fig.add_subplot(1, 3, i+1)
        GX, GY = np.meshgrid(xg, yg)
        Q = np.zeros((GX.size, 3)); order = [a for a in range(3) if a != fix_ax]
        Q[:, order[0]] = GX.ravel(); Q[:, order[1]] = GY.ravel(); Q[:, fix_ax] = fix_v
        Z = idw(Q).reshape(GX.shape)
        im = ax.pcolormesh(GX, GY, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"face: {ttl}", fontsize=9)
    fig.colorbar(im, ax=fig.axes, shrink=0.6, label="switched")
    fig.suptitle("Tuned-Kruger switching phase diagram - three cube faces", fontsize=10)
    out = ANA / "phase_three_faces.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--all-channels", action="store_true",
                    help="Modulate all four exchange channels (Grueneisen-scaled, "
                         "lambda2_X = |X|/|K| * lambda) instead of K only.")
    args = ap.parse_args()
    set_campaign(args.all_channels)
    print(f"Magnetoelastic channels: {'ALL (Grueneisen-scaled)' if ALL_CHANNELS else 'K only'}; "
          f"integrator={INTEGRATOR}; campaign dir={CAMP.name}", flush=True)
    print("Building/verifying 3Q seeds per J7:", flush=True)
    seeds = {}
    for j7 in J7_GRID:
        seed, r3 = seed_for(j7); seeds[j7] = seed
        print(f"  J7={j7:.2f}: r3={r3:.3f} {'OK' if r3 > 0.6 else 'BAD'}", flush=True)
    if not args.plot_only:
        labs = []
        for j7 in J7_GRID:
            for lam in LAMBDA_GRID:
                for e0 in E0_GRID:
                    write_md(j7, lam, e0, seeds[j7]); labs.append(label_of(j7, lam, e0))
        print(f"Running {len(labs)} MD runs (workers={args.workers}) ...", flush=True)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_md, l) for l in labs]
            for _ in concurrent.futures.as_completed(futs):
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(labs)}", flush=True)
    rows = [r for j7 in J7_GRID for lam in LAMBDA_GRID for e0 in E0_GRID
            if (r := analyze(j7, lam, e0)) is not None]
    ANA.mkdir(parents=True, exist_ok=True)
    with (ANA / "phase_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} rows to phase_summary.csv", flush=True)
    make_plots(rows)
    print("\nThreshold E0_c(J7, lambda):")
    for j7 in J7_GRID:
        line = f"  J7={j7:.2f}: "
        for lam in LAMBDA_GRID:
            sw = [r["E0"] for r in rows if r["J7"] == j7 and r["lambda_K2"] == lam and r["switched"]]
            line += f"l{lam}={min(sw) if sw else '>16'} "
        print(line, flush=True)


if __name__ == "__main__":
    main()
