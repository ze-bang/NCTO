#!/usr/bin/env python3
"""Clean (no-defect) pump polarization / fluence switching study.

Same tuned-Kruger point and E1 magnetoelastic model as the L36 switching
cross-check, but with NO extended defect and NO disorder: a pristine 3Q lattice
is driven and we map the switched/unswitched outcome over (pump polarization
theta, fluence E0).  The study is run twice -- with the J7 ring-exchange phonon
coupling OFF (lambda_E1_J7_0 = 0) and ON (lambda_E1_J7_0 = (J7/K) lambda_K2,
the Grueneisen-matched value) -- to isolate the polarization-isotropic effect of
transiently tuning the near-degeneracy knob J7.

Outputs a CSV and a comparison figure (theta x E0 switching maps, off vs on).
"""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "spin_solver"
OUT = ROOT / "NCTO_project" / "tuned_kruger_campaign" / "polarization_fluence_study"
CFG = OUT / "configs"; RUN = OUT / "runs"; INIT = OUT / "initial_states"; ANA = OUT / "analysis"
for d in (CFG, RUN, INIT, ANA):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "NCTO_project" / "scripts" / "np_campaign"))
sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from analysis_utils import M_VECTORS                  # noqa: E402
from reader_strain_lattice import generate_honeycomb_positions as hc_positions  # noqa: E402

# Relaxed L18 3Q reference at the working point (J7=-0.40); tiles cleanly to L36.
REF_3Q_L18 = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "phase_diagram"
              / "analysis" / "seed3Q_J70400.txt")

# Tuned-Kruger Hamiltonian (meV) and corrected signed-Grueneisen bilinear coupling.
J, K, GAMMA, GAMMAP = 0.68, -7.89, 3.07, -2.94
J2_A, J2_B, J3, J7 = -0.06, -0.70, 0.52, -0.40
LAMBDA_K2 = 0.02
LAMBDA_J2  = LAMBDA_K2 * J      / K
LAMBDA_G2  = LAMBDA_K2 * GAMMA  / K
LAMBDA_GP2 = LAMBDA_K2 * GAMMAP / K
# Grueneisen-matched ring-exchange (isotropic breathing) coupling: same fractional
# sensitivity as the bilinear channels, lambda_J7,0 = (J7/K) lambda_K2.
LAMBDA_J7_0_ON = LAMBDA_K2 * J7 / K            # = +0.001014
OMEGA_E1, GAMMA_E1, ZSTAR = 4.0, 0.08488263631567752, 1.0
T0_PUMP, SIGMA_PUMP, ALPHA = 50.0, 15.0, 0.05
MD_END, DT, SAVE = 150.0, 0.005, 100


def r3_of(spins, xy, n):
    vals = []
    for q in M_VECTORS:
        ph = xy[:, 0] * q[0] + xy[:, 1] * q[1]
        sq = np.sum(spins * np.exp(1j * ph[:, None]), axis=0)
        vals.append(float(np.real(sq @ np.conj(sq))) / n)
    return min(vals) / max(vals) if max(vals) > 0 else 1.0


def clean_3q_init(lattice, xy, n):
    """Pristine 3Q start: tile the relaxed L18 reference (r3~0.94) to the target
    lattice.  The analytic triple-Q ansatz relaxes into zigzag at this working
    point, so we use the genuine relaxed 3Q reference directly (no re-anneal)."""
    out = INIT / f"L{lattice}_clean3q"
    spin_out = out / "sample_0" / "spins_T=0.txt"
    if spin_out.exists():
        return spin_out
    if lattice % 18 != 0:
        raise ValueError("lattice must be a multiple of 18 to tile the L18 reference")
    ref = np.loadtxt(REF_3Q_L18).reshape(18, 18, 2, 3)
    reps = lattice // 18
    tiled = np.tile(ref, (reps, reps, 1, 1)).reshape(n, 3)
    tiled /= np.linalg.norm(tiled, axis=1)[:, None]
    (out / "sample_0").mkdir(parents=True, exist_ok=True)
    np.savetxt(spin_out, tiled, fmt="%.10e")
    return spin_out


def label(coupling, theta_deg, e0):
    return f"L_pol_{coupling}_th{int(round(theta_deg)):03d}_E{int(round(e0*100)):05d}"


def write_md(lattice, coupling, theta_deg, e0, init_spin):
    lam_j7 = LAMBDA_J7_0_ON if coupling == "j7on" else 0.0
    theta_rad = np.deg2rad(theta_deg)
    lab = label(coupling, theta_deg, e0)
    cfg = CFG / f"{lab}.param"
    cfg.write_text(f"""system = NCTO
lattice_size = {lattice},{lattice},1
simulation_mode = MD
output_dir = {(RUN / lab).relative_to(ROOT)}
num_trials = 1
J = {J}\nK = {K}\nGamma = {GAMMA}\nGammap = {GAMMAP}
J2_A = {J2_A}\nJ2_B = {J2_B}\nJ3 = {J3}\nJ7 = {J7}
field_strength = 0.0\nfield_direction = 0,0,1
omega_E1 = {OMEGA_E1}\ngamma_E1 = {GAMMA_E1}\nlambda_E1_quartic = 0.0\nZ_star = {ZSTAR}
lambda_E1_K_2 = {LAMBDA_K2:.12g}
lambda_E1_J_2 = {LAMBDA_J2:.12g}
lambda_E1_Gamma_2 = {LAMBDA_G2:.12g}
lambda_E1_Gammap_2 = {LAMBDA_GP2:.12g}
lambda_E1_K_0 = 0.0\nlambda_E1_J_0 = 0.0\nlambda_E1_Gamma_0 = 0.0\nlambda_E1_Gammap_0 = 0.0
lambda_E1_J7_0 = {lam_j7:.12g}
alpha_gilbert = {ALPHA}\nlangevin_temperature = 0.0
pump_amplitude = {e0:.12g}\npump_frequency = {OMEGA_E1}\npump_time = {T0_PUMP}
pump_width = {SIGMA_PUMP}\npump_phase = 0.0\npump_polarization = {theta_rad:.12g}
probe_amplitude = 0.0
md_time_start = 0.0\nmd_time_end = {MD_END}\nmd_timestep = {DT}\nmd_save_interval = {SAVE}
md_integrator = rk4\nmd_abs_tol = 1e-8\nmd_rel_tol = 1e-8
initial_spin_config = {init_spin.relative_to(ROOT)}
relax_phonons = false\nadiabatic_phonons = false\nphonon_only_relax = false
""")
    return cfg, lab


def _read_h5(h5):
    import h5py
    with h5py.File(h5, "r") as f:
        sp = f["trajectory/spins"][-1]
        q2 = float((f["phonon_trajectory/Qx_E1"][:]**2 + f["phonon_trajectory/Qy_E1"][:]**2).max())
    return sp, q2


def run_one(args_tuple):
    lattice, coupling, theta_deg, e0, init_spin, xy, n = args_tuple
    cfg, lab = write_md(lattice, coupling, theta_deg, e0, init_spin)
    h5 = RUN / lab / "sample_0" / "trajectory.h5"
    for attempt in range(2):
        if not h5.exists():
            subprocess.run([str(EXE), str(cfg)], capture_output=True, cwd=str(ROOT), text=True)
        if not h5.exists():
            return None
        try:
            sp, q2 = _read_h5(h5)
            break
        except OSError:                      # corrupt/partial h5 -> delete and rerun once
            try: h5.unlink()
            except OSError: pass
    else:
        return None
    r3 = r3_of(sp, xy, n)
    dj7_frac = LAMBDA_J7_0_ON * q2 / J7 if coupling == "j7on" else 0.0
    return dict(coupling=coupling, theta_deg=theta_deg, e0=e0, r3=r3,
                switched=int(r3 < 0.2), peak_Q2=q2, dJ7_frac_peak=dj7_frac)


def make_plot(rows, lattice):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    thetas = sorted(set(r["theta_deg"] for r in rows))
    e0s = sorted(set(r["e0"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, coup, ttl in zip(axes[0], ["j7off", "j7on"],
                             ["J7 phonon OFF", f"J7 phonon ON ($\\lambda_{{J7,0}}$={LAMBDA_J7_0_ON:.4f})"]):
        Z = np.full((len(thetas), len(e0s)), np.nan)
        sw = {(r["theta_deg"], r["e0"]): r["switched"] for r in rows if r["coupling"] == coup}
        for i, th in enumerate(thetas):
            for j, e0 in enumerate(e0s):
                if (th, e0) in sw:
                    Z[i, j] = sw[(th, e0)]
        im = ax.pcolormesh(e0s, thetas, Z, cmap="RdBu_r", vmin=0, vmax=1, shading="nearest")
        ax.set_xlabel("$E_0$ (pump fluence)"); ax.set_ylabel(r"polarization $\theta$ (deg)")
        ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="switched ($r_3<0.2$)")
    fig.suptitle(f"Clean $L={lattice}$ polarization/fluence switching: J7 ring-exchange phonon coupling", fontsize=11)
    png = ANA / "polarization_fluence_switching.png"
    fig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"saved {png.relative_to(ROOT)}", flush=True)


def main():
    import concurrent.futures
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", type=int, default=36)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--theta-deg", nargs="+", type=float, default=[0, 15, 30, 45, 60, 75, 90])
    ap.add_argument("--e0-values", nargs="+", type=float, default=[0, 4, 6, 8, 10, 12, 14, 16, 20])
    args = ap.parse_args()

    n = 2 * args.lattice * args.lattice
    xy = hc_positions(n)[:, :2]
    init_spin = clean_3q_init(args.lattice, xy, n)
    print(f"clean 3Q init: {init_spin.relative_to(ROOT)}  (r3={r3_of(np.loadtxt(init_spin), xy, n):.3f})", flush=True)

    jobs = [(args.lattice, coup, th, e0, init_spin, xy, n)
            for coup in ("j7off", "j7on") for th in args.theta_deg for e0 in args.e0_values]
    print(f"L{args.lattice} clean polarization study: {len(jobs)} runs (workers={args.workers})", flush=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, rec in enumerate(pool.map(run_one, jobs), 1):
            if rec:
                rows.append(rec)
                if k % 10 == 0:
                    print(f"  {k}/{len(jobs)}", flush=True)
    csv_path = ANA / "polarization_fluence.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)
    make_plot(rows, args.lattice)
    # threshold summary
    for coup in ("j7off", "j7on"):
        print(f"\n{coup} switching threshold E0_c(theta):", flush=True)
        for th in args.theta_deg:
            sw = [r["e0"] for r in rows if r["coupling"] == coup and r["theta_deg"] == th and r["switched"]]
            print(f"  theta={th:5.1f}deg: E0_c={min(sw) if sw else '>max'}", flush=True)


if __name__ == "__main__":
    main()
