#!/usr/bin/env python3
"""Unified NCTO tuned-Kruger campaign driver (all studies, consistent L=36).

Subcommands
-----------
  static-pd     No-drive 3Q-vs-ZZ ground-state phase diagram, J7 in [0,-0.7].
  drive-pd      Driven switching phase diagram over (J7, lambda_K2, E0).
  switching     Enhanced-|K| line + background disorder: switched fraction vs E0.
  polarization  Clean lattice: pump polarization x fluence, J7 phonon off vs on.
  all           Run static-pd -> drive-pd -> switching -> polarization.

Everything shares ncto_common (single source of Hamiltonian + corrected
signed-Grueneisen lambda couplings) and a single OMP-pinned worker pool.

    python3 run_campaign.py all --workers "$(nproc)"
    python3 run_campaign.py drive-pd --all-channels --workers "$(nproc)"
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ncto_common as nc
from ncto_common import (CAMPAIGN, ROOT, LATTICE, LATTICE as L_DEFAULT, LAMBDA_K2_DEFAULT,
                         J, K, GAMMA, GAMMAP, J7_DEFAULT, LatticeContext,
                         me_lambda2, lam_j7_0, md_config, sa_relax, energy_of,
                         run_solver, r3_of, zz_fraction, parallel_run, default_workers)

# --------------------------------------------------------------------------- #
# Grids (physics axes).
# --------------------------------------------------------------------------- #
J7_STATIC = [round(-0.05 * i, 3) for i in range(15)]          # 0.0 .. -0.70
J7_DRIVE = [-0.40, -0.45, -0.50, -0.55, -0.60, -0.70]         # near-SU(2) window
LAMBDA_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
E0_DRIVE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0]

_CTX: dict[int, LatticeContext] = {}


def ctx_for(lattice: int) -> LatticeContext:
    if lattice not in _CTX:
        _CTX[lattice] = LatticeContext(lattice)
    return _CTX[lattice]


def read_final_spins(h5_path: Path) -> np.ndarray | None:
    import h5py
    try:
        with h5py.File(h5_path, "r") as f:
            return np.array(f["trajectory/spins"][-1])
    except Exception:
        return None


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ===========================================================================
# 1. Static (no-drive) 3Q-vs-ZZ phase diagram
# ===========================================================================
#   The quadratic E1 striction dX ~ eps^2 vanishes at the eps=0 equilibrium
#   (below the pseudo-JT threshold lambda*_{K,2} ~ 1.55, far above our span),
#   so the *static* 3Q/ZZ boundary is independent of lambda_{K,2}: it is set by
#   J7 alone.  We therefore relax both branches once per J7 and present the map
#   flat along the requested lambda axis.  dE(J7)=E_3Q-E_ZZ is the driving force
#   that feeds the CNT critical-radius estimate.
def _static_one(item):
    lattice, j7 = item
    ctx = ctx_for(lattice)
    base = CAMPAIGN / "static_phase_diagram" / "relax" / f"J7{int(abs(j7)*1000):04d}"
    q3 = sa_relax(lattice=lattice, j7=j7, work_dir=base / "q3",
                  seed_spins=nc.tile_l18_reference(nc.REF_3Q_L18, lattice),
                  n_deterministics=2000)
    zz = sa_relax(lattice=lattice, j7=j7, work_dir=base / "zz",
                  seed_spins=ctx.zz, n_deterministics=2000)
    if q3 is None or zz is None:
        return None
    e3q = energy_of(lattice=lattice, j7=j7, work_dir=base / "q3_E", spins=q3)
    ezz = energy_of(lattice=lattice, j7=j7, work_dir=base / "zz_E", spins=zz)
    if e3q is None or ezz is None:
        return None
    n = 2 * lattice * lattice
    return dict(J7=j7, E_3Q_meV=e3q, E_ZZ_meV=ezz,
                E_3Q_per_site=e3q / n, E_ZZ_per_site=ezz / n,
                dE_meV=e3q - ezz, dE_per_site=(e3q - ezz) / n,
                r3_3Q=r3_of(q3, ctx), r3_ZZ=r3_of(zz, ctx),
                zzfrac_3Q=zz_fraction(q3, ctx), zzfrac_ZZ=zz_fraction(zz, ctx),
                ground_state="ZZ" if ezz < e3q else "3Q")


def cmd_static_pd(args):
    ana = CAMPAIGN / "static_phase_diagram" / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    items = [(args.lattice, j7) for j7 in args.j7_values]
    rows = [r for _, r in parallel_run(_static_one, items, args.workers, "static-pd")
            if r is not None]
    rows.sort(key=lambda r: -r["J7"])
    csv_path = ana / "static_phase_diagram.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} J7 points)", flush=True)

    plt = _mpl()
    j7s = [r["J7"] for r in rows]
    de = [r["dE_per_site"] for r in rows]
    winner = np.array([[0.0 if r["ground_state"] == "3Q" else 1.0] for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax0.axhline(0.0, color="0.6", lw=0.8)
    ax0.plot(j7s, de, "o-", color="C0")
    ax0.set_xlabel("$J_7$"); ax0.set_ylabel(r"$(E_{3Q}-E_{ZZ})/N$ (meV/site)")
    ax0.set_title("Static driving force (>0: ZZ lower)")
    lam = args.lambda_values
    grid = np.tile(winner, (1, len(lam)))     # flat along lambda (static-inert)
    im = ax1.pcolormesh(lam, j7s, grid, cmap="RdBu_r", vmin=0, vmax=1, shading="nearest")
    ax1.set_xlabel(r"$\lambda_{K,2}$ (static-inert)"); ax1.set_ylabel("$J_7$")
    ax1.set_title("Ground state (blue=3Q, red=ZZ)")
    fig.colorbar(im, ax=ax1, ticks=[0, 1], label="0=3Q  1=ZZ")
    fig.suptitle(f"Static (no-drive) 3Q vs ZZ phase diagram, L={args.lattice}", fontsize=11)
    png = ana / "static_phase_diagram_3q_zz.png"
    fig.tight_layout(); fig.savefig(png, dpi=180); plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)}", flush=True)
    for r in rows:
        print(f"  J7={r['J7']:+.2f}: dE/site={r['dE_per_site']:+.4f} meV  "
              f"gs={r['ground_state']}  r3(3Q)={r['r3_3Q']:.2f}", flush=True)


# ===========================================================================
# 2. Driven switching phase diagram over (J7, lambda_K2, E0)
# ===========================================================================
def _drive_seed(item):
    lattice, j7 = item
    spins = sa_relax(lattice=lattice, j7=j7,
                     work_dir=CAMPAIGN / "phase_diagram_allchan" / "seeds" / f"J7{int(abs(j7)*1000):04d}",
                     seed_spins=nc.tile_l18_reference(nc.REF_3Q_L18, lattice),
                     n_deterministics=4000)
    return j7, spins


def _drive_label(j7, lam, e0):
    return f"J7{int(abs(j7)*1000):04d}_lK{int(lam*1000):03d}_E{int(e0*100):05d}"


def cmd_drive_pd(args):
    sub = "phase_diagram_allchan" if args.all_channels else "phase_diagram_konly"
    base = CAMPAIGN / sub
    cfg_dir, run_dir, ana = base / "configs", base / "runs", base / "analysis"
    for d in (cfg_dir, run_dir, ana):
        d.mkdir(parents=True, exist_ok=True)
    integrator = "rk4" if args.all_channels else "dopri5"
    lattice = args.lattice
    ctx = ctx_for(lattice)

    seeds = {j7: sp for j7, sp in
             (_drive_seed((lattice, j7)) for j7 in args.j7_values)}
    for j7, sp in seeds.items():
        if sp is None:
            print(f"[seed FAIL] J7={j7}", flush=True)
        else:
            print(f"  seed J7={j7:+.2f}: r3={r3_of(sp, ctx):.3f}", flush=True)
    seed_paths = {j7: (CAMPAIGN / sub / "seeds" / f"J7{int(abs(j7)*1000):04d}"
                       / "sample_0" / "spins_T=0.txt") for j7 in seeds}

    combos = [(j7, lam, e0) for j7 in args.j7_values
              for lam in args.lambda_values for e0 in args.e0_values
              if seeds.get(j7) is not None]

    def _run(combo):
        j7, lam, e0 = combo
        lab = _drive_label(j7, lam, e0)
        out = run_dir / lab
        h5 = out / "sample_0" / "trajectory.h5"
        if not h5.exists():
            cfg = cfg_dir / f"{lab}.param"
            cfg.write_text(md_config(lattice=lattice, j7=j7, output_dir=out,
                                     initial_spin=seed_paths[j7],
                                     lam2=me_lambda2(lam, args.all_channels),
                                     lam_j7=0.0, e0=e0, integrator=integrator))
            run_solver(cfg)
        sp = read_final_spins(h5)
        if sp is None:
            return None
        r3 = r3_of(sp, ctx)
        return dict(J7=j7, lambda_K2=lam, E0=e0, r3=r3, switched=int(r3 < 0.2))

    rows = [r for _, r in parallel_run(_run, combos, args.workers, "drive-pd")
            if r is not None]
    fieldnames = list(rows[0].keys())
    # write both names: plot_phase_barrier_cube.py reads phase_summary_full.csv
    for name in ("phase_summary.csv", "phase_summary_full.csv"):
        with (ana / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    print(f"wrote {(ana / 'phase_summary_full.csv').relative_to(ROOT)} ({len(rows)} rows); "
          f"plot with plot_phase_barrier_cube.py "
          f"{'--all-channels' if args.all_channels else ''}", flush=True)


# ===========================================================================
# 3. Enhanced-|K| line + disorder switching (Figs 2c/4)
# ===========================================================================
_SW = CAMPAIGN / "pinning_switching_crosscheck_L36"


def _defect_increment(dtype: str, strength: float):
    if dtype == "kred":
        return 0.0, -strength * abs(K), 0.0, 0.0
    if dtype == "kflip":
        return 0.0, +2.0 * strength * abs(K), 0.0, 0.0
    if dtype == "vacancy":
        return -strength * J, -strength * K, -strength * GAMMA, -strength * GAMMAP
    if dtype == "nematic":
        return 0.0, +strength * abs(K), 0.0, 0.0
    raise ValueError(dtype)


def _selected_bonds(ctx, dtype, half_width):
    sel = [k for k in range(len(ctx.bonds)) if abs(ctx.bmid[k, 0] - ctx.xc) <= half_width]
    if dtype == "nematic":
        sel = [k for k in sel if ctx.classes[k] == 0]
    return sel


def _build_disorder(ctx, dtype, strength, half_width, sigma_k, seed, mode, def_dir):
    mode_tag = "" if mode == "selected-centered" else f"_{mode}"
    tag = (f"L{ctx.lattice}_{dtype}_s{strength:.3f}_hw{half_width:.2f}"
           f"{mode_tag}_sig{sigma_k:.3f}_seed{seed:03d}").replace(".", "p")
    path = def_dir / f"{tag}.txt"
    selected = set(_selected_bonds(ctx, dtype, half_width))
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    dJ0, dK0, dG0, dGp0 = _defect_increment(dtype, strength)
    with path.open("w") as h:
        h.write("# site partner dJ dK dGamma dGammap\n")
        h.write(f"# L={ctx.lattice}; {dtype} s={strength:g} hw={half_width:g}; "
                f"mode={mode}; sigma_K={sigma_k:g}; seed={seed}\n")
        for k, (i, j) in enumerate(ctx.bonds):
            dJ = dG = dGp = dK = 0.0
            if mode == "global-zero-k" and sigma_k > 0:
                dK += rng.normal(0.0, sigma_k)
            if k in selected:
                dJ += dJ0; dK += dK0; dG += dG0; dGp += dGp0
                if mode == "selected-centered" and sigma_k > 0:
                    dK += rng.normal(0.0, sigma_k)
            if any(abs(v) > 1e-14 for v in (dJ, dK, dG, dGp)):
                h.write(f"{i} {j} {dJ:.12e} {dK:.12e} {dG:.12e} {dGp:.12e}\n")
    return path


@dataclass(frozen=True)
class DriveRun:
    lattice: int
    dtype: str
    strength: float
    half_width: float
    e0: float
    sigma_k: float
    seed: int
    mode: str

    @property
    def label(self):
        mode_tag = "" if self.mode == "selected-centered" else f"_{self.mode}"
        return (f"L{self.lattice}_{self.dtype}_s{self.strength:.3f}_hw{self.half_width:.2f}"
                f"{mode_tag}_allJKGG_E{self.e0:.2f}_sig{self.sigma_k:.3f}_seed{self.seed:03d}"
                ).replace(".", "p")


def _texture_metrics(ctx, spins, band_mask):
    zz = np.einsum("ij,ij->i", spins, ctx.zz) > 0.8
    off = ~band_mask
    on_d = float(zz[band_mask].mean()) if np.any(band_mask) else 0.0
    off_d = float(zz[off].mean()) if np.any(off) else 0.0
    tot = int(zz.sum())
    return dict(r3=r3_of(spins, ctx), zz_frac=float(zz.mean()),
                zz_on_band_density=on_d, zz_off_band_density=off_d,
                zz_on_band_pct=100.0 * int((zz & band_mask).sum()) / tot if tot else 0.0,
                pinning_enrichment=on_d / max(off_d, 1e-9))


def _sw_analyze(ctx, run, disorder_file, init_spin, band_mask, cfg_dir, run_dir, q_dir, no_quench):
    lab = run.label
    out = run_dir / lab
    h5 = out / "sample_0" / "trajectory.h5"
    if not h5.exists():
        cfg = cfg_dir / f"{lab}.param"
        cfg.write_text(md_config(lattice=ctx.lattice, j7=J7_DEFAULT, output_dir=out,
                                 initial_spin=init_spin,
                                 lam2=me_lambda2(LAMBDA_K2_DEFAULT, all_channels=True),
                                 lam_j7=0.0, e0=run.e0, integrator="rk4",
                                 disorder_file=disorder_file))
        if not run_solver(cfg):
            return None
    final = read_final_spins(h5)
    if final is None:
        return None
    rec = dict(lattice=ctx.lattice, dtype=run.dtype, strength=run.strength,
               half_width=run.half_width, e0=run.e0, sigma_k=run.sigma_k,
               seed=run.seed, label=lab)
    for k, v in _texture_metrics(ctx, final, band_mask).items():
        rec[f"final_{k}"] = v
    if not no_quench:
        q = sa_relax(lattice=ctx.lattice, j7=J7_DEFAULT, work_dir=q_dir / lab,
                     seed_spins=final, n_deterministics=800, disorder_file=disorder_file)
        if q is not None:
            for k, v in _texture_metrics(ctx, q, band_mask).items():
                rec[f"quench_{k}"] = v
    qzz = rec.get("quench_zz_frac", rec["final_zz_frac"])
    qon = rec.get("quench_zz_on_band_density", rec["final_zz_on_band_density"])
    qoff = rec.get("quench_zz_off_band_density", rec["final_zz_off_band_density"])
    rec["written"] = int(qzz > 0.06)
    rec["pinned_written"] = int((qzz > 0.06) and (qon > max(0.10, 2.0 * qoff)))
    return rec


def _sw_aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["e0"], r["sigma_k"]), []).append(r)
    base_seed = {}
    baselines = {}
    for (e0, sig), rs in groups.items():
        if abs(e0) < 1e-12:
            for r in rs:
                base_seed[(sig, int(r["seed"]))] = dict(
                    qzz=r.get("quench_zz_frac", r["final_zz_frac"]),
                    on=r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]),
                    off=r.get("quench_zz_off_band_density", r["final_zz_off_band_density"]))
            baselines[sig] = dict(
                mean_quench_zz=float(np.mean([r.get("quench_zz_frac", r["final_zz_frac"]) for r in rs])),
                mean_on_band_density=float(np.mean([
                    r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]) for r in rs])))
    out = []
    for (e0, sig), rs in sorted(groups.items()):
        written, pinned = [], []
        for r in rs:
            qzz = r.get("quench_zz_frac", r["final_zz_frac"])
            on = r.get("quench_zz_on_band_density", r["final_zz_on_band_density"])
            off = r.get("quench_zz_off_band_density", r["final_zz_off_band_density"])
            b = base_seed.get((sig, int(r["seed"])), dict(qzz=0.0, on=0.0, off=0.0))
            is_w = (qzz - b["qzz"]) > 0.05
            is_p = is_w and (on - b["on"]) > 0.05 and on > max(0.10, 2.0 * off)
            written.append(float(is_w)); pinned.append(float(is_p))
        bl = baselines.get(sig, dict(mean_quench_zz=0.0, mean_on_band_density=0.0))
        mqzz = float(np.mean([r.get("quench_zz_frac", r["final_zz_frac"]) for r in rs]))
        mon = float(np.mean([r.get("quench_zz_on_band_density", r["final_zz_on_band_density"]) for r in rs]))
        out.append(dict(
            lattice=int(rs[0].get("lattice", 36)), e0=e0, sigma_k=sig, n=len(rs),
            write_fraction=float(np.mean([r["written"] for r in rs])),
            pinned_write_fraction=float(np.mean([r["pinned_written"] for r in rs])),
            drive_write_fraction=float(np.mean(written)),
            drive_pinned_fraction=float(np.mean(pinned)),
            mean_quench_zz=mqzz, excess_quench_zz=mqzz - bl["mean_quench_zz"],
            mean_on_band_density=mon, excess_on_band_density=mon - bl["mean_on_band_density"],
            mean_on_band_pct=float(np.mean([r.get("quench_zz_on_band_pct", r["final_zz_on_band_pct"]) for r in rs])),
            mean_enrichment=float(np.mean([r.get("quench_pinning_enrichment", r["final_pinning_enrichment"]) for r in rs]))))
    return out


def cmd_switching(args):
    ctx = ctx_for(args.lattice)
    cfg_dir, run_dir, def_dir = _SW / "configs", _SW / "runs", _SW / "disorder_files"
    init_dir, q_dir, ana = _SW / "initial_states", _SW / "quenches", _SW / "analysis"
    for d in (cfg_dir, run_dir, def_dir, init_dir, q_dir, ana):
        d.mkdir(parents=True, exist_ok=True)
    band_mask = np.abs(ctx.x - ctx.xc) <= args.half_width + 1.5

    # disorder files + relaxed 3Q initial states, one per (sigma_k, seed)
    items = []
    for sig in args.sigma_k_values:
        for seed in range(args.seeds if sig > 0 else 1):
            df = _build_disorder(ctx, args.dtype, args.strength, args.half_width,
                                 sig, seed, args.disorder_mode, def_dir)
            items.append((sig, seed, df))

    def _relax(item):
        sig, seed, df = item
        tag = (f"L{ctx.lattice}_{args.dtype}_s{args.strength:.3f}_hw{args.half_width:.2f}"
               f"_sig{sig:.3f}_seed{seed:03d}").replace(".", "p")
        init = sa_relax(lattice=ctx.lattice, j7=J7_DEFAULT, work_dir=init_dir / tag,
                        seed_spins=ctx.q3, n_deterministics=1200, disorder_file=df)
        return (sig, seed), (df, init_dir / tag / "sample_0" / "spins_T=0.txt")

    init_map = {}
    for _, (key, val) in parallel_run(_relax, items, args.workers, "switching:relax"):
        init_map[key] = val

    jobs = []
    for sig in args.sigma_k_values:
        for seed in range(args.seeds if sig > 0 else 1):
            df, init = init_map[(sig, seed)]
            for e0 in args.e0_values:
                jobs.append((DriveRun(ctx.lattice, args.dtype, args.strength,
                                      args.half_width, e0, sig, seed, args.disorder_mode), df, init))

    def _run(job):
        run, df, init = job
        return _sw_analyze(ctx, run, df, init, band_mask, cfg_dir, run_dir, q_dir, args.no_quench)

    rows = [r for _, r in parallel_run(_run, jobs, args.workers, "switching:drive")
            if r is not None]
    tag = (f"L{ctx.lattice}_{args.dtype}_s{args.strength:.3f}_hw{args.half_width:.2f}").replace(".", "p")
    if args.output_suffix:
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in args.output_suffix)
        tag = f"{tag}_{clean}"
    csv_path = ana / f"{tag}_drive_crosscheck.csv"
    summary_path = ana / f"{tag}_drive_crosscheck_summary.json"
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    summary_path.write_text(json.dumps(_sw_aggregate(rows), indent=2))
    print(f"wrote {csv_path.relative_to(ROOT)}\nwrote {summary_path.relative_to(ROOT)}", flush=True)


# ===========================================================================
# 4. Clean polarization / fluence study, J7 phonon off vs on (Fig pol)
# ===========================================================================
def cmd_polarization(args):
    base = CAMPAIGN / "polarization_fluence_study"
    cfg_dir, run_dir, init_dir, ana = base / "configs", base / "runs", base / "initial_states", base / "analysis"
    for d in (cfg_dir, run_dir, init_dir, ana):
        d.mkdir(parents=True, exist_ok=True)
    lattice = args.lattice
    ctx = ctx_for(lattice)
    init_spins = nc.tile_l18_reference(nc.REF_3Q_L18, lattice)
    init_path = init_dir / f"L{lattice}_clean3q.txt"
    if not init_path.exists():
        np.savetxt(init_path, init_spins, fmt="%.10e")
    print(f"clean 3Q init r3={r3_of(init_spins, ctx):.3f}", flush=True)

    lam2 = me_lambda2(LAMBDA_K2_DEFAULT, all_channels=True)
    jobs = [(coup, th, e0) for coup in ("j7off", "j7on")
            for th in args.theta_deg for e0 in args.e0_values]

    def _run(job):
        coup, th, e0 = job
        lab = f"L_pol_{coup}_th{int(round(th)):03d}_E{int(round(e0*100)):05d}"
        out = run_dir / lab
        h5 = out / "sample_0" / "trajectory.h5"
        if not h5.exists():
            cfg = cfg_dir / f"{lab}.param"
            cfg.write_text(md_config(lattice=lattice, j7=J7_DEFAULT, output_dir=out,
                                     initial_spin=init_path, lam2=lam2,
                                     lam_j7=lam_j7_0(LAMBDA_K2_DEFAULT) if coup == "j7on" else 0.0,
                                     e0=e0, pump_polarization=np.deg2rad(th), integrator="rk4"))
            run_solver(cfg)
        sp = read_final_spins(h5)
        if sp is None:
            return None
        r3 = r3_of(sp, ctx)
        return dict(coupling=coup, theta_deg=th, e0=e0, r3=r3, switched=int(r3 < 0.2))

    rows = [r for _, r in parallel_run(_run, jobs, args.workers, "polarization")
            if r is not None]
    csv_path = ana / "polarization_fluence.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)

    plt = _mpl()
    thetas = sorted(set(r["theta_deg"] for r in rows))
    e0s = sorted(set(r["e0"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, coup, ttl in zip(axes[0], ["j7off", "j7on"],
                             ["J7 phonon OFF", f"J7 phonon ON ($\\lambda_{{J7,0}}$={lam_j7_0(LAMBDA_K2_DEFAULT):.4f})"]):
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
    fig.suptitle(f"Clean $L={lattice}$ polarization/fluence switching", fontsize=11)
    png = ana / "polarization_fluence_switching.png"
    fig.savefig(png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================
def build_parser():
    # Shared options accepted both before AND after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=default_workers(),
                        help="parallel solver processes (default: all cores)")
    common.add_argument("--lattice", type=int, default=LATTICE, help="lattice size L (default 36)")

    # --workers/--lattice live on the subcommands (used as `<cmd> --workers N`),
    # so they are not added to the top-level parser to avoid a silent override.
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("static-pd", parents=[common], help="no-drive 3Q-vs-ZZ phase diagram")
    s.add_argument("--j7-values", nargs="+", type=float, default=J7_STATIC)
    s.add_argument("--lambda-values", nargs="+", type=float, default=LAMBDA_GRID)
    s.set_defaults(func=cmd_static_pd)

    s = sub.add_parser("drive-pd", parents=[common], help="driven switching phase diagram")
    s.add_argument("--all-channels", action="store_true",
                   help="modulate all four channels with signed Grueneisen ratios (X/K)")
    s.add_argument("--j7-values", nargs="+", type=float, default=J7_DRIVE)
    s.add_argument("--lambda-values", nargs="+", type=float, default=LAMBDA_GRID)
    s.add_argument("--e0-values", nargs="+", type=float, default=E0_DRIVE)
    s.set_defaults(func=cmd_drive_pd)

    s = sub.add_parser("switching", parents=[common], help="enhanced-|K| line + disorder switching fraction")
    s.add_argument("--dtype", choices=["kred", "kflip", "vacancy", "nematic"], default="kred")
    s.add_argument("--strength", type=float, default=0.5)
    s.add_argument("--half-width", type=float, default=2.0)
    s.add_argument("--sigma-k-values", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    s.add_argument("--e0-values", nargs="+", type=float,
                   default=[0, 6, 8, 9, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20, 22, 25])
    s.add_argument("--seeds", type=int, default=25)
    s.add_argument("--disorder-mode", choices=["selected-centered", "global-zero-k"], default="global-zero-k")
    s.add_argument("--no-quench", action="store_true")
    s.add_argument("--output-suffix", default="allJKGG_err10")
    s.set_defaults(func=cmd_switching)

    s = sub.add_parser("polarization", parents=[common], help="clean polarization/fluence, J7 phonon off vs on")
    s.add_argument("--theta-deg", nargs="+", type=float, default=[0, 15, 30, 45, 60, 75, 90])
    s.add_argument("--e0-values", nargs="+", type=float, default=[0, 4, 6, 8, 10, 12, 14, 16, 20])
    s.set_defaults(func=cmd_polarization)

    s = sub.add_parser("all", parents=[common], help="static-pd -> drive-pd -> switching -> polarization")
    s.add_argument("--all-channels", action="store_true", default=True)
    s.set_defaults(func=cmd_all)
    return p


def cmd_all(args):
    # static-pd
    cmd_static_pd(argparse.Namespace(workers=args.workers, lattice=args.lattice,
                                     j7_values=J7_STATIC, lambda_values=LAMBDA_GRID))
    # drive-pd (all channels)
    cmd_drive_pd(argparse.Namespace(workers=args.workers, lattice=args.lattice,
                                    all_channels=True, j7_values=J7_DRIVE,
                                    lambda_values=LAMBDA_GRID, e0_values=E0_DRIVE))
    # switching (final-story protocol)
    cmd_switching(argparse.Namespace(workers=args.workers, lattice=args.lattice,
                                     dtype="kred", strength=0.5, half_width=2.0,
                                     sigma_k_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                                     e0_values=[0, 6, 8, 9, 10, 10.5, 11, 11.5, 12, 12.5, 13,
                                                13.5, 14, 15, 16, 17, 18, 19, 20, 22, 25],
                                     seeds=25, disorder_mode="global-zero-k",
                                     no_quench=False, output_suffix="allJKGG_err10"))
    # polarization
    cmd_polarization(argparse.Namespace(workers=args.workers, lattice=args.lattice,
                                        theta_deg=[0, 15, 30, 45, 60, 75, 90],
                                        e0_values=[0, 4, 6, 8, 10, 12, 14, 16, 20]))


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
