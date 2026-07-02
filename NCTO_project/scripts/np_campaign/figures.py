"""Manuscript figure generators for the NCTO Nature-Physics campaign.

One make_fXX() per manuscript figure.  Each function reads only the
CSV / HDF5 outputs of phases C1..C9 already present under
NCTO_project/np_campaign_out/ and writes to ../figs/.

If the upstream data is missing or empty, a placeholder panel is
emitted instead of raising, so the driver can keep going and report
which phases need to be re-run.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_utils import read_csv
from common import CAMPAIGN_OUT

FIG_DIR = CAMPAIGN_OUT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CM = 1.0 / 2.54
NATURE_SINGLE = 8.6 * CM   # single column (inches)
NATURE_DOUBLE = 17.8 * CM  # double column

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=_PALETTE),
    })


def save(fig, name: str, formats=("pdf", "png")) -> list:
    out = []
    for fmt in formats:
        p = FIG_DIR / f"{name}.{fmt}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def missing_panel(ax, msg: str):
    ax.text(0.5, 0.5, f"missing data:\n{msg}",
            ha="center", va="center", fontsize=7,
            transform=ax.transAxes, color="#888888")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")


def _placeholder(name: str, msg: str, *, double=False, formats=("pdf", "png")):
    w = NATURE_DOUBLE if double else NATURE_SINGLE
    fig, ax = plt.subplots(figsize=(w, w * 0.6))
    missing_panel(ax, msg)
    return save(fig, name, formats=formats)


# ====================================================================== F2
def make_f2(formats=("pdf", "png")) -> list:
    csv = CAMPAIGN_OUT / "C1" / "results.csv"
    if not csv.exists():
        return _placeholder("F2_microscopic_switching_map",
                            f"{csv.relative_to(CAMPAIGN_OUT.parent)}",
                            double=True, formats=formats)
    rows = read_csv(csv)
    if not rows:
        return _placeholder("F2_microscopic_switching_map",
                            "C1/results.csv is empty",
                            double=True, formats=formats)
    chans = sorted({r["channel"] for r in rows if r.get("channel")})
    if not chans:
        return _placeholder("F2_microscopic_switching_map",
                            "no 'channel' column in C1 results",
                            double=True, formats=formats)
    fig, axes = plt.subplots(1, len(chans),
                             figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 3.5),
                             sharey=True, constrained_layout=True)
    if len(chans) == 1:
        axes = [axes]
    last_im = None
    for ax, ch in zip(axes, chans):
        sub = [r for r in rows if r["channel"] == ch]
        E0s = sorted({float(r["E0"]) for r in sub})
        ths = sorted({float(r["theta"]) for r in sub})
        Z = np.full((len(E0s), len(ths)), np.nan)
        cnt = np.zeros_like(Z)
        for r in sub:
            i = E0s.index(float(r["E0"])); j = ths.index(float(r["theta"]))
            Z[i, j] = (Z[i, j] if Z[i, j] == Z[i, j] else 0.0) + float(r.get("switched", 0))
            cnt[i, j] += 1
        Z = np.where(cnt > 0, Z / np.where(cnt == 0, 1, cnt), np.nan)
        last_im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma",
                            vmin=0, vmax=1,
                            extent=[min(ths), max(ths), min(E0s), max(E0s)])
        ax.set_title(str(ch).replace("LAMBDA_", r"$\lambda_{") + "}$")
        ax.set_xlabel(r"$\theta_{\rm pump}$ [rad]")
    axes[0].set_ylabel(r"$E_0$")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.85, label="switching rate")
    return save(fig, "F2_microscopic_switching_map", formats=formats)


# ====================================================================== F3
def make_f3(formats=("pdf", "png")) -> list:
    csv = CAMPAIGN_OUT / "C1" / "results.csv"
    if not csv.exists():
        return _placeholder("F3_hamiltonian_robustness",
                            "C1/results.csv", double=True, formats=formats)
    rows = read_csv(csv)
    if not rows:
        return _placeholder("F3_hamiltonian_robustness",
                            "C1/results.csv is empty",
                            double=True, formats=formats)
    # threshold E0 per (J,K,Gamma,Gammap,channel) = min E0 with switched=1
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        key = (r.get("J"), r.get("K"), r.get("Gamma"),
               r.get("Gammap"), r.get("channel"))
        try:
            grp[key].append((float(r["E0"]), int(r["switched"])))
        except (KeyError, ValueError, TypeError):
            continue
    thr_pts = []
    for key, pairs in grp.items():
        pairs.sort()
        thr = next((E for E, sw in pairs if sw), None)
        if thr is None:
            continue
        J, K, G, Gp, ch = key
        thr_pts.append({"J": J, "K": K, "Gamma": G, "Gammap": Gp,
                        "channel": ch, "thr": thr})
    if not thr_pts:
        return _placeholder("F3_hamiltonian_robustness",
                            "no rows with switched==1 in C1 results",
                            double=True, formats=formats)
    axes_names = ["J", "K", "Gamma", "Gammap"]
    fig, axarr = plt.subplots(1, 4,
                              figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 4.5),
                              constrained_layout=True)
    for ax, axis in zip(axarr, axes_names):
        groups = {}
        for p in thr_pts:
            v = p[axis]
            if v is None:
                continue
            groups.setdefault(float(v), []).append(p["thr"])
        if not groups:
            missing_panel(ax, f"no {axis} samples")
            continue
        xs = sorted(groups)
        data = [groups[x] for x in xs]
        bp = ax.boxplot(data, positions=xs,
                        widths=[0.6 * (xs[1] - xs[0]) if len(xs) > 1 else 0.4] * len(xs),
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(_PALETTE[0]); patch.set_alpha(0.5)
        ax.set_xlabel(axis)
        ax.set_ylabel(r"$E_0^{\star}$" if axis == "J" else "")
    fig.suptitle("Threshold robustness across Hamiltonian axes", y=1.05)
    return save(fig, "F3_hamiltonian_robustness", formats=formats)


# ====================================================================== F4
def make_f4(formats=("pdf", "png")) -> list:
    csv = CAMPAIGN_OUT / "C2" / "barriers.csv"
    if not csv.exists():
        return _placeholder("F4_gneb_barrier", "C2/barriers.csv",
                            double=True, formats=formats)
    rows = read_csv(csv)
    if not rows:
        return _placeholder("F4_gneb_barrier", "C2/barriers.csv empty",
                            double=True, formats=formats)
    fig, (ax_a, ax_b) = plt.subplots(1, 2,
                                     figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 2.5),
                                     constrained_layout=True)
    # (a) overlay any energy_path CSV/TXT we can find
    runs = sorted((CAMPAIGN_OUT / "C2" / "runs").glob("*"))
    plotted = 0
    for r in runs[:6]:
        path = None
        for cand in ("energy_path.csv", "gneb/energy_path.csv",
                     "energy.txt"):
            p = r / cand
            if p.exists():
                path = p; break
        if path is None:
            continue
        try:
            arr = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None,
                             skiprows=1 if path.suffix == ".csv" else 0)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        ax_a.plot(arr[:, 0], arr[:, 1] - arr[:, 1].min(), label=r.name)
        plotted += 1
    if plotted == 0:
        missing_panel(ax_a, "no energy_path files in C2/runs/*")
    else:
        ax_a.set_xlabel("reaction coordinate")
        ax_a.set_ylabel(r"$E - E_{\min}$")
        ax_a.legend(fontsize=6, ncol=2)
        ax_a.set_title("(a) GNEB paths")

    # (b) ΔE‡ heatmap from CSV
    Q2 = sorted({float(r["Q2_proxy"]) for r in rows
                 if r.get("Q2_proxy") is not None})
    ths = sorted({float(r["theta"]) for r in rows
                  if r.get("theta") is not None})
    Z = np.full((len(Q2), len(ths)), np.nan)
    for r in rows:
        b = r.get("barrier")
        if b is None or (isinstance(b, float) and b != b):
            continue
        try:
            Z[Q2.index(float(r["Q2_proxy"])),
              ths.index(float(r["theta"]))] = float(b)
        except (ValueError, KeyError):
            continue
    if np.all(np.isnan(Z)):
        missing_panel(ax_b, "all barriers NaN")
    else:
        im = ax_b.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                         extent=[min(ths), max(ths), min(Q2), max(Q2)])
        ax_b.set_xlabel(r"$\theta_{Eg}$ [rad]")
        ax_b.set_ylabel(r"$|Q|^2$ proxy")
        ax_b.set_title("(b) Activation barrier")
        fig.colorbar(im, ax=ax_b, label=r"$\Delta E^{\ddagger}$")
    return save(fig, "F4_gneb_barrier", formats=formats)


# ====================================================================== F5
def make_f5(formats=("pdf", "png")) -> list:
    base = CAMPAIGN_OUT / "C3"
    cg = base / "coarse_grained.csv"
    h5_3q = base / "pt_3q" / "pt_output.h5"
    h5_zz = base / "pt_zz" / "pt_output.h5"
    fig, axes = plt.subplots(1, 3,
                             figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 3.2),
                             constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # (a) PT energy histograms
    have_pt = False
    if h5_3q.exists() and h5_zz.exists():
        try:
            import h5py
            with h5py.File(h5_3q, "r") as h:
                T3 = h["temperatures"][:]; E3 = h["energies"][:]
            with h5py.File(h5_zz, "r") as h:
                Tz = h["temperatures"][:]; Ez = h["energies"][:]
            T_low = float(min(T3.min(), Tz.min()))
            i3 = int(np.argmin(np.abs(T3 - T_low)))
            iz = int(np.argmin(np.abs(Tz - T_low)))
            ax_a.hist(E3[i3, :], bins=40, alpha=0.6, label=f"3Q @ T={T_low:.3g}",
                      color=_PALETTE[0])
            ax_a.hist(Ez[iz, :], bins=40, alpha=0.6, label=f"ZZ @ T={T_low:.3g}",
                      color=_PALETTE[1])
            ax_a.set_xlabel("E"); ax_a.set_ylabel("count"); ax_a.legend()
            ax_a.set_title("(a) PT energy")
            have_pt = True
        except Exception as exc:
            missing_panel(ax_a, f"HDF5 read failed:\n{exc}")
    if not have_pt:
        missing_panel(ax_a, "C3/pt_*/pt_output.h5")

    # (b) summary table from coarse_grained.csv
    if cg.exists():
        rows = read_csv(cg)
        if rows:
            r = rows[0]
            keys = ["Delta_f0_per_site", "sigma_wall", "M_wall",
                    "E0_threshold_median"]
            vals = [r.get(k, float("nan")) for k in keys]
            ypos = np.arange(len(keys))
            ax_b.barh(ypos, [v if isinstance(v, (int, float)) and v == v else 0
                             for v in vals], color=_PALETTE[2])
            ax_b.set_yticks(ypos); ax_b.set_yticklabels(keys, fontsize=7)
            ax_b.set_title("(b) Coarse-grained params")
        else:
            missing_panel(ax_b, "coarse_grained.csv empty")
    else:
        missing_panel(ax_b, "C3/coarse_grained.csv")

    # (c) Delta_f0 vs T — placeholder if no T-grid CSV present
    df0_csv = base / "delta_f0_vs_T.csv"
    if df0_csv.exists():
        rs = read_csv(df0_csv)
        if rs:
            T = [r["T"] for r in rs]
            df = [r["Delta_f0"] for r in rs]
            ax_c.plot(T, df, "o-")
            ax_c.set_xlabel("T"); ax_c.set_ylabel(r"$\Delta f_0$")
            ax_c.set_title("(c) Δf₀(T)")
        else:
            missing_panel(ax_c, "delta_f0_vs_T.csv empty")
    else:
        missing_panel(ax_c, "C3/delta_f0_vs_T.csv\n(future deliverable)")
    return save(fig, "F5_coarse_grained", formats=formats)


# ====================================================================== F6
def make_f6(formats=("pdf", "png")) -> list:
    csv = CAMPAIGN_OUT / "C4" / "stress.csv"
    if not csv.exists():
        csv = CAMPAIGN_OUT / "C4" / "results.csv"
    if not csv.exists():
        return _placeholder("F6_stress_tests", "C4/stress.csv",
                            formats=formats)
    rows = read_csv(csv)
    if not rows:
        return _placeholder("F6_stress_tests", "C4 CSV empty",
                            formats=formats)
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE, NATURE_SINGLE * 0.75),
                           constrained_layout=True)
    # Look for either disorder_xi or axis=="disorder"
    sub = [r for r in rows if r.get("axis") == "disorder"
           or "disorder_xi" in r or "disorder_strength" in r]
    if not sub:
        sub = rows
    xs, ys = [], []
    for r in sub:
        x = r.get("xi") or r.get("disorder_xi") or r.get("disorder_strength")
        y = r.get("r3") or r.get("m_min_over_max") or r.get("switched")
        if x is None or y is None:
            continue
        try:
            xs.append(float(x)); ys.append(float(y))
        except (TypeError, ValueError):
            continue
    if not xs:
        missing_panel(ax, "no disorder samples in C4")
    else:
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-",
                color=_PALETTE[1])
        ax.set_xlabel(r"$\sigma_{\rm disorder}$")
        ax.set_ylabel(r"$r_3$ / switching")
        ax.set_title("Stress: disorder")
    return save(fig, "F6_stress_tests", formats=formats)


# ====================================================================== F7
def make_f7(formats=("pdf", "png")) -> list:
    base = CAMPAIGN_OUT / "C5"
    raw = base / "switching_vs_fluence.csv"
    obs = base / "f_obs_vs_fluence_and_spot.csv"
    fit = base / "kjma_fit.csv"
    fig, axes = plt.subplots(1, 3,
                             figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 3.2),
                             constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    if raw.exists():
        rs = read_csv(raw)
        F = [r["F"] for r in rs if r.get("switched", -1) >= 0]
        S = [r["switched"] for r in rs if r.get("switched", -1) >= 0]
        if F:
            ax_a.plot(F, S, "o-", color=_PALETTE[0])
            ax_a.set_xlabel("F (homogeneous)"); ax_a.set_ylabel("switched")
            ax_a.set_title("(a) f(F)")
        else:
            missing_panel(ax_a, "no successful C5 runs")
    else:
        missing_panel(ax_a, "C5/switching_vs_fluence.csv")

    if obs.exists():
        rs = read_csv(obs)
        ws = sorted({r["w"] for r in rs})
        for k, w in enumerate(ws):
            sub = sorted((r for r in rs if r["w"] == w),
                         key=lambda r: r["F_peak"])
            ax_b.plot([r["F_peak"] for r in sub],
                      [r["f_obs"] for r in sub], "o-",
                      label=f"w={w}", color=_PALETTE[k % len(_PALETTE)])
        ax_b.set_xlabel(r"$F_{\rm peak}$"); ax_b.set_ylabel(r"$f_{\rm obs}$")
        ax_b.legend(fontsize=6); ax_b.set_title("(b) Beam-convolved")
    else:
        missing_panel(ax_b, "C5/f_obs_vs_fluence_and_spot.csv")

    if obs.exists() and fit.exists():
        rs = read_csv(obs)
        # Avrami plot on middle w
        ws = sorted({r["w"] for r in rs})
        w0 = ws[len(ws) // 2]
        sub = [r for r in rs if r["w"] == w0 and 0 < r["f_obs"] < 1]
        if len(sub) >= 3:
            x = np.log([r["F_peak"] for r in sub])
            y = np.log(-np.log(1 - np.array([r["f_obs"] for r in sub])))
            ax_c.plot(x, y, "o", color=_PALETTE[2])
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 32)
            ax_c.plot(xx, slope * xx + intercept, "-", color=_PALETTE[2],
                      label=f"n = {slope:.2f}")
            ax_c.set_xlabel(r"$\ln F$")
            ax_c.set_ylabel(r"$\ln(-\ln(1-f))$")
            ax_c.legend(); ax_c.set_title("(c) Avrami")
        else:
            missing_panel(ax_c, "not enough (0<f<1) points")
    else:
        missing_panel(ax_c, "C5/kjma_fit.csv")
    return save(fig, "F7_kjma_forward", formats=formats)


# ====================================================================== F8
def make_f8(formats=("pdf", "png")) -> list:
    base = CAMPAIGN_OUT / "C6"
    clean = base / "droplet_radius_clean.csv"
    dis = base / "droplet_radius_disordered.csv"
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE, NATURE_SINGLE * 0.75),
                           constrained_layout=True)
    plotted = False
    if clean.exists():
        rs = read_csv(clean)
        t = np.array([r["t"] for r in rs])
        R = np.array([r["R"] for r in rs])
        if len(t):
            ax.plot(t, R, "-", color=_PALETTE[0], label="clean")
            plotted = True
            # AC fit: R^2 = R0^2 - 2 lambda (t - t0)
            mask = R > 1.0
            if mask.sum() > 4:
                p = np.polyfit(t[mask], R[mask] ** 2, 1)
                ax.plot(t[mask], np.sqrt(np.maximum(p[0] * t[mask] + p[1], 0)),
                        "--", color=_PALETTE[0], alpha=0.6,
                        label=f"AC fit (slope {p[0]:.2g})")
    if dis.exists():
        rs = read_csv(dis)
        seeds = sorted({int(r["seed"]) for r in rs if r.get("seed") is not None})
        for k, s in enumerate(seeds):
            sub = sorted((r for r in rs if int(r["seed"]) == s),
                         key=lambda r: r["t"])
            ax.plot([r["t"] for r in sub], [r["R"] for r in sub],
                    "-", color=_PALETTE[1 + k % 6], alpha=0.7,
                    label=f"dis seed {s}")
            plotted = True
    if not plotted:
        missing_panel(ax, "C6 droplet CSVs not found")
    else:
        ax.set_xlabel("t"); ax.set_ylabel("R(t)")
        ax.legend(fontsize=6)
        ax.set_title("Droplet recovery (mechanisms A vs B)")
    return save(fig, "F8_recovery_competition", formats=formats)


# ====================================================================== F9
def make_f9(formats=("pdf", "png")) -> list:
    base = CAMPAIGN_OUT / "C7" / "runs" / "2dcs"
    fig, (ax_a, ax_b) = plt.subplots(1, 2,
                                     figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 2.6),
                                     constrained_layout=True)
    # 2D map
    img = None
    if base.exists():
        for cand in list(base.rglob("chi3*.csv")) + list(base.rglob("twoD_amp*.csv")):
            try:
                arr = np.loadtxt(cand, delimiter=",")
                if arr.ndim == 2 and min(arr.shape) > 4:
                    img = arr; break
            except Exception:
                continue
        if img is None:
            for cand in list(base.rglob("*.h5")):
                try:
                    import h5py
                    with h5py.File(cand, "r") as h:
                        for key in ("chi3", "twoD_amp", "amplitude"):
                            if key in h:
                                img = np.abs(h[key][:]); break
                    if img is not None: break
                except Exception:
                    continue
    if img is None:
        files = list(base.rglob("*")) if base.exists() else []
        msg = "no parseable 2DCS output"
        if files:
            msg += f"\n({len(files)} files in {base.name}/)"
        missing_panel(ax_a, msg)
    else:
        im = ax_a.imshow(img, origin="lower", aspect="auto", cmap="magma")
        fig.colorbar(im, ax=ax_a, label=r"|$\chi^{(3)}$|")
        ax_a.set_xlabel(r"$\omega_{\rm probe}$"); ax_a.set_ylabel(r"$\omega_{\rm pump}$")
        ax_a.set_title("(a) 2DCS")

    # 1D pump-probe placeholder (no companion C7 1D run yet)
    one_d = CAMPAIGN_OUT / "C7" / "linear_response.csv"
    if one_d.exists():
        rs = read_csv(one_d)
        if rs:
            t = [r["t"] for r in rs]; y = [r["dR"] for r in rs]
            ax_b.plot(t, y, "-", color=_PALETTE[0])
            ax_b.set_xlabel("t"); ax_b.set_ylabel(r"$\Delta R(t)$")
            ax_b.set_title("(b) Linear pump-probe")
        else:
            missing_panel(ax_b, "linear_response.csv empty")
    else:
        missing_panel(ax_b, "C7/linear_response.csv\n(optional)")
    return save(fig, "F9_observables", formats=formats)


# ====================================================================== F10
def make_f10(formats=("pdf", "png")) -> list:
    verdicts = CAMPAIGN_OUT / "C8" / "verdicts.csv"
    freq = CAMPAIGN_OUT / "C8" / "K1_freq_sweep.csv"
    fig, (ax_a, ax_b) = plt.subplots(1, 2,
                                     figsize=(NATURE_DOUBLE, NATURE_DOUBLE / 3.0),
                                     constrained_layout=True)
    if verdicts.exists():
        rs = read_csv(verdicts)
        if rs:
            labels = [r["test"] for r in rs]
            passed = [int(r["passed"]) for r in rs]
            exp = [int(r["expected"]) for r in rs]
            x = np.arange(len(labels))
            ax_a.bar(x - 0.2, exp, width=0.4, color=_PALETTE[7], label="expected")
            ax_a.bar(x + 0.2, passed, width=0.4, color=_PALETTE[2], label="observed")
            ax_a.set_xticks(x); ax_a.set_xticklabels(labels, rotation=30,
                                                     ha="right", fontsize=6)
            ax_a.set_ylim(0, 1.2); ax_a.legend()
            ax_a.set_title("(a) Falsifier verdicts")
        else:
            missing_panel(ax_a, "verdicts.csv empty")
    else:
        missing_panel(ax_a, "C8/verdicts.csv")
    if freq.exists():
        rs = read_csv(freq)
        if rs:
            f = [r["freq"] for r in rs]; s = [r["switched"] for r in rs]
            ax_b.plot(f, s, "o-", color=_PALETTE[1])
            ax_b.set_xlabel("pump frequency"); ax_b.set_ylabel("switched")
            ax_b.set_title("(b) K1 — resonance")
        else:
            missing_panel(ax_b, "K1_freq_sweep.csv empty")
    else:
        missing_panel(ax_b, "C8/K1_freq_sweep.csv")
    return save(fig, "F10_falsifiers", formats=formats)


# ====================================================================== F11
def make_f11(formats=("pdf", "png")) -> list:
    csv = CAMPAIGN_OUT / "C9" / "predictions.csv"
    if not csv.exists():
        return _placeholder("F11_geometry_predictions",
                            "C9/predictions.csv", double=True, formats=formats)
    rs = read_csv(csv)
    if not rs:
        return _placeholder("F11_geometry_predictions",
                            "C9 predictions empty",
                            double=True, formats=formats)
    axes_names = sorted({r["axis"] for r in rs if r.get("axis")})
    n = max(len(axes_names), 1)
    fig, axarr = plt.subplots(1, n,
                              figsize=(NATURE_DOUBLE, NATURE_DOUBLE / max(n, 3)),
                              constrained_layout=True)
    if n == 1:
        axarr = [axarr]
    for ax, axis in zip(axarr, axes_names):
        sub = sorted((r for r in rs if r["axis"] == axis),
                     key=lambda r: float(r["value"]))
        x = [float(r["value"]) for r in sub]
        for k, key in enumerate(["tau_A", "tau_B", "tau_C"]):
            y = [r.get(key) for r in sub]
            y = [yi for yi in y if isinstance(yi, (int, float)) and yi == yi]
            if len(y) == len(x):
                ax.plot(x, y, "o-", color=_PALETTE[k], label=key)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(axis); ax.set_ylabel(r"$\tau$")
        ax.set_title(axis); ax.legend(fontsize=6)
    return save(fig, "F11_geometry_predictions", formats=formats)


# ====================================================================== registry
ALL_FIGURES = {
    "F2": make_f2,
    "F3": make_f3,
    "F4": make_f4,
    "F5": make_f5,
    "F6": make_f6,
    "F7": make_f7,
    "F8": make_f8,
    "F9": make_f9,
    "F10": make_f10,
    "F11": make_f11,
}
