# RUN PLAN — remaining simulations

Everything below uses the **corrected** magnetoelastic coupling
(`λ_{X,2}=(X/K)·λ_{K,2}`, `λ_{K,2}=0.02`), already baked into the scripts.

## Prerequisites (once per machine)

1. Build the solver: `git clone ClassicalSpin_Cpp`, then
   `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target spin_solver`.
2. Overlay this repo: copy `NCTO_project/` and `NCTO_project_paper/` on top of the
   ClassicalSpin_Cpp checkout (the scripts resolve the repo root as their
   great-grandparent dir and read `build/spin_solver`, `util/readers_new/`).
3. Python: `numpy h5py matplotlib scipy`.

Everything is **resumable** — the solver skips existing outputs. On a cluster,
`sbatch NCTO_project/cluster_campaign/run_campaign.sbatch` runs steps 1–3 below.

## Status

| # | Task | Figure | Status |
|---|------|--------|--------|
| 1 | Phase-diagram cube (clean, corrected coupling) | Fig 1 | ✅ done locally — rerun on cluster for provenance |
| 2 | L36 `err10` switching (defect + global disorder) | Figs 2c, 4 | ✅ done locally — rerun on cluster for provenance |
| 3 | Clean polarization/fluence study, J7 phonon **off vs on** | Fig `pol` | ⬜ **NOT RUN — main remaining item** |
| 4 | Update paper result numbers from new summaries | — | ⬜ pending |
| — | ~~Wide fluence/disorder heatmap~~ (Fig 5) | — | ❌ removed this session |

## Step 1 — phase cube (Fig 1)

```bash
python3 NCTO_project/scripts/run_tuned_kruger_phase_diagram.py --all-channels --workers "$(nproc)"
cp NCTO_project/tuned_kruger_campaign/phase_diagram_allchan/analysis/phase_summary.csv \
   NCTO_project/tuned_kruger_campaign/phase_diagram_allchan/analysis/phase_summary_full.csv
python3 NCTO_project/scripts/plot_phase_barrier_cube.py --all-channels
```
Grid: 6 J7 × 7 λ × 11 E0 = 462 L18 MD runs.

## Step 2 — L36 err10 switching (Figs 2c, 4)

```bash
python3 NCTO_project/scripts/cross_validate_pinned_switching_l36.py \
  --lattice 36 --dtype kred --strength 0.5 --half-width 2.0 \
  --disorder-mode global-zero-k --seeds 25 \
  --sigma-k-values 0.0 0.1 0.2 0.3 0.4 0.5 \
  --e0-values 0 6 8 9 10 10.5 11 11.5 12 12.5 13 13.5 14 15 16 17 18 19 20 22 25 \
  --output-suffix allJKGG_err10 --workers "$(nproc)"

A=NCTO_project/tuned_kruger_campaign/pinning_switching_crosscheck_L36/analysis
S=$A/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck_summary.json
python3 NCTO_project/scripts/plot_l36_switching_rounding.py --summary "$S" --out-dir "$A/l36_switching_allJKGG_err10"
python3 NCTO_project/scripts/plot_one_picture_signatures.py --switch-summary "$S" \
  --out-dir "$A/one_picture_allJKGG_err10" --max-switch-sigma 0.20
```
Work: 2646 L36 MD + 2646 quenches (+ ~126 relaxations). ~6–10 h at 32 cores.

## Step 3 — clean polarization/fluence study (Fig `pol`)  ← MAIN REMAINING

No extended defect, no disorder. Scans pump polarization θ × fluence E0, run
twice: J7 ring-exchange phonon coupling **off** (`λ_{J7,0}=0`) and **on**
(`λ_{J7,0}=(J7/K)λ_{K,2}=1.0×10⁻³`).

```bash
python3 NCTO_project/scripts/run_polarization_fluence_study.py --lattice 36 --workers "$(nproc)"
# defaults: theta = 0 15 30 45 60 75 90 deg ; E0 = 0 4 6 8 10 12 14 16 20
```
Outputs:
`NCTO_project/tuned_kruger_campaign/polarization_fluence_study/analysis/polarization_fluence.csv`
and `.../polarization_fluence_switching.png` (the two θ×E0 maps, off vs on).

Expected signature: the bilinear striction makes the threshold θ-dependent; the
isotropic J7 breathing coupling shifts E0ᶜ(θ) roughly uniformly in θ. Their
difference is the phonon-tuned ring-exchange contribution.

Work: 2 couplings × 7 θ × 9 E0 = 126 L36 MD runs (~30 min at 32 cores).

## Step 4 — update the paper

After Steps 1–3, fill the paper with the new numbers (currently placeholders /
predictive):
- abstract + body: clean fixed-line threshold, σ-resolved switching %, 25–75% widths;
- Fig `pol` section: on-vs-off E0ᶜ(θ) thresholds;
- confirm `polarization_fluence_switching.png` is present so the paper builds.
