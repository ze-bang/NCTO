# RUN PLAN

Status of the Na₂Co₂TeO₆ optically-written-zigzag campaign. All scripts use the
**corrected** quadratic E1 magnetoelastic coupling `λ_{X,2}=(X/K)·λ_{K,2}`
(`λ_{K,2}=0.02`), already baked in.

## Prerequisites (once per machine)

1. Build the solver: clone `ClassicalSpin_Cpp`, then
   `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target spin_solver`.
2. Overlay this repo onto the solver checkout (`cp -r NCTO/NCTO_project NCTO/NCTO_project_paper ClassicalSpin_Cpp/`),
   so scripts resolve `build/spin_solver` and `util/readers_new/`.
3. Python: `numpy h5py matplotlib scipy`. Everything is resumable (the solver
   skips existing outputs).

---

## ✅ DONE — GNEB depinning barrier (relaxation, extended defect)

The 36×36 GNEB minimum-energy path from the pinned zigzag strip to relaxed 3Q,
for an enhanced-|K| line defect swept over half-width, is **correct and final**
(Fig 3, `selected_kred_mep_L36.png`; barrier grows 3.8→21.2 meV over hw=0.5→2).
This is a static (spin-only) transition-state calculation — it does **not**
involve the phonon drive.

Regenerate if needed:
```bash
python3 NCTO_project/scripts/selected_gneb_mep_l36_width_sweep.py   # barrier vs width
python3 NCTO_project/scripts/selected_gneb_mep_l36.py               # the selected hw=2 MEP + figure
```
Data lives in `tuned_kruger_campaign/defect_catalogue_L36/`; reference states in
`tuned_kruger_campaign/kinetic_barrier/`.

---

## 🔨 TODO — the driving (phonon-pump) studies

These are the remaining physics. All are driven-dynamics calculations with the
E1 pump; none are finalized.

### 1. Clean switching phase diagram (Fig 1)
Driven switch/no-switch over (J7, λ_{K,2}, E0) on a clean lattice.
```bash
python3 NCTO_project/scripts/run_tuned_kruger_phase_diagram.py --all-channels --workers "$(nproc)"
python3 NCTO_project/scripts/plot_phase_barrier_cube.py --all-channels
```

### 2. Switching fraction vs fluence at different disorder (Figs 2c, 4)
Fixed enhanced-|K| line + zero-mean Gaussian background disorder on all NN bonds;
switched fraction vs E0 for several σ_K.
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
~2646 L36 MD + quenches, ~6–10 h at 32 cores.

### 3. Polarization study ± J7 phonon tuning (Fig `pol`)
Clean lattice (no defect, no disorder). Scan pump polarization θ × fluence E0,
run twice: J7 ring-exchange phonon coupling **off** (`λ_{J7,0}=0`) vs **on**
(`λ_{J7,0}=(J7/K)λ_{K,2}=1.0×10⁻³`, the Grüneisen-matched value). The bilinear
striction makes the threshold θ-dependent; the isotropic J7 breathing coupling
`δJ7(Q)=λ_{J7,0}|Q|²` shifts E0ᶜ(θ) uniformly in θ (phonon-amplitude tuning of
the ring exchange). Their difference is the J7-tuning contribution.
```bash
python3 NCTO_project/scripts/run_polarization_fluence_study.py --lattice 36 --workers "$(nproc)"
```

**Cluster:** `sbatch NCTO_project/cluster_campaign/run_campaign.sbatch` runs 1–3.

---

## 📝 After the driving runs

Update `NCTO_project_paper/phonon_engineering_ncto.tex` with the new numbers
(clean threshold, σ-resolved switching %, 25–75% widths, Fig `pol` E0ᶜ(θ)) and
confirm `polarization_fluence_switching.png` exists so the paper builds.
