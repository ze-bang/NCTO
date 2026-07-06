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

All runs are consolidated into **one driver**, `run_campaign.py`, sharing
`ncto_common.py` (single source of the Hamiltonian and the corrected
signed-Grüneisen couplings `λ_{X,2}=(X/K)·λ_{K,2}`). Everything is **L=36**,
resumable, and parallelised across all cores with one OMP-pinned worker pool
(`OMP_NUM_THREADS=1` per solver process — no oversubscription).

Run the whole campaign:
```bash
python3 NCTO_project/scripts/run_campaign.py all --workers "$(nproc)"
```
or any single study via its subcommand (below).

### 0. Static (no-drive) 3Q-vs-ZZ phase diagram — NEW baseline
Ground-state 3Q vs ZZ over J7 ∈ [0, −0.7] (Δ=0.05, 15 points), same λ_{K,2}
span. No pump: the quadratic E1 striction `δX∝ε²` vanishes at the ε=0
equilibrium (below the pseudo-JT threshold λ*_{K,2}≈1.55), so the static
boundary is **λ-independent** — set by J7 alone. Emits `dE(J7)=E_3Q−E_ZZ`
(the CNT driving force) and the winner map.
```bash
python3 NCTO_project/scripts/run_campaign.py static-pd --workers "$(nproc)"
```
~30 SA relaxations + energy evals (15 J7 × {3Q,ZZ}); minutes.
→ `tuned_kruger_campaign/static_phase_diagram/analysis/static_phase_diagram.{csv,png}`

### 1. Clean switching phase diagram (Fig 1)
Driven switch/no-switch over (J7, λ_{K,2}, E0), all four channels (signed
Grüneisen). 6 J7 × 7 λ × **18 E0 (E0→40)** = 756 L36 MD. The E0 grid runs to 40
because the L=36 threshold is ~16 (vs ~8 at L18) — the extension is now the
default.
```bash
python3 NCTO_project/scripts/run_campaign.py drive-pd --all-channels --workers "$(nproc)"
python3 NCTO_project/scripts/plot_phase_barrier_cube.py --all-channels
```

### 2. Switching fraction vs fluence at different disorder (Figs 2c, 4)
Fixed enhanced-|K| line (`dK=−0.5|K|`, hw=2.0) + zero-mean Gaussian background
disorder on all NN bonds; switched fraction vs E0 for several σ_K.
```bash
python3 NCTO_project/scripts/run_campaign.py switching --workers "$(nproc)"
# (defaults: kred, strength 0.5, half-width 2.0, global-zero-k, seeds 25,
#  σ_K 0..0.5, the 21-point E0 grid, suffix allJKGG_err10)

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
(`λ_{J7,0}=(J7/K)λ_{K,2}=+1.014×10⁻³`, the Grüneisen-matched value). The bilinear
striction makes the threshold θ-dependent; the isotropic J7 breathing coupling
`δJ7(Q)=λ_{J7,0}|Q|²` shifts E0ᶜ(θ) uniformly in θ. Their difference is the
J7-tuning contribution. 2×7×9 = 126 L36 MD.
```bash
python3 NCTO_project/scripts/run_campaign.py polarization --workers "$(nproc)"
```

**Cluster:** `sbatch NCTO_project/cluster_campaign/run_campaign.sbatch` (update it to
call `run_campaign.py all`).

---

## 📝 After the driving runs

Update `NCTO_project_paper/phonon_engineering_ncto.tex` with the new numbers
(clean threshold, σ-resolved switching %, 25–75% widths, Fig `pol` E0ᶜ(θ)) and
confirm `polarization_fluence_switching.png` exists so the paper builds.
