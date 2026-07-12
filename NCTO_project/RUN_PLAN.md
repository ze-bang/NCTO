# NCTO campaign — final deliverables

All results use the tuned-Kruger Hamiltonian (meV) `J=0.68, K=-7.89, Γ=3.07,
Γ'=-2.94, J2_A=-0.06, J2_B=-0.70, J3=0.52` at **L=36**, with the corrected
signed-Grüneisen E1 coupling `λ_{X,2}=(X/K)·λ_{K,2}` on all four channels
(`λ_{J7,0}=0` everywhere except the polarization study). Solver =
`ClassicalSpin_Cpp` (`build/spin_solver` + `build/gneb_field_eval`), overlaid
via `build`/`util` symlinks; runtime env `cluster_campaign/env_fir.sh`;
regenerate any run with `run_campaign.py <subcommand>` (resumable).

| # | Deliverable | Figure | Driver / plotter |
|---|---|---|---|
| 1 | Optical switching threshold map E0c(J7, λ) — clean, all-channel | `phase_diagram_allchan/analysis/switching_threshold_map.png` (+ 3D cube) | `run_campaign.py drive-pd --all-channels` → `plot_switching_phase.py`, `plot_phase_barrier_cube.py` |
| 2 | Static 3Q-vs-zigzag phase diagram (no drive) | `static_phase_diagram/analysis/static_phase_diagram_3q_zz.png` | `run_campaign.py static-pd` |
| 3 | Kinetic depinning barrier vs defect width (continuation GNEB) | `kinetic_barrier/kinetic_barrier_vs_width.png` + `width_sweep/mep_profiles_per_hw.png` | `gneb_barrier_width_sweep.py` → `plot_mep_profiles.py`, `plot_kinetic_barrier_width.py` |
| 4 | Defect width calibrated to the 0.8 ms / 10 K lifetime | (same figure, right panel) | — |
| 5 | Switching fraction vs fluence at increasing disorder (hw=2.0) | `pinning_switching_crosscheck_L36/analysis/switching_fraction_vs_fluence.png` | `run_campaign.py switching --half-width 2.0` → `plot_switching_fraction.py` |
| S | Polarization switching map r3(θ, E0), J7 modulation off/on | `polarization_fluence_study/analysis/polarization_threshold_vs_theta.png` | `run_campaign.py polarization` → `plot_polarization_threshold.py` |

## Result summary

1. **Threshold map (clean, defect-free).** Switching is confined to a wedge near
   the 3Q/ZZ degeneracy: E0c=14 at (J7=-0.40, λ=0.01) rising to ~38 by λ=0.04;
   no switching for J7 ≤ -0.48 or λ ≥ 0.045 (up to E0=40). Stronger E1 coupling
   *raises* the threshold (drive-renormalized 3Q stabilization).
2. **Static phase diagram.** Zigzag ground state for J7 ≳ -0.42, 3Q below;
   boundary λ-independent (quadratic striction inert at ε=0). Working point
   J7=-0.40 sits just on the ZZ side of near-degeneracy (dE ≈ +0.002 meV/site);
   the 3Q branch is metastable only for J7 ≤ -0.40.
3. **Kinetic barrier vs width.** Continuation GNEB (each hw warm-started from
   the previous) gives one converged MEP family: ΔG = 9.4 / 11.3 / 16.3 / 16.3 /
   17.4 meV at hw = 1.0 / 1.5 / 2.0 / 2.5 / 3.0, saturating ≈ 17.4 meV.
   Quarter-widths (0.75-2.75) are **discarded**: their unbalanced K-enhanced
   bond row makes the pinned strip the *ground state* (end state above strip →
   no decay barrier); hw=0.5 strip is not metastable. Per-hw MEP profiles are
   the convergence record.
4. **Lifetime calibration.** ΔG* = 17.7 meV for τ=0.8 ms at 10 K (τ0=ħ/J≈1 ps).
   The saturated barrier 17.4 meV gives τ ≈ 0.5 ms — the 0.8 ms target within
   the attempt-time band (τ0≈1.5 ps exact). Switching study uses hw=2.0
   (ΔG=16.3 meV). NB the barrier is extensive in defect *length* (0.53 meV/cell
   at hw=2), so lifetimes quote a fixed physical defect length (ℓ_d = box = 36).
5. **Disorder rounding (hw=2.0).** Clean line: sharp threshold at E0≈16-17.
   Zero-mean Gaussian disorder on the NN Kitaev coupling (σ_K per bond)
   progressively rounds the step and moves the onset to lower fluence
   (σ_K=0.5 writes from E0≈6) — threshold-free switching from disorder.
S. **Polarization (clean, ± J7 modulation).** Converged r3(θ,E0) maps show
   *windowed* switching: clean near θ=0 and 90°, stable partial-nematic states
   (r3≈0.2-0.5) at intermediate θ. Grüneisen-matched J7 breathing modulation
   (λ_{J7,0}=+1.014e-3) lowers thresholds but fragments the switching region.
   Verified clean geometry and full dynamical convergence.
