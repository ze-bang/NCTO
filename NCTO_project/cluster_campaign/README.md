# NCTO corrected-coupling campaign — remote cluster run

Self-contained instructions to regenerate the driven-dynamics figures of
`phonon_engineering_ncto.tex` on an HPC cluster (SLURM), using the **corrected
magnetoelastic coupling**.

## 1. What this recomputes and why

The E1 magnetoelastic coupling is the quadratic (nonlinear-phononic) exchange
striction implemented in `src/core/phonon_lattice.cpp`:

    delta X_gamma(Q) = lambda_{X,0} |Q|^2
                     + lambda_{X,2} [ (Qx^2 - Qy^2) cos 2*theta_gamma
                                      + 2 Qx Qy sin 2*theta_gamma ],
    X in {J, K, Gamma, Gamma'},  theta_x=0, theta_y=2pi/3, theta_z=4pi/3.

The published runs used `lambda_{X,2} = (|X|/|K|) lambda_{K,2}`, which has the
**wrong sign** on the J and Gamma channels. The correct isotropic-Grueneisen
choice (common fractional modulation delta X / X) is the *signed* ratio

    lambda_{X,2} = (X/K) * lambda_{K,2},   lambda_{K,2} = 0.02
    => lambda_{J,2} = -0.0017, lambda_{Gamma,2} = -0.0078, lambda_{Gamma',2} = +0.0075

This is already baked into the two driver scripts in this bundle. A local
spot-check showed the switching threshold roughly doubles (E0 ~ 4 -> ~ 8), so
the figures below must be regenerated.

Affected figures: **Fig 1** (phase cube), **Fig 2c** + **Fig 4** (L36 switching).
NOT affected (spin-only, no phonon): the GNEB barrier / lifetime / defect
geometry (Fig 3 and the 21.22 / 24.15 meV numbers) — do not recompute.

## 2. Prerequisites on the cluster

- Toolchain: C++17 compiler, CMake >= 3.16, MPI, HDF5 (C++), Boost, Eigen3, OpenMP.
- Python 3 with: `numpy`, `h5py`, `matplotlib`, `scipy`.
- Two repos: the solver (`ClassicalSpin_Cpp`) and this campaign (`NCTO`). The
  campaign is overlaid onto the solver checkout so the scripts resolve their
  paths (`build/spin_solver`, `util/readers_new/`).

## 3. Workflow

### (a) Get both repos onto the cluster and overlay
    git clone git@github.com:ze-bang/ClassicalSpin_Cpp.git
    git clone git@github.com:ze-bang/NCTO.git
    # overlay the campaign on top of the solver checkout:
    cp -r NCTO/NCTO_project NCTO/NCTO_project_paper ClassicalSpin_Cpp/
    cd ClassicalSpin_Cpp
    # (`make_bundle.sh` is an alternative if overlaying from a live working tree)

### (c) Build the solver
    # EDIT module names in build.sbatch to match your site, then:
    sbatch NCTO_project/cluster_campaign/build.sbatch
    # verify: ls -la build/spin_solver

### (d) Run the campaign
    # EDIT module/venv lines + #SBATCH partition/account in run_campaign.sbatch
    sbatch NCTO_project/cluster_campaign/run_campaign.sbatch
    # includes the clean polarization/fluence study (J7 phonon on vs off);
    # skip it with: RUN_POL=0 sbatch ...

### (e) Regenerate the figures (cheap; on the cluster or back home)
    A=NCTO_project/tuned_kruger_campaign
    L=$A/pinning_switching_crosscheck_L36/analysis
    python3 NCTO_project/scripts/plot_phase_barrier_cube.py --all-channels
    python3 NCTO_project/scripts/plot_l36_switching_rounding.py \
        --summary $L/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck_summary.json \
        --out-dir $L/l36_switching_allJKGG_err10
    python3 NCTO_project/scripts/plot_one_picture_signatures.py \
        --switch-summary $L/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck_summary.json \
        --out-dir $L/one_picture_allJKGG_err10 --max-switch-sigma 0.20
Plotting is a separate light step (the plotters also pull in the coupling-
independent GNEB / defect-geometry inputs, which are unchanged).

## 4. Resource / runtime notes

- Single node, one solver process per core (`OMP_NUM_THREADS=1`, thread-pool
  parallelism). `--cpus-per-task=32` is a good default; raise to the node width.
- Work: Fig 1 = 462 L18 MD runs (~minutes). Figs 2c/4 = 2646 L36 MD + 2646
  quenches + ~126 initial-state relaxations. Estimated ~6-10 h on 32 cores
  (L36 MD ~3-4 min each). Walltime is set to 24 h.
- **Resumable**: outputs are skipped if present, so a walltime kill + resubmit
  continues. Safe to run (d) repeatedly until the log prints `ALL DATA DONE`.

## 5. Outputs

- Fig 1 data: `.../phase_diagram_allchan/analysis/phase_summary.csv`
- Figs 2c/4 data:
  `.../pinning_switching_crosscheck_L36/analysis/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck{.csv,_summary.json}`
- Bring the summary CSV/JSON (and, if generated there, the PNG/PDF figures) back
  and update the result numbers in `phonon_engineering_ncto.tex`
  (thresholds, sigma-resolved switching percentages, 25-75% widths).

## 6. Scaling further (optional)

For many nodes, convert to a SLURM job array over `.param` configs: pre-generate
configs with `--plot-only`-style dry runs, then have each array task run a slice
and aggregate at the end. Not needed for the grids above — one fat node suffices.
