# Nature-Physics Computational Campaign

Scripts implementing the campaign laid out in
[`docs/ncto_nature_physics_campaign.tex`](../../docs/ncto_nature_physics_campaign.tex).
Each phase has a single Python entry point under this directory.

## Layout

| File | Phase | What it produces |
|---|---|---|
| `common.py` | infra | param-templating + `spin_solver` runner |
| `analysis_utils.py` | infra | M-point structure factor, harmonic fits, CSV IO |
| `C1_param_robustness.py` | C1 | `np_campaign_out/C1/{results.csv, threshold_map.png}` |
| `C2_barrier_string.py` | C2 | `np_campaign_out/C2/{paths.csv, barrier_map.png}` |
| `C3_extract_coarse_grained.py` | C3 | `np_campaign_out/C3/coarse_grained.csv` |
| `C4_stress_tests.py` | C4 | `np_campaign_out/C4/{stress.csv, stress.png}` |
| `C5_kjma_forward.py` | C5 | `np_campaign_out/C5/{kjma_rise.csv, fmax_F_theta.csv, *.png}` |
| `C6_recovery_competition.py` | C6 | `np_campaign_out/C6/{recovery_curves.csv, tau_off_F.csv, recovery_compare.png}` |
| `C7_observables.py` | C7 | `np_campaign_out/C7/{harmonics.csv, intensities.csv, freq_response.csv, observables.png}` |
| `C8_kill_alternatives.py` | C8 | `np_campaign_out/C8/{verdict.csv, K{1..4}_results.csv, kill.png}` |
| `C9_geometry_predictions.py` | C9 | `np_campaign_out/C9/{predictions.csv, decision_tree.txt, scaling.png}` |
| `run_all.sh` | driver | runs all phases sequentially |

## Quick start

```bash
# sanity-check sweep (~minutes wall time)
bash NCTO_project/scripts/np_campaign/run_all.sh quick

# publication-grade sweep (hours-to-days depending on machine)
bash NCTO_project/scripts/np_campaign/run_all.sh full
```

Each phase script also accepts `--quick` / `--full` directly.

## Conventions

* The C++ solver and analyser are taken from `build/spin_solver` and
  `build/analyze_phonon_m_order`.
* All template substitutions go through `common.render_param`, which
  fills `NCTO_project/configs/np_campaign_base.param` placeholders from
  `common.DEFAULTS` and per-phase overrides.
* All outputs land under `NCTO_project/np_campaign_out/`.
* Switching is declared by `r_3 = m_min/m_max < 0.2`.
* Pure-Python phases (C5, C6, C7, C9) do not call the C++ solver and
  can be developed/iterated independently.

## Dependencies between phases

```
C1 ─┐
    ├─> C3 ─> C5 ─> C7 ─> C9
C2 ─┘            └─ C6 ─┘
C4 (independent, complements C1)
C8 (reuses C1, C5, C6 drivers)
```

## Status of approximations

Two implementation choices were made for tractability and are flagged
in the corresponding script docstrings:

* **C2 barrier** is estimated by SLERP-interpolated chains relaxed
  under a static effective bias rather than a true string method.
* **C4 ellipticity** is approximated as an effective amplitude
  rescaling; full ellipticity support would require extending the
  C++ pump driver to accept independent (Ex, Ey) channels.

Both can be upgraded without changing the surrounding orchestration.
