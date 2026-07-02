#!/bin/bash
# Package the untracked campaign "overlay" (scripts, helper modules, reference
# states) that is NOT in git, so it can be dropped on top of a fresh clone of
# ClassicalSpin_Cpp on the cluster.  Run this LOCALLY from the repo root.
#
#   bash NCTO_project/cluster_campaign/make_bundle.sh
#
# Produces:  ncto_campaign_overlay.tar.gz   (in the repo root)
# The corrected signed-Grueneisen coupling lambda_{X,2}=(X/K) lambda_{K,2} is
# already baked into the two driver scripts included here.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT=ncto_campaign_overlay.tar.gz
CAMP=NCTO_project/tuned_kruger_campaign

FILES=(
  # --- python helper modules (imported by the drivers) ---
  util/readers_new/reader_strain_lattice.py
  NCTO_project/scripts/np_campaign/analysis_utils.py
  NCTO_project/scripts/np_campaign/initial_states.py
  # --- the two campaign drivers (contain the corrected coupling) ---
  NCTO_project/scripts/run_tuned_kruger_phase_diagram.py
  NCTO_project/scripts/cross_validate_pinned_switching_l36.py
  NCTO_project/scripts/run_polarization_fluence_study.py
  # --- cluster job scripts ---
  NCTO_project/cluster_campaign/campaign_steps.sh
  NCTO_project/cluster_campaign/run_campaign.sbatch
  NCTO_project/cluster_campaign/build.sbatch
  NCTO_project/cluster_campaign/README.md
  # --- coupling-independent reference states the drivers read ---
  "$CAMP/kinetic_barrier/zz_relax/sample_0/spins_T=0.txt"     # REF_ZZ_L18
)
# 3Q seeds (6): let the phase driver skip template-based regeneration
for s in "$CAMP"/phase_diagram/analysis/seed3Q_J7*.txt; do FILES+=("$s"); done

echo "Bundling ${#FILES[@]} files ..."
missing=0
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "  MISSING: $f"; missing=1; }; done
[ "$missing" = 0 ] || { echo "Aborting: missing files above."; exit 1; }

tar -czf "$OUT" "${FILES[@]}"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "Transfer to the cluster and extract INSIDE the cloned repo root:"
echo "  scp $OUT user@cluster:/path/to/ClassicalSpin_Cpp/"
echo "  ssh user@cluster 'cd /path/to/ClassicalSpin_Cpp && tar -xzf $OUT'"
