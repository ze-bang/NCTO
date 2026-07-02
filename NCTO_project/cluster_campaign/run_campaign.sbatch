#!/bin/bash
#SBATCH --job-name=ncto_campaign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=ncto_campaign_%j.log
### --- SITE PLACEHOLDERS: set for your cluster ---
###SBATCH --partition=<PARTITION>
###SBATCH --account=<ACCOUNT>
# ---------------------------------------------------------------------------
# Corrected-coupling recompute (single fat node).  The campaign launches many
# independent single-rank spin_solver processes via an internal Python thread
# pool, so it is embarrassingly parallel on one node -- no mpirun needed.
#
# RESUMABLE: the solver skips existing outputs. If the 24h walltime is hit,
# just resubmit this same script; it continues where it stopped.
#
# Optional: export RUN_FIG5=1 before sbatch to also run the wide heatmap.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- EDIT: load your cluster's runtime (must match build.sbatch) ---
module purge || true
module load gcc openmpi hdf5 boost eigen python || \
  echo "WARN: adjust module names to your site"

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # repo root

# Python needs: numpy, h5py, matplotlib, scipy. If using a venv/conda, activate here:
#   source ~/ncto-venv/bin/activate      # or: conda activate ncto
python3 -c "import numpy,h5py" || { echo "FATAL: need numpy + h5py"; exit 1; }

export WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS=1                  # one thread per solver process; pool provides parallelism

bash NCTO_project/cluster_campaign/campaign_steps.sh
