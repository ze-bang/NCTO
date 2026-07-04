#!/bin/bash
#SBATCH --job-name=ncto_campaign
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=ncto_campaign_%j.log
# ---------------------------------------------------------------------------
# Full NCTO tuned-Kruger campaign on ONE Fir node (192 cores).  All studies are
# embarrassingly parallel single-rank spin_solver processes dispatched by the
# driver's OMP-pinned thread pool -- no mpirun.
#   static-pd (fast) -> drive-pd 462 -> switching 2646+quench -> polarization 126.
# RESUMABLE: the solver skips existing outputs; just resubmit to continue.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 -c "import numpy,h5py; print('env OK: numpy',numpy.__version__,'h5py',h5py.__version__,h5py.version.hdf5_version)" \
  || { echo "FATAL: python env"; exit 1; }
export WORKERS="${SLURM_CPUS_PER_TASK:-192}"
echo "==== $(date) launch campaign, WORKERS=$WORKERS ===="
bash NCTO_project/cluster_campaign/campaign_steps.sh
