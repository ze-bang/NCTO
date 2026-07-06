#!/bin/bash
#SBATCH --job-name=ncto_sw08
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=ncto_sw08_%j.log
# Switching phase diagram at the NAILED geometry: hw=1.8 (dG~18.4 meV => 0.8 ms
# @10K, tau0~ hbar/J). Same kred / s=0.5 / global-zero-k / seeds 25 / E0->50.
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/run_campaign.py switching \
  --half-width 1.8 --output-suffix kred_0p8ms_hw1p8 --workers "${SLURM_CPUS_PER_TASK:-192}"
echo "==== $(date) SWITCHING (nailed hw=1.8) DONE ===="
