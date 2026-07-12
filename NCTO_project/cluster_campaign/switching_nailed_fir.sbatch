#!/bin/bash
#SBATCH --job-name=ncto_sw
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=ncto_sw_%j.log
# Switching phase diagram at hw=2.0 (converged-branch barrier 16.3 meV,
# tau ~ 0.2 ms @10K with tau0=hbar/J). kred s=0.5, global-zero-k, seeds 25, E0->50.
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/run_campaign.py switching \
  --half-width 2.0 --output-suffix kred_hw2p0 --workers "${SLURM_CPUS_PER_TASK:-192}"
python3 NCTO_project/scripts/plot_switching_fraction.py \
  --summary "$REPO/NCTO_project/tuned_kruger_campaign/pinning_switching_crosscheck_L36/analysis/L36_kred_s0p500_hw2p00_kred_hw2p0_drive_crosscheck_summary.json"
echo "==== $(date) SWITCHING (hw=2.0) DONE ===="
