#!/bin/bash
#SBATCH --job-name=ncto_dpdhr
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=ncto_dpdhr_%j.log
# High-resolution clean all-channel drive phase diagram (final cross-check):
# J7 step 0.01, lambda step 0.005, E0 step 2. Existing points reused (resumable).
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/run_campaign.py drive-pd --all-channels \
  --workers "${SLURM_CPUS_PER_TASK:-192}" \
  --j7-values -0.40 -0.41 -0.42 -0.43 -0.44 -0.45 -0.46 -0.47 -0.48 -0.49 -0.50 \
  --lambda-values 0.005 0.010 0.015 0.020 0.025 0.030 0.035 0.040 0.045 \
  --e0-values 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
python3 NCTO_project/scripts/plot_switching_phase.py
echo "==== $(date) DRIVE-PD HIRES DONE ===="
