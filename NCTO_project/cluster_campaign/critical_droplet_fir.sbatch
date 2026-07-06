#!/bin/bash
#SBATCH --job-name=ncto_droplet
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=ncto_droplet_%j.log
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
# Seed-independent critical droplet dE(R), J7 spanning off-degeneracy so R* is
# finite/tractable at the large-Df end and diverges toward the -0.42 crossing.
python3 NCTO_project/scripts/critical_droplet.py --lattice 54 \
  --j7-values -0.45 -0.50 -0.60 -0.70 -0.80 -1.00 \
  --r-values 2 3 4 5 6 7 8 9 10 12 14 16 18 20 24
echo "==== $(date) DROPLET DONE ===="
