#!/bin/bash
#SBATCH --job-name=ncto_wsweep
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=ncto_wsweep_%j.log
# Continuation GNEB width sweep (SERIAL: each hw warm-starts from the previous
# one -> one continuous MEP family, no basin-hopping scatter).
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/gneb_barrier_width_sweep.py --lattice 36 --force
python3 NCTO_project/scripts/plot_kinetic_barrier_width.py
echo "==== $(date) WIDTH SWEEP (continuation) DONE ===="
