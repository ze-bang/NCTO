#!/bin/bash
#SBATCH --job-name=ncto_refine
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ncto_refine_%j.log
# Convergence check: refine each width-sweep MEP 25 -> 41 images (warm start,
# independent per hw -> parallel). Converged iff the barrier is stable.
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
for hw in 1.0 1.5 2.0 2.5 3.0; do
  python3 NCTO_project/scripts/gneb_refine_width.py --hw "$hw" > refine_hw${hw}.log 2>&1 &
done
wait
echo "==== $(date) REFINE DONE ===="
grep -h 'refined barrier' refine_hw*.log
