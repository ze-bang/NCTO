#!/bin/bash
#SBATCH --job-name=ncto_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --output=ncto_build_%j.log
### --- SITE PLACEHOLDERS: set for your cluster ---
###SBATCH --partition=<PARTITION>
###SBATCH --account=<ACCOUNT>
# ---------------------------------------------------------------------------
# Build build/spin_solver from a fresh clone.  Dependencies:
#   C++17 compiler, CMake >= 3.16, MPI, HDF5 (C++), Boost, Eigen3, OpenMP.
# Adjust the `module load` lines to your cluster's module names.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- EDIT: load your cluster's toolchain ---
module purge || true
module load cmake gcc openmpi hdf5 boost eigen || \
  echo "WARN: adjust module names to your site (cmake/gcc/openmpi/hdf5/boost/eigen)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"          # run from the repo root
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target spin_solver -j "${SLURM_CPUS_PER_TASK:-8}"

echo "Built: $(ls -la build/spin_solver)"
./build/spin_solver --help 2>/dev/null | head -3 || true
