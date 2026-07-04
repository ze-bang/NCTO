# Fir (Digital Research Alliance) runtime env for the NCTO campaign.
# Source this in the SAME shell that runs the solver / python (module is an
# eval'd shell function -- never pipe `module load ... | grep`).
module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 eigen/3.4.0 boost-mpi/1.82.0 \
            hdf5-mpi/1.14.6 python/3.11.5 scipy-stack/2024b
# hdf5-mpi sets only LIBRARY_PATH; spin_solver + h5py need libhdf5*.so at runtime:
export LD_LIBRARY_PATH="${EBROOTHDF5:+$EBROOTHDF5/lib:}${LD_LIBRARY_PATH:-}"
# use the module numpy/h5py (matched to HDF5 1.14.6), not the broken ~/.local one:
export PYTHONNOUSERSITE=1
# one thread per solver process; the driver's pool provides parallelism:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
