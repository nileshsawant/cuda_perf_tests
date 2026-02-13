#!/bin/bash

# Enable GPU-aware MPI support for Cray MPICH
export MPICH_GPU_SUPPORT_ENABLED=1

# Use the mpi built with cuda support
srun -n 2 ./mpiHostDevice.exe
