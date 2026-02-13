#!/bin/bash

# Enable GPU-aware MPI support for Cray MPICH
export MPICH_GPU_SUPPORT_ENABLED=1
# Optional: May improve performance depending on the code
export MPICH_OFI_NIC_POLICY=GPU

# Use the mpi built with cuda support
srun -n 2 ./mpiHostDevice_nvhpc.exe
