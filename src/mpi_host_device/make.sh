#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load craype-x86-milan
module load cuda/12.3

# On Cray systems with PrgEnv-gnu, get MPI paths from environment
# Use nvcc directly for CUDA compilation
MPI_INCLUDE="-I${CRAY_MPICH_DIR}/include"
MPI_LINK="-L${CRAY_MPICH_DIR}/lib -lmpich"
CUDA_LINK="-L${CUDA_HOME}/lib64 -lcudart"
# Try to find and link GTL for GPU-aware MPI support
GTL_LINK="-L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_cuda"

nvcc -arch=sm_90 -x cu -std=c++17 --expt-extended-lambda \
     $MPI_INCLUDE $MPI_LINK $CUDA_LINK $GTL_LINK mpiHostDevice.cpp -o mpiHostDevice.exe

