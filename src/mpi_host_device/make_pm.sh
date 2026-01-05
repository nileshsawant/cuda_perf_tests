#!/bin/bash

MPI_INCLUDE=$(cc --cray-print-opts=cflags)
MPI_LINK="-L$PE_PERFTOOLS_MPICH_LIBDIR -lmpi $PE_MPICH_GTL_DIR_nvidia80 -lmpi_gtl_cuda"
CUDA_MPI_LINK="-L$CUDATOOLKIT_HOME -lcudart"
nvcc -arch=sm_80 -x cu -std=c++17 --expt-extended-lambda \
     $MPI_INCLUDE $MPI_LINK $CUDA_MPI_LINK mpiHostDevice.cpp -o mpiHostDevice.exe
