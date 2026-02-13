#!/bin/bash

# Based on https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Environments/gpubuildandrun/#mpicudaaware
# Use PrgEnv-nvhpc for CUDA-aware MPI with NVIDIA compilers

module load gcc
module load PrgEnv-nvhpc
module load cray-libsci/23.05.1.4
module load binutils

# Compile with CC (Cray wrapper that uses NVIDIA backend compilers)
# -gpu=cc90 targets H100 (compute capability 9.0)
# -cuda enables CUDA support
# -target-accel=nvidia90 specifies GPU architecture

CC -gpu=cc90 -cuda -target-accel=nvidia90 -c mpiHostDevice.cpp
CC -gpu=cc90 -cuda -target-accel=nvidia90 -lcudart -lcuda mpiHostDevice.o -o mpiHostDevice_nvhpc.exe

