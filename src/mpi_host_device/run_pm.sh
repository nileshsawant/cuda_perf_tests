#!/bin/bash

# Get interactive allocation
salloc --nodes 1 --qos interactive --time 00:05:00 \
       --constraint gpu --gpus-per-node=2 --account=m4106_g

# Run with two ranks
srun -n2 ./mpiHostDevice.exe
