# mpi_host_device

The following code implements message passing of a buffer in 3 different manners. First, the buffer
is sent directly between CPU ranks. Second, the buffer is copied from GPU to CPU and then passed to a
separate CPU rank, which copies the buffer to a separate GPU. Finally, the buffer is sent directly from
GPU to GPU. 

The following output is obtained on 2 CPU ranks with 2 A100 GPUs on the NERSC Perlmutter machine.
```
CPU-CPU MPI transfer time (us): 74
GPU/CPU-CPU/GPU MPI transfer time (us): 7254
GPU-GPU MPI transfer time (us): 278
```

The following output is obtained on 2 CPU ranks with 2 H100 GPUs on the NREL Kestrel machine using nvcc with GTL:
```
CPU-CPU MPI transfer time (us): 72
GPU/CPU-CPU/GPU MPI transfer time (us): 4119
GPU-GPU MPI transfer time (us): 859
```

The following output is obtained on 2 CPU ranks with 2 H100 GPUs on the NREL Kestrel machine using PrgEnv-nvhpc:
```
CPU-CPU MPI transfer time (us): 84
GPU/CPU-CPU/GPU MPI transfer time (us): 4295
GPU-GPU MPI transfer time (us): 912
```

## Building on NREL Kestrel

Two build approaches are provided for Kestrel's H100 GPU nodes:

### Method 1: nvcc with GTL (make.sh)
Uses nvcc directly with Cray MPICH and manually links the GTL library for GPU-aware MPI support.
```bash
source make.sh
source run.sh
```

### Method 2: PrgEnv-nvhpc (make_nvhpc.sh)
Uses Cray's `PrgEnv-nvhpc` with NVIDIA compilers. GPU-aware MPI is handled automatically by the Cray wrappers. Based on [Kestrel GPU documentation](https://natlabrockies.github.io/HPC/Documentation/Systems/Kestrel/Environments/gpubuildandrun/#mpicudaaware).
```bash
source make_nvhpc.sh
source run_nvhpc.sh
```

Both methods deliver similar performance.