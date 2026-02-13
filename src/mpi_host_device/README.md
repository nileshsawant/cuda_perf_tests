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

The following output is obtained on 2 CPU ranks with 2 H100 GPUs on the NREL Kestrel machine.
```
CPU-CPU MPI transfer time (us): 72
GPU/CPU-CPU/GPU MPI transfer time (us): 4119
GPU-GPU MPI transfer time (us): 859
```