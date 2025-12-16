# matrix_mult

The following code implements 2 different matrix multiplication tests (**C = A x B**). The first is a
naive approach that requires each thread to load a row and column of the matrices to perform a dot
product; this is the most intensive for global memory access. The second approach uses shared memory
to load tiles of **A** and **B** and compute the dot product on each tile and then summed over tiles.

The following output is obtained on a single A100 GPU on the NERSC Perlmutter machine.
```
Naive matrix multiplication compute time (ms): 261
Shared memory matrix multiplication compute time (ms): 62
```