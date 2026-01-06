# matrix_mult

The following code implements 3 different matrix multiplication tests (**C = A x B**). The first is a
naive approach that requires each thread to load a row and column of the matrices to perform a dot
product and does NOT allow coalesced memory access of **C**. The second approach parallels the first
but modifies the indexing to USE coalesced memory access of **C**. The third approach uses shared
memory to load tiles of **A** and **B** and compute the dot product on each tile and then sum
over tiles.

The following output is obtained on a single A100 GPU on the NERSC Perlmutter machine.
```
Naive matrix multiplication compute time (ms): 546
Coalesced memory matrix multiplication compute time (ms): 61
Shared memory matrix multiplication compute time (ms): 48
```