# matrix_add

The following code implements 5 different matrix summation tests. The first is the standard
**C = A + B** that is completed element by element. Therefore, each thread must load 2 elements
from global memory and write the sum. The other 4 tests involve summations over the columns
and rows of **A** and **B** with and without shared memory. In the *non-shared* memory
implementation, each thread must load the entire columns/rows of the matrices. In the *shared*
memory implementation, each thread must only load a few elements ($\propto NROW/TILEY$) from
global memory and then sum over each tile.  

The following output is obtained on a single A100 GPU on the NERSC Perlmutter machine.
```
Element matrix add compute time (ms): 28
Sum matrix col compute time (ms): 264
Sum matrix row compute time (ms): 227
Sum matrix col shared compute time (ms): 62
Sum matrix row shared compute time (ms): 34
```