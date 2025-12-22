# lambda_kernel

The following code employs a lambda to forward a function for execution in a kernel.
This approach is broadly utilized with abstraction layers like AMReX and KOKKOS. The performance
aspect here reuses the shared memory column sum to show how a user could employ such techniques
within a lambda function that is utilized by an abstraction layer.

The following output is obtained on a single A100 GPU on the NERSC Perlmutter machine.
```
Lambda w/out shared compute time (ms): 2483
Lambda w/    shared compute time (ms): 2859
```