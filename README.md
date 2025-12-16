# cuda_perf_tests

Basic performance tests comparing optimized to naive kernel implementations.
All tests were performed with [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)
and the reported timings were obtained from runs on the
[Perlmutter](https://www.nersc.gov/what-we-do/computing-for-science/perlmutter) HPC system.

    
## Getting started
    
### Prerequisites

Functioning device & host (<span style="color:red">nvcc & g++</span>) compilers are all that is
needed to run the examples.

### Getting cuda_perf_tests

The following command may be utilized to clone the repository
```
git clone https://github.com/AMLattanzi/cuda_perf_tests.git
```

### Running an example

The code tree is given below where each subdirectory inside the **src** directory contains
a particular test whose timings are documented in **README** and a shell file **make.sh** for
compilation.
```
cuda_perf_tests/
├── README.md
└── src
    ├── matrix_add
    └── matrix_mult
```

To compile and run a given example inside the src directory, one may execute the following commands:
```
cd src/<case_name>
source make.sh
./<case_name>.exe
```

