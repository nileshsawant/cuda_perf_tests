#include <chrono>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#define GPU_DEVICE __device__
#define GPU_HOST_DEVICE __host__ __device__
#define GPU_GLOBAL __global__
#define SHARED __shared__
#define SYNCTHREADS __syncthreads()

#define TILEX 32
#define TILEY 32


// Template function to add 1 on HorD
template <typename T>
inline
GPU_HOST_DEVICE
T
addOne (T& val)
{
  return (val + static_cast<T>(1));
}


// parallelForKernel
template <typename F>
GPU_GLOBAL
void
parallelForKernel (F f)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  f(row,col);
}


// Wrapper for parallelForKernel
template <typename F>
void
parallelFor (int nrow,
             int ncol,
             F f)
{
  dim3 block(TILEX,TILEY);
  dim3 grid((ncol+TILEX-1)/TILEX,(nrow+TILEY-1)/TILEY);
  parallelForKernel<<<grid,block>>>(f);
}


// Main routine
int
main ()
{
  // Sizes
  int nrow  = 1<<12;
  int ncol  = 1<<12;
  int ncell = nrow * ncol;
  auto size = ncell * sizeof(double);

  // Allocate pinned host
  double* h_mat;
  cudaMallocHost((void**) &h_mat, size);

  // Initialize pinned host
  for (int idx(0); idx < ncell; ++idx) {
    h_mat[idx] = (double)idx;
  }

  // Allocate device
  double* d_mat;
  cudaMalloc((void**) &d_mat, size);

  // Async copy H2D
  cudaMemcpyAsync(d_mat, h_mat, size, cudaMemcpyHostToDevice);

  // Kernel wrapper for column sum with capture by value
  // for the device pointer and sizes
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();
  parallelFor(nrow, ncol, [=] GPU_DEVICE (int row, int col)
  {
    if (row < nrow) {
      double sum = 0.;
      for (int icol(0); icol<ncol; ++icol) {
        sum += d_mat[ncol*row + icol];
      }
      d_mat[ncol*row + col] = sum;
    }
  });
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Lambda w/out shared compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;


  // Reset the device matrix
  cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);


  // Kernel wrapper for column sum with shared memory
  // capture by value for the device pointer and sizes
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  parallelFor(nrow, ncol, [=] GPU_DEVICE (int row, int col)
  {
    if (row < nrow) {
      int scol   = threadIdx.x;
      int srow   = threadIdx.y;
      double sum = 0.;

      int ntilex = (ncol + TILEX - 1)/TILEX;
      for (int itilex(0); itilex<ntilex; ++itilex) {
        SHARED double data[TILEY][TILEX];

        int gcol = scol + itilex*TILEX;

        if (gcol < ncol) {
          data[srow][scol] = d_mat[ncol*row + gcol];
        } else {
          data[srow][scol] = 0.;
        }

        SYNCTHREADS;

        for (int icol(0); icol<TILEX; ++icol) {
          sum += data[srow][icol];
        }

        SYNCTHREADS;
      } // for tile

      d_mat[ncol*row + col] = sum;
    } // row < nrow
  });
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Lambda w/    shared compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Async copy D2H
  cudaMemcpyAsync(h_mat, d_mat, size, cudaMemcpyDeviceToHost);

  // Sync
  cudaDeviceSynchronize();

  // Clean up
  cudaFreeHost(h_mat);
  cudaFree(d_mat);

  return 0;
}
