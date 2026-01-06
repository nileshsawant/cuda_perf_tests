#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <exception>
#include <cuda_runtime.h>

#define TILEY 32
#define TILEX 32


// Matrix multiplication kernel
// Layout: matrix(row,col)
__global__
void
matrixMult (int nrow,
            int ncol,
            double* dA,
            double* dB,
            double* dC)
{
  // NOT coalesced when accessing dC!
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < nrow && col < ncol) {
    double sum = 0.;
    for (int k(0); k<nrow; ++k) {
      sum += dA[row*ncol + k] + dB[k*ncol + col];
    }
    dC[row*ncol + col] = sum;
  }
}


// Coalesced matrix multiplication kernel
// Layout: matrix(row,col)
__global__
void
matrixMultCoalesced (int nrow,
                     int ncol,
                     double* dA,
                     double* dB,
                     double* dC)
{
  // IS coalesced when accessing dC!
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < nrow && col < ncol) {
    double sum = 0.;
    for (int k(0); k<nrow; ++k) {
      sum += dA[row*ncol + k] + dB[k*ncol + col];
    }
    dC[row*ncol + col] = sum;
  }
}


// Add each column with shared memory
__global__
void
matrixMultShared (int nrow,
                  int ncol,
                  double* dA,
                  double* dB,
                  double* dC)
{
  // Global index owned by this thread
  int grow = blockDim.y * blockIdx.y + threadIdx.y;
  int gcol = blockDim.x * blockIdx.x + threadIdx.x;
  
  // Shared memory index inside the block
  int srow = threadIdx.y;
  int scol = threadIdx.x;

  // Sum on each thread
  double sum = 0.;
  
  // Loop over tiles
  int ntiley = (nrow+TILEY-1)/TILEY;
  for (int itiley(0); itiley<ntiley; ++itiley) {

    // Shared memory over the block
    __shared__ double shared_A[TILEY][TILEX];
    __shared__ double shared_B[TILEY][TILEX];

    // Local memory index for A
    int arow = grow;
    int acol = scol + itiley*TILEY;

    // Local memory index for B
    int brow = srow + itiley*TILEY;
    int bcol = gcol;

    // Populate shared memory matrix
    if (grow < nrow && gcol < ncol) {
      shared_A[srow][scol] = dA[ncol*arow + acol];
      shared_B[srow][scol] = dB[ncol*brow + bcol];
    } else {
      shared_A[srow][scol] = 0.;
      shared_B[srow][scol] = 0.;
    }

    // Sync the threads
    __syncthreads();

    // Sum the tile
    for (int k(0); k<TILEY; ++k) {
      sum += shared_A[srow][k] + shared_B[k][scol];
    }
    
    // Sync the threads
    __syncthreads();
  }

  // Set the value of C
  dC[ncol*grow + gcol] = sum;
}


// Confirm solution matches naive
__global__
void
verifySolutions(int* derr,
                int nrow,
                int ncol,
                double* dC,
                double* dCs)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int val = 0;
  if (row < nrow && col < ncol) {
    val = (std::fabs(dC[row*ncol + col] - dCs[row*ncol + col]) > 1.0e-12);
  }
  atomicAdd(derr, val);
}


// Initialize host data
// NOTE: Data is contiuous along col
void
init_host_data (int nrow,
                int ncol,
                double* hA)
{
  for (int i(0); i<nrow; i++) {
    for (int j(0); j<ncol; ++j) {
      int idx = i * ncol + j;
      hA[idx] = (double) idx;
    }
  }
}


int
main ()
{
  // Matrix dimensions
  int nrow = 1<<12;
  int ncol = 1<<12;
  int m_size = nrow * ncol * sizeof(double);
  
  // Allocate host data
  double* hA;
  double* hB;
  double* hC;
  hA = (double*)malloc(m_size);
  hB = (double*)malloc(m_size);
  hC = (double*)malloc(m_size);

  // Initialize host matrices
  init_host_data(nrow, ncol, hA);
  init_host_data(nrow, ncol, hB);

  // Allocate device data
  double* dA;
  double* dB;
  double* dC;
  double* dCs;
  cudaMalloc((void**) &dA , m_size);
  cudaMalloc((void**) &dB , m_size);
  cudaMalloc((void**) &dC , m_size);
  cudaMalloc((void**) &dCs, m_size);

  // Allocate for solution checking on device
  int  herr = 0;
  int* derr;
  cudaMalloc((void**)  &derr, sizeof(int));
  cudaMemcpy(derr, &herr, sizeof(int), cudaMemcpyHostToDevice);

  // Copy host to device
  cudaMemcpy(dA, hA, m_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, m_size, cudaMemcpyHostToDevice);

  // Test naive implementation
  //========================================================
  // Start timer
  auto start = std::chrono::high_resolution_clock::now();
  
  // Launch the matrix mult kernel
  dim3 block(TILEX,TILEY);
  dim3 grid((ncol + block.x - 1)/block.x, (nrow + block.y - 1)/block.y);
  matrixMult<<<grid,block>>>(nrow, ncol, dA, dB, dC);

  // Copy device to host
  cudaMemcpy(hC, dC, m_size, cudaMemcpyDeviceToHost);
  
  // End timer
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Naive matrix multiplication compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  //========================================================

  
  // Test coalesced version
  //========================================================
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixMultCoalesced<<<grid,block>>>(nrow, ncol, dA, dB, dCs);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Coalesced memory matrix multiplication compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;

  // Confirm the methods give the same answer
  verifySolutions<<<grid,block>>>(derr, nrow, ncol, dC, dCs);
  cudaMemcpy(&herr, derr, sizeof(int), cudaMemcpyDeviceToHost);
  if (herr>0) {
    std::cout << "Number of differences detected: " << herr << "\n";
  }

  // Clean up
  cudaMemset(dCs, 0., m_size);
  cudaMemset(derr, 0, sizeof(int));
  //========================================================

  
  // Test shared memory version
  //========================================================
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixMultShared<<<grid,block>>>(nrow, ncol, dA, dB, dCs);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Shared memory matrix multiplication compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;

  // Confirm the methods give the same answer
  verifySolutions<<<grid,block>>>(derr, nrow, ncol, dC, dCs);
  cudaMemcpy(&herr, derr, sizeof(int), cudaMemcpyDeviceToHost);
  if (herr>0) {
    std::cout << "Number of differences detected: " << herr << "\n";
  }
  //========================================================
  
  // Clean up memory
  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dCs);
  cudaFree(derr);
  
  return herr;
}
