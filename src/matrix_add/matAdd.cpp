#include <chrono>
#include <cstdio>
#include <iostream>
#include <exception>
#include <cuda_runtime.h>

#define TILEY 32
#define TILEX 32

// Matrix addition kernel
// Layout: matrix(row,col)
__global__
void
matrixAdd (int nrow,
           int ncol,
           double* dA,
           double* dB,
           double* dC)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < nrow && col < ncol) {
    dC[row*ncol + col] = dA[row*ncol + col] + dB[row*ncol + col];
  }
}


// Add each column
__global__
void
matrixAddCol (int nrow,
              int ncol,
              double* dA,
              double* dB,
              double* dC)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < nrow && col < ncol) {
    dC[row*ncol + col] = 0.;
    for (int i(0); i<ncol; ++i) {
      // Each thread individually moves up and down a row
      // Threads will access contiguous data (coalescing)
      dC[row*ncol + col] += dA[i*ncol + col] + dB[i*ncol + col];
    }
  }
}


// Add each row
__global__
void
matrixAddRow (int nrow,
              int ncol,
              double* dA,
              double* dB,
              double* dC)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < nrow && col < ncol) {
    dC[row*ncol + col] = 0.;
    for (int i(0); i<nrow; ++i) {
      // Each thread individually moves along a row
      // Threads will NOT access contiguous data (NO coalescing)
      dC[row*ncol + col] += dA[row*ncol + i] + dB[row*ncol + i];
    }
  }
}


// Add each column with shared memory
__global__
void
matrixAddColShared (int nrow,
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

    // Local memory index
    int lrow = srow + itiley*TILEY;
    int lcol = gcol;

    // Populate shared memory matrix
    if (grow < nrow && gcol < ncol) {
      shared_A[srow][scol] = dA[ncol*lrow + lcol];
      shared_B[srow][scol] = dB[ncol*lrow + lcol];
    } else {
      shared_A[srow][scol] = 0.;
      shared_B[srow][scol] = 0.;
    }

    // Sync the threads
    __syncthreads();

    // Sum the tile
    for (int irow(0); irow<TILEY; ++irow) {
      sum += shared_A[irow][scol] + shared_B[irow][scol];
    }
    
    // Sync the threads
    __syncthreads();
  }

  // Set the value of C
  dC[ncol*grow + gcol] = sum;
}


// Add each column with shared memory
__global__
void
matrixAddRowShared (int nrow,
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
  int ntilex = (ncol+TILEX-1)/TILEX;
  for (int itilex(0); itilex<ntilex; ++itilex) {

    // Shared memory over the block
    __shared__ double shared_A[TILEY][TILEX];
    __shared__ double shared_B[TILEY][TILEX];

    // Local memory index
    int lrow = grow;
    int lcol = scol + itilex*TILEX;

    // Populate shared memory matrix
    if (grow < nrow && gcol < ncol) {
      shared_A[srow][scol] = dA[ncol*lrow + lcol];
      shared_B[srow][scol] = dB[ncol*lrow + lcol];
    } else {
      shared_A[srow][scol] = 0.;
      shared_B[srow][scol] = 0.;
    }

    // Sync the threads
    __syncthreads();

    // Sum the tile
    for (int icol(0); icol<TILEX; ++icol) {
      sum += shared_A[srow][icol] + shared_B[srow][icol];
    }
    
    // Sync the threads
    __syncthreads();
  }

  // Set the value of C
  dC[ncol*grow + gcol] = sum;
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


// Test the solution
int
test_host_sol (int nrow,
               int ncol,
               double* hC)
{
  double eps = std::numeric_limits<double>::epsilon();
  for (int i(0); i<nrow; i++) {
    for (int j(0); j<ncol; ++j) {
      int idx = i * ncol + j;
      if (std::fabs(hC[idx] - (double) (2*idx)) > eps) {
        std::cout << "Exceed precision diff at: (" << i << " , " << j << ")\n";
        std::cout << "Value is: " << hC[idx] << "\n";
        return 0;
      }
    }
  }
  return 1;
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

  // Allocate device data;
  double* dA;
  double* dB;
  double* dC;
  cudaMalloc((void**) &dA, m_size);
  cudaMalloc((void**) &dB, m_size);
  cudaMalloc((void**) &dC, m_size);

  // Copy host to device
  cudaMemcpy(dA, hA, m_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, m_size, cudaMemcpyHostToDevice);

  // Start timer
  auto start = std::chrono::high_resolution_clock::now();
  
  // Launch the kernel
  dim3 block(TILEX,TILEY);
  dim3 grid((ncol + block.x - 1)/block.x, (nrow + block.y - 1)/block.y);
  matrixAdd<<<grid,block>>>(nrow, ncol, dA, dB, dC);

  // Copy device to host
  cudaMemcpy(hC, dC, m_size, cudaMemcpyDeviceToHost);

  // End timer
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Element matrix add compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Confirm solution
  int success = test_host_sol(nrow, ncol, hC);

  // Test row sum
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixAddCol<<<grid,block>>>(nrow, ncol, dA, dB, dC);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Sum matrix col compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;

  // Test column sum
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixAddRow<<<grid,block>>>(nrow, ncol, dA, dB, dC);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Sum matrix row compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;

  // Test col sum shared
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixAddColShared<<<grid,block>>>(nrow, ncol, dA, dB, dC);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Sum matrix col shared compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Test row sum shared
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  matrixAddRowShared<<<grid,block>>>(nrow, ncol, dA, dB, dC);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "Sum matrix row shared compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Clean up memory
  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  
  return success;
}
