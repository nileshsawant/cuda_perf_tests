#include <chrono>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>

#define TILEX 32
#define TILEY 32


// Initialize device data to index
__global__
void
initKernelIdx (int ncol,
               double* data)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col<ncol) {
    data[col] = static_cast<double>(col);
  }
}


// Print values to confirm
__global__
void
printKernelB (int ncol,
              int myRank,
              double* data)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col<5) {
    printf("Device w/ buffer rank %i at %i the value is %f\n",myRank,col,data[col]);
    data[col] *= 1.0;
  }
}


// Print values to confirm
__global__
void
printKernelNB (int ncol,
               int myRank,
               double* data)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col<5) {
    printf("Device w/out buffer rank %i at %i the value is %f\n",myRank,col,data[col]);
    data[col] *= 1.0;
  }
}


// Main driver
int
main (int argc, char *argv[])
{
  int myRank;
  int srcRank{0};
  int destRank{1};
  int tag=99;
  MPI_Status status;

  // Size of buffer
  int n = 1<<12;
  auto size = n * sizeof(double);

  // Host data
  double* h_data = static_cast<double*>(malloc(size));
  memset(h_data, 0, size);

  // Device data
  double* d_data;
  cudaMalloc((void**) &d_data, size);
  cudaMemset(d_data, 0, size);

  // Default stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Init MPI
  MPI_Init(&argc, &argv);
    
  // Rank ID
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  
  // Host 2 Host transfer
  //==============================================================
  // Rank 0 populates
  if (myRank == srcRank) {
    // Host
    for (int idx(0); idx<n; ++idx) {
      h_data[idx] = static_cast<double>(idx);
    }
  }

  // Barrier
  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::high_resolution_clock::now();
  
  // Rank 0 sends
  if (myRank == srcRank) {
    // Send the data
    MPI_Send(h_data, n, MPI_DOUBLE, destRank, tag, MPI_COMM_WORLD);
  }

  // Rank 1 receives and prints
  if (myRank == destRank) {
    // Receive the message
    MPI_Recv(h_data, n, MPI_DOUBLE, srcRank, tag, MPI_COMM_WORLD, &status);
  }

  // Verify data transferred
  if (myRank == destRank) {
    // Host
    for (int idx(0); idx<5; ++idx) {
      std::cout << "Host rank " << myRank << " at " << idx
                << " the value is " << h_data[idx] << "\n";
    }
  }

  // Barrier
  MPI_Barrier(MPI_COMM_WORLD);
  auto end = std::chrono::high_resolution_clock::now();
  if (myRank == destRank) {
      std::cout << "CPU-CPU MPI transfer time (us): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << std::endl;
  }
  //==============================================================

  
  // Device 2 Device transfer w/ Host buffer
  //==============================================================
  memset(h_data, 0, size);
  // Rank 0 populates
  if (myRank == srcRank) {
    // Device
    dim3 block(TILEX);
    dim3 grid((n+TILEX-1)/TILEX);
    initKernelIdx<<<grid,block,0,stream>>>(n, d_data);
  }

  // Barrier
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  
  // Rank 0 sends
  if (myRank == srcRank) {
    // Send the data
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    MPI_Send(h_data, n, MPI_DOUBLE, destRank, tag, MPI_COMM_WORLD);
  }

  // Rank 1 receives and prints
  if (myRank == destRank) {
    // Receive the message
    MPI_Recv(h_data, n, MPI_DOUBLE, srcRank, tag, MPI_COMM_WORLD, &status);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
  }

  // Verify data transferred
  if (myRank == destRank) {
    // Device
    dim3 block(TILEX);
    dim3 grid((n+TILEX-1)/TILEX);
    printKernelB<<<grid,block,0,stream>>>(n, myRank, d_data);
    cudaStreamSynchronize(stream);
  }

  // Barrier
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  if (myRank == destRank) {
      std::cout << "GPU/CPU-CPU/GPU MPI transfer time (us): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << std::endl;
  }
  //==============================================================


  // Device 2 Device transfer w/out Host buffer
  //==============================================================
  cudaMemset(d_data, 0, size);
  // Rank 0 populates
  if (myRank == srcRank) {
    // Device
    dim3 block(TILEX);
    dim3 grid((n+TILEX-1)/TILEX);
    initKernelIdx<<<grid,block,0,stream>>>(n, d_data);
    cudaStreamSynchronize(stream);
  }

  // Barrier
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();

  // Rank 0 sends
  if (myRank == srcRank) {
    // Send the data
    MPI_Send(d_data, n, MPI_DOUBLE, destRank, tag, MPI_COMM_WORLD);
  }

  // Rank 1 receives and prints
  if (myRank == destRank) {
    // Receive the message
    MPI_Recv(d_data, n, MPI_DOUBLE, srcRank, tag, MPI_COMM_WORLD, &status);
    cudaDeviceSynchronize();
  }
  
  // Verify data transferred
  if (myRank == destRank) {
    // Device
    dim3 block(TILEX);
    dim3 grid((n+TILEX-1)/TILEX);
    printKernelNB<<<grid,block,0,stream>>>(n, myRank, d_data);
    cudaStreamSynchronize(stream);
  }

  // Barrier
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  if (myRank == destRank) {
      std::cout << "GPU-GPU MPI transfer time (us): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << std::endl;
  }
  //==============================================================

  
  // Clean up
  free(h_data);
  cudaFree(d_data);
  
  // Finalize
  MPI_Finalize();

  return 0;
}
