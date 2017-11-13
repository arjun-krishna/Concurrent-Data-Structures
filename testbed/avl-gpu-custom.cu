#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../gpu/avl.cu"

//__device__ node* root;

__global__ void custom_kernel() {
  
  int tid = blockIdx.x*1000 + threadIdx.x;
  if (tid == 0) {
    // insert(root, 0);
    global_Root = new_node(0, NULL);
  }
  __syncthreads();
  if (tid != 0) {
    coarse_insert(tid);
  }
  __syncthreads();
  if (tid == 0) {
    printf("In-order\n");
    in_order(global_Root);
    printf("Pre-order\n");
    pre_order(global_Root);
  }
  __syncthreads();
}

int main(int argc, char* argv[]) {
  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
  //custom_kernel<<<1,10>>>();
  custom_kernel<<<10,1000>>>();
  cudaError_t err = cudaGetLastError();  
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();
  return 0;
}