#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../gpu/avl.cu"

//__device__ node* root;

__global__ void initialize_kernel() {
  
  int tid = threadIdx.x;
  if (tid == 0) {
    global_Root = new_node(0, NULL);
  }
}

__global__ void small_insert_N_kernel() {
  int tid = threadIdx.x;
  if (tid != 0) {
    coarse_insert(tid);
  }
}

__global__ void large_insert_kernel() {
  
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if (tid != 0) {
    coarse_insert(tid);
  }
}


__global__ void print_kernel() {
  // printf("In-order\n");
  in_order(global_Root);
  printf("\n");
}


// Less than 10-threads
__global__ void custom_insert() {
  
  int tid = threadIdx.x;
  if (tid != 0) {
    switch(tid) {
      case 1 :
        coarse_insert(1);
        break;
      case 2 :
        coarse_insert(tid);
        break;
      case 3 :
        coarse_insert(tid);
        break;
      case 4 :
        coarse_insert(tid);
        break;
      case 5 :
        coarse_insert(tid);
        break;
      case 6 :
        coarse_insert(tid);
        break;
      case 7 :
        coarse_insert(tid);
        break;
      case 8 :
        coarse_insert(tid);
        break;
      default :
        coarse_insert(tid);
        break;
    }
  }
}


__global__ void find_kernel() {
  int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  find(global_Root, tid);
}

// Less than 10-threads
__global__ void custom_find() {
  
  int tid = threadIdx.x;
  node* f;
  if (tid != 0) {
    switch(tid) {
      case 1 :
        f = find(global_Root, 1);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 2 :
        f = find(global_Root, 2);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 3 :
        f = find(global_Root, 3);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 4 :
        f = find(global_Root, 4);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 5 :
        f = find(global_Root, 5);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 6 :
        f = find(global_Root, 6);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 7 :
        f = find(global_Root, 7);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      case 8 :
        f = find(global_Root, 8);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
      default :
        f = find(global_Root, 9);
        if(f==NULL)
          printf("Faiure\n");
        else
          printf("Success\n");
        break;
    }
  }
}


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
  /*custom_kernel<<<10,1000>>>();
  cudaError_t err = cudaGetLastError();  
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  cudaDeviceSynchronize();*/
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
  
  initialize_kernel<<<1,1>>>();
  cudaDeviceSynchronize();

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  GPUTimer time_insert, time_delete, time_find;
  
  time_insert.Start();
  large_insert_kernel<<<10,100, 0, s1>>>();
  time_insert.Stop();
  
  time_find.Start();
  find_kernel<<<10, 100, 0, s1>>>();
  time_find.Stop();

  cudaError_t err = cudaGetLastError();  
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  
  cudaDeviceSynchronize();
  printf("Insert Kernel ran in: %f ms\n", time_insert.Elapsed());
  printf("Find   Kernel ran in: %f ms\n", time_find.Elapsed());

  // print_kernel<<<1,1>>>();
  // cudaDeviceSynchronize();
  return 0;
}