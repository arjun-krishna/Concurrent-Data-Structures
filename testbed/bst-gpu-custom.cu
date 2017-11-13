#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../gpu/bst.cu"

#include <curand.h>
#include <curand_kernel.h>

__device__ node* root = NULL;

__global__ void initialize_kernel() {
  
  int tid = threadIdx.x;
  if (tid == 0) {
    root = new_node(0, NULL);
  }
}

__global__ void small_insert_N_kernel() {
  int tid = threadIdx.x;
  if (tid != 0) {
    insert(root, tid);
  }
}

__global__ void large_insert_kernel() {
  
  int tid = blockIdx.x*1000+threadIdx.x;
  if (tid != 0) {
    insert(root,tid);
  }
}


__global__ void insert_random() {
  int tid = threadIdx.x;
  curandState_t state;

  curand_init(tid, 0, 0, &state);

  if (tid != 0) {
    int r = curand(&state)%10000;
    insert(root, r);
  }
}

__global__ void small_delete_N_kernel() {
  int tid = threadIdx.x;
  if (tid%4 == 1) {
    bst_delete(root, tid);
  }
}

__global__ void large_delete_kernel() {
  
  int tid = blockIdx.x*1000+threadIdx.x;
  if (tid%4 == 1) {
    bst_delete(root, tid);
  }
}

__global__ void delete_random() {
  int tid = threadIdx.x;
  curandState_t state;

  curand_init(tid, 0, 0, &state);
  if (tid != 0) {
    int r = curand(&state)%10000;
    insert(root, r);
  }
}

__global__ void print_kernel() {
  // printf("In-order\n");
  in_order(root);
  printf("\n");
}


// Less than 10-threads
__global__ void custom_insert() {
  
  int tid = threadIdx.x;
  if (tid != 0) {
    switch(tid) {
      case 1 :
        insert(root, 1);
        break;
      case 2 :
        insert(root, 2);
        break;
      case 3 :
        insert(root, 3);
        break;
      case 4 :
        insert(root, 4);
        break;
      case 5 :
        insert(root, 5);
        break;
      case 6 :
        insert(root, 6);
        break;
      case 7 :
        insert(root, 7);
        break;
      case 8 :
        insert(root, 8);
        break;
      default :
        insert(root, tid);
        break;
    }
  }
}

__global__ void custom_delete() {
  int tid = threadIdx.x;
  if (tid != 0) {
    switch (tid) {
      case 1 :
        bst_delete(root, 1);
        break;
      case 2 :
        bst_delete(root, 2);
        break;
      case 3 :
        bst_delete(root, 3);
        break;
      case 4 :
        bst_delete(root, 4);
        break;
      case 5 :
        bst_delete(root, 5);
        break;
    }
  }
}

__global__ void find_kernel() {
  int tid = blockIdx.x*blockDim.x+ threadIdx.x;
  find(root, tid);
}

// Less than 10-threads
__global__ void custom_find() {
  
  int tid = threadIdx.x;
	node* f;
  if (tid != 0) {
    switch(tid) {
      case 1 :
        f = find(root, 1);
				if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 2 :
				f = find(root, 2);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 3 :
        f = find(root, 3);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 4 :
        f = find(root, 4);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 5 :
        f = find(root, 5);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 6 :
        f = find(root, 6);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 7 :
        f = find(root, 7);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      case 8 :
        f = find(root, 8);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
      default :
        f = find(root, 9);
        if(f==NULL)
					printf("Faiure\n");
				else
					printf("Success\n");
        break;
    }
  }
}

int main(int argc, char* argv[]) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
  
  initialize_kernel<<<1,1>>>();
  cudaDeviceSynchronize();

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);
  GPUTimer time_insert, time_delete, time_find;
  
  time_insert.Start();
  // custom_insert<<<1,10, 0, s1>>>();
  large_insert_kernel<<<10, 1000, 0, s1>>>();
  time_insert.Stop();
  
  time_delete.Start();
  // custom_delete<<<1,10, 0, s1>>>();
  large_delete_kernel<<<10, 1000, 0, s1>>>();
  time_delete.Stop();

  time_find.Start();
  find_kernel<<<10, 1000, 0, s1>>>();
  time_find.Stop();

  cudaError_t err = cudaGetLastError();  
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  
  cudaDeviceSynchronize();
  printf("Insert Kernel ran in: %f ms\n", time_insert.Elapsed());
  printf("delete Kernel ran in: %f ms\n", time_delete.Elapsed());
  printf("Find   Kernel ran in: %f ms\n", time_find.Elapsed());

  // print_kernel<<<1,1>>>();
  // cudaDeviceSynchronize();
  return 0;
}
