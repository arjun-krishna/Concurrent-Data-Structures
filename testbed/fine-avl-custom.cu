#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../gpu/avl-fine.cu"

__global__ void custom_kernel() {
	
	int tid = threadIdx.x;
	if (tid == 0) {
		// insert(root, 0);
		global_root = new_node(0, NULL);
	}
	__syncthreads();
	if (tid != 0) {
		insert(global_root, tid);
	}
	__syncthreads();
	if (tid == 0) {
		printf("In-order\n");
		in_order(global_root);
		printf("Pre-order\n");
		pre_order(global_root);
	}
	__syncthreads();
}

int main(int argc, char* argv[]) {
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
	//custom_kernel<<<1,1000>>>();
	custom_kernel<<<1,5>>>();
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	return 0;
}