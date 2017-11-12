#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../gpu/bst.cu"

__device__ node* root = NULL;

__global__ void custom_kernel() {
	
	int tid = threadIdx.x;
	if (tid == 0) {
		// insert(root, 0);
		root = new_node(0);
	}
	__syncthreads();
	if (tid != 0) {
		insert(root, tid);
	}
	__syncthreads();
	if (tid == 0) {
		printf("In-order\n");
		in_order(root);
		printf("Pre-order\n");
		pre_order(root);
	}
}

int main(int argc, char* argv[]) {
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
	custom_kernel<<<1,1024>>>();
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	return 0;
}