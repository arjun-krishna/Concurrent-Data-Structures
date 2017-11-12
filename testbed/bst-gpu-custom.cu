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
	switch (tid) {
		case 0 : break;
		case 1 : insert(&root, 1); break;
		case 2 : insert(&root, 2); break;
		case 3 : insert(&root, 3); break;
		case 4 : insert(&root, 4); break;
		case 5 : insert(&root, 5); break;
		case 6 : insert(&root, 6); break;
		case 7 : insert(&root, 7); break;
		case 8 : insert(&root, 8); break;
		case 9 : insert(&root, 9); break;
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
	custom_kernel<<<1,10>>>();
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	return 0;
}