#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "../gpu/bst.cu"

__device__ node* root = NULL;

__global__ void initialize_kernel() {
	
	int tid = threadIdx.x;
	if (tid == 0) {
		root = new_node(0, NULL);
	}
}

__global__ void insert_kernel() {
	
	int tid = blockIdx.x*1000+threadIdx.x;
	if (tid != 0) {
		insert(root,tid);
		/*int count = 0;
		while(count<10){
			insert(root, tid+count*1000);
			count++;
		}*/
	}
}

__global__ void delete_kernel() {
	
	int tid = blockIdx.x*1000+threadIdx.x;
	if (tid%4 == 1) {
		bst_delete(root, tid);
	}
}

__global__ void print_kernel() {
	
	printf("In-order\n");
	in_order(root);
}



__global__ void custom_kernel() {
	
	int tid = threadIdx.x;
	if (tid == 0) {
		// insert(root, 0);
		root = new_node(0, NULL);
	}
	__syncthreads();
	if (tid != 0) {
		insert(root, tid);
	}
	__syncthreads();
	if (tid == 0) {
		//printf("Find %d\n",find(root,2)->data);
		//printf("Parent %d\n",find(root,2)->parent->data);
		//bst_delete(root,2);
		printf("In-order\n");
		in_order(root);
		printf("Pre-order\n");
		pre_order(root);
	}
	__syncthreads();
}

int main(int argc, char* argv[]) {
	// cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8000000); 
	//custom_kernel<<<1,1000>>>();
	initialize_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	//cudaStream_t s1, s2;
	//cudaStreamCreate(&s1);
	//cudaStreamCreate(&s2);
	GPUTimer timer;
  timer.Start();
	insert_kernel<<<10,1000>>>();
	delete_kernel<<<10,1000>>>();
	timer.Stop();
	printf("The kernel ran in: %f ms\n", timer.Elapsed());
	cudaError_t err = cudaGetLastError();  
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	cudaDeviceSynchronize();
	print_kernel<<<1,1>>>();
	cudaDeviceSynchronize();
	return 0;
}
