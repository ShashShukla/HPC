/*
CUDA code for mutiplying bit-matrix A with bit-vector x
over the GF(2) field.
Author: Shashwat Shukla
22nd Dec 2016

Multiplication is performed using the Four Russians Method (M4R).

A*x is computed as x^T*A^T (^T means transpose) for much faster and efficient computation.

C++ does not have a data-type of size one bit.
std::bitset is a pseudo-container that stores bits inside integer variables. However, they
can't be passed by reference and hence can't be used for GPU compute.
std::vector<bool> also suffers from this issue. It also does store bits in contiguous spaces and thus
would drastically slow down GPU memory access as coalescing will not be possible.

A custom implementation of a bit-set is hence used, which also uses unsigned integers to store and
access bits.
Note that we could store one bit in one int variable. But this would be wasteful as each integer
is made of 32 bits.
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
using namespace std;


#define n 10000 //The size of the matrix.
#define N  ((n + 31) / 32) //Number of unsigned integers (32 bits each) needed to store the bits.
#define Nb ((n + 7) / 8) //Number of blocks to deploy.
#define threads 1024 //Number of threads per block for preprocessing.

//Define matrices and vectors
unsigned int A[Nb * 8][N]; //Not A[n][N] because we need to pad the matrix and ensure that each block has 8 rows
unsigned int x[N];
unsigned int *Ad, *xd, *cd, *cache;


//This kernel creates the cache
__global__ void kernelCache(unsigned int *Ad, unsigned int *xd, unsigned int *cd, unsigned int *cache) {

	//Thread id
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int offset = N * 8 * by + tx;
	int iter = (N + threads - 1) / threads;

	//Initialise cache to zero
	for (int k = 0; k < iter; k++) { //Traverse along the length of the matrix A
		if ((k * threads + tx) < N) {//Padding
			for (int i = 0; i < 256; i++) {
				cache[by * 256 * N + i * N + k * threads + tx] = 0;
			}
		}
	}

	__syncthreads();

	//Create the cache
	for (int k = 0; k < iter; k++) { //Traverse along the length of the matrix A
		if ((k * threads + tx) < N) {//Padding
#pragma unroll
			for (int i = 0; i < 256; i++) { //2^8 = 256
				for (int j = 0; j < 8; j++) { //Bit by bit Xor
					if (((i >> j) & 1) == 1) {
						cache[by * 256 * N + i * N + k * threads + tx] = cache[by * 256 * N + i * N + k * threads + tx] ^ Ad[offset + k * threads + j * N];
					}
				}
			}
		}
	}

	__syncthreads();
}

//This kernel performs the multiplication
__global__ void kernelMultiply(unsigned int *Ad, unsigned int *xd, unsigned int *cd, unsigned int *cache) {

	//Thread id
	int tx = threadIdx.x;

	int iter = ((N + 1024 - 1) / 1024);

	//Initialise temp variable cd to zero
	for (int k = 0; k < iter; k++) { //Traverse along the length of the matrix A
		if ((k * 1024 + tx) < N) {//Padding
			cd[k * 1024 + tx] = 0;
		}
	}

	__syncthreads();

#pragma unroll
	//Perform the multiplication
	for (int i = 0; i < Nb; i++) {
		int p = (i / 4); int q = (i % 4);
		int z = (xd[p] >> (8 * q)) & 255;

		for (int k = 0; k < iter; k++) { //Traverse along the length of cd
			if ((k * 1024 + tx) < N) {//Padding
				cd[k * 1024 + tx] = cd[k * 1024 + tx] ^ cache[i * 256 * N + z * N + k * 1024 + tx];
			}
		}
	}

	__syncthreads();

	//Store the result back into xd
	for (int k = 0; k < iter; k++) { //Traverse along the length of the matrix A
		if ((k * 1024 + tx) < N) {//Padding
			xd[k * 1024 + tx] = cd[k * 1024 + tx];
		}
	}

	//__syncthreads();
}

//Launch the kernel and check for errors
cudaError_t launchKernel() {

	cudaError_t cudaStatus;

	//Define kernel grid dimensions
	size_t size = N * sizeof(unsigned int);

	dim3 threadsPerBlock(threads);
	dim3 numBlocks(1, Nb);

	// Choose the GPU to run code on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Check if there is a CUDA enabled GPU?");
		getchar();
		goto Error;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&Ad, Nb * 8 * size); //The matrix A
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for Ad!");
		getchar();
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&xd, size); //Vector x
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for xd!");
		getchar();
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&cd, size); //Temporary storage variable
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for cd!");
		getchar();
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&cache, Nb * 256 * size); //Cache to store the preprocessed data
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for cache!");
		getchar();
		goto Error;
	}

	// Copy input vectors from host memory to GPU.
	cudaStatus = cudaMemcpy(Ad, A, Nb * 8 * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (host to GPU) for A failed!");
		getchar();
		goto Error;
	}

	cudaStatus = cudaMemcpy(xd, x, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (host to GPU) for x failed!");
		getchar();
		goto Error;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start, 0);
	//Launch kernel to create cache
	kernelCache << < numBlocks, 1024 >> > (Ad, xd, cd, cache);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelCache launch failed: %s\n", cudaGetErrorString(cudaStatus));
		getchar();
		goto Error;
	}

	//Launch kernel to perform the multiplication
	kernelMultiply << < 1, threadsPerBlock >> > (Ad, xd, cd, cache);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Kernel Execution time(ms) = " << milliseconds << endl;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelCache launch failed: %s\n", cudaGetErrorString(cudaStatus));
		getchar();
		goto Error;
	}


	// Copy output vectors from GPU to host memory.
	cudaStatus = cudaMemcpy(x, xd, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (GPU to host) failed!");
		getchar();
		goto Error;
	}

Error:
	cudaFree(Ad);
	cudaFree(xd);
	cudaFree(cd);
	cudaFree(cache);

	return cudaStatus;
}

int main() {

	//The following is some initialisation for A and x. Change appropriately.
	for (int i = 0; i < Nb * 8; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = 0;
		}
	}
	A[0][0] = 1;
	for (int i = 0; i < N; i++)
		x[i] = 0;
	x[0] = 1;

	//Now transpose A and also pad this new matrix. Use this matrix as the matrix A henceforth

	//Launch the kernel
	cudaError_t cudaStatus = launchKernel();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "The kernel could not be launched!");
		getchar();
		return 1;
	}

	//Show the resultant vector(in decimal format, grouped as 32 bit words)
	for (int i = 0; i < N; i++)
		cout << x[i] << endl;

	cout << "Press enter to exit. ";
	getchar();

	return 0;
}
