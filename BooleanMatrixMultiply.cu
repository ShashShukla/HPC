/*
CUDA code for mutiplying bit-matrix A with bit-vector x
over the GF(2) field.
Author: Shashwat Shukla
15th Dec 2016

Tiled multiplication is performed.
32x32 tiles have been used.

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


#define n 10000 //The size of the matrix
#define N  (n + 31) / 32 //Number of unsigned integers (32 bits each) needed to store the bits.

//Define matrices and vectors
unsigned int A[n][N];
unsigned int x[N];
unsigned int *Ad, *xd;


//Calculates the xor of the 32 bits of an integer
__device__ unsigned int xorBits(unsigned int x) {
	unsigned int temp = 0;
	for (int i = 0; i < 32; i++) {
		temp = temp + (x >> i);
	}
	temp = temp % 2;
	return temp;
}

//The kernel
__global__ void kernel(unsigned int *Ad, unsigned int *xd) {
	__shared__ unsigned int B[32][32]; //Stores a sub-matrix of A
	__shared__ unsigned int v[32]; //Stores a part of the vector x
	__shared__ unsigned int c[32]; //Stores the result for this block

	//Thread id
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int offset = N * 32 * by + N * ty + tx;

	//Initialise the vector c to zero
	if (ty == 0)
		c[tx] = 0;

	for (int i = 0; i < (N + 31) / 32; i++) { //Traverse the entire width of the array A (along the column)

		if ((i * 32 + tx) < N && (by * 32 + ty) < n) //Padding
			B[ty][tx] = Ad[offset + i * 32];
		else
			B[ty][tx] = 0;

		if (ty == 0)
			v[tx] = xd[i * 32 + tx];

		__syncthreads(); //Wait for all threads to finish copying data to shared memory

		c[ty] = c[ty] ^ (B[ty][tx] & v[tx]);

		__syncthreads(); //Wait for all threads to finish computation and store the result in shared memory

		if (ty == 0) //Compress the resultant vector
			c[tx] = xorBits(c[tx]);

		__syncthreads();

	}

	if (tx == 0 && ty == 0) { //Find the resultant vector from each block
		unsigned int temp = 0;
		temp = c[31];
		for (int i = 30; i >= 0; i--) {
			temp = temp << 1;
			temp = temp + c[i];
		}
		xd[by] = temp; //Copy result from each block to the final result
	}
}

//Launch the kernel and check for errors
cudaError_t launchKernel() {

	cudaError_t cudaStatus;

	//Define kernel grid dimensions
	size_t size = N * sizeof(unsigned int);
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(1, N);


	// Choose the GPU to run code on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Check if there is a CUDA enabled GPU?");
		getchar();
		goto Error;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&Ad, n * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		getchar();
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&xd, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		getchar();
		goto Error;
	}

	// Copy input vectors from host memory to GPU.
	cudaStatus = cudaMemcpy(Ad, A, size * n, cudaMemcpyHostToDevice);
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

	//Launch kernel
	kernel << < numBlocks, threadsPerBlock >> > (Ad, xd);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

	return cudaStatus;
}

int main() {

	//The following is some initialisation for A and x. Change appropriately.
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = 0;
		}
	}
	A[0][0] = 1;
	for (int i = 0; i < N; i++)
		x[i] = 0;
	x[0] = 1;

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