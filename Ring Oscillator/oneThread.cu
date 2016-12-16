#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
using namespace std;

/////////////////////////////////////////////////////////////////////////////////////
//MOSFET Transconductances
__device__ float pMOS_didbydvsg(float Vs, float Vg, float Vd, float V_T, float K, float lambda) {
	float Vsg = Vs - Vg, Vsd = Vs - Vd;
	float gm_p;
	if (Vsg < -1 * V_T) {       //cutoff
		gm_p = 0;
	}
	else if (Vsd < Vsg + V_T) {     //linear
		gm_p = K * Vsd;
	}
	else {              //saturation
		gm_p = K * (Vsg + V_T) * (1 + lambda * (Vsd));
	}
	return gm_p;
}

__device__ float pMOS_didbydvsd(float Vs, float Vg, float Vd, float V_T, float K, float lambda) {
	float Vsg = Vs - Vg, Vsd = Vs - Vd;
	float go_p;
	if (Vsg < -1 * V_T) {       //cutoff
		go_p = 0;
	}
	else if (Vsd < Vsg + V_T) {     //linear
		go_p = K * ((Vsg + V_T) - Vsd);
	}
	else {              //saturation
		go_p = K / 2 * lambda * (Vsg + V_T) * (Vsg + V_T);
	}
	return go_p;
}

__device__ float nMOS_didbydvgs(float Vs, float Vg, float Vd, float V_T, float K, float lambda) {
	float Vgs = Vg - Vs, Vds = Vd - Vs;
	float gm_n;
	if (Vgs < V_T) {        //cutoff
		gm_n = 0;
	}
	else if (Vds < Vgs - V_T) {     //linear
		gm_n = K * Vds;
	}
	else {              //saturation
		gm_n = K * (Vgs - V_T) * (1 + lambda * (Vds));
	}
	return gm_n;
}

__device__ float nMOS_didbydvds(float Vs, float Vg, float Vd, float V_T, float K, float lambda) {
	float Vgs = Vg - Vs, Vds = Vd - Vs;
	float go_n;
	if (Vgs < V_T) {        //cutoff
		go_n = 0;
	}
	else if (Vds < Vgs + V_T) {     //linear
		go_n = K * (Vgs - V_T) * (1 + lambda * (Vds));
	}
	else {              //saturation
		go_n = K / 2 * lambda * (Vgs - V_T) * (Vgs - V_T);
	}
	return go_n;
}
/////////////////////////////////////////////////////////////////////////////////////////

//LHS of MNA matrix equation
__device__ void cmos_ring_osc_lhs(float lhs[5][5], float V1, float V2, float V3, float V4, float C1, float C2, float C3, float delta_t, float V_T_p, float V_T_n, float K_p, float K_n, float lambda) {
	//Initialize trasnconductances
	float gm_p1 = 0, gm_p2 = 0, gm_p3 = 0;
	float go_p1 = 0, go_p2 = 0, go_p3 = 0;
	float gm_n1 = 0, gm_n2 = 0, gm_n3 = 0;
	float go_n1 = 0, go_n2 = 0, go_n3 = 0;
	//Update transconductances
	gm_p1 = pMOS_didbydvsg(V1, V4, V2, V_T_p, K_p, lambda);
	go_p1 = pMOS_didbydvsd(V1, V4, V2, V_T_p, K_p, lambda);
	gm_n1 = nMOS_didbydvgs(0, V4, V2, V_T_n, K_n, lambda);
	go_n1 = nMOS_didbydvds(0, V4, V2, V_T_n, K_n, lambda);

	gm_p2 = pMOS_didbydvsg(V1, V2, V3, V_T_p, K_p, lambda);
	go_p2 = pMOS_didbydvsd(V1, V2, V3, V_T_p, K_p, lambda);
	gm_n2 = nMOS_didbydvgs(0, V2, V3, V_T_n, K_n, lambda);
	go_n2 = nMOS_didbydvds(0, V2, V3, V_T_n, K_n, lambda);

	gm_p3 = pMOS_didbydvsg(V1, V3, V4, V_T_p, K_p, lambda);
	go_p3 = pMOS_didbydvsd(V1, V3, V4, V_T_p, K_p, lambda);
	gm_n3 = nMOS_didbydvgs(0, V3, V4, V_T_n, K_n, lambda);
	go_n3 = nMOS_didbydvds(0, V3, V4, V_T_n, K_n, lambda);
	//Update LHS matrix
	lhs[0][0] = gm_p1 + go_p1 + gm_p2 + go_p2 + gm_p3 + go_p3;
	lhs[0][1] = -gm_p2 - go_p1;
	lhs[0][2] = -gm_p3 - go_p2;
	lhs[0][3] = -gm_p1 - go_p3;
	lhs[0][4] = 1;

	lhs[1][0] = gm_p1 + go_p1;
	lhs[1][1] = -go_p1 - go_n1 - C1 / delta_t;
	lhs[1][3] = -gm_p1 - gm_n1;

	lhs[2][0] = gm_p2 + go_p2;
	lhs[2][1] = -gm_p2 - gm_n2;
	lhs[2][2] = -go_p2 - go_n2 - C2 / delta_t;

	lhs[3][0] = gm_p3 + go_p3;
	lhs[3][2] = -gm_p3 - gm_n3;
	lhs[3][3] = -go_p3 - go_n3 - C3 / delta_t;
	lhs[4][0] = 1;
}

//RHS of MNA matrix equation
__device__ void cmos_ring_osc_rhs(float* rhs, float V, float V1, float V2_prev, float V2, float V3_prev, float V3, float V4_prev, float V4, float ip1_hat, float ip2_hat, float ip3_hat, float in1_hat, float in2_hat, float in3_hat, float C1, float C2, float C3, float delta_t, float V_T_p, float V_T_n, float K_p, float K_n, float lambda) {
	//Initialize transconductances
	float gm_p1 = 0, gm_p2 = 0, gm_p3 = 0;
	float go_p1 = 0, go_p2 = 0, go_p3 = 0;
	float gm_n1 = 0, gm_n2 = 0, gm_n3 = 0;
	float go_n1 = 0, go_n2 = 0, go_n3 = 0;
	//Computing transconductances
	gm_p1 = pMOS_didbydvsg(V1, V4, V2, V_T_p, K_p, lambda);
	go_p1 = pMOS_didbydvsd(V1, V4, V2, V_T_p, K_p, lambda);
	gm_n1 = nMOS_didbydvgs(0, V4, V2, V_T_n, K_n, lambda);
	go_n1 = nMOS_didbydvds(0, V4, V2, V_T_n, K_n, lambda);

	gm_p2 = pMOS_didbydvsg(V1, V2, V3, V_T_p, K_p, lambda);
	go_p2 = pMOS_didbydvsd(V1, V2, V3, V_T_p, K_p, lambda);
	gm_n2 = nMOS_didbydvgs(0, V2, V3, V_T_n, K_n, lambda);
	go_n2 = nMOS_didbydvds(0, V2, V3, V_T_n, K_n, lambda);

	gm_p3 = pMOS_didbydvsg(V1, V3, V4, V_T_p, K_p, lambda);
	go_p3 = pMOS_didbydvsd(V1, V3, V4, V_T_p, K_p, lambda);
	gm_n3 = nMOS_didbydvgs(0, V3, V4, V_T_n, K_n, lambda);
	go_n3 = nMOS_didbydvds(0, V3, V4, V_T_n, K_n, lambda);
	//Computing RHS vector
	rhs[0] = -ip1_hat - ip2_hat - ip3_hat + (gm_p1 + go_p1 + gm_p2 + go_p2 + gm_p3 + go_p3) * V1 - (gm_p2 + go_p1) * V2 - (gm_p3 + go_p2) * V3 - (gm_p3 + go_p3) * V4;
	rhs[1] = -ip1_hat + in1_hat + (gm_p1 + go_p1) * V1 - (gm_p1 + gm_n1) * V4 - (go_p1 + go_n1) * V2 - C2 / delta_t*V2_prev;
	rhs[2] = -ip2_hat + in2_hat + (gm_p2 + go_p2) * V1 - (gm_p2 + gm_n2) * V2 - (go_p2 + go_n2) * V3 - C2 / delta_t*V3_prev;
	rhs[3] = -ip3_hat + in3_hat + (gm_p3 + go_p3) * V1 - (gm_p3 + gm_n3) * V3 - (go_p3 + go_n3) * V4 - C3 / delta_t*V4_prev;
	rhs[4] = V;
}

//Function to compute norm of a vector
__device__ float error(float* x_new, float* x)
{
	float sum = 0;
	float sum_of_errors = 0;
	for (int i = 0; i < 5; i++)
	{
		sum_of_errors = sum_of_errors + (x_new[i] - x[i]) * (x_new[i] - x[i]);
		sum = sum + x_new[i] * x_new[i];
	}

	return sqrt(sum_of_errors) / sqrt(sum);
}

////////////////////////////////////////////////////////////////////
//Solve Ax=b using Gaussian elimination
__device__ void l_soln(float lhs[5][5], float x[5], float rhs[5]) {
	int i, j, k;
	float a[5][6];        //a is the augmented-matrix
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			a[i][j] = lhs[i][j];
		}
		a[i][5] = rhs[i];
	}
	/*for (i = 0; i<5; i++){
	for(j=0;j<6; j++){
	cout<<a[i][j]<<setw(16);
	}
	cout<<"\n";
	}
	cout<<endl<<endl; */
	float m;
	int f;
	for (i = 0; i < 5 - 1; i++) {
		m = a[i][i];
		f = 0;
		for (j = i + 1; j < 5; j++) { //Find Pivot element
			if (fabs(a[j][i]) > fabs(m)) {
				m = a[j][i];
				f = j;
			}
		}
		if (f != 0) {
			for (k = 0; k <= 5; k++) { //Partial pivoting
				float temp = a[i][k];
				a[i][k] = a[f][k];
				a[f][k] = temp;
			}
		}
		/*for (int r = 0; r<5; r++){
		for(int s=0;s<6; s++){
		cout<<a[r][s]<<setw(16);
		}
		cout<<"\n";
		}
		cout<<endl<<endl; */

		for (k = i + 1; k < 5; k++) //Gaussian elimination
		{
			float t = a[k][i] / a[i][i];
			a[k][i] = 0;
			for (j = i + 1; j <= 5; j++)
				a[k][j] = a[k][j] - t * a[i][j];
		}

	}

	/*cout<<"\n\nThe matrix after gauss-elimination is as follows:\n";
	for (i=0; i<5; i++)            //print the new matrix
	{
	for (j=0; j<=5; j++)
	cout<<a[i][j]<<setw(16);
	cout<<"\n";
	} */
	for (i = 5 - 1; i >= 0; i--) //Back-substitution
	{
		x[i] = a[i][5];
		for (j = i + 1; j < 5; j++)
			x[i] = x[i] - a[i][j] * x[j];
		x[i] = x[i] / a[i][i];
	}
	/*cout<<"\nThe values of the variables are as follows:\n";
	for (i=0;i<5;i++)
	cout<<x[i]<<endl;
	int p;
	cout << "Press any key to continue: ";
	cin >> p; */
}



__global__ void kernel(float* V2d, float* V3d, float* V4d) {

	float V = 5;      //Voltage source
	float C = 0.9e-15;
	float C1 = C, C2 = C, C3 = C; //Individual Capacitors

								  //Simulation parameters
								  //float Time = 5e-10;    //Simulation time
	float delta_t = 1e-12; //Simulation time step
	enum { final_T = 500 };     //Total simulation time steps

	float is_hat, ip1_hat, ip2_hat, ip3_hat, in1_hat, in2_hat, in3_hat;
	float V1 = 5;
	float V2[final_T], V3[final_T], V4[final_T];

	float lhs[5][5] = { 0 };
	float rhs[5] = { 0 };

	//MOSFET Parameters
	float V_T_p = -0.8;    //pMOS threshold voltage
	float V_T_n = 0.8; //nMOS threshold voltage
	float K_p = 0.02e-3; //pMOS constant
	float K_n = 0.01e-3; //nMOS constant
	float lambda = 0;
	//int n = 5;

	//Initial conditions
	V2[0] = 0; V3[0] = 5; V4[0] = 0;
	is_hat = ip1_hat = ip2_hat = ip3_hat = in1_hat = in2_hat = in3_hat = 0;

	int max_iter = 10;  //Maximum number of iterations for solving matrix equation

	float V2_prev = V2[0];
	float V3_prev = V3[0];
	float V4_prev = V4[0];
	float vector[5] = { 0 };
	float vector1[5] = { 0 };

	for (int i = 0; i < final_T; i++) {
		int j = 1;
		float err = 1;
		float V2_hat = V2[i], V3_hat = V3[i], V4_hat = V4[i];

		while ((j < max_iter) && (err > 1e-2)) {
			vector1[0] = V1;
			vector1[1] = V2_hat;
			vector1[2] = V3_hat;
			vector1[3] = V4_hat;
			vector1[4] = is_hat;

			//LHS and RHS
			cmos_ring_osc_lhs(lhs, V1, V2_hat, V3_hat, V4_hat, C1, C2, C3, delta_t, V_T_p, V_T_n, K_p, K_n, lambda);
			cmos_ring_osc_rhs(rhs, V, V1, V2_prev, V2_hat, V3_prev, V3_hat, V4_prev, V4_hat, ip1_hat, ip2_hat, ip3_hat, in1_hat, in2_hat, in3_hat, C1, C2, C3, delta_t, V_T_p, V_T_n, K_p, K_n, lambda);
			//Solve MNA
			l_soln(lhs, vector, rhs);
			//cout<<vector[0]<<'\t'<<vector[1]<<'\t'<<vector[2]<<'\t'<<vector[3]<<'\t'<<vector[4]<<'\n';

			V1 = vector[0]; V2[i + 1] = vector[1]; V3[i + 1] = vector[2]; V4[i + 1] = vector[3]; is_hat = vector[4];
			err = error(vector, vector1); //Compute error
										  //cout<<err<<endl;
			j = j + 1;
			V2_hat = V2[i + 1]; V3_hat = V3[i + 1]; V4_hat = V4[i + 1]; //Update node voltages for next cycle


																		//Update current values for next cycle
			float V_sg_p1 = V1 - V4[i + 1], V_sd_p1 = V1 - V2[i + 1];
			if (V_sg_p1 <= -1 * V_T_p) {
				ip1_hat = 0;
			}
			else if (V_sd_p1 < (V_sg_p1 + V_T_p)) {
				ip1_hat = (1 + lambda * V_sd_p1) * K_p * ((V_sg_p1 + V_T_p) * V_sd_p1 - V_sd_p1 * V_sd_p1 / 2);
			}
			else {
				ip1_hat = (1 + lambda * V_sd_p1) * K_p * (V_sg_p1 + V_T_p) * (V_sg_p1 + V_T_p) / 2;
			}

			float V_sg_p2 = V1 - V2[i + 1], V_sd_p2 = V1 - V3[i + 1];
			if (V_sg_p2 <= -1 * V_T_p) {
				ip2_hat = 0;
			}
			else if (V_sd_p2 < (V_sg_p2 + V_T_p)) {
				ip2_hat = (1 + lambda * V_sd_p2) * K_p * ((V_sg_p2 + V_T_p) * V_sd_p2 - V_sd_p2 * V_sd_p2 / 2);
			}
			else {
				ip2_hat = (1 + lambda * V_sd_p2) * K_p * (V_sg_p2 + V_T_p) * (V_sg_p2 + V_T_p) / 2;
			}

			float V_sg_p3 = V1 - V3[i + 1], V_sd_p3 = V1 - V4[i + 1];
			if (V_sg_p3 <= -1 * V_T_p) {
				ip3_hat = 0;
			}
			else if (V_sd_p3 < (V_sg_p3 + V_T_p)) {
				ip3_hat = (1 + lambda * V_sd_p3) * K_p * ((V_sg_p3 + V_T_p) * V_sd_p3 - V_sd_p3 * V_sd_p3 / 2);
			}
			else {
				ip3_hat = (1 + lambda * V_sd_p3) * K_p * (V_sg_p3 + V_T_p) * (V_sg_p3 + V_T_p) / 2;
			}

			float V_gs_n1 = V4[i + 1], V_ds_n1 = V2[i + 1];
			if (V_gs_n1 <= V_T_n) {
				in1_hat = 0;
			}
			else if (V_ds_n1 < V_gs_n1 - V_T_n) {
				in1_hat = (1 + lambda * V_ds_n1) * K_n * ((V_gs_n1 - V_T_n) * V_ds_n1 - V_ds_n1 * V_ds_n1 / 2);
			}
			else {
				in1_hat = (1 + lambda * V_ds_n1) * K_n * (V_gs_n1 - V_T_n) * (V_gs_n1 - V_T_n) / 2;
			}

			float V_gs_n2 = V2[i + 1], V_ds_n2 = V3[i + 1];
			if (V_gs_n2 <= V_T_n) {
				in2_hat = 0;
			}
			else if (V_ds_n2 < V_gs_n2 - V_T_n) {
				in2_hat = (1 + lambda * V_ds_n2) * K_n * ((V_gs_n2 - V_T_n) * V_ds_n2 - V_ds_n2 * V_ds_n2 / 2);
			}
			else {
				in2_hat = (1 + lambda * V_ds_n2) * K_n * (V_gs_n2 - V_T_n) * (V_gs_n2 - V_T_n) / 2;
			}

			float V_gs_n3 = V3[i + 1], V_ds_n3 = V4[i + 1];
			if (V_gs_n3 <= V_T_n) {
				in3_hat = 0;
			}
			else if (V_ds_n3 < V_gs_n3 - V_T_n) {
				in3_hat = (1 + lambda * V_ds_n3) * K_n * ((V_gs_n3 - V_T_n) * V_ds_n3 - V_ds_n3 * V_ds_n3 / 2);
			}
			else {
				in3_hat = (1 + lambda * V_ds_n3) * K_n * (V_gs_n3 - V_T_n) * (V_gs_n3 - V_T_n) / 2;
			}
		}
		V2_prev = V2[i];
		V3_prev = V3[i];
		V4_prev = V4[i];
	}

	for (int i = 0; i < final_T; i++) {
		V2d[i] = V2[i];
		V3d[i] = V3[i];
		V4d[i] = V4[i];
	} 
}

enum { final_T = 500 };
//MNA Variables
float V1 = 5;
float V2h[final_T]; //h = host
float V3h[final_T];
float V4h[final_T];
float *V2d, *V3d, *V4d; //d = device

cudaError_t launchKernel() //Launches the kernel and checks for associated errors
{
	ofstream myfile;
	myfile.open("ring_osc_out.csv");
	myfile << "V1 V2 V3 V4" << endl;

	cudaError_t cudaStatus;

	//Define kernel grid dimensions
	size_t size = final_T * sizeof(float);
	dim3 threadsPerBlock = 1;
	dim3 numBlocks = 1;

	// Choose the GPU to run code on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Check if there is a CUDA enabled GPU?");
		goto Error;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&V2d, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&V3d, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&V4d, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch kernel
	kernel << < numBlocks, threadsPerBlock >> >(V2d, V3d, V4d);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	/* Disabled cudaDeviceSynchronize as it stalls the GPU pipeline
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
	} */

	// Copy output vectors from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(V2h, V2d, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(V3h, V3d, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(V4h, V4d, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	for (int i = 0; i < final_T; i++)
		myfile << V1 << " " << V2h[i] << " " << V3h[i] << " " << V4h[i] << endl;

	myfile.close();

Error:
	cudaFree(V2d);
	cudaFree(V3d);
	cudaFree(V4d);

	return cudaStatus;
}


int main() {

	cudaError_t cudaStatus = launchKernel();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "The kernel could not be launched!");
		return 1;
	}

	char c;
	cout << "Hit 0 to exit. ";
	cin >> c;
	return 0;
}


