#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void kernel_mul(int* matrixA, int* matrixB, int* matrixC, int* matrixD, int* matrixE, int p, int q, int r)
{
	int tidx = threadIdx.x / 32;
	int tidy = threadIdx.x % 32;
	int x = tidx + blockIdx.x * 32;
	int y = tidy + blockIdx.y * 32;
	int sum = 0;
	if(x < p && y < r)
	{
		for(int i = 0; i < q; i++)
		{
			sum += matrixA[x*q + i] * matrixB[i*r + y];	
			sum += matrixC[x*q + i] * matrixD[i*r + y];
			//A warp of 32 threads compute from E[x][y] to E[x][y+31]
			//Thus the warp-threads access the same row x in matrixA and matrixC
			//But we obtain coalesced global memory access as the warp-threads access column y - y+31 which have the values in contiguous memory locations 
			//for a given i in the for loop we access B[i][y]-B[i][y+31] and D[i][y]-D[i][y+31] which are in contiguous memory locations
		}
		matrixE[x*r + y] = sum;
	}
}

__global__ void kernel_transpose(int* matrix_T,int* matrix, int r, int q)
{
	__shared__ int tile[32][32];
	int tidx = threadIdx.x / 32;
	int tidy = threadIdx.x % 32;
	int x = tidx + blockIdx.x * 32;	// 0 - r+
	int y = tidy + blockIdx.y * 32; // 0 - q+
	int id = x*q + y;
	int x_t = tidx + blockIdx.y * 32;	// 0 - q+
	int y_t = tidy + blockIdx.x * 32;	// 0 - r+
	int id_t = x_t*r + y_t;
	if(x < r && y < q)
	{
		tile[tidx][tidy] = matrix[id];		//Coalesced global memory access as warp-threads access along the same row of size 32
	}
	__syncthreads();
	if(x_t < q && y_t < r)
	{
		matrix_T[id_t] = tile[tidy][tidx];	//Coalesced global memory access as warp-threads access along the same row of size 32
	}
	
}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	// Memory coallescing and shared memory optimization

	int p_t = (p+31)/32;
	int q_t = (q+31)/32;
	int r_t = (r+31)/32;

	int* d_matrixDt;
	cudaMalloc(&d_matrixDt, r * q * sizeof(int));

	dim3 dimGrid(r_t, q_t, 1);
	dim3 dimBlock(32*32);

	kernel_transpose<<<dimGrid, dimBlock>>>(d_matrixDt, d_matrixD, r, q); 	//d_matrixDt is the transpose of d_matrixD
	
	cudaDeviceSynchronize();

	dim3 dimGrid1(p_t, r_t, 1);

	kernel_mul<<<dimGrid1, dimBlock>>>(d_matrixA, d_matrixB, d_matrixC, d_matrixDt, d_matrixE, p, q, r);	//d_matrixE = d_matrixA * d_matrixB + d_matrixC * d_matrixDt

	cudaDeviceSynchronize();

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
