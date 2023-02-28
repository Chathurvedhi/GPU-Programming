/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    int r1 = blockIdx.x;
    int r2 = threadIdx.x;
    for(int c1 = 0; c1 < n; c1++){
        for(int c2 = 0; c2 < n; c2++){
            C[m*n*n*r1 + m*n*c2 + m*c1 + r2] = A[n*r1+c1] * B[n*r2+c2];
        }
    }

}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    int c1 = blockIdx.x;
    int c2 = threadIdx.x;
    int temp = threadIdx.y;
    if(temp == 1) c2 += ceil(float(n) / 2);
    if(c2 >= n) return;
    for(int r1 = 0; r1 < m; r1++){
        for(int r2 = 0; r2 < m; r2++){
            C[m*n*n*r1 + m*n*c2 + m*c1 + r2] = A[n*r1+c1] * B[n*r2+c2];
        }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    int bx = blockIdx.x;        // (n*n)/16
    int gd = gridDim.x;         // (n*n)/16
    int by = blockIdx.y;        // (m*m)/64
    int tx = threadIdx.x;       // 64
    int ty = threadIdx.y;       // 16
    long pos = ty + tx * 16 + bx * 16 * 64 + by * 16 * 64 * gd;
    if(pos >= m * m * n * n) return;
    int r = pos / (m * n);
    int c = pos % (m * n);
    int r1 = r / n;
    int c2 = r % n;
    int c1 = c / m;
    int r2 = c % m;
    C[pos] = A[n * r1 + c1] * B[n * r2 + c2];
    
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    // --> Allocate memory for A on device 
    cudaMalloc(&d_a, sizeof(long int)*(m * n));
    // --> Allocate memory for B on device 
    cudaMalloc(&d_b, sizeof(long int)*(m * n));
    // --> Allocate memory for C on device 
    cudaMalloc(&d_c, m * m * n * n * sizeof(long int));

    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    cudaMemcpy(d_a, h_a, sizeof(long int)*(m * n), cudaMemcpyHostToDevice);
    // --> Copy B from Host to Device 
    cudaMemcpy(d_b, h_b, sizeof(long int)*(m * n), cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/


    // --> Set the launch configuration 

    double starttime = rtclock();  

    // --> Launch the kernel 
    per_row_AB_kernel<<<m,m>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 

    cudaMemcpy(h_c, d_c, sizeof(long int)*(m * m * n * n), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/

    long val = ceil(float(n) / 2);
    
    
    // --> Set the launch configuration 

    starttime = rtclock(); 

    // --> Launch the kernel 
    per_column_AB_kernel<<<n,dim3(val,2,1)>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // --> Copy C from Device to Host

    cudaMemcpy(h_c, d_c, sizeof(long int)*(m * m * n * n), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  

    // --> Launch the kernel 
    per_element_kernel<<<grid3,block3>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c, d_c, sizeof(long int)*(m * m * n * n), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}