#include <iostream>

__global__ vecAddKernel(float* A, float* B, float* C, int vec_len) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < vec_len) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ vecAdd(float* A_h, float* B_h, float* C_h, int vec_len) {
    int size = n * sizeof(float);
    float* A_d, B_d, C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyDeviceToHost);

    int dimBlock = 256
    int dimGrid = ceil(n / 256.0);
    vecAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, vec_len);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyHostToDevice);
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}
int main(int argc, char *argv[])
{
    
    return 0;
}



