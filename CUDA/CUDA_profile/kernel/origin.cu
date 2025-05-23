#include "./kernel.h"

// CUDA Kernel unoptimized

__global__ void CopyBufferOrigin(unsigned int* src, unsigned int* dst) {
    int id = (int)blockIdx.x * blockDim.x + threadIdx.x;
    dst[id] = src[id];
}

__global__ void showBuiltInVariables() {
    // Block indices
    int blockIndexX = blockIdx.x;
    int blockIndexY = blockIdx.y;
    int blockIndexZ = blockIdx.z;

    // Thread indices
    int threadIndexX = threadIdx.x;
    int threadIndexY = threadIdx.y;
    int threadIndexZ = threadIdx.z;

    // Block dimensions
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int blockDimZ = blockDim.z;

    // Grid dimensions
    int gridDimX = gridDim.x;
    int gridDimY = gridDim.y;
    int gridDimZ = gridDim.z;

    // Global thread index (unique across the entire grid)
    int globalThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // Total number of threads in the grid
    int totalNumThreads = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;

    // Print information
    printf("BlockIdx: (%d, %d, %d), ThreadIdx: (%d, %d, %d)\n", blockIndexX, blockIndexY, blockIndexZ, threadIndexX, threadIndexY, threadIndexZ);
    printf("BlockDim: (%d, %d, %d), GridDim: (%d, %d, %d)\n", blockDimX, blockDimY, blockDimZ, gridDimX, gridDimY, gridDimZ);
    printf("GlobalThreadIndex: %d, TotalNumThreads: %d\n", globalThreadIndex, totalNumThreads);
}
int main(int argc, char *argv[])
{
        
    size_t size = ARR_LEN * sizeof(unsigned int);
    unsigned int* h_A = (unsigned int*)malloc(size);
    unsigned int* h_B = (unsigned int*)malloc(size);
    generateRandomUnsignedIntArray(h_A, ARR_LEN, 0, 1000);
    // ShowArr(h_A, ARR_LEN, "h_A");
    unsigned int *d_A, *d_B;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void** )&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threadPerBlock = 512;
    int blockPerGrid = (ARR_LEN + threadPerBlock - 1) / threadPerBlock;
    CopyBufferOrigin<<<blockPerGrid, threadPerBlock>>>(d_A, d_B);

    // showBuiltInVariables<<<blockPerGrid, threadPerBlock>>>();
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    // performApproxTest(h_A, h_B, ARR_LEN);
    // performIdenticalTest(h_A, h_B, ARR_LEN);
    // ShowArr(h_B, ARR_LEN, "h_B");
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    return 0;
}
