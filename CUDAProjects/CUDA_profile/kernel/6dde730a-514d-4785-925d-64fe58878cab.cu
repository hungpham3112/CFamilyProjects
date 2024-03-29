#include "./kernel.h"

// CUDA kernel optimized 1

__global__ void CopyBufferOpt1(const unsigned int* src, unsigned int* dst) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int coarseningFactor = 4;
    int coarsenedId = id * coarseningFactor;
    for (int i = 0; i < coarseningFactor; i++) {
        dst[coarsenedId + i] = src[coarsenedId + i];
    }
}

__global__ void showBlockIndices() {
    // Access block indices
    int blockIndexX = blockIdx.x;
    int blockIndexY = blockIdx.y;
    int blockIndexZ = blockIdx.z;

    // Print block indices
    printf("BlockIdx: (%d, %d, %d)\n", blockIndexX, blockIndexY, blockIndexZ);
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
    CopyBufferOpt1<<<blockPerGrid, threadPerBlock>>>(d_A, d_B);
    // showBlockIndices<<<blockPerGrid, threadPerBlock>>>();

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
