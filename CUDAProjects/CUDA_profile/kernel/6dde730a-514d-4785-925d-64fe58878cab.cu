// CUDA kernel optimized 1

__global__ void CopyBuffer(const unsigned int* src, unsigned int* dst) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int coarseningFactor = 4;

    int coarsenedId = id * coarseningFactor;

    for (int i = 0; i < coarseningFactor; i++) {
        dst[coarsenedId + i] = src[coarsenedId + i];
    }
}
