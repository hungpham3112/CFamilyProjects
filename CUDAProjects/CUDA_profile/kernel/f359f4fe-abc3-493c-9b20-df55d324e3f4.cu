// CUDA kernel optimized 2

__global__ void CopyBuffer(const unsigned int* src, unsigned int* dst) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < 512) {
        dst[id] = src[id];
    } else {
        dst[id] = 0;
    }
}
