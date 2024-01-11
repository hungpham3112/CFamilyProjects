// CUDA Kernel unoptimized

__global__ void CopyBuffer(unsigned int* src, unsigned int* dst, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        dst[id] = src[id];
    }
}

