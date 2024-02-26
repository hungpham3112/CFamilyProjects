// #include "cadna.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
// #include "cadna_gpu.cu"

namespace cg = cooperative_groups;



unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
    // printf("%d\n", threads);
    // printf("%d\n", blocks);

  if ((float)threads * blocks >
      (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }
}

__global__ void reduce0(float *g_idata, float *g_odata, unsigned int n) {
  
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

 extern __shared__ float sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    // modulo arithmetic is slow!
    if ((tid % (2 * s)) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0){
      g_odata[blockIdx.x] = sdata[0];
  }
}

void reduce(int size, int threads, int blocks, float *d_idata,
            float *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  reduce0<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
}

float reduceCPU(float *data, int size) {
  float sum = data[0];
  float c = (float)0.0;

  for (int i = 1; i < size; i++) {
    float y = data[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

float benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, bool cpuFinalReduction, int cpuFinalThreshold,
                  float *h_odata, float *d_idata, float *d_odata) {
  float gpu_result = 0;
  bool needReadBack = true;

  float *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(float) * numBlocks);
    gpu_result = 0;

    cudaDeviceSynchronize();

    // execute the kernel
    reduce(n, numThreads, numBlocks, d_idata, d_odata);

    if (cpuFinalReduction) {
      // sum partial sums from each block on CPU
      // copy result from device to host
      cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float),
                                 cudaMemcpyDeviceToHost);

      for (int i = 0; i < numBlocks; i++) {
        gpu_result += h_odata[i];
      }

      needReadBack = false;
    } else {
      // sum partial block sums on GPU
      int s = numBlocks;

      while (s > cpuFinalThreshold) {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks,
                               threads);
        cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(float),
                                   cudaMemcpyDeviceToDevice);
        reduce(s, threads, blocks, d_intermediateSums, d_odata);

      s = (s + threads - 1) / threads;
      }

      if (s > 1) {
        // copy result from device to host
        cudaMemcpy(h_odata, d_odata, s * sizeof(float),
                                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < s; i++) {
          gpu_result += h_odata[i];
        }

        needReadBack = false;
      }
    }

    cudaDeviceSynchronize();
  if (needReadBack) {
    cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_intermediateSums);
  return gpu_result;
}

int main (int argc, char **argv)
{
    //InitCuda(1, 512);
  int size = 1 << 4;   // number of elements to reduce
  int maxThreads = 256; // number of threads per block
  int maxBlocks = 64;
  bool cpuFinalReduction = false;
  int cpuFinalThreshold = 1;


  // cadna_init(-1);

  // Data initialization (on the CPU)
    unsigned int bytes = size * sizeof(float);

    float *h_idata = (float *)malloc(bytes);

    for (int i = 0; i < size; i++) {
      // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (float)rand() / RAND_MAX;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks,

                           numThreads);
    float *h_odata = (float *)malloc(numBlocks * sizeof(float));
    float *d_idata = NULL;
    float *d_odata = NULL;

    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, numBlocks * sizeof(float));
  // for (int i = 0; i < 10; ++i) {
  //     printf("%f\n", h_idata[i]);
  // }

  cudaMemcpy(d_idata,h_idata,sizeof(float) * size,cudaMemcpyHostToDevice);
  // cudaMemcpy(d_odata, h_idata,sizeof(float) * numBlocks,cudaMemcpyHostToDevice);

  reduce(size, numThreads, numBlocks, d_idata, d_odata);

  cudaMemcpy(h_odata, d_odata, sizeof(float)* numBlocks, cudaMemcpyDeviceToHost);
    float gpu_result =
        benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                           cpuFinalReduction, cpuFinalThreshold, 
                           h_odata, d_idata, d_odata);

    float cpu_result = reduceCPU(h_idata, size);
    int precision = 8;
    // double threshold = 1e-8 * size;
    // double diff = 0;
      printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
      printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

     // printf(diff)


  // for (int i = 0; i < VECTOR_SIZE [>VECTOR_SIZE<]; i++) {
  //   printf("  S_CPU[%d] = ",i);
  //   S_CPU[i].display();
  //   printf("\n  S_fromGPU[%d] = ",i);
  //   S_CPU_fromGPU[i].display();
  //   printf("%s \n",S_CPU_fromGPU[i].str(s));
  // }

free(h_idata);
free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  return(EXIT_SUCCESS);
}

