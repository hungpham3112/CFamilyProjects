#include <cstdlib>
using namespace std;

#include <cadna_gpu_half.cu>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <cadna.h>
#include <iostream>
#include <cadna_commun.h>
#include "utils.cpp"


namespace cg = cooperative_groups;


__global__ void reduceKernel(half_gpu_st *g_idata_old, half_gpu_st *g_odata, unsigned int n) {
    cadna_init_gpu();
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ half_gpu_st sdata[];
    if (i < n) {
        half_gpu_st g_idata = __float_gpu_st2half_gpu_st(g_idata_old[i]);
        sdata[tid] = g_idata;
    } else {
        sdata[tid] = (half_gpu_st)0.0f;
    }

    cg::sync(cta);
    // do reduction in shared mem

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2 * s)) == 0) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0){
        g_odata[blockIdx.x] = __half_gpu_st2float_gpu_st(sdata[0]);
    }
}

void reduce(int size, int threads, int blocks, float_gpu_st *d_idata,
            float_gpu_st *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float_gpu_st) : threads * sizeof(float_gpu_st);
  printf("%d", smemSize);

  reduceKernel<<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
}

float_st reduceCPU(float_st *data, int size) {
  float_st sum = data[0];
  float_st c = (float_st)0.0f;

  for (int i = 1; i < size; i++) {
    float_st y = data[i] - c;
    float_st t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

template <class T>
T benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, int whichKernel, int testIterations,
                  bool cpuFinalReduction, int cpuFinalThreshold,
                  StopWatchInterface *timer, T *h_odata, T *d_idata,
                  T *d_odata) {
  T gpu_result = 0;
  bool needReadBack = true;

  T *d_intermediateSums;
  checkCudaErrors(
      cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks));

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;

    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    // execute the kernel
    reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    if (cpuFinalReduction) {
      // sum partial sums from each block on CPU
      // copy result from device to host
      checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T),
                                 cudaMemcpyDeviceToHost));

      for (int i = 0; i < numBlocks; i++) {
        gpu_result += h_odata[i];
      }

      needReadBack = false;
    } else {
      // sum partial block sums on GPU
      int s = numBlocks;
      int kernel = whichKernel;

      while (s > cpuFinalThreshold) {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks,
                               threads);
        checkCudaErrors(cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T),
                                   cudaMemcpyDeviceToDevice));
        reduce<T>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

        if (kernel < 3) {
          s = (s + threads - 1) / threads;
        } else {
          s = (s + (threads * 2 - 1)) / (threads * 2);
        }
      }

      if (s > 1) {
        // copy result from device to host
        checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T),
                                   cudaMemcpyDeviceToHost));

        for (int i = 0; i < s; i++) {
          gpu_result += h_odata[i];
        }

        needReadBack = false;
      }
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
  }

  if (needReadBack) {
    // copy final sum from device to host
    checkCudaErrors(
        cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaFree(d_intermediateSums));
  return gpu_result;
}

int main (int argc, char **argv)
{
    cadna_init(-1);

    int size = 1 << 6;   // number of elements to reduce
    int maxThreads = 16; // number of threads per block
    int maxBlocks = 8;
    bool cpuFinalReduction = false;
    int cpuFinalThreshold = 1;


    printf("- Initialization\n");
    unsigned int bytes = size * sizeof(float_st);

    srand(42);
    float_st *h_idata = (float_st *)malloc(bytes);

    for (int i = 0; i < size; i++) {
      // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (float_st)rand() / (float_st)RAND_MAX;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);
    float_st *h_odata = (float_st *)malloc(numBlocks * sizeof(float_st));
    float_gpu_st *d_idata = NULL;
    float_gpu_st *d_odata = NULL;

    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, numBlocks * sizeof(float_st));
  // for (int i = 0; i < 10; ++i) {
  //     printf("%f\n", h_idata[i]);
  // }

    // Warm-up computing
    cudaMemcpy(d_idata,h_idata,sizeof(float_st) * size,cudaMemcpyHostToDevice);
    reduce(size, numThreads, numBlocks, d_idata, d_odata);
    cudaMemcpy(h_odata, d_odata, sizeof(float_st)* numBlocks, cudaMemcpyDeviceToHost);

    // Actual computing
    float_st gpu_result = benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks, cpuFinalReduction, cpuFinalThreshold, h_odata, d_idata, d_odata);

    float_st cpu_result = reduceCPU(h_idata, size);

    float_st diff = 0.0f;
    printf("\nGPU result = %s, number of significant digits: %d\n", strp(gpu_result), gpu_result.nb_significant_digit());
    printf("CPU result = %s, number of significant digits: %d\n", strp(cpu_result), cpu_result.nb_significant_digit());

    float x, y, z;
    x = gpu_result.getx();
    y = gpu_result.gety();
    z = gpu_result.getz();
    std::cout << "x = "<< x << " y = " << y << " z = "<< " "<< z << std::endl;
    gpu_result.display(); 
    cpu_result.display();
    float_st zero = 0.;
    std::cout << (gpu_result.computedzero() ? "computational zero" : "no computational zero") << std::endl;
    std::cout << zero << std::endl;

    diff = fabs(gpu_result - cpu_result);
    printf("Difference between gpu(double_st) and cpu(double_st), %s\n", strp(diff));

    cadna_end();
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    return(EXIT_SUCCESS);
}

