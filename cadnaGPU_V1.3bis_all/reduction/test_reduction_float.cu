#include <cuda_runtime.h>

#include "reduction.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <stdlib.h>
#include <cadna.h>
#include <iostream>
#include <cadna_gpu.cu>
#include <curand_kernel.h>

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

  if (threads * blocks > (unsigned int)prop.maxGridSize[0] * (unsigned int)prop.maxThreadsPerBlock) {
    printf("threads: %d, blocks: %d, propmaxgrid: %d, propmaxthread: %d",
            threads, blocks, prop.maxGridSize[0], prop.maxThreadsPerBlock);
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
    std::cout << "Threads: " << threads << "\nBlocks: " << blocks << std::endl;
}

template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {
  cadna_init_gpu();
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // load shared mem
  extern __shared__ T sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    sdata[tid] = g_idata[i];
  } else {
    sdata[tid] = static_cast<T>(0.0);
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
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}

template <class T>
__global__ void casting(int threads, int blocks, float_gpu_st *d_idata,
                        float_gpu_st *d_odata, T *g_idata, T *g_odata, int size,
                        int mode) {
  // for (int j = 0; j < 10; j++) {
  //     printf("%f\n", (float)d_idata[j]);
  // }

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure that the index is within the array bounds
    if constexpr (std::is_same<T, half_gpu_st>::value) {
        if (mode == 0) {
            if (i < size) {
              g_idata[i] = __float_gpu_st2half_gpu_st(d_idata[i]);
              g_odata[i] = __float_gpu_st2half_gpu_st(d_odata[i]);
            }

        } else {
            if (i < size) {
              d_idata[i] = __half_gpu_st2float_gpu_st(g_idata[i]);
              d_odata[i] = __half_gpu_st2float_gpu_st(g_odata[i]);
            }
        }
    } else if constexpr (std::is_same<T, float_gpu_st>::value) {
        if (mode == 0) {
            if (i < size) {
              g_idata[i] = d_idata[i];
              g_odata[i] = d_odata[i];
            }

        } else {
            if (i < size) {
              d_idata[i] = g_idata[i];
              d_odata[i] = g_odata[i];
            }
        }
    }
  for (int j = 0; j < 10; j++) {
      printf("%f\n", (float)d_idata[j]);
  }
}

template <class T>
void reduce(int size, int threads, int blocks, float_gpu_st *d_idata,
            float_gpu_st *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float_gpu_st)
                                 : threads * sizeof(float_gpu_st);
  T *g_idata, *g_odata;
  cudaMalloc((void **)&g_idata, sizeof(T) * size);
  cudaMalloc((void **)&g_odata, sizeof(T) * size);

  casting<<<dimGrid, dimBlock>>>(threads, blocks, d_idata, d_odata, g_idata,
                                 g_odata, size, 0);

  reduce0<<<dimGrid, dimBlock, smemSize>>>(g_idata, g_odata, size);

  casting<<<dimGrid, dimBlock>>>(threads, blocks, d_idata, d_odata, g_idata,
                                 g_odata, size, 1);
}

template <class T>
double_st reduceCPU(T *data, int size) {
  double_st sum = (double_st)data[0];
  double_st c = (double_st)0.0f;

  for (int i = 1; i < size; i++) {
    double_st y = (double_st)data[i] - c;
    double_st t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

float_st benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, bool cpuFinalReduction, int cpuFinalThreshold,
                  float_st *h_odata, float_gpu_st *d_idata, float_gpu_st *d_odata) {
  float_st gpu_result = (float_st)0.0f;
  bool needReadBack = true;

  float_gpu_st *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(float_st) * numBlocks);

    cudaDeviceSynchronize();

    // execute the kernel
    reduce<float_gpu_st>(n, numThreads, numBlocks, d_idata, d_odata);

    if (cpuFinalReduction) {
      // sum partial sums from each block on CPU
      // copy result from device to host
      cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float_st),
                                 cudaMemcpyDeviceToHost);

      for (int i = 0; i < numBlocks; i++) {
        gpu_result = gpu_result + h_odata[i];
      }

      needReadBack = false;
    } else {
      // sum partial block sums on GPU
      int s = numBlocks;

      while (s > cpuFinalThreshold) {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks,
                               threads);
        cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(float_st),
                                   cudaMemcpyDeviceToDevice);
        reduce<float_gpu_st>(s, threads, blocks, d_intermediateSums, d_odata);

      s = (s + threads - 1) / threads;
      }

      if (s > 1) {
        // copy result from device to host
        cudaMemcpy(h_odata, d_odata, s * sizeof(float_st),
                                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < s; i++) {
          gpu_result = gpu_result + h_odata[i];
        }

        needReadBack = false;
      }
    }

    cudaDeviceSynchronize();
  if (needReadBack) {
    cudaMemcpy(&gpu_result, d_odata, sizeof(float_st), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_intermediateSums);
  return gpu_result;
}

int main (int argc, char **argv) {

    cadna_init(-1);

    int size = 1 << 10;   // number of elements to reduce
    std::cout << "Number of elements to reduce: " << size << "\n";
    int maxThreads = 1024; // number of threads per block
    int maxBlocks = 1024;
    bool cpuFinalReduction = true;
    int cpuFinalThreshold = 1;

    // Seed to reproduce
    srand(42);

    unsigned int bytes = size * sizeof(float_st);

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
    //   printf("%s\n", strp(h_idata[i]));
    // }

    cudaMemcpy(d_idata,h_idata,sizeof(float_st) * size,cudaMemcpyHostToDevice);
    // Warm-up
    reduce(size, numThreads, numBlocks, d_idata, d_odata);
    cudaMemcpy(h_odata, d_odata, sizeof(float_st)* numBlocks, cudaMemcpyDeviceToHost);

    float_st gpu_result = benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks, cpuFinalReduction, cpuFinalThreshold, h_odata, d_idata, d_odata);

    // float x_gpu, y_gpu, z_gpu;
    // x_gpu = gpu_result.getx();
    // y_gpu = gpu_result.gety();
    // z_gpu = gpu_result.getz();
    // std::cout << "x_gpu: " << std::fixed << std::setprecision(15) << x_gpu << std::endl;
    // std::cout << "y_gpu: " << std::fixed << std::setprecision(15) << y_gpu << std::endl;
    // std::cout << "z_gpu: " << std::fixed << std::setprecision(15) << z_gpu << std::endl;

    double_st cpu_result = reduceCPU(h_idata, size);
    // float x_cpu, y_cpu, z_cpu;
    // x_cpu = cpu_result.getx();
    // y_cpu = cpu_result.gety();
    // z_cpu = cpu_result.getz();
    // std::cout << "x_cpu: " << std::fixed << std::setprecision(15) << x_cpu << std::endl;
    // std::cout << "y_cpu: " << std::fixed << std::setprecision(15) << y_cpu << std::endl;
    // std::cout << "z_cpu: " << std::fixed << std::setprecision(15) << z_cpu << std::endl;
    
    // int precision = 8;
    // float_st threshold = 1e-8 * size;
    double_st diff = fabs(static_cast<double_st>(gpu_result) - cpu_result);
      printf("\nGPU result(float_st, float_st) = %s\nNumber of significant digits in GPU result: %d\n", strp(gpu_result), gpu_result.nb_significant_digit());
      printf("\nCPU result(float_st, float_st) = %s\nNumber of significant digits in CPU result: %d\n", strp(cpu_result), cpu_result.nb_significant_digit());
      // float x, y, z;
      // x = gpu_result.getx();
      // y = gpu_result.gety();
      // z = gpu_result.getz();
      // std::cout << "x = "<< x << " y = " << y << " z = "<< " "<< z << std::endl;
      // gpu_result.display();
      // cpu_result.display();
      // float_st zero = -422550099332258165946469842944.000000;
      // std::cout << (gpu_result.computedzero() ? "computational zero" : "no computational zero") << std::endl;
      // std::cout << zero << " sig figs: " << zero.nb_significant_digit() << std::endl;

     printf("\nDifference between CPU result and GPU result = %s\n"
             "Number of difference' significant figures: %d\n", strp(diff), diff.nb_significant_digit());


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
    cadna_end();
    return(EXIT_SUCCESS);
}
