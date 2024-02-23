#include <cstdlib>
using namespace std;

#include <cadna_gpu_half.cu>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <cadna.h>
#include <iostream>
#include <cadna_commun.h>


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
}

__global__ void reduceKernel(float_gpu_st *g_idata, float_gpu_st *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();


  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    cadna_init_gpu();

  extern __shared__ half_gpu_st sdata[];

  sdata[tid] = (i < n) ? __float_gpu_st2half_gpu_st(g_idata[i]) : (half_gpu_st)0.0f;

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

float_st benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, bool cpuFinalReduction, int cpuFinalThreshold,
                  float_st *h_odata, float_gpu_st *d_idata, float_gpu_st *d_odata) {
  float_st gpu_result = (float_st)0.0f;
  bool needReadBack = true;

  float_gpu_st *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(float_st) * numBlocks);

    cudaDeviceSynchronize();

    // execute the kernel
    reduce(n, numThreads, numBlocks, d_idata, d_odata);

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
        reduce(s, threads, blocks, d_intermediateSums, d_odata);

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

int main (int argc, char **argv)
{
    cadna_init(-1);

    //InitCuda(1, 512);
  int size = 1 << 6;   // number of elements to reduce
  int maxThreads = 16; // number of threads per block
  int maxBlocks = 8;
  bool cpuFinalReduction = false;
  int cpuFinalThreshold = 1;


  // cadna_init(-1);

  // Data initialization (on the CPU)
  printf("- Initialization\n");
    unsigned int bytes = size * sizeof(float_st);

    float_st *h_idata = (float_st *)malloc(bytes);

    for (int i = 0; i < size; i++) {
      // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (float_st)rand() / (float_st)RAND_MAX;
    }

    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks,

                           numThreads);
    float_st *h_odata = (float_st *)malloc(numBlocks * sizeof(float_st));
    float_gpu_st *d_idata = NULL;
    float_gpu_st *d_odata = NULL;

    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, numBlocks * sizeof(float_st));
  // for (int i = 0; i < 10; ++i) {
  //     printf("%f\n", h_idata[i]);
  // }

  cudaMemcpy(d_idata,h_idata,sizeof(float_st) * size,cudaMemcpyHostToDevice);
  // cudaMemcpy(d_odata, h_idata,sizeof(float) * numBlocks,cudaMemcpyHostToDevice);

  reduce(size, numThreads, numBlocks, d_idata, d_odata);

  cudaMemcpy(h_odata, d_odata, sizeof(float_st)* numBlocks, cudaMemcpyDeviceToHost);
    float_st gpu_result =
        benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                           cpuFinalReduction, cpuFinalThreshold, 
                           h_odata, d_idata, d_odata);

    float_st cpu_result = reduceCPU(h_idata, size);
    // int precision = 8;
    // float_st threshold = 1e-8 * size;
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



     // diff = fabs((double_st)gpu_result - (double_st)cpu_result);
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
  cadna_end();
  return(EXIT_SUCCESS);

  cadna_end();
}

