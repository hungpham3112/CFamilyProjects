#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <cadna.h>
#include <cadna_gpu.cu>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <typeinfo>
// #include <cadna_gpu_half.cu>

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

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = std::min(maxBlocks, blocks);
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
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
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
    int index = 2 * s * tid;

    if (index < blockDim.x) {
      sdata[index] = sdata[index] + sdata[index + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    sdata[tid] = g_idata[i];
  } else {
    sdata[tid] = static_cast<T>(0.0);
  }

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ T sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum;

  if (i < n) {
    T mySum = g_idata[i];
  } else {
    T mySum = static_cast<T>(0.0);
  }

  if (i + blockDim.x < n) mySum = mySum + g_idata[i + blockDim.x];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
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
    // if constexpr (std::is_same<T, half_gpu_st>::value) {
    //     if (mode == 0) {
    //         if (i < size) {
    //           g_idata[i] = __float_gpu_st2half_gpu_st(d_idata[i]);
    //           g_odata[i] = __float_gpu_st2half_gpu_st(d_odata[i]);
    //         }

    //     } else {
    //         if (i < size) {
    //           d_idata[i] = __half_gpu_st2float_gpu_st(g_idata[i]);
    //           d_odata[i] = __half_gpu_st2float_gpu_st(g_odata[i]);
    //         }
    //     }
    //   }
    if constexpr (std::is_same<T, float_gpu_st>::value) {
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
}

template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  // as kernel 9 - multi_warp_cg_reduce cannot work for more than 64 threads
  // we choose to set kernel 7 for this purpose.
  if (threads < 64 && whichKernel == 9)
  {
    whichKernel = 7;
  }

  T *g_idata, *g_odata;
  cudaMalloc((void **)&g_idata, sizeof(T) * size);
  cudaMalloc((void **)&g_odata, sizeof(T) * size);

  casting<<<dimGrid, dimBlock>>>(threads, blocks, d_idata, d_odata, g_idata,
                                 g_odata, size, 0);


  // choose which of the optimized versions of reduction to launch
  switch (whichKernel) {
    case 0:
      reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 3:
      reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
  }
  casting<<<dimGrid, dimBlock>>>(threads, blocks, d_idata, d_odata, g_idata,
                                 g_odata, size, 1);
}

template <class T>
float_st benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                         int maxBlocks, int whichKernel, bool cpuFinalReduction,
                         int cpuFinalThreshold, float_st *h_odata,
                         float_gpu_st *d_idata, float_gpu_st *d_odata) {
  float_st gpu_result = (float_st)0.0f;
  bool needReadBack = true;

  float_gpu_st *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(float_st) * numBlocks);

  cudaDeviceSynchronize();

  // execute the kernel
  reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

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
      getNumBlocksAndThreads(0, s, maxBlocks, maxThreads, blocks, threads);
      cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(float_st),
                 cudaMemcpyDeviceToDevice);
      reduce<T>(s, threads, blocks, d_intermediateSums, d_odata);
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

double_st reduceCPU(float_st *data, int size) {
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

template <class T> void get3Numbers(double_st cpu_result, T gpu_result) {
  auto x = gpu_result.getx();
  auto y = gpu_result.gety();
  auto z = gpu_result.getz();
  std::cout << "x = " << x << " y = " << y << " z = "
            << " " << z << std::endl;
  gpu_result.display();
  cpu_result.display();
}

template <class T> double_st calDiff(double_st cpu_result, T gpu_result) {
  return fabs(static_cast<double_st>(gpu_result) - cpu_result);
}

void writeToCSV(const std::string& fileName, const std::vector<std::string>& values) {
    // Open the file for writing
    std::ofstream outputFile(fileName, std::ios::app);

    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return;
    }

    // Write values to the CSV file
    for (size_t i = 0; i < values.size() - 1; ++i) {
        outputFile << values[i] << ",";
    }
    outputFile << values.back() << std::endl;

    // Close the file
    outputFile.close();

    std::cout << "Data written to the CSV file successfully." << std::endl;
}

template<typename T>
std::string getTypeName() {
    if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "std::string";
    } else if constexpr (std::is_same_v<T, float_st>) {
        return "float_st";
    } else if constexpr (std::is_same_v<T, double_st>) {
        return "double_st";
    } else {
        return "Unknown Type";
    }
}

// template<class T>
// bool isclose(float a, float b, float atol, float rtol,
//                        const char *mode) {
//   float diff = std::abs(a - b);
//   if (std::strcmp(mode, "weak") == 0) {
//     float tolerance = (atol + rtol * std::max(std::abs(a), std::abs(b))) / 2;
//     return diff <= tolerance;
//   } else if (std::strcmp(mode, "strong") == 0) {
//     float tolerance = (atol + rtol * std::min(std::abs(a), std::abs(b))) / 2;
//     return diff <= tolerance;
//   }
//   return false;
// }

int main(int argc, char **argv) {
  cadna_init(-1);

  int size = 1 << 15;   // number of elements to reduce
  std::cout << "Size of array: " << size << std::endl;
  int maxThreads = 512; // number of threads per block
  int maxBlocks = 1024;
  bool cpuFinalReduction = true;
  int cpuFinalThreshold = 1;

  unsigned int bytes = size * sizeof(float_st);

  srand(42);
  float_st *h_idata = (float_st *)malloc(bytes);

  for (int i = 0; i < size; i++) {
    // Keep the numbers small so we don't get truncation error in the sum
    h_idata[i] = (float_st)rand() / (float_st)RAND_MAX;
  }

  int numBlocks = 0;
  int numThreads = 0;
  int kernel = 0;
  std::vector<int> lst_threads = {1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
  getNumBlocksAndThreads(kernel, size, maxBlocks, maxThreads, numBlocks, numThreads);
  float_st *h_odata = (float_st *)malloc(numBlocks * sizeof(float_st));
  float_gpu_st *d_idata = NULL;
  float_gpu_st *d_odata = NULL;

  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, numBlocks * sizeof(float_st));

  // Warm-up computing
  cudaMemcpy(d_idata, h_idata, sizeof(float_st) * size, cudaMemcpyHostToDevice);
  reduce<float_gpu_st>(size, numThreads, numBlocks, d_idata, d_odata);
  cudaMemcpy(h_odata, d_odata, sizeof(float_st) * numBlocks,
             cudaMemcpyDeviceToHost);

  auto gpu_result = benchmarkReduce<float_gpu_st>(
      size, numThreads, numBlocks, maxThreads, maxBlocks, cpuFinalReduction,
      cpuFinalThreshold, h_odata, d_idata, d_odata);
  auto cpu_result = reduceCPU(h_idata, size);

  // double_st diff = calDiff<float_st>(cpu_result, gpu_result);
  double_st diff = fabs(static_cast<double_st>(gpu_result) - cpu_result);
  printf("\nGPU result = %s, number of significant digits: %d\n",
         strp(gpu_result), gpu_result.nb_significant_digit());
  printf("CPU result = %s, number of significant digits: %d\n",
         strp(cpu_result), cpu_result.nb_significant_digit());
  printf("Difference between gpu(double_st) and cpu(double_st), %s, number of significant digits: %d\n", strp(diff), diff.nb_significant_digit());

  // write to file
    std::string fileName = "output.csv";
    std::vector<std::string> values = {std::to_string(size), std::to_string(numThreads),
        std::to_string(numBlocks), std::to_string(kernel),
        getTypeName<decltype(cpu_result)>(), getTypeName<decltype(gpu_result)>(),
        strp(gpu_result), std::to_string(gpu_result.nb_significant_digit()), 
        strp(cpu_result),std::to_string(cpu_result.nb_significant_digit()),
        strp(diff), std::to_string(diff.nb_significant_digit())};


    writeToCSV(fileName, values);


  cadna_end();
  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);
  cudaFree(d_odata);
  return (EXIT_SUCCESS);
}

