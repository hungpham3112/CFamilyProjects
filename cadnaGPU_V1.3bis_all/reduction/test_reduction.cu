// CUDA Runtime
#include <cuda_runtime.h>
#include <fstream>

// Utilities and system includes
#include <algorithm>
#include <ios>
#include <random>
#include <iomanip>

// includes, project
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif

template <class T>
__global__ void reduce0(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

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
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

extern "C" bool isPow2(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
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
  if (threads < 64 && whichKernel == 9) {
    whichKernel = 7;
  }

  // choose which of the optimized versions of reduction to launch
  switch (whichKernel) {
  case 0:
    reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;

  // case 1:
  //   reduce1<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //   break;

  // case 2:
  //   reduce2<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //   break;

  // case 3:
  //   reduce3<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //   break;

  // case 4:
  //   switch (threads) {
  //   case 512:
  //     reduce4<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 256:
  //     reduce4<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 128:
  //     reduce4<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 64:
  //     reduce4<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 32:
  //     reduce4<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 16:
  //     reduce4<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 8:
  //     reduce4<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 4:
  //     reduce4<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 2:
  //     reduce4<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 1:
  //     reduce4<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;
  //   }

  //   break;

  // case 5:
  //   switch (threads) {
  //   case 512:
  //     reduce5<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 256:
  //     reduce5<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 128:
  //     reduce5<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 64:
  //     reduce5<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 32:
  //     reduce5<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 16:
  //     reduce5<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 8:
  //     reduce5<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 4:
  //     reduce5<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 2:
  //     reduce5<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 1:
  //     reduce5<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;
  //   }

  //   break;

  // case 6:
  //   if (isPow2(size)) {
  //     switch (threads) {
  //     case 512:
  //       reduce6<T, 512, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 256:
  //       reduce6<T, 256, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 128:
  //       reduce6<T, 128, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 64:
  //       reduce6<T, 64, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 32:
  //       reduce6<T, 32, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 16:
  //       reduce6<T, 16, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 8:
  //       reduce6<T, 8, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 4:
  //       reduce6<T, 4, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 2:
  //       reduce6<T, 2, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 1:
  //       reduce6<T, 1, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     }
  //   } else {
  //     switch (threads) {
  //     case 512:
  //       reduce6<T, 512, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 256:
  //       reduce6<T, 256, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 128:
  //       reduce6<T, 128, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 64:
  //       reduce6<T, 64, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 32:
  //       reduce6<T, 32, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 16:
  //       reduce6<T, 16, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 8:
  //       reduce6<T, 8, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 4:
  //       reduce6<T, 4, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 2:
  //       reduce6<T, 2, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 1:
  //       reduce6<T, 1, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     }
  //   }

  //   break;

  // case 7:
  //   // For reduce7 kernel we require only blockSize/warpSize
  //   // number of elements in shared memory
  //   smemSize = ((threads / 32) + 1) * sizeof(T);
  //   if (isPow2(size)) {
  //     switch (threads) {
  //     case 1024:
  //       reduce7<T, 1024, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     case 512:
  //       reduce7<T, 512, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 256:
  //       reduce7<T, 256, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 128:
  //       reduce7<T, 128, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 64:
  //       reduce7<T, 64, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 32:
  //       reduce7<T, 32, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 16:
  //       reduce7<T, 16, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 8:
  //       reduce7<T, 8, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 4:
  //       reduce7<T, 4, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 2:
  //       reduce7<T, 2, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 1:
  //       reduce7<T, 1, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     }
  //   } else {
  //     switch (threads) {
  //     case 1024:
  //       reduce7<T, 1024, true>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     case 512:
  //       reduce7<T, 512, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 256:
  //       reduce7<T, 256, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 128:
  //       reduce7<T, 128, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 64:
  //       reduce7<T, 64, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 32:
  //       reduce7<T, 32, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 16:
  //       reduce7<T, 16, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 8:
  //       reduce7<T, 8, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 4:
  //       reduce7<T, 4, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 2:
  //       reduce7<T, 2, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;

  //     case 1:
  //       reduce7<T, 1, false>
  //           <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //       break;
  //     }
  //   }

  //   break;
  // case 8:
  //   cg_reduce<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //   break;
  // case 9:
  //   constexpr int numOfMultiWarpGroups = 2;
  //   smemSize = numOfMultiWarpGroups * sizeof(T);
  //   switch (threads) {
  //   case 1024:
  //     multi_warp_cg_reduce<T, 1024, 1024 / numOfMultiWarpGroups>
  //         <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 512:
  //     multi_warp_cg_reduce<T, 512, 512 / numOfMultiWarpGroups>
  //         <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 256:
  //     multi_warp_cg_reduce<T, 256, 256 / numOfMultiWarpGroups>
  //         <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 128:
  //     multi_warp_cg_reduce<T, 128, 128 / numOfMultiWarpGroups>
  //         <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

  //   case 64:
  //     multi_warp_cg_reduce<T, 64, 64 / numOfMultiWarpGroups>
  //         <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
  //     break;

    default:
      printf("thread block size of < 64 is not supported for this kernel\n");
      break;
    // }
    // break;
  }
}
enum ReduceType {
  REDUCE_FLOAT,
  REDUCE_DOUBLE,
};

template <class T>
void generate_random_array(T *array, const int &size, const int &seed) {
    std::cout << "Seed: " << seed << std::endl;
  std::mt19937 engine(seed);
  std::normal_distribution<T> generator(0, 1);
  for (int i = 0; i < size; i++)
    array[i] = generator(engine);
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
template <class T> T runTest(int argc, char **argv, ReduceType datatype, float* data);

#define MAX_BLOCK_DIM_SIZE 65535

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

const char *getReduceTypeString(const ReduceType type) {
  switch (type) {
  case REDUCE_FLOAT:
    return "float";
  case REDUCE_DOUBLE:
    return "double";
  default:
    return "unknown";
  }
}

template <typename T>
void appendVectorToFile(const std::vector<T>& dataVector, const std::string& filePath) {
    // Open the file in append mode on the host (CPU) side
    std::ofstream outfile(filePath, std::ios::app);

    // Check if the file is opened successfully
    if (outfile.is_open()) {
        // Write each element to a new line
        for (const auto& value : dataVector) {
            outfile << value << std::endl;
        }

        // Close the file on the host (CPU) side
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << filePath << std::endl;
    }
}
template <typename T>
std::string formatNumericToString(T value, size_t precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

template <typename T>
T calculateMean(const std::vector<T>& numbers) {
    return numbers.empty() ? T() : std::accumulate(numbers.begin(), numbers.end(), T()) / static_cast<T>(numbers.size());
}

template <typename T>
T calculateVariance(std::vector<T> &samples)
{
     int size = samples.size();

     T variance = 0;
     T t = samples[0];
     for (int i = 1; i < size; i++)
     {
          t += samples[i];
          T diff = ((i + 1) * samples[i]) - t;
          variance += (diff * diff) / ((i + 1.0) *i);
     }

     return variance / (size - 1);
}

template <typename T>
T calculateStandardDeviation(std::vector<T>& samples) {
    T variance = calculateVariance(samples);
    return std::sqrt(variance);
}

std::vector<std::string> findCommonCharacters(const std::vector<std::string>& vector1, const std::vector<std::string>& vector2) {
    std::vector<std::string> commonCharacters;

    // Iterate through each pair of elements and store the common characters
    for (size_t i = 0; i < std::min(vector1.size(), vector2.size()); ++i) {
        const std::string& str1 = vector1[i];
        const std::string& str2 = vector2[i];

        std::string commonChars;
        
        // Find common characters from left to right
        size_t minLength = std::min(str1.length(), str2.length());
        for (size_t j = 0; j < minLength; ++j) {
            if (str1[j] == str2[j]) {
                commonChars += str1[j];
            } else {
                break; // Stop when a non-matching pair is found
            }
        }

        commonCharacters.push_back(commonChars);
    }

    return commonCharacters;
}

std::vector<int> countDecimalDigits(const std::vector<std::string>& numbers) {
    std::vector<int> decimalCounts;

    for (const std::string& number : numbers) {
        size_t decimalPointPos = number.find('.');

        // If a decimal point is found, count the characters after it
        int count = (decimalPointPos != std::string::npos) ? static_cast<int>(number.substr(decimalPointPos + 1).length()) : 0;
        decimalCounts.push_back(count);
    }

    return decimalCounts;
}

int main(int argc, char **argv) {
  int size = 1 << 15;
  unsigned int bytes = sizeof(float) * size;
  std::vector<std::string> double_string_vec = {};
  std::vector<std::string> float_string_vec = {};
  std::vector<std::string> err_string_vec = {};
  std::vector<double> err_vec = {};
  int precision = 30;
    std::string fileName = "output.txt";
  for (int seed = 1; seed < 11; seed++) {
        // Double array handling
      ReduceType datatype = REDUCE_DOUBLE;
      printf("\nReducing array of type %s\n", getReduceTypeString(datatype));
      float *data1 = (float*)malloc(bytes);
      generate_random_array(data1, size, seed);
      double gpu_result_double = (double)0.0;
        gpu_result_double = runTest<double>(argc, argv, datatype,data1);
        std::cout << "GPU_result_double: " << std::fixed <<  std::setprecision(precision) <<gpu_result_double << "\n"<< std::endl;
        std::string formatted_double = formatNumericToString(gpu_result_double, precision);
        double_string_vec.push_back(formatted_double);
      // for (int i = 0; i < 4; i++) {
      //     std::cout << "First: \n" << data2[i] << std::endl;
      // }
      // for (int i = 0; i < 4; i++) {
      //     std::cout << "second: \n" << (double)data2[i] << std::endl;
      // }



        // Float array handling
      datatype = REDUCE_FLOAT;
      printf("Reducing array of type %s\n", getReduceTypeString(datatype));
      float *data2 = (float *)malloc(bytes);
      generate_random_array(data2, size, seed);
      float gpu_result_float = (float)0.0f;  // Initialize outside the switch
        gpu_result_float = runTest<float>(argc, argv, datatype,data2);
        std::cout << "CPU_result_float: "<< std::fixed <<  std::setprecision(precision) << gpu_result_float << "\n" <<std::endl;
        std::string formatted_float = formatNumericToString(gpu_result_float, precision);
        float_string_vec.push_back(formatted_float);

        // Absolute error handling
        double err = std::abs((double)gpu_result_float - gpu_result_double);
      printf("Absolute error for seed %d: %.30lf\n", seed, err);
        std::string formatted_err = formatNumericToString(err, precision);
        err_string_vec.push_back(formatted_err);
        err_vec.push_back(err);


  }


  double mean = calculateMean(err_vec);
  double var = calculateVariance(err_vec);
  double stddev = calculateStandardDeviation(err_vec);
  std::vector<std::string> accumulate_vec = {formatNumericToString(mean, precision), formatNumericToString(var,
          precision), formatNumericToString(stddev, precision)};
std::vector<std::string> common_vec = findCommonCharacters(float_string_vec, double_string_vec);
std::vector<int> fraction_vec = countDecimalDigits(common_vec);
    appendVectorToFile(double_string_vec, fileName);
    appendVectorToFile(float_string_vec, fileName);
    appendVectorToFile(err_string_vec, fileName);
    appendVectorToFile(accumulate_vec, fileName);
    appendVectorToFile(common_vec, fileName);
    appendVectorToFile(fraction_vec, fileName);

}

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  if (whichKernel < 3) {
    threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
  }

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

  if (whichKernel >= 6) {
    blocks = std::min(maxBlocks, blocks);
  }
  std::cout << "Threads: " << threads << "\nBlocks: " << blocks << std::endl;

}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, int whichKernel, int testIterations,
                  bool cpuFinalReduction, int cpuFinalThreshold, T *h_odata,
                  T *d_idata, T *d_odata) {
  T gpu_result = 0;
  bool needReadBack = true;

  T *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks);

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;

    cudaDeviceSynchronize();

    reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    if (cpuFinalReduction) {
      // sum partial sums from each block on CPU
      // copy result from device to host
      cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T),
                 cudaMemcpyDeviceToHost);

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
        cudaMemcpy(d_intermediateSums, d_odata, s * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        reduce<T>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

        if (kernel < 3) {
          s = (s + threads - 1) / threads;
        } else {
          s = (s + (threads * 2 - 1)) / (threads * 2);
        }
      }

      if (s > 1) {
        // copy result from device to host
        cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost);

        for (int i = 0; i < s; i++) {
          gpu_result += h_odata[i];
        }

        needReadBack = false;
      }
    }

    cudaDeviceSynchronize();
  }

  if (needReadBack) {
    // copy final sum from device to host
    cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
  }
  cudaFree(d_intermediateSums);
  return gpu_result;
}

template <class T> T runTest(int argc, char **argv, ReduceType datatype, float* random_data) {
  int size = 1 << 15;   // number of elements to reduce
  std::cout << "Size: " << size << std::endl;
  int maxThreads = 128; // number of threads per block
  int whichKernel = 0;
  int maxBlocks = 1024;
  bool cpuFinalReduction = false;
  std::cout << (cpuFinalReduction ? "1-step" : "N-steps") << std::endl;
  int cpuFinalThreshold = 1;

  unsigned int bytes = size * sizeof(T);

  T *h_idata = (T *)malloc(bytes);
  for (int i = 0; i < size; i++) {
      h_idata[i] = (T) random_data[i];
  }

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks,
                         numThreads);

  if (numBlocks == 1) {
    cpuFinalThreshold = 1;
  }

  // allocate mem for the result on host side
  T *h_odata = (T *)malloc(numBlocks * sizeof(T));

  // allocate device memory and data
  T *d_idata = NULL;
  T *d_odata = NULL;

  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, numBlocks * sizeof(T));

  // copy data directly to device memory
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T), cudaMemcpyHostToDevice);

  // // warm-up
  reduce<T>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

  int testIterations = 1;

  T gpu_result = (T)0.0f;
  gpu_result =
      benchmarkReduce<T>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                         whichKernel, testIterations, cpuFinalReduction,
                         cpuFinalThreshold, h_odata, d_idata, d_odata);


    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);
  return gpu_result;
}
