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
#include <numeric>
#include <tuple>

#include <cadna.h>
#include <cadna_gpu.cu>

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
    mySum = mySum + __shfl_down_sync(mask, mySum, offset);
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
    cadna_init_gpu();
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    sdata[tid] = g_idata[i];
  } else {
      sdata[tid] = (T)0.0;
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
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce1(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    sdata[tid] = g_idata[i];
  } else {
      sdata[tid] = (T)0.0;
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

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    sdata[tid] = g_idata[i];
  } else {
      sdata[tid] = (T)0.0;
  }

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = sdata[tid] + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum;
  if (i < n) {
    mySum = g_idata[i];
  } else {
      mySum = (T)0.0;
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
template <class T, unsigned int blockSize>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum;
  if (i < n) {
    mySum = g_idata[i];
  } else {
      mySum = (T)0.0;
  }

  if (i + blockSize < n) mySum = mySum + g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum = mySum + sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum = mySum + (T)tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize>
__global__ void reduce5(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

  T mySum;
  if (i < n) {
    mySum = g_idata[i];
  } else {
      mySum = (T)0.0;
  }

  if (i + blockSize < n) mySum = mySum + g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum = mySum + sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum = mySum + (T)tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  T mySum = (T)0.0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum = mySum + g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum = mySum + g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum = mySum + g_idata[i];
      i += gridSize;
    }
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum = mySum + (T)tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = (T)0.0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      mySum = mySum + g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        mySum = mySum + g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      mySum = mySum + g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
      (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = mySum;
  }
}


// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
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
  case 8:
    reduce0<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
    break;
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

  case 4:
    switch (threads) {
    case 512:
      reduce4<T, 512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 256:
      reduce4<T, 256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 128:
      reduce4<T, 128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 64:
      reduce4<T, 64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 32:
      reduce4<T, 32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 16:
      reduce4<T, 16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 8:
      reduce4<T, 8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 4:
      reduce4<T, 4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 2:
      reduce4<T, 2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;

    case 1:
      reduce4<T, 1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
      break;
    }

    break;
    case 5:
      switch (threads) {
        case 512:
          reduce5<T, 512>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 256:
          reduce5<T, 256>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 128:
          reduce5<T, 128>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 64:
          reduce5<T, 64>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 32:
          reduce5<T, 32>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 16:
          reduce5<T, 16>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 8:
          reduce5<T, 8>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 4:
          reduce5<T, 4>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 2:
          reduce5<T, 2>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;

        case 1:
          reduce5<T, 1>
              <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
          break;
      }

      break;

  case 6:
    if (isPow2(size)) {
      switch (threads) {
      case 512:
        reduce6<T, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduce6<T, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduce6<T, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduce6<T, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduce6<T, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduce6<T, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduce6<T, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduce6<T, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduce6<T, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduce6<T, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      }
    } else {
      switch (threads) {
      case 512:
        reduce6<T, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduce6<T, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduce6<T, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduce6<T, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduce6<T, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduce6<T, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduce6<T, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduce6<T, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduce6<T, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduce6<T, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      }
    }

    break;

  case 7:
    // For reduce7 kernel we require only blockSize/warpSize
    // number of elements in shared memory
    smemSize = ((threads / 32) + 1) * sizeof(T);
    if (isPow2(size)) {
      switch (threads) {
      case 1024:
        reduce7<T, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        reduce7<T, 512, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduce7<T, 256, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduce7<T, 128, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduce7<T, 64, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduce7<T, 32, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduce7<T, 16, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduce7<T, 8, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduce7<T, 4, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduce7<T, 2, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduce7<T, 1, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      }
    } else {
      switch (threads) {
      case 1024:
        reduce7<T, 1024, true>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      case 512:
        reduce7<T, 512, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 256:
        reduce7<T, 256, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 128:
        reduce7<T, 128, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 64:
        reduce7<T, 64, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 32:
        reduce7<T, 32, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 16:
        reduce7<T, 16, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 8:
        reduce7<T, 8, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 4:
        reduce7<T, 4, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 2:
        reduce7<T, 2, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;

      case 1:
        reduce7<T, 1, false>
            <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
        break;
      }
    }

    break;
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

    // default:
    //   printf("thread block size of < 64 is not supported for this kernel\n");
    //   break;
    // }
    // break;
  }
}

enum ReduceType {
  REDUCE_FLOAT,
  REDUCE_DOUBLE,
  REDUCE_FLOAT_CADNA,
  REDUCE_DOUBLE_CADNA,
};

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  if (whichKernel < 3 || whichKernel == 8) {
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

template <class T>
void generate_random_array(T *array, const int &size, const int &seed) {
    std::cout << "Seed: " << seed << std::endl;
  std::mt19937 engine(seed);
  std::normal_distribution<T> generator(0, 1);
  for (int i = 0; i < size; i++)
    array[i] = generator(engine);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward

#define MAX_BLOCK_DIM_SIZE 65535

extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

const char *getReduceTypeString(const ReduceType type) {
  switch (type) {
  case REDUCE_FLOAT:
    return "float";
  case REDUCE_DOUBLE:
    return "double";
  case REDUCE_FLOAT_CADNA:
    return "float_st";
  case REDUCE_DOUBLE_CADNA:
    return "double_st";
  default:
    return "unknown";
  }
}

template <typename T>
std::vector<std::string> convertVectorElementsToString(const std::vector<T>& inputVector) {
    std::vector<std::string> result;

    // Convert each element in the input vector to string
    for (const auto& element : inputVector) {
        std::stringstream ss;
        ss << element;
        result.push_back(ss.str());
    }

    return result;
}

void mergeAndAppendToCSV(const std::string &fileName, const std::vector<std::vector<std::string>> &vectors) {
    // Open file for writing
    std::ofstream file(fileName, std::ios::app);


    // Check if the file is valid
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Iterate through vectors and write to CSV
    for (size_t i = 0; i < vectors[0].size(); ++i) {
        for (size_t j = 0; j < vectors.size(); ++j) {
            file << vectors[j][i];

            // If not the last vector, add a comma separator
            if (j < vectors.size() - 1) {
                file << ",";
            }
        }

        file << "\n";
    }
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
          t = t + samples[i];
          T diff = ((T)(i + 1) * samples[i]) - t;
          variance = variance + (diff * diff) / ((i + 1.0) *i);
     }

     return variance / (T)(size - 1);
}

template <typename T>
T calculateStandardDeviation(std::vector<T>& samples) {
    T variance = calculateVariance(samples);
    return sqrt(variance);
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

std::vector<std::string> extractDecimalDigits(const std::vector<std::string>& numbers) {
    std::vector<std::string> decimalParts;

    for (const std::string& number : numbers) {
        size_t decimalPointPos = number.find('.');

        // If a decimal point is found, extract the characters after it
        std::string decimalPart = (decimalPointPos != std::string::npos) ? number.substr(decimalPointPos + 1) : "";
        decimalParts.push_back(std::to_string(decimalPart.length()));
    }

    return decimalParts;
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T, class U>
T benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                  int maxBlocks, int whichKernel, int testIterations,
                  bool cpuFinalReduction, int cpuFinalThreshold, T *h_odata,
                  U *d_idata, U *d_odata) {
  T gpu_result = (T)0.0f;
  bool needReadBack = true;

  U *d_intermediateSums;
  cudaMalloc((void **)&d_intermediateSums, sizeof(T) * numBlocks);

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = (T)0.0f;

    cudaDeviceSynchronize();

    reduce<U>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    if (cpuFinalReduction) {
      // sum partial sums from each block on CPU
      // copy result from device to host
      cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T),
                 cudaMemcpyDeviceToHost);

      for (int i = 0; i < numBlocks; i++) {
        gpu_result = gpu_result + h_odata[i];
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
        reduce<U>(s, threads, blocks, kernel, d_intermediateSums, d_odata);

        if (kernel < 3 || kernel == 8) {
          s = (s + threads - 1) / threads;
        } else {
          s = (s + (threads * 2 - 1)) / (threads * 2);
        }
      }

      if (s > 1) {
        // copy result from device to host
        cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost);

        for (int i = 0; i < s; i++) {
          gpu_result = gpu_result + h_odata[i];
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

template <class T, class U> 
std::tuple<int, int, int, T> runTest(int argc, char **argv, ReduceType datatype, T* random_data, int step_kind, int kernel, int thread) {
  int size = 1 << 15;   // number of elements to reduce
  std::cout << "Size: " << size << std::endl;
  int maxThreads = thread; // number of threads per block
  int whichKernel = kernel;
  std::cout << "Kernel: " << kernel << std::endl;
  int maxBlocks = 1024;
  bool cpuFinalReduction = step_kind ? false : true;
  std::cout << (cpuFinalReduction ? "N-steps" : "1-step") << std::endl;
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
      U *d_idata = NULL;
      U *d_odata = NULL;

      cudaMalloc((void **)&d_idata, bytes);
      cudaMalloc((void **)&d_odata, numBlocks * sizeof(T));

      // copy data directly to device memory
      cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(T), cudaMemcpyHostToDevice);

      // // warm-up
      reduce<U>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

      int testIterations = 1;

      T gpu_result = (T)0.0f;
      gpu_result =
          benchmarkReduce<T, U>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                             whichKernel, testIterations, cpuFinalReduction,
                             cpuFinalThreshold, h_odata, d_idata, d_odata);
    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);
  return std::make_tuple(numThreads, numBlocks, whichKernel, gpu_result);


}

template <typename T>
std::string formatNumericToString(T value, size_t precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

int main(int argc, char **argv) {
  int size = 1 << 15;
  unsigned int bytes = sizeof(float) * size;

  for (int kernel = 7; kernel < 8; kernel++) {
  for (int thread = 32; thread <= 256; thread*=2) {
  for (int step = 0; step < 2; step++) {
    std::vector<double_st> double_vec = {};
    std::vector<int> double_sig_fig = {};
    std::vector<float_st> float_vec = {};
    std::vector<int> float_sig_fig = {};
    std::vector<double_st> err_vec = {};
    std::vector<int> err_sig_fig = {};
    std::vector<int> thread_vec = {};
    std::vector<int> block_vec = {};
    std::vector<int> test_kernel_vec = {};
    std::vector<int> reference_kernel_vec = {};
    std::vector<char> step_vec = {};
    std::vector<int> seed_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int precision = 30;
    std::string fileName = "output.csv";
    for (auto seed: seed_vec) {
          // Double array handling
        ReduceType datatype = REDUCE_DOUBLE_CADNA;
        printf("\nReducing array of type %s\n", getReduceTypeString(datatype));
        float *data = (float*)malloc(bytes);
        generate_random_array(data, size, seed);
        double_st *data1 = (double_st *)malloc(sizeof(double_st) * size);
        for (int i = 0; i < size; i++) {
            data1[i] = (double_st)data[i];
        }
        auto [thread1, block1, kernel1, gpu_result_double] = runTest<double_st, double_gpu_st>(argc, argv, datatype,data1, step, 8, thread);
          std::cout << "GPU_result_double: " << std::fixed <<  std::setprecision(precision) <<gpu_result_double << "\n"<< std::endl;
          double_vec.push_back(gpu_result_double);
          double_sig_fig.push_back(gpu_result_double.nb_significant_digit());


          // Float array handling
        datatype = REDUCE_FLOAT_CADNA;
        printf("Reducing array of type %s\n", getReduceTypeString(datatype));
        float_st *data2 = (float_st *)malloc(sizeof(float_st) * size);
        for (int i = 0; i < size; i++) {
            data2[i] = (float_st)data[i];
        }
        auto [thread2, block2, kernel2, gpu_result_float] = runTest<float_st, float_gpu_st>(argc, argv, datatype,data2, step, kernel, thread);
          std::cout << "GPU_result_float: "<< std::fixed <<  std::setprecision(precision) << gpu_result_float << "\n" <<std::endl;
          float_vec.push_back(gpu_result_float);
          float_sig_fig.push_back(gpu_result_float.nb_significant_digit());

          // Absolute error handling
          double_st err = fabs((double_st)gpu_result_float - gpu_result_double);
          std::cout << "Absolute error for seed " << seed << ": " << err << "\n";
          err_vec.push_back(err);
          err_sig_fig.push_back(err.nb_significant_digit());

        // miscellaneous
        thread_vec.push_back(thread2);
        block_vec.push_back(block2);
        test_kernel_vec.push_back(kernel2);
        reference_kernel_vec.push_back(kernel1);
        step_vec.push_back(step ? 'N' : '1');
        double err = std::abs((double)gpu_result_float - gpu_result_double);
        printf("Absolute error for seed %d: %s\n", seed, strp(err));
        std::string formatted_err = formatNumericToString(err, precision);
        err_string_vec.push_back(formatted_err);
        err_vec.push_back(err);

    }


    double_st mean = calculateMean(err_vec);
    double_st var = calculateVariance(err_vec);
    double_st stddev = calculateStandardDeviation(err_vec);
    std::vector<double_st> accumulate_vec = {mean, var, stddev};
    std::vector<std::string> common_vec = findCommonCharacters(convertVectorElementsToString(float_vec), convertVectorElementsToString(double_vec));
    std::vector<std::string> fraction_vec = extractDecimalDigits(common_vec);
    std::vector<std::vector<std::string>> dataframe = {
                                      convertVectorElementsToString(thread_vec), 
                                      convertVectorElementsToString(block_vec), 
                                      convertVectorElementsToString(step_vec), 
                                      convertVectorElementsToString(seed_vec), 
                                      convertVectorElementsToString(test_kernel_vec), 
                                      convertVectorElementsToString(float_vec), 
                                      convertVectorElementsToString(float_sig_fig), 
                                      convertVectorElementsToString(reference_kernel_vec), 
                                      convertVectorElementsToString(double_vec), 
                                      convertVectorElementsToString(double_sig_fig), 
                                      convertVectorElementsToString(err_vec), 
                                      convertVectorElementsToString(err_sig_fig), 
                                      convertVectorElementsToString(common_vec), 
                                      convertVectorElementsToString(fraction_vec)
                                      };
    mergeAndAppendToCSV(fileName, dataframe);
  }
  }
}
}