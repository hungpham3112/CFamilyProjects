// Copyright 2019   J.-M. Chesneaux, P. Eberhart, F. Jezequel, J.-L. Lamotte, S. Zhou

// This file is part of CADNA.

// CADNA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// CADNA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with CADNA.  If not, see <http://www.gnu.org/licenses/>.
// variables globales
# include "cadna_gpu_float.h"
#include "cadna_gpu_double.h"

#ifdef HALF


#elif HALF2

#else
#define MAX_THREAD_PER_BLOCK 1024 // 512

//#define MAX_BLOCK_SIZE_X 1024// 512

#define MAX_THREAD (gridDim.x*blockDim.x * gridDim.y*blockDim.y * gridDim.z*blockDim.z)

#define CUDA_ERROR(cuda_call) {					\
    cudaError_t error = cuda_call;				\
    if(error != cudaSuccess){					\
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(error));  \
      exit(EXIT_FAILURE);					\
    }								\
  }

__device__ unsigned int _cadna_TauswortheStep(unsigned int, unsigned int,
					      unsigned int,
					      unsigned int,
					      unsigned int) __attribute__((always_inline));
__device__ inline unsigned int _cadna_TauswortheStep(unsigned int seed, unsigned int s1,
						     unsigned int s2,
						     unsigned int s3,
						     unsigned int m){
  unsigned int b = (((seed << s1) ^ seed) >> s2);
  return  (((seed & m) << s3) ^ b);
}

__device__ inline unsigned int _cadna_LCGStep(unsigned int, unsigned int,
					      unsigned int) __attribute__((always_inline));
__device__ inline unsigned int _cadna_LCGStep(unsigned int seed, unsigned int a,
					      unsigned int c){
  return (a * seed + c);
}

__device__ __shared__ unsigned int seed[MAX_THREAD_PER_BLOCK];
__device__ __shared__ unsigned char _cadna_random_counter[MAX_THREAD_PER_BLOCK];

__device__ void cadna_init_gpu()
{
  unsigned int idx, init;

  idx = // (gridDim.x*blockDim.x * gridDim.y*blockDim.y) * blockIdx.z +
    // (gridDim.x*blockDim.x) * blockIdx.y +
    // blockDim.x * blockIdx.x +
    (blockDim.x*blockDim.y) * threadIdx.z +
    blockDim.x * threadIdx.y +
    threadIdx.x ;
  init = (gridDim.x*blockDim.x * gridDim.y*blockDim.y) * blockIdx.z +
    (gridDim.x*blockDim.x) * blockIdx.y +
    blockDim.x * blockIdx.x +
    (blockDim.x*blockDim.y) * threadIdx.z +
    blockDim.x * threadIdx.y +
    threadIdx.x ;

  seed[idx] = init*1099087573UL;
  _cadna_random_counter[idx] = 0;
  
  // __syncthreads();
}

__device__ inline  unsigned int RANDOMGPU()
{

    unsigned int idx;
    idx = // (gridDim.x*blockDim.x * gridDim.y*blockDim.y) * blockIdx.z +
      // (gridDim.x*blockDim.x) * blockIdx.y +
      // blockDim.x * blockIdx.x +
      (blockDim.x*blockDim.y) * threadIdx.z +
      blockDim.x * threadIdx.y +
      threadIdx.x ;

    // /!\ N'est plus static sur GPU
    const unsigned int Taus1S1 = 13;
    const unsigned int Taus1S2 = 19;
    const unsigned int Taus1S3 = 12;
    const unsigned int Taus1M  = 429496729U;
    const unsigned int Taus2S1 = 2;
    const unsigned int Taus2S2 = 25;
    const unsigned int Taus2S3 = 4;
    const unsigned int Taus2M  = 4294967288U;
    const unsigned int Taus3S1 = 3;
    const unsigned int Taus3S2 = 11;
    const unsigned int Taus3S3 = 17;
    const unsigned int Taus3M  = 429496280U;
    const unsigned int LCGa    = 1664525;
    const unsigned int LCGc    = 1013904223U;

    unsigned int z1, z2, z3, z4;

    // À supposer qu'il est peut coûteux de modifier seed. On économise un tableau de _cadna_random.
    // Test : si le compteur est à 0 ou 32 :
    if((_cadna_random_counter[idx]&0xF)==0){
        _cadna_random_counter[idx] = 0;

        z1 = _cadna_TauswortheStep(seed[idx], Taus1S1, Taus1S2, Taus1S3, Taus1M);
        z2 = _cadna_TauswortheStep(seed[idx], Taus2S1, Taus2S2, Taus2S3, Taus2M);
        z3 = _cadna_TauswortheStep(seed[idx], Taus3S1, Taus3S2, Taus3S3, Taus3M);
        z4 = _cadna_LCGStep(seed[idx], LCGa, LCGc);
        seed[idx] = (z1^ z2 ^ z3 ^ z4);
    }

    return seed[idx]>>((_cadna_random_counter[idx]++)*2)&3;
}
#endif

/////////////////////////////////////////////////////

__device__ float_gpu_st operator+(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,b.x);
  else res.x=__fadd_rd(a.x,b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,b.y);
    res.z=__fadd_rd(a.z,b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,b.y);
    res.z=__fadd_ru(a.z,b.z);
  }

  res.error=a.error | b.error;

  return res;
}


__device__ float_gpu_st operator+(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,b);
  else res.x=__fadd_rd(a.x,b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,b);
    res.z=__fadd_rd(a.z,b);;
  }
  else {
    res.y=__fadd_rd(a.y,b);
    res.z=__fadd_ru(a.z,b);
  }
  res.error=a.error;
  return res;
}


__device__ float_gpu_st operator+(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,b.x);
  else res.x=__fadd_rd(a,b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,b.y);
    res.z=__fadd_rd(a,b.z);;
  }
  else {
    res.y=__fadd_rd(a,b.y);
    res.z=__fadd_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__ float_gpu_st operator+=(float_gpu_st& a, const float_gpu_st& b)
{
  //float_gpu_st res;
  unsigned char random;

  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fadd_ru(a.x,b.x);
  else a.x=__fadd_rd(a.x,b.x);

  if (random>>1) {
    a.y=__fadd_ru(a.y,b.y);
    a.z=__fadd_rd(a.z,b.z);;
  }
  else {
    a.y=__fadd_rd(a.y,b.y);
    a.z=__fadd_ru(a.z,b.z);
  }

  a.error=a.error | b.error;

  return a;
}


__device__ float_gpu_st operator+=(float_gpu_st& a, const float& b)
{
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__fadd_ru(a.x,b);
  else a.x=__fadd_rd(a.x,b);

  if (random>>1) {
    a.y=__fadd_ru(a.y,b);
    a.z=__fadd_rd(a.z,b);;
  }
  else {
    a.y=__fadd_rd(a.y,b);
    a.z=__fadd_ru(a.z,b);
  }
  //res.error=a.error;
  return a;
}



/////////////////////////////////////////////////////


__device__ float_gpu_st operator-(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ float_gpu_st operator-(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  res.error= a.error;

  return res;

}

__device__ float_gpu_st operator-(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  res.error= b.error;
  return res;
}

__device__ float_gpu_st operator-=(float_gpu_st& a, const float_gpu_st& b)
{
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fadd_ru(a.x,-b.x);
  else a.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    a.y=__fadd_ru(a.y,-b.y);
    a.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    a.y=__fadd_rd(a.y,-b.y);
    a.z=__fadd_ru(a.z,-b.z);
  }

  a.error= a.error | b.error;
  return a;
}


__device__ float_gpu_st operator-=(float_gpu_st& a, const float& b)
{
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__fadd_ru(a.x,-b);
  else a.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    a.y=__fadd_ru(a.y,-b);
    a.z=__fadd_rd(a.z,-b);;
  }
  else {
    a.y=__fadd_rd(a.y,-b);
    a.z=__fadd_ru(a.z,-b);
  }
  //res.error= a.error;

  return a;

}


/////////////////////////////////////////////////////

__device__ float_gpu_st operator*(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fmul_ru(a.x,b.x);
  else res.x=__fmul_rd(a.x,b.x);
  if (random>>1) {
    res.y=__fmul_ru(a.y,b.y);
    res.z=__fmul_rd(a.z,b.z);;
  }
  else {
    res.y=__fmul_rd(a.y,b.y);
    res.z=__fmul_ru(a.z,b.z);
  }


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  if (a.accuracy==DIGIT_NOT_COMPUTED	)
    a.approx_digit();
  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=(a.accuracy==0 &&  b.accuracy==0		 ) ? CADNA_MUL : 0;
  res.error=a.error | b.error | inst;
  return res;
}



__device__ float_gpu_st operator*(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fmul_ru(a.x,b);
  else res.x=__fmul_rd(a.x,b);
  if (random>>1) {
    res.y=__fmul_ru(a.y,b);
    res.z=__fmul_rd(a.z,b);;
  }
  else {
    res.y=__fmul_rd(a.y,b);
    res.z=__fmul_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ float_gpu_st operator*(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fmul_ru(a,b.x);
  else res.x=__fmul_rd(a,b.x);
  if (random>>1) {
    res.y=__fmul_ru(a,b.y);
    res.z=__fmul_rd(a,b.z);;
  }
  else {
    res.y=__fmul_rd(a,b.y);
    res.z=__fmul_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}
__device__ float_gpu_st operator*=(float_gpu_st& a, const float_gpu_st& b)
{
  //float_gpu_st res;
  unsigned char random;

  unsigned int inst;

  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fmul_ru(a.x,b.x);
  else a.x=__fmul_rd(a.x,b.x);
  if (random>>1) {
    a.y=__fmul_ru(a.y,b.y);
    a.z=__fmul_rd(a.z,b.z);;
  }
  else {
    a.y=__fmul_rd(a.y,b.y);
    a.z=__fmul_ru(a.z,b.z);
  }


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  if (a.accuracy==DIGIT_NOT_COMPUTED	)
    a.approx_digit();
  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=(a.accuracy==0 &&  b.accuracy==0		 ) ? CADNA_MUL : 0;
  a.error=a.error | b.error | inst;
  return a;
}



__device__ float_gpu_st operator*=(float_gpu_st& a, const float& b)
{
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fmul_ru(a.x,b);
  else a.x=__fmul_rd(a.x,b);
  if (random>>1) {
    a.y=__fmul_ru(a.y,b);
    a.z=__fmul_rd(a.z,b);;
  }
  else {
    a.y=__fmul_rd(a.y,b);
    a.z=__fmul_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


///////////////////////////////////////////

__device__ float_gpu_st operator/(const float_gpu_st& a, const float_gpu_st& b)
{
  unsigned int inst;
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fdiv_ru(a.x,b.x);
  else res.x=__fdiv_rd(a.x,b.x);
  if (random>>1) {
    res.y=__fdiv_ru(a.y,b.y);
    res.z=__fdiv_rd(a.z,b.z);;
  }
  else {
    res.y=__fdiv_rd(a.y,b.y);
    res.z=__fdiv_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  res.error=a.error | b.error | inst;

  return res;
}


__device__ float_gpu_st operator/(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fdiv_ru(a.x,b);
  else res.x=__fdiv_rd(a.x,b);
  if (random>>1) {
    res.y=__fdiv_ru(a.y,b);
    res.z=__fdiv_rd(a.z,b);;
  }
  else {
    res.y=__fdiv_rd(a.y,b);
    res.z=__fdiv_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}


__device__ float_gpu_st operator/(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__fdiv_ru(a,b.x);
  else res.x=__fdiv_rd(a,b.x);
  if (random>>1) {
    res.y=__fdiv_ru(a,b.y);
    res.z=__fdiv_rd(a,b.z);;
  }
  else {
    res.y=__fdiv_rd(a,b.y);
    res.z=__fdiv_ru(a,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_DIV : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

__device__ float_gpu_st operator/=(float_gpu_st& a, const float_gpu_st& b)
{
  unsigned int inst;
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fdiv_ru(a.x,b.x);
  else a.x=__fdiv_rd(a.x,b.x);
  if (random>>1) {
    a.y=__fdiv_ru(a.y,b.y);
    a.z=__fdiv_rd(a.z,b.z);;
  }
  else {
    a.y=__fdiv_rd(a.y,b.y);
    a.z=__fdiv_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  a.error=a.error | b.error | inst;

  return a;
}


__device__ float_gpu_st operator/=(float_gpu_st& a, const float& b)
{
  //float_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__fdiv_ru(a.x,b);
  else a.x=__fdiv_rd(a.x,b);
  if (random>>1) {
    a.y=__fdiv_ru(a.y,b);
    a.z=__fdiv_rd(a.z,b);;
  }
  else {
    a.y=__fdiv_rd(a.y,b);
    a.z=__fdiv_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


///////////////////////////////////////

__device__ int operator==(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  return res.computedzero();
}

__device__ int operator==(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  return res.computedzero();
}

__device__ int operator==(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  return res.computedzero();
}

///////////////////////////////////////

///////////////////////////////////////

__device__ int operator!=(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  return !res.computedzero();
}

__device__ int operator!=(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  return !res.computedzero();
}

__device__ int operator!=(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  return !res.computedzero();
}

///////////////////////////////////////

__device__ int operator>(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( a.x + a.y + a.z ) >	( b.x + b.y + b.z ));
}


__device__ int operator>(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }


  return !r && ( ( a.x + a.y + a.z ) > 3*b );
}



__device__ int operator>(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*a  >	( b.x + b.y + b.z ));
}




///////////////////////////////////////


__device__ int operator>=(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( a.x + a.y + a.z ) >=	( b.x + b.y + b.z ));
}


__device__ int operator>=(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( a.x + a.y + a.z ) >=	3*b);
}



__device__ int operator>=(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*a  >=	( b.x + b.y + b.z ));
}




///////////////////////////////////////

__device__ int operator<(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( a.x + a.y + a.z ) <	( b.x + b.y + b.z ));
}


__device__ int operator<(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( ( a.x + a.y + a.z ) < 3*b );
}



__device__ int operator<(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*a  <	( b.x + b.y + b.z ));
}




///////////////////////////////////////


__device__ int operator<=(const float_gpu_st& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b.x);
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b.y);
    res.z=__fadd_rd(a.z,-b.z);;
  }
  else {
    res.y=__fadd_rd(a.y,-b.y);
    res.z=__fadd_ru(a.z,-b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( a.x + a.y + a.z ) <=	( b.x + b.y + b.z ));
}


__device__ int operator<=(const float_gpu_st& a, const float& b)
{
  float_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a.x,-b);
  else res.x=__fadd_rd(a.x,-b);

  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);
    res.z=__fadd_rd(a.z,-b);;
  }
  else {
    res.y=__fadd_rd(a.y,-b);
    res.z=__fadd_ru(a.z,-b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( a.x + a.y + a.z ) <=	3*b);
}



__device__ int operator<=(const float& a, const float_gpu_st& b)
{
  float_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__fadd_ru(a,-b.x);
  else res.x=__fadd_rd(a,-b.x);

  if (random>>1) {
    res.y=__fadd_ru(a,-b.y);
    res.z=__fadd_rd(a,-b.z);;
  }
  else {
    res.y=__fadd_rd(a,-b.y);
    res.z=__fadd_ru(a,-b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*a  <=	( b.x + b.y + b.z ));
}

///////////////////////////////////////
 __device__  float_gpu_st fabsf(const  float_gpu_st& a) 
{ 
   float_gpu_st res; 
   res.x = fabsf(a.x); 
   res.y = fabsf(a.y); 
   res.z = fabsf(a.z); 
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

 __device__  float_gpu_st fabs(const  float_gpu_st& a) 
{ 
   float_gpu_st res; 
   res.x = fabs(a.x); 
   res.y = fabs(a.y); 
   res.z = fabs(a.z); 
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

 __device__  float_gpu_st sqrtf(const  float_gpu_st& a) 
{ 
  float_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__fsqrt_ru(a.x);	    
  else res.x=__fsqrt_rd(a.x);

  if (random>>1) {
     res.y=__fsqrt_ru(a.y);				 
     res.z=__fsqrt_rd(a.z);				 
  }							 
  else {							 
    res.y=__fsqrt_rd(a.y);				 
    res.z=__fsqrt_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}
///////////////////////////////////////
__device__  float_gpu_st fmaxf(const float& a, const float_gpu_st& b) 
{ 
  float_gpu_st res;   
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a,-b.x);			 
  else res.x=__fadd_rd(a,-b.x);
  if (random>>1) {						 
    res.y=__fadd_ru(a,-b.y);				 
    res.z=__fadd_rd(a,-b.z);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a,-b.y);				 
    res.z=__fadd_ru(a,-b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=7;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*a > (b.x + b.y + b.z)) {
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=7;
      res.error=0;	
    }
    else							 
      res=b;	
  }
  return(res); 
}


__device__  float_gpu_st fmaxf(const  float_gpu_st& a, const  float& b) 
{ 
  float_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a.x,-b);			 
  else res.x=__fadd_rd(a.x,-b);
  if  (random>>1) {
    res.y=__fadd_ru(a.y,-b);				 
    res.z=__fadd_rd(a.z,-b);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a.y,-b);				 
    res.z=__fadd_ru(a.z,-b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=7;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if (( a.x + a.y + a.z ) > 3.f*b ) {
	res=a;	
    }
    else {								 
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=7;
      res.error=0;	
    }
  }
  return(res); 
}


__device__  float_gpu_st fmaxf(const  float_gpu_st& a, const  float_gpu_st& b) 
{ 
  float_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a.x,-b.x);			 
  else res.x=__fadd_rd(a.x,-b.x);				 
  if (random>>1) {						 
    res.y=__fadd_ru(a.y,-b.y);				 
    res.z=__fadd_rd(a.z,-b.z);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a.y,-b.y);				 
    res.z=__fadd_ru(a.z,-b.z);					
  }			
  if (res.isnumericalnoise()){
    if (a.accuracy==DIGIT_NOT_COMPUTED)
      a.nb_significant_digit();						 
    if (b.accuracy==DIGIT_NOT_COMPUTED)
      b.nb_significant_digit();
    if (a.accuracy > b.accuracy ){					 
	res=a;		
        res.error=a.error|CADNA_BRANCHING;
        }			 							 
    else {			 					 
	res=b;	
        res.error=b.error|CADNA_BRANCHING;
        }
  }
  else { 
    if (( a.x + a.y + a.z ) > ( b.x + b.y + b.z )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

//////

__device__  float_gpu_st fminf(const float& a, float_gpu_st& b) 
{ 
  float_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a,-b.x);			 
  else res.x=__fadd_rd(a,-b.x);
  if (random>>1) {						 
    res.y=__fadd_ru(a,-b.y);				 
    res.z=__fadd_rd(a,-b.z);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a,-b.y);				 
    res.z=__fadd_ru(a,-b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=7;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*a < (b.x + b.y + b.z)) {
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=7;
      res.error=0;	
    }
    else							 
      res=b;	
  }
  return(res); 
}


__device__  float_gpu_st fminf(const  float_gpu_st& a, const  float& b) 
{ 
  float_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a.x,-b);			 
  else res.x=__fadd_rd(a.x,-b);
  if (random>>1) {
    res.y=__fadd_ru(a.y,-b);				 
    res.z=__fadd_rd(a.z,-b);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a.y,-b);				 
    res.z=__fadd_ru(a.z,-b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=7;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if (( a.x + a.y + a.z ) < 3.f*b ) {
	res=a;	
    }
    else {								 
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=7;
      res.error=0;	
    }
  }
  return(res); 
}



__device__  float_gpu_st fminf(const  float_gpu_st& a, const  float_gpu_st& b) 
{ 
  float_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__fadd_ru(a.x,-b.x);			 
  else res.x=__fadd_rd(a.x,-b.x);

  if (random>>1) {				 
    res.y=__fadd_ru(a.y,-b.y);				 
    res.z=__fadd_rd(a.z,-b.z);				 
  }					       	 
  else {							 
    res.y=__fadd_rd(a.y,-b.y);				 
    res.z=__fadd_ru(a.z,-b.z);					
  }			
  if (res.isnumericalnoise()){
    if (a.accuracy==DIGIT_NOT_COMPUTED)
      a.nb_significant_digit();						 
    if (b.accuracy==DIGIT_NOT_COMPUTED)
      b.nb_significant_digit();
    if (a.accuracy > b.accuracy ){					 
	res=a;		
        res.error=a.error|CADNA_BRANCHING;
        }			 							 
    else {			 					 
	res=b;	
        res.error=b.error|CADNA_BRANCHING;
        }
  }
  else { 
    if (( a.x + a.y + a.z ) < ( b.x + b.y + b.z )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

///////////////////////////////////////


__device__ void float_gpu_st::modify(const int &a)
{
  accuracy |=a;
}


__device__ float_gpu_st& float_gpu_st::operator=(const float &a)
{
  x=a;
  y=a;
  z=a;
  accuracy=7;
  error=0;
  return *this ;
}

__device__ int  float_gpu_st::nb_significant_digit() const
{
  float x0,x1,x2,xx;

  xx=x+y+z;

  accuracy=0;
  if (xx==0.0){
    if ((x==y) &&(x==z) ) accuracy=7;
  }
  else {
    xx=3/xx;
    x0=x*xx-1;
    x1=y*xx-1;
    x2=z*xx-1;
    //FJ 4 Mar 2014:
    float yy=(x0*x0+x1*x1+x2*x2)*(float)3.08546617;
    if (yy<=1.e-14)  accuracy=7;
    else {
      yy= -log10(yy);
      if (yy>=0.) accuracy=(int)((yy+(float)1.)*(float)0.5);
    }
  }
  return accuracy;
}

