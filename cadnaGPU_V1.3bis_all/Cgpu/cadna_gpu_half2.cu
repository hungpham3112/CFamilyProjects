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
# include "cadna_gpu_half2.h"
#define MAX_THREAD_PER_BLOCK 1024 // 512
/*
__device__ __half2 INF2;
__device__ __half2 SUP2;

//#define MAX_BLOCK_SIZE_X 1024// 512

//#define INF2 __float2half2(0.9990234375f) // 1-2*u
//#define SUP2 __float2half2(1.0009765625f) // 1+2*u





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
  INF2 = __float2half2_rn(0.9990234375f);
  SUP2 = __float2half2_rn(1.0009765625f);

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
*/
/////////////////////////////////////////////////////
__device__ __half2 __float2half2_rd(const float a)
{

	half2 x = __float2half2_rn(a);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __float2half2_ru(const float a)
{
	half2 x = __float2half2_rn(a);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __floats2half2_rd(const float a, const float b)
{

	half2 x = __floats2half2_rn(a, b);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __floats2half2_ru(const float a, const float b)
{
	half2 x = __floats2half2_rn(a, b);
	x = __hmul2(x,SUP2);
	return x;
}



__device__ __half2 __hadd2_rd(const __half2 a, const __half2 b)
{

	half2 x = __hadd2(a,b);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __hadd2_ru(const __half2 a, const __half2 b)
{
	half2 x = __hadd2(a,b);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __hadd2_sat_rd(const __half2 a, const __half2 b)
{

	half2 x = __hadd2_sat(a,b);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __hadd2_sat_ru(const __half2 a, const __half2 b)
{
	half2 x = __hadd2_sat(a,b);
	x = __hmul2(x,SUP2);
	return x;
}
__device__ __half2 __hsub2_rd(const __half2 a, const __half2 b)
{

	half2 x = __hsub2(a,b);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __hsub2_ru(const __half2 a, const __half2 b)
{
	half2 x = __hsub2(a,b);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __hsub2_sat_rd(const __half2 a, const __half2 b)
{

	half2 x = __hsub2_sat(a,b);
	x = __hmul2(x,INF2);
	return x;
}

__device__ __half2 __hsub2_sat_ru(const __half2 a, const __half2 b)
{
	half2 x = __hsub2_sat(a,b);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __hmul2_rd(const __half2 a, const __half2 b)
{
	half2 x = __hmul2(a,b);
	x = __hmul2(x,INF2);
	return x;
}
__device__ __half2 __hmul2_ru(const __half2 a, const __half2 b)
{
	half2 x = __hmul2(a,b);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __hmul2_sat_rd(const __half2 a, const __half2 b)
{
	half2 x = __hmul2_sat(a,b);
	x = __hmul2(x,INF2);
	return x;
}
__device__ __half2 __hmul2_sat_ru(const __half2 a, const __half2 b)
{
	half2 x = __hmul2_sat(a,b);
	x = __hmul2(x,SUP2);
	return x;
}
__device__ __half2 __h2div_rd(const __half2 a, const __half2 b)
{
	half2 x = __h2div(a,b);
	x = __hmul2(x,INF2);
	return x;
}
__device__ __half2 __h2div_ru(const __half2 a, const __half2 b)
{
	half2 x = __h2div(a,b);
	x = __hmul2(x,SUP2);
	return x;
}

__device__ __half2 __hneg2_rd(const __half2 a)
{
	half2 x = __hneg2(a);
	//x = __hmul2(x,INF2);
	return x;
}
__device__ __half2 __hneg2_ru(const __half2 a)
{
	half2 x = __hneg2(a);
	//x = __hmul2(x,SUP2);
	return x;
}
__device__ __half2  __h2sqrt_ru(const half2 a)
{
	return __hmul2(h2sqrt(a),SUP2);
}
__device__ __half2  __h2sqrt_rd(const half2 a)
{
	return __hmul2(h2sqrt(a),INF2);
}

__device__ __half2  __hfma2_ru(const half2 a,const half2 b, const half2 c)
{
	return __hmul2(__hfma2(a,b,c),SUP2);
}
__device__ __half2  __hfma2_rd(const half2 a,const half2 b, const half2 c)
{
	return __hmul2(__hfma2(a,b,c),INF2);
}

__device__ __half2  __hfma2_sat_ru(const half2 a,const half2 b, const half2 c)
{
	return __hmul2(__hfma2_sat(a,b,c),SUP2);
}
__device__ __half2  __hfma2_sat_rd(const half2 a,const half2 b, const half2 c)
{
	return __hmul2(__hfma2_sat(a,b,c),INF2);
}

__device__ __half2  h2sin_ru(const half2 a)
{
	return __hmul2(h2sin(a),SUP2);
}
__device__ __half2  h2sin_rd(const half2 a)
{
	return __hmul2(h2sin(a),INF2);
}
__device__ __half2  h2cos_ru(const half2 a)
{
	return __hmul2(h2cos(a),SUP2);
}
__device__ __half2  h2cos_rd(const half2 a)
{
	return __hmul2(h2cos(a),INF2);
}
__device__ __half2  h2exp_ru(const half2 a)
{
	return __hmul2(h2exp(a),SUP2);
}
__device__ __half2  h2exp_rd(const half2 a)
{
	return __hmul2(h2exp(a),INF2);
}
__device__ __half2  h2log_ru(const half2 a)
{
	return __hmul2(h2log(a),SUP2);
}
__device__ __half2  h2log_rd(const half2 a)
{
	return __hmul2(h2log(a),INF2);
}
__device__ __half2  h2log2_ru(const half2 a)
{
	return __hmul2(h2log2(a),SUP2);
}
__device__ __half2  h2log2_rd(const half2 a)
{
	return __hmul2(h2log2(a),INF2);
}
__device__ __half2  h2log10_ru(const half2 a)
{
	return __hmul2(h2log10(a),SUP2);
}
__device__ __half2  h2log10_rd(const half2 a)
{
	return __hmul2(h2log10(a),INF2);
}
__device__ __half2  h2rcp_ru(const half2 a)
{
	return __hmul2(h2rcp(a),SUP2);
}
__device__ __half2  h2rcp_rd(const half2 a)
{
	return __hmul2(h2rcp(a),INF2);
}



//////////////////////////////////////////////////

__device__ half2_gpu_st operator+(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a.x,b.x);
  else res.x=__hadd2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd2_ru(a.y,b.y);
    res.z=__hadd2_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd2_rd(a.y,b.y);
    res.z=__hadd2_ru(a.z,b.z);
  }

  res.error=a.error | b.error;

  return res;
}


__device__ half2_gpu_st operator+(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a.x,b);
  else res.x=__hadd2_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd2_ru(a.y,b);
    res.z=__hadd2_rd(a.z,b);;
  }
  else {
    res.y=__hadd2_rd(a.y,b);
    res.z=__hadd2_ru(a.z,b);
  }
  res.error=a.error;
  return res;
}


__device__ half2_gpu_st operator+(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a,b.x);
  else res.x=__hadd2_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd2_ru(a,b.y);
    res.z=__hadd2_rd(a,b.z);;
  }
  else {
    res.y=__hadd2_rd(a,b.y);
    res.z=__hadd2_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__ half2_gpu_st operator+=(half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a.x,b.x);
  else res.x=__hadd2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd2_ru(a.y,b.y);
    res.z=__hadd2_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd2_rd(a.y,b.y);
    res.z=__hadd2_ru(a.z,b.z);
  }

  res.error=a.error | b.error;
  a = res;
  return res;
}


__device__ half2_gpu_st operator+=(half2_gpu_st& a, const half2& b)
{
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__hadd2_ru(a.x,b);
  else a.x=__hadd2_rd(a.x,b);

  if (random>>1) {
    a.y=__hadd2_ru(a.y,b);
    a.z=__hadd2_rd(a.z,b);;
  }
  else {
    a.y=__hadd2_rd(a.y,b);
    a.z=__hadd2_ru(a.z,b);
  }
  //res.error=a.error;
  return a;
}


__device__  half2_gpu_st __hadd2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a.x,b.x);
  else res.x=__hadd2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd2_ru(a.y,b.y);
    res.z=__hadd2_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd2_rd(a.y,b.y);
    res.z=__hadd2_ru(a.z,b.z);
  }

  res.error=a.error | b.error;

  return res;
}

__device__ half2_gpu_st __hadd2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a.x,b);
  else res.x=__hadd2_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd2_ru(a.y,b);
    res.z=__hadd2_rd(a.z,b);;
  }
  else {
    res.y=__hadd2_rd(a.y,b);
    res.z=__hadd2_ru(a.z,b);
  }
  res.error=a.error;
  return res;
}


__device__ half2_gpu_st __hadd2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_ru(a,b.x);
  else res.x=__hadd2_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd2_ru(a,b.y);
    res.z=__hadd2_rd(a,b.z);;
  }
  else {
    res.y=__hadd2_rd(a,b.y);
    res.z=__hadd2_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__  half2_gpu_st __hadd2_sat(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_sat_ru(a.x,b.x);
  else res.x=__hadd2_sat_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd2_sat_ru(a.y,b.y);
    res.z=__hadd2_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd2_sat_rd(a.y,b.y);
    res.z=__hadd2_sat_ru(a.z,b.z);
  }
	



  res.error=a.error | b.error;

  return res;
}

__device__ half2_gpu_st __hadd2_sat(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_sat_ru(a.x,b);
  else res.x=__hadd2_sat_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd2_sat_ru(a.y,b);
    res.z=__hadd2_sat_rd(a.z,b);;
  }
  else {
    res.y=__hadd2_sat_rd(a.y,b);
    res.z=__hadd2_sat_ru(a.z,b);
  }



  res.error=a.error;
  return res;
}


__device__ half2_gpu_st __hadd2_sat(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd2_sat_ru(a,b.x);
  else res.x=__hadd2_sat_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd2_sat_ru(a,b.y);
    res.z=__hadd2_sat_rd(a,b.z);;
  }
  else {
    res.y=__hadd2_sat_rd(a,b.y);
    res.z=__hadd2_sat_ru(a,b.z);
  }

 


  res.error=b.error;
  return res;
}




/////////////////////////////////////////////////////


__device__ half2_gpu_st operator-(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half2_gpu_st operator-(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half2_gpu_st operator-(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}
__device__ half2_gpu_st operator-=(half2_gpu_st& a, const half2_gpu_st& b)
{
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hsub2_ru(a.x,b.x);
  else a.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    a.y=__hsub2_ru(a.y,b.y);
    a.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    a.y=__hsub2_rd(a.y,b.y);
    a.z=__hsub2_ru(a.z,b.z);
  }

  a.error= a.error | b.error;
  return a;
}


__device__ half2_gpu_st operator-=(half2_gpu_st& a, const half2& b)
{
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__hsub2_ru(a.x,b);
  else a.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    a.y=__hsub2_ru(a.y,b);
    a.z=__hsub2_rd(a.z,b);;
  }
  else {
    a.y=__hsub2_rd(a.y,b);
    a.z=__hsub2_ru(a.z,b);
  }
  //res.error= a.error;

  return a;

}


__device__ half2_gpu_st __hsub2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half2_gpu_st __hsub2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half2_gpu_st __hsub2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}

__device__ half2_gpu_st __hsub2_sat(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_sat_ru(a.x,b.x);
  else res.x=__hsub2_sat_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_sat_ru(a.y,b.y);
    res.z=__hsub2_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_sat_rd(a.y,b.y);
    res.z=__hsub2_sat_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half2_gpu_st __hsub2_sat(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_sat_ru(a.x,b);
  else res.x=__hsub2_sat_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_sat_ru(a.y,b);
    res.z=__hsub2_sat_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_sat_rd(a.y,b);
    res.z=__hsub2_sat_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half2_gpu_st __hsub2_sat(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_sat_ru(a,b.x);
  else res.x=__hsub2_sat_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_sat_ru(a,b.y);
    res.z=__hsub2_sat_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_sat_rd(a,b.y);
    res.z=__hsub2_sat_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}

/////////////////////////////////////////////////////

__device__ half2_gpu_st operator*(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a.x,b.x);
  else res.x=__hmul2_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul2_ru(a.y,b.y);
    res.z=__hmul2_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul2_rd(a.y,b.y);
    res.z=__hmul2_ru(a.z,b.z);
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



__device__ half2_gpu_st operator*(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a.x,b);
  else res.x=__hmul2_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul2_ru(a.y,b);
    res.z=__hmul2_rd(a.z,b);;
  }
  else {
    res.y=__hmul2_rd(a.y,b);
    res.z=__hmul2_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half2_gpu_st operator*(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a,b.x);
  else res.x=__hmul2_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul2_ru(a,b.y);
    res.z=__hmul2_rd(a,b.z);;
  }
  else {
    res.y=__hmul2_rd(a,b.y);
    res.z=__hmul2_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}
__device__ half2_gpu_st operator*=(half2_gpu_st& a, const half2_gpu_st& b)
{
  //half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hmul2_ru(a.x,b.x);
  else a.x=__hmul2_rd(a.x,b.x);
  if (random>>1) {
    a.y=__hmul2_ru(a.y,b.y);
    a.z=__hmul2_rd(a.z,b.z);;
  }
  else {
    a.y=__hmul2_rd(a.y,b.y);
    a.z=__hmul2_ru(a.z,b.z);
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



__device__ half2_gpu_st operator*=(half2_gpu_st& a, const half2& b)
{
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hmul2_ru(a.x,b);
  else a.x=__hmul2_rd(a.x,b);
  if (random>>1) {
    a.y=__hmul2_ru(a.y,b);
    a.z=__hmul2_rd(a.z,b);;
  }
  else {
    a.y=__hmul2_rd(a.y,b);
    a.z=__hmul2_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


__device__ half2_gpu_st __hmul2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a.x,b.x);
  else res.x=__hmul2_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul2_ru(a.y,b.y);
    res.z=__hmul2_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul2_rd(a.y,b.y);
    res.z=__hmul2_ru(a.z,b.z);
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



__device__ half2_gpu_st __hmul2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a.x,b);
  else res.x=__hmul2_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul2_ru(a.y,b);
    res.z=__hmul2_rd(a.z,b);;
  }
  else {
    res.y=__hmul2_rd(a.y,b);
    res.z=__hmul2_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half2_gpu_st __hmul2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_ru(a,b.x);
  else res.x=__hmul2_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul2_ru(a,b.y);
    res.z=__hmul2_rd(a,b.z);;
  }
  else {
    res.y=__hmul2_rd(a,b.y);
    res.z=__hmul2_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__ half2_gpu_st __hmul2_sat(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_sat_ru(a.x,b.x);
  else res.x=__hmul2_sat_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul2_sat_ru(a.y,b.y);
    res.z=__hmul2_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul2_sat_rd(a.y,b.y);
    res.z=__hmul2_sat_ru(a.z,b.z);
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



__device__ half2_gpu_st __hmul2_sat(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_sat_ru(a.x,b);
  else res.x=__hmul2_sat_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul2_sat_ru(a.y,b);
    res.z=__hmul2_sat_rd(a.z,b);;
  }
  else {
    res.y=__hmul2_sat_rd(a.y,b);
    res.z=__hmul2_sat_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half2_gpu_st __hmul2_sat(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul2_sat_ru(a,b.x);
  else res.x=__hmul2_sat_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul2_sat_ru(a,b.y);
    res.z=__hmul2_sat_rd(a,b.z);;
  }
  else {
    res.y=__hmul2_sat_rd(a,b.y);
    res.z=__hmul2_sat_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}



///////////////////////////////////////////

__device__ half2_gpu_st operator/(const half2_gpu_st& a, const half2_gpu_st& b)
{
  unsigned int inst;
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a.x,b.x);
  else res.x=__h2div_rd(a.x,b.x);
  if (random>>1) {
    res.y=__h2div_ru(a.y,b.y);
    res.z=__h2div_rd(a.z,b.z);;
  }
  else {
    res.y=__h2div_rd(a.y,b.y);
    res.z=__h2div_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  res.error=a.error | b.error | inst;

  return res;
}


__device__ half2_gpu_st operator/(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a.x,b);
  else res.x=__h2div_rd(a.x,b);
  if (random>>1) {
    res.y=__h2div_ru(a.y,b);
    res.z=__h2div_rd(a.z,b);;
  }
  else {
    res.y=__h2div_rd(a.y,b);
    res.z=__h2div_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}


__device__ half2_gpu_st operator/(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a,b.x);
  else res.x=__h2div_rd(a,b.x);
  if (random>>1) {
    res.y=__h2div_ru(a,b.y);
    res.z=__h2div_rd(a,b.z);;
  }
  else {
    res.y=__h2div_rd(a,b.y);
    res.z=__h2div_ru(a,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_DIV : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}
__device__ half2_gpu_st operator/=(half2_gpu_st& a, const half2_gpu_st& b)
{
  unsigned int inst;
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__h2div_ru(a.x,b.x);
  else a.x=__h2div_rd(a.x,b.x);
  if (random>>1) {
    a.y=__h2div_ru(a.y,b.y);
    a.z=__h2div_rd(a.z,b.z);;
  }
  else {
    a.y=__h2div_rd(a.y,b.y);
    a.z=__h2div_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  a.error=a.error | b.error | inst;

  return a;
}


__device__ half2_gpu_st operator/=(half2_gpu_st& a, const half2& b)
{
  //half2_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__h2div_ru(a.x,b);
  else a.x=__h2div_rd(a.x,b);
  if (random>>1) {
    a.y=__h2div_ru(a.y,b);
    a.z=__h2div_rd(a.z,b);;
  }
  else {
    a.y=__h2div_rd(a.y,b);
    a.z=__h2div_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


__device__ half2_gpu_st __h2div(const half2_gpu_st& a, const half2_gpu_st& b)
{
  unsigned int inst;
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a.x,b.x);
  else res.x=__h2div_rd(a.x,b.x);
  if (random>>1) {
    res.y=__h2div_ru(a.y,b.y);
    res.z=__h2div_rd(a.z,b.z);;
  }
  else {
    res.y=__h2div_rd(a.y,b.y);
    res.z=__h2div_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  res.error=a.error | b.error | inst;

  return res;
}


__device__ half2_gpu_st __h2div(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a.x,b);
  else res.x=__h2div_rd(a.x,b);
  if (random>>1) {
    res.y=__h2div_ru(a.y,b);
    res.z=__h2div_rd(a.z,b);;
  }
  else {
    res.y=__h2div_rd(a.y,b);
    res.z=__h2div_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}


__device__ half2_gpu_st __h2div(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__h2div_ru(a,b.x);
  else res.x=__h2div_rd(a,b.x);
  if (random>>1) {
    res.y=__h2div_ru(a,b.y);
    res.z=__h2div_rd(a,b.z);;
  }
  else {
    res.y=__h2div_rd(a,b.y);
    res.z=__h2div_ru(a,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_DIV : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half2_gpu_st __hfma2(const half2_gpu_st& a, const half2_gpu_st& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a.x,b.x,c.x);
  else res.x=__hfma2_rd(a.x,b.x,c.x);
  if (random>>1) {
    res.y=__hfma2_ru(a.y,b.y,c.y);
    res.z=__hfma2_rd(a.z,b.z,c.z);;
  }
  else {
    res.y=__hfma2_rd(a.y,b.y,c.y);
    res.z=__hfma2_ru(a.z,b.z,c.z);
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

__device__ half2_gpu_st __hfma2(const half2_gpu_st& a, const half2& b,const half2& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a.x,b,c);
  else res.x=__hfma2_rd(a.x,b,c);
  if (random>>1) {
    res.y=__hfma2_ru(a.y,b,c);
    res.z=__hfma2_rd(a.z,b,c);;
  }
  else {
    res.y=__hfma2_rd(a.y,b,c);
    res.z=__hfma2_ru(a.z,b,c);
  }
  res.error=a.error;

  return res;
}
__device__ half2_gpu_st __hfma2(const half2_gpu_st& a, const half2& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a.x,b,c.x);
  else res.x=__hfma2_rd(a.x,b,c.x);
  if (random>>1) {
    res.y=__hfma2_ru(a.y,b,c.y);
    res.z=__hfma2_rd(a.z,b,c.z);;
  }
  else {
    res.y=__hfma2_rd(a.y,b,c.y);
    res.z=__hfma2_ru(a.z,b,c.z);
  }
  res.error=a.error;

  return res;
}
__device__ half2_gpu_st __hfma2(const half2_gpu_st& a, const half2_gpu_st& b, const half2& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a.x,b.x,c);
  else res.x=__hfma2_rd(a.x,b.x,c);
  if (random>>1) {
    res.y=__hfma2_ru(a.y,b.y,c);
    res.z=__hfma2_rd(a.z,b.z,c);;
  }
  else {
    res.y=__hfma2_rd(a.y,b.y,c);
    res.z=__hfma2_ru(a.z,b.z,c);
  }
  res.error=a.error;

  return res;
}

__device__ half2_gpu_st __hfma2(const half2& a, const half2_gpu_st& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a,b.x,c.x);
  else res.x=__hfma2_rd(a,b.x,c.x);
  if (random>>1) {
    res.y=__hfma2_ru(a,b.y,c.y);
    res.z=__hfma2_rd(a,b.z,c.z);;
  }
  else {
    res.y=__hfma2_rd(a,b.y,c.y);
    res.z=__hfma2_ru(a,b.z,c.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}


__device__ half2_gpu_st __hfma2(const half2& a, const half2& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a,b,c.x);
  else res.x=__hfma2_rd(a,b,c.x);
  if (random>>1) {
    res.y=__hfma2_ru(a,b,c.y);
    res.z=__hfma2_rd(a,b,c.z);;
  }
  else {
    res.y=__hfma2_rd(a,b,c.y);
    res.z=__hfma2_ru(a,b,c.z);
  }

  
  res.error=c.error;
  return res;
}

__device__ half2_gpu_st __hfma2(const half2& a, const half2_gpu_st& b, const half2& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_ru(a,b.x,c);
  else res.x=__hfma2_rd(a,b.x,c);
  if (random>>1) {
    res.y=__hfma2_ru(a,b.y,c);
    res.z=__hfma2_rd(a,b.z,c);;
  }
  else {
    res.y=__hfma2_rd(a,b.y,c);
    res.z=__hfma2_ru(a,b.z,c);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half2_gpu_st __hfma2_sat(const half2_gpu_st& a, const half2_gpu_st& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a.x,b.x,c.x);
  else res.x=__hfma2_sat_rd(a.x,b.x,c.x);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a.y,b.y,c.y);
    res.z=__hfma2_sat_rd(a.z,b.z,c.z);;
  }
  else {
    res.y=__hfma2_sat_rd(a.y,b.y,c.y);
    res.z=__hfma2_sat_ru(a.z,b.z,c.z);
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

__device__ half2_gpu_st __hfma2_sat(const half2_gpu_st& a, const half2& b,const half2& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a.x,b,c);
  else res.x=__hfma2_sat_rd(a.x,b,c);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a.y,b,c);
    res.z=__hfma2_sat_rd(a.z,b,c);;
  }
  else {
    res.y=__hfma2_sat_rd(a.y,b,c);
    res.z=__hfma2_sat_ru(a.z,b,c);
  }
  res.error=a.error;

  return res;
}
__device__ half2_gpu_st __hfma2_sat(const half2_gpu_st& a, const half2& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a.x,b,c.x);
  else res.x=__hfma2_sat_rd(a.x,b,c.x);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a.y,b,c.y);
    res.z=__hfma2_sat_rd(a.z,b,c.z);;
  }
  else {
    res.y=__hfma2_sat_rd(a.y,b,c.y);
    res.z=__hfma2_sat_ru(a.z,b,c.z);
  }
  res.error=a.error;

  return res;
}
__device__ half2_gpu_st __hfma2_sat(const half2_gpu_st& a, const half2_gpu_st& b, const half2& c)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a.x,b.x,c);
  else res.x=__hfma2_sat_rd(a.x,b.x,c);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a.y,b.y,c);
    res.z=__hfma2_sat_rd(a.z,b.z,c);;
  }
  else {
    res.y=__hfma2_sat_rd(a.y,b.y,c);
    res.z=__hfma2_sat_ru(a.z,b.z,c);
  }
  res.error=a.error;

  return res;
}

__device__ half2_gpu_st __hfma2_sat(const half2& a, const half2_gpu_st& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a,b.x,c.x);
  else res.x=__hfma2_sat_rd(a,b.x,c.x);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a,b.y,c.y);
    res.z=__hfma2_sat_rd(a,b.z,c.z);;
  }
  else {
    res.y=__hfma2_sat_rd(a,b.y,c.y);
    res.z=__hfma2_sat_ru(a,b.z,c.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}


__device__ half2_gpu_st __hfma2_sat(const half2& a, const half2& b, const half2_gpu_st& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a,b,c.x);
  else res.x=__hfma2_sat_rd(a,b,c.x);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a,b,c.y);
    res.z=__hfma2_sat_rd(a,b,c.z);;
  }
  else {
    res.y=__hfma2_sat_rd(a,b,c.y);
    res.z=__hfma2_sat_ru(a,b,c.z);
  }

  
  res.error=c.error;
  return res;
}

__device__ half2_gpu_st __hfma2_sat(const half2& a, const half2_gpu_st& b, const half2& c)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma2_sat_ru(a,b.x,c);
  else res.x=__hfma2_sat_rd(a,b.x,c);
  if (random>>1) {
    res.y=__hfma2_sat_ru(a,b.y,c);
    res.z=__hfma2_sat_rd(a,b.z,c);;
  }
  else {
    res.y=__hfma2_sat_rd(a,b.y,c);
    res.z=__hfma2_sat_ru(a,b.z,c);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half2_gpu_st __hneg2(const half2_gpu_st& a)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hneg2_ru(a.x);
  else res.x=__hneg2_rd(a.x);
  if (random>>1) {
    res.y=__hneg2_ru(a.y);
    res.z=__hneg2_rd(a.z);;
  }
  else {
    res.y=__hneg2_rd(a.y);
    res.z=__hneg2_ru(a.z);
  }

  res.error=a.error;
 return res;	
}






///////////////////////////////////////

__device__ int operator==(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int operator==(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int operator==(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return res.computedzero();
}


__device__ int __hbeq2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int __hbeq2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int __hbeq2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return res.computedzero();
}

__device__ int __hbequ2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int __hbequ2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int __hbequ2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return res.computedzero();
}

///////////////////////////////////////

///////////////////////////////////////

__device__ int operator!=(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int operator!=(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int operator!=(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return !res.computedzero();
}

__device__ int __hbne2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int __hbne2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int __hbne2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return !res.computedzero();
}

__device__ int __hbneu2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int __hbneu2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int __hbneu2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  return !res.computedzero();
}



///////////////////////////////////////

__device__ int operator>(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z)) >	((__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}


__device__ int operator>(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }


  return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z) ) > 3*(__low2float(b)+__high2float(b)) );
}



__device__ int operator>(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*(__low2float(a)+__high2float(a))  >	( (__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}

__device__ int __hbgt2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }
	return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z)) >	((__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}



__device__ int __hbgt2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }
  return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z) ) > 3*(__low2float(b)+__high2float(b)) );
}





__device__ int __hbgt2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( 3*(__low2float(a)+__high2float(a))  >	( (__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}


__device__ int __hbgtu2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }
	return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z)) >	((__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}



__device__ int __hbgtu2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }
  return !r && ( (__low2float(a.x) + __high2float(a.x) + __low2float(a.y) + __high2float(a.y)+__low2float(a.z) + __high2float(a.z) ) > 3*(__low2float(b)+__high2float(b)) );
}





__device__ int __hbgtu2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( 3*(__low2float(a)+__high2float(a))  >	( (__low2float(b.x) + __high2float(b.x) + __low2float(b.y) + __high2float(b.y)+__low2float(b.z) + __high2float(b.z) )));
}

///////////////////////////////////////


__device__ int operator>=(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int operator>=(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	3*(__low2float(b)+__high2float(b)));
}



__device__ int operator>=(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}

__device__ int __hbge2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hbge2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	3*(__low2float(b)+__high2float(b)));
}



__device__ int __hbge2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}

__device__ int __hbgeu2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hbgeu2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) >=	3*(__low2float(b)+__high2float(b)));
}



__device__ int __hbgeu2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  >=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}
///////////////////////////////////////

__device__ int operator<(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int operator<(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) < 3*(__low2float(b)+__high2float(b)) );
}



__device__ int operator<(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*(__low2float(a)+__high2float(a))  <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hblt2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hblt2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) < 3*(__low2float(b)+__high2float(b)) );
}



__device__ int __hblt2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*(__low2float(a)+__high2float(a))  <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}

__device__ int __hbltu2(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hbltu2(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) < 3*(__low2float(b)+__high2float(b)) );
}



__device__ int __hbltu2(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*(__low2float(a)+__high2float(a))  <	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}
///////////////////////////////////////


__device__ int operator<=(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int operator<=(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	3*(__low2float(b)+__high2float(b)));
}



__device__ int operator<=(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}

__device__ int __hble(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hble(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	3*(__low2float(b)+__high2float(b)));
}



__device__ int __hble(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}

__device__ int __hbleu(const half2_gpu_st& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b.x);
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b.y);
    res.z=__hsub2_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub2_rd(a.y,b.y);
    res.z=__hsub2_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}


__device__ int __hbleu(const half2_gpu_st& a, const half2& b)
{
  half2_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a.x,b);
  else res.x=__hsub2_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);
    res.z=__hsub2_rd(a.z,b);;
  }
  else {
    res.y=__hsub2_rd(a.y,b);
    res.z=__hsub2_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) <=	3*(__low2float(b)+__high2float(b)));
}



__device__ int __hbleu(const half2& a, const half2_gpu_st& b)
{
  half2_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub2_ru(a,b.x);
  else res.x=__hsub2_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub2_ru(a,b.y);
    res.z=__hsub2_rd(a,b.z);;
  }
  else {
    res.y=__hsub2_rd(a,b.y);
    res.z=__hsub2_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*(__low2float(a)+__high2float(a))  <=	( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) ));
}
///////////////////////////////////////
 __device__  half2_gpu_st fabsf(const  half2_gpu_st& a) 
{ 
   half2_gpu_st res; 
   res.x = __float2half2_rn(fabsf((__low2float(a.x)+__high2float(a.x)))); 
   res.y = __float2half2_rn(fabsf((__low2float(a.y)+__high2float(a.y)))); 
   res.z = __float2half2_rn(fabsf((__low2float(a.z)+__high2float(a.z)))); 
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

 __device__  half2_gpu_st fabs(const  half2_gpu_st& a) 
{ 
   half2_gpu_st res; 
   res.x = __float2half2_rn(fabs((__low2float(a.x)+__high2float(a.x)))); 
   res.y = __float2half2_rn(fabs((__low2float(a.y)+__high2float(a.y)))); 
   res.z = __float2half2_rn(fabs((__low2float(a.z)+__high2float(a.z)))); 
 /*  res.x = fabs(a.x); 
   res.y = fabs(a.y); 
   res.z = fabs(a.z); */
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half2_gpu_st h2sqrt(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__h2sqrt_ru(a.x);	    
  else res.x=__h2sqrt_rd(a.x);

  if (random>>1) {
     res.y=__h2sqrt_ru(a.y);				 
     res.z=__h2sqrt_rd(a.z);				 
  }							 
  else {							 
    res.y=__h2sqrt_rd(a.y);				 
    res.z=__h2sqrt_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


///////////////////////////////////////

__device__  half2_gpu_st h2sin(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2sin_ru(a.x);	    
  else res.x=h2sin_rd(a.x);

  if (random>>1) {
     res.y=h2sin_ru(a.y);				 
     res.z=h2sin_rd(a.z);				 
  }							 
  else {							 
    res.y=h2sin_rd(a.y);				 
    res.z=h2sin_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half2_gpu_st h2cos(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2cos_ru(a.x);	    
  else res.x=h2cos_rd(a.x);

  if (random>>1) {
     res.y=h2cos_ru(a.y);				 
     res.z=h2cos_rd(a.z);				 
  }							 
  else {							 
    res.y=h2cos_rd(a.y);				 
    res.z=h2cos_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half2_gpu_st h2exp(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2exp_ru(a.x);	    
  else res.x=h2exp_rd(a.x);

  if (random>>1) {
     res.y=h2exp_ru(a.y);				 
     res.z=h2exp_rd(a.z);				 
  }							 
  else {							 
    res.y=h2exp_rd(a.y);				 
    res.z=h2exp_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half2_gpu_st h2log(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2log_ru(a.x);	    
  else res.x=h2log_rd(a.x);

  if (random>>1) {
     res.y=h2log_ru(a.y);				 
     res.z=h2log_rd(a.z);				 
  }							 
  else {							 
    res.y=h2log_rd(a.y);				 
    res.z=h2log_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half2_gpu_st h2log2(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2log2_ru(a.x);	    
  else res.x=h2log2_rd(a.x);

  if (random>>1) {
     res.y=h2log2_ru(a.y);				 
     res.z=h2log2_rd(a.z);				 
  }							 
  else {							 
    res.y=h2log2_rd(a.y);				 
    res.z=h2log2_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half2_gpu_st h2log10(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2log10_ru(a.x);	    
  else res.x=h2log10_rd(a.x);

  if (random>>1) {
     res.y=h2log10_ru(a.y);				 
     res.z=h2log10_rd(a.z);				 
  }							 
  else {							 
    res.y=h2log10_rd(a.y);				 
    res.z=h2log10_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half2_gpu_st h2rcp(const  half2_gpu_st& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=h2rcp_ru(a.x);	    
  else res.x=h2rcp_rd(a.x);

  if (random>>1) {
     res.y=h2rcp_ru(a.y);				 
     res.z=h2rcp_rd(a.z);				 
  }							 
  else {							 
    res.y=h2rcp_rd(a.y);				 
    res.z=h2rcp_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

///////////////////////////////////////
__device__  half2_gpu_st fmaxf(const half2& a, const half2_gpu_st& b) 
{ 
  half2_gpu_st res;   
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a,b.x);			 
  else res.x=__hsub2_rd(a,b.x);
  if (random>>1) {						 
    res.y=__hsub2_ru(a,b.y);				 
    res.z=__hsub2_rd(a,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a,b.y);				 
    res.z=__hsub2_ru(a,b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*(__low2float(a)+__high2float(a)) > ((__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)))) {
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=0;	
    }
    else							 
      res=b;	
  }
  return(res); 
}


__device__  half2_gpu_st fmaxf(const  half2_gpu_st& a, const  half2& b) 
{ 
  half2_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a.x,b);			 
  else res.x=__hsub2_rd(a.x,b);
  if  (random>>1) {
    res.y=__hsub2_ru(a.y,b);				 
    res.z=__hsub2_rd(a.z,b);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a.y,b);				 
    res.z=__hsub2_ru(a.z,b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if (((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) > 3.f*(__low2float(b)+__high2float(b)) ) {
	res=a;	
    }
    else {								 
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=0;	
    }
  }
  return(res); 
}


__device__  half2_gpu_st fmaxf(const  half2_gpu_st& a, const  half2_gpu_st& b) 
{ 
  half2_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a.x,b.x);			 
  else res.x=__hsub2_rd(a.x,b.x);				 
  if (random>>1) {						 
    res.y=__hsub2_ru(a.y,b.y);				 
    res.z=__hsub2_rd(a.z,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a.y,b.y);				 
    res.z=__hsub2_ru(a.z,b.z);					
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
    if (((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) > ( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

//////

__device__  half2_gpu_st fminf(const half2& a, half2_gpu_st& b) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a,b.x);			 
  else res.x=__hsub2_rd(a,b.x);
  if (random>>1) {						 
    res.y=__hsub2_ru(a,b.y);				 
    res.z=__hsub2_rd(a,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a,b.y);				 
    res.z=__hsub2_ru(a,b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*(__low2float(a)+__high2float(a)) < ((__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)))) {
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=0;	
    }
    else							 
      res=b;	
  }
  return(res); 
}


__device__  half2_gpu_st fminf(const  half2_gpu_st& a, const  half2& b) 
{ 
  half2_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a.x,b);			 
  else res.x=__hsub2_rd(a.x,b);
  if (random>>1) {
    res.y=__hsub2_ru(a.y,b);				 
    res.z=__hsub2_rd(a.z,b);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a.y,b);				 
    res.z=__hsub2_ru(a.z,b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if (( (__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) < 3.f*(__low2float(b)+__high2float(b)) ) {
	res=a;	
    }
    else {								 
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=0;	
    }
  }
  return(res); 
}



__device__  half2_gpu_st fminf(const  half2_gpu_st& a, const  half2_gpu_st& b) 
{ 
  half2_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub2_ru(a.x,b.x);			 
  else res.x=__hsub2_rd(a.x,b.x);

  if (random>>1) {				 
    res.y=__hsub2_ru(a.y,b.y);				 
    res.z=__hsub2_rd(a.z,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub2_rd(a.y,b.y);				 
    res.z=__hsub2_ru(a.z,b.z);					
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
    if (((__low2float(a.x)+__high2float(a.x)) + (__low2float(a.y)+__high2float(a.y)) + (__low2float(a.z)+__high2float(a.z)) ) < ( (__low2float(b.x)+__high2float(b.y)) + (__low2float(b.y)+__high2float(b.y)) + (__low2float(b.z)+__high2float(b.z)) )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

///////////////////////////////////////

__device__  half2_gpu_st __float2half2_gpu_st(const  float& a) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x = __float2half2_ru(a);	    
  else res.x=__float2half2_rd(a);

  if (random>>1) {
     res.y=__float2half2_ru(a);				 
     res.z=__float2half2_rd(a);				 
  }							 
  else {							 
    res.y=__float2half2_rd(a);				 
    res.z=__float2half2_ru(a);					
  }				
   //res.accuracy=DIGIT_NOT_COMPUTED; 
   //res.error=a.error;
   return(res); 
}



__device__  half2_gpu_st __floats2half2_gpu_st(const  float& a, const float& b) 
{ 
  half2_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x = __floats2half2_ru(a, b);	    
  else res.x=__floats2half2_rd(a, b);

  if (random>>1) {
     res.y=__floats2half2_ru(a, b);				 
     res.z=__floats2half2_rd(a, b);				 
  }							 
  else {							 
    res.y=__floats2half2_rd(a, b);				 
    res.z=__floats2half2_ru(a, b);					
  }				
   //res.accuracy=DIGIT_NOT_COMPUTED; 
   //res.error=a.error;
   return(res); 
}



__device__  float __half2_gpu_st2float(const  half2_gpu_st& a) 
{ 
  float res = (__low2float(a.x)+__high2float(a.x));
 /* half2_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=half22float_ru(a.x);	    
  else res.x=half22float_rd(a.x);

  if (random>>1) {
     res.y=half22float_ru(a.y);				 
     res.z=half22float_rd(a.z);				 
  }							 
  else {							 
    res.y=half22float_rd(a.y);				 
    res.z=half22float_ru(a.z);					
  }
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   */
   return(res); 
}


__device__  float_gpu_st __half2_gpu_st2float_gpu_st_low(const  half2_gpu_st& a) 
{ 
 
 
  float_gpu_st res;
 /* res.x = (__low2float(a.x)+__high2float(a.x));
  res.y = (__low2float(a.y)+__high2float(a.y));
  res.z = (__low2float(a.z)+__high2float(a.z));
*/
  
  res.x = (__low2float(a.x));
  res.y = (__low2float(a.y));
  res.z = (__low2float(a.z));



/*  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=half22float_ru(a.x);	    
  else res.x=half22float_rd(a.x);

  if (random>>1) {
     res.y=half22float_ru(a.y);				 
     res.z=half22float_rd(a.z);				 
  }							 
  else {							 
    res.y=half22float_rd(a.y);				 
    res.z=half22float_ru(a.z);					
  }
  */
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   
   return(res); 
}


__device__  float_gpu_st __half2_gpu_st2float_gpu_st_high(const  half2_gpu_st& a) 
{ 
 
 
  float_gpu_st res;
 /* res.x = (__low2float(a.x)+__high2float(a.x));
  res.y = (__low2float(a.y)+__high2float(a.y));
  res.z = (__low2float(a.z)+__high2float(a.z));
*/
  
  res.x = (__high2float(a.x));
  res.y = (__high2float(a.y));
  res.z = (__high2float(a.z));



/*  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=half22float_ru(a.x);	    
  else res.x=half22float_rd(a.x);

  if (random>>1) {
     res.y=half22float_ru(a.y);				 
     res.z=half22float_rd(a.z);				 
  }							 
  else {							 
    res.y=half22float_rd(a.y);				 
    res.z=half22float_ru(a.z);					
  }
  */
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   
   return(res); 
}



__device__  half2_gpu_st __float_gpu_st2half2_gpu_st(const  float_gpu_st& a) 
{ 
 
  half2_gpu_st res;
 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__float2half2_ru(a.x);	    
  else res.x=__float2half2_rd(a.x);

  if (random>>1) {
     res.y=__float2half2_ru(a.y);				 
     res.z=__float2half2_rd(a.z);				 
  }							 
  else {							 
    res.y=__float2half2_rd(a.y);				 
    res.z=__float2half2_ru(a.z);					
  }
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half2_gpu_st __float_gpu_st2half2_gpu_st(const  float_gpu_st& a, const float_gpu_st& b) 
{ 
 
  half2_gpu_st res;
 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x= {__float2half_ru(a.x), __float2half_ru(b.x)};	    
  else res.x= {__float2half_rd(a.x), __float2half_rd(b.x)};

  if (random>>1) {
     res.y= {__float2half_ru(a.y), __float2half_ru(b.y)};				 
     res.z= {__float2half_rd(a.z), __float2half_rd(b.z)};				 
  }							 
  else {							 
    res.y={__float2half_rd(a.y), __float2half_rd(b.y)};				 
    res.z={__float2half_ru(a.z), __float2half_ru(b.z)};					
  }
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error | b.error;
   return(res); 
}






///////////////////////////////////////

__device__ void half2_gpu_st::modify(const int &a)
{
  accuracy |=a;
}


__device__ half2_gpu_st& half2_gpu_st::operator=(const half2 &a)
{
  x=a;
  y=a;
  z=a;
  accuracy=3;
  error=0;
  return *this ;
}

__device__ int  half2_gpu_st::nb_significant_digit() const
{
  float x0,x1,x2,xx;
	float x_h,y_h,z_h;
	x_h = __low2float(x)+__high2float(x); 	
	y_h = __low2float(y)+__high2float(y); 	
	z_h = __low2float(z)+__high2float(z); 	
  xx=x_h+y_h+z_h;

  accuracy=0;
  if (xx==0.0){
    if ((x_h==y_h) &&(x_h==z_h) ) accuracy=3;
  }
  else {
    xx=3/xx;
    x0=x_h*xx-1;
    x1=y_h*xx-1;
    x2=z_h*xx-1;
    //FJ 4 Mar 2014:
    float yy=(x0*x0+x1*x1+x2*x2)*(float)3.08546617;
    if (yy<=1.e-6)  accuracy=3;
    else {
      yy= -log10(yy);
      if (yy>=0.) accuracy=(int)((yy+(float)1.)*(float)0.5);
    }
  }
  return accuracy;
}

