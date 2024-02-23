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
#include "cadna_gpu_half.h"
#include "cadna_gpu_half2.h"
#define MAX_THREAD_PER_BLOCK 1024 // 512

//#define MAX_BLOCK_SIZE_X 1024// 512

//#define INF __float2half(0.9990234375f) // 1-2*u
//#define SUP __float2half(1.0009765625f) // 1+2*u

__device__ __half INF;
__device__ __half SUP;

__device__ __half2 INF2;
__device__ __half2 SUP2;


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
  INF = __float2half(0.9990234375f);
  SUP = __float2half(1.0009765625f);
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

/////////////////////////////////////////////////////

__device__ __half __hadd_rd(const __half a, const __half b)
{

	half x = __hadd(a,b);
	x = __hmul(x,INF);
	return x;
}

__device__ __half __hadd_ru(const __half a, const __half b)
{
	half x = __hadd(a,b);
	x = __hmul(x,SUP);
	return x;
}

__device__ __half __hadd_sat_rd(const __half a, const __half b)
{

	half x = __hadd_sat(a,b);
	x = __hmul(x,INF);
	return x;
}

__device__ __half __hadd_sat_ru(const __half a, const __half b)
{
	half x = __hadd_sat(a,b);
	x = __hmul(x,SUP);
	return x;
}
__device__ __half __hsub_rd(const __half a, const __half b)
{

	half x = __hsub(a,b);
	x = __hmul(x,INF);
	return x;
}

__device__ __half __hsub_ru(const __half a, const __half b)
{
	half x = __hsub(a,b);
	x = __hmul(x,SUP);
	return x;
}

__device__ __half __hsub_sat_rd(const __half a, const __half b)
{

	half x = __hsub_sat(a,b);
	x = __hmul(x,INF);
	return x;
}

__device__ __half __hsub_sat_ru(const __half a, const __half b)
{
	half x = __hsub_sat(a,b);
	x = __hmul(x,SUP);
	return x;
}

__device__ __half __hmul_rd(const __half a, const __half b)
{
	half x = __hmul(a,b);
	x = __hmul(x,INF);
	return x;
}
__device__ __half __hmul_ru(const __half a, const __half b)
{
	half x = __hmul(a,b);
	x = __hmul(x,SUP);
	return x;
}

__device__ __half __hmul_sat_rd(const __half a, const __half b)
{
	half x = __hmul_sat(a,b);
	x = __hmul(x,INF);
	return x;
}
__device__ __half __hmul_sat_ru(const __half a, const __half b)
{
	half x = __hmul_sat(a,b);
	x = __hmul(x,SUP);
	return x;
}
__device__ __half __hdiv_rd(const __half a, const __half b)
{
	half x = __hdiv(a,b);
	x = __hmul(x,INF);
	return x;
}
__device__ __half __hdiv_ru(const __half a, const __half b)
{
	half x = __hdiv(a,b);
	x = __hmul(x,SUP);
	return x;
}

__device__ __half __hneg_rd(const __half a)
{
	half x = __hneg(a);
	//x = __hmul(x,INF);
	return x;
}
__device__ __half __hneg_ru(const __half a)
{
	half x = __hneg(a);
	//x = __hmul(x,SUP);
	return x;
}
/*
__device__ __half  __hsqrt_ru(const half a)
{
	return __hmul(__float2half(__fsqrt_ru(__half2float(a))),SUP);
}
__device__ __half  __hsqrt_rd(const half a)
{
	return __hmul(__float2half(__fsqrt_rd(__half2float(a))),INF);
}
*/
__device__ __half  __hsqrt_ru(const half a)
{
	return __hmul(hsqrt(a),SUP);
}
__device__ __half  __hsqrt_rd(const half a)
{
	return __hmul(hsqrt(a),INF);
}

__device__ __half  __hfma_ru(const half a,const half b, const half c)
{
	return __hmul(__hfma(a,b,c),SUP);
}
__device__ __half  __hfma_rd(const half a,const half b, const half c)
{
	return __hmul(__hfma(a,b,c),INF);
}

__device__ __half  __hfma_sat_ru(const half a,const half b, const half c)
{
	return __hmul(__hfma_sat(a,b,c),SUP);
}
__device__ __half  __hfma_sat_rd(const half a,const half b, const half c)
{
	return __hmul(__hfma_sat(a,b,c),INF);
}

__device__ __half  hsin_ru(const half a)
{
	return __hmul(hsin(a),SUP);
}
__device__ __half  hsin_rd(const half a)
{
	return __hmul(hsin(a),INF);
}
__device__ __half  hcos_ru(const half a)
{
	return __hmul(hcos(a),SUP);
}
__device__ __half  hcos_rd(const half a)
{
	return __hmul(hcos(a),INF);
}
__device__ __half  hexp_ru(const half a)
{
	return __hmul(hexp(a),SUP);
}
__device__ __half  hexp_rd(const half a)
{
	return __hmul(hexp(a),INF);
}
__device__ __half  hlog_ru(const half a)
{
	return __hmul(hlog(a),SUP);
}
__device__ __half  hlog_rd(const half a)
{
	return __hmul(hlog(a),INF);
}
__device__ __half  hlog2_ru(const half a)
{
	return __hmul(hlog2(a),SUP);
}
__device__ __half  hlog2_rd(const half a)
{
	return __hmul(hlog2(a),INF);
}
__device__ __half  hlog10_ru(const half a)
{
	return __hmul(hlog10(a),SUP);
}
__device__ __half  hlog10_rd(const half a)
{
	return __hmul(hlog10(a),INF);
}
__device__ __half  hrcp_ru(const half a)
{
	return __hmul(hrcp(a),SUP);
}
__device__ __half  hrcp_rd(const half a)
{
	return __hmul(hrcp(a),INF);
}



//////////////////////////////////////////////////

__device__ half_gpu_st operator+(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a.x,b.x);
  else res.x=__hadd_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd_ru(a.y,b.y);
    res.z=__hadd_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd_rd(a.y,b.y);
    res.z=__hadd_ru(a.z,b.z);
  }

  res.error=a.error | b.error;

  return res;
}


__device__ half_gpu_st operator+(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a.x,b);
  else res.x=__hadd_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd_ru(a.y,b);
    res.z=__hadd_rd(a.z,b);;
  }
  else {
    res.y=__hadd_rd(a.y,b);
    res.z=__hadd_ru(a.z,b);
  }
  res.error=a.error;
  return res;
}


__device__ half_gpu_st operator+(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a,b.x);
  else res.x=__hadd_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd_ru(a,b.y);
    res.z=__hadd_rd(a,b.z);;
  }
  else {
    res.y=__hadd_rd(a,b.y);
    res.z=__hadd_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__ half_gpu_st operator+=(half_gpu_st& a, const half_gpu_st& b)
{
  //half_gpu_st res;
  unsigned char random;

  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hadd_ru(a.x,b.x);
  else a.x=__hadd_rd(a.x,b.x);

  if (random>>1) {
    a.y=__hadd_ru(a.y,b.y);
    a.z=__hadd_rd(a.z,b.z);;
  }
  else {
    a.y=__hadd_rd(a.y,b.y);
    a.z=__hadd_ru(a.z,b.z);
  }

  a.error=a.error | b.error;

  return a;
}


__device__ half_gpu_st operator+=(half_gpu_st& a, const half& b)
{
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__hadd_ru(a.x,b);
  else a.x=__hadd_rd(a.x,b);

  if (random>>1) {
    a.y=__hadd_ru(a.y,b);
    a.z=__hadd_rd(a.z,b);;
  }
  else {
    a.y=__hadd_rd(a.y,b);
    a.z=__hadd_ru(a.z,b);
  }
  //res.error=a.error;
  return a;
}


__device__  half_gpu_st __hadd(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a.x,b.x);
  else res.x=__hadd_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd_ru(a.y,b.y);
    res.z=__hadd_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd_rd(a.y,b.y);
    res.z=__hadd_ru(a.z,b.z);
  }

  res.error=a.error | b.error;

  return res;
}

__device__ half_gpu_st __hadd(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a.x,b);
  else res.x=__hadd_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd_ru(a.y,b);
    res.z=__hadd_rd(a.z,b);;
  }
  else {
    res.y=__hadd_rd(a.y,b);
    res.z=__hadd_ru(a.z,b);
  }
  res.error=a.error;
  return res;
}


__device__ half_gpu_st __hadd(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_ru(a,b.x);
  else res.x=__hadd_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd_ru(a,b.y);
    res.z=__hadd_rd(a,b.z);;
  }
  else {
    res.y=__hadd_rd(a,b.y);
    res.z=__hadd_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__  half_gpu_st __hadd_sat(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_sat_ru(a.x,b.x);
  else res.x=__hadd_sat_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hadd_sat_ru(a.y,b.y);
    res.z=__hadd_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hadd_sat_rd(a.y,b.y);
    res.z=__hadd_sat_ru(a.z,b.z);
  }
	



  res.error=a.error | b.error;

  return res;
}

__device__ half_gpu_st __hadd_sat(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hadd_sat_ru(a.x,b);
  else res.x=__hadd_sat_rd(a.x,b);

  if (random>>1) {
    res.y=__hadd_sat_ru(a.y,b);
    res.z=__hadd_sat_rd(a.z,b);;
  }
  else {
    res.y=__hadd_sat_rd(a.y,b);
    res.z=__hadd_sat_ru(a.z,b);
  }



  res.error=a.error;
  return res;
}


__device__ half_gpu_st __hadd_sat(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hadd_sat_ru(a,b.x);
  else res.x=__hadd_sat_rd(a,b.x);

  if (random>>1) {
    res.y=__hadd_sat_ru(a,b.y);
    res.z=__hadd_sat_rd(a,b.z);;
  }
  else {
    res.y=__hadd_sat_rd(a,b.y);
    res.z=__hadd_sat_ru(a,b.z);
  }

  /*
  if(__half2float(res.x) > 1.f)
  {
	  res.x = __float2half(1.f);
  }
  if(__half2float(res.y) > 1.f)
  {
	  res.y = __float2half(1.f);
  }
  if(__half2float(res.z) > 1.f)
  {
	  res.z = __float2half(1.f);
  }
  if(__half2float(res) > 1.f)
  {
	  res = __float2half(1.f);
  }

  if(__half2float(res.x) < 0.f)
  {
	  res.x = __float2half(0.f);
  }
  if(__half2float(res.y) < 0.f)
  {
	  res.y = __float2half(0.f);
  }
  if(__half2float(res.z) < 0.f)
  {
	  res.z = __float2half(0.f);
  }
  if(__half2float(res) < 0.f)
  {
	  res = __float2half(0.f);
  }
*/



  res.error=b.error;
  return res;
}




/////////////////////////////////////////////////////


__device__ half_gpu_st operator-(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half_gpu_st operator-(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half_gpu_st operator-(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}
__device__ half_gpu_st operator-=(half_gpu_st& a, const half_gpu_st& b)
{
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hsub_ru(a.x,b.x);
  else a.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    a.y=__hsub_ru(a.y,b.y);
    a.z=__hsub_rd(a.z,b.z);;
  }
  else {
    a.y=__hsub_rd(a.y,b.y);
    a.z=__hsub_ru(a.z,b.z);
  }

  a.error= a.error | b.error;
  return a;
}


__device__ half_gpu_st operator-=(half_gpu_st& a, const half& b)
{
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;

  random = RANDOMGPU();
  if (random&1) a.x=__hsub_ru(a.x,b);
  else a.x=__hsub_rd(a.x,b);

  if (random>>1) {
    a.y=__hsub_ru(a.y,b);
    a.z=__hsub_rd(a.z,b);;
  }
  else {
    a.y=__hsub_rd(a.y,b);
    a.z=__hsub_ru(a.z,b);
  }
  //res.error= a.error;

  return a;

}


__device__ half_gpu_st __hsub(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half_gpu_st __hsub(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half_gpu_st __hsub(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}

__device__ half_gpu_st __hsub_sat(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_sat_ru(a.x,b.x);
  else res.x=__hsub_sat_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_sat_ru(a.y,b.y);
    res.z=__hsub_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_sat_rd(a.y,b.y);
    res.z=__hsub_sat_ru(a.z,b.z);
  }

  res.error= a.error | b.error;
  return res;
}


__device__ half_gpu_st __hsub_sat(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;

  random = RANDOMGPU();
  if (random&1) res.x=__hsub_sat_ru(a.x,b);
  else res.x=__hsub_sat_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_sat_ru(a.y,b);
    res.z=__hsub_sat_rd(a.z,b);;
  }
  else {
    res.y=__hsub_sat_rd(a.y,b);
    res.z=__hsub_sat_ru(a.z,b);
  }
  res.error= a.error;

  return res;

}

__device__ half_gpu_st __hsub_sat(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_sat_ru(a,b.x);
  else res.x=__hsub_sat_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_sat_ru(a,b.y);
    res.z=__hsub_sat_rd(a,b.z);;
  }
  else {
    res.y=__hsub_sat_rd(a,b.y);
    res.z=__hsub_sat_ru(a,b.z);
  }
  res.error= b.error;
  return res;
}

/////////////////////////////////////////////////////

__device__ half_gpu_st operator*(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a.x,b.x);
  else res.x=__hmul_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul_ru(a.y,b.y);
    res.z=__hmul_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul_rd(a.y,b.y);
    res.z=__hmul_ru(a.z,b.z);
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



__device__ half_gpu_st operator*(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a.x,b);
  else res.x=__hmul_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul_ru(a.y,b);
    res.z=__hmul_rd(a.z,b);;
  }
  else {
    res.y=__hmul_rd(a.y,b);
    res.z=__hmul_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half_gpu_st operator*(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a,b.x);
  else res.x=__hmul_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul_ru(a,b.y);
    res.z=__hmul_rd(a,b.z);;
  }
  else {
    res.y=__hmul_rd(a,b.y);
    res.z=__hmul_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}
__device__ half_gpu_st operator*=(half_gpu_st& a, const half_gpu_st& b)
{
  //half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hmul_ru(a.x,b.x);
  else a.x=__hmul_rd(a.x,b.x);
  if (random>>1) {
    a.y=__hmul_ru(a.y,b.y);
    a.z=__hmul_rd(a.z,b.z);;
  }
  else {
    a.y=__hmul_rd(a.y,b.y);
    a.z=__hmul_ru(a.z,b.z);
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



__device__ half_gpu_st operator*=(half_gpu_st& a, const half& b)
{
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hmul_ru(a.x,b);
  else a.x=__hmul_rd(a.x,b);
  if (random>>1) {
    a.y=__hmul_ru(a.y,b);
    a.z=__hmul_rd(a.z,b);;
  }
  else {
    a.y=__hmul_rd(a.y,b);
    a.z=__hmul_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


__device__ half_gpu_st __hmul(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a.x,b.x);
  else res.x=__hmul_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul_ru(a.y,b.y);
    res.z=__hmul_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul_rd(a.y,b.y);
    res.z=__hmul_ru(a.z,b.z);
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



__device__ half_gpu_st __hmul(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a.x,b);
  else res.x=__hmul_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul_ru(a.y,b);
    res.z=__hmul_rd(a.z,b);;
  }
  else {
    res.y=__hmul_rd(a.y,b);
    res.z=__hmul_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half_gpu_st __hmul(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_ru(a,b.x);
  else res.x=__hmul_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul_ru(a,b.y);
    res.z=__hmul_rd(a,b.z);;
  }
  else {
    res.y=__hmul_rd(a,b.y);
    res.z=__hmul_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}

__device__ half_gpu_st __hmul_sat(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_sat_ru(a.x,b.x);
  else res.x=__hmul_sat_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hmul_sat_ru(a.y,b.y);
    res.z=__hmul_sat_rd(a.z,b.z);;
  }
  else {
    res.y=__hmul_sat_rd(a.y,b.y);
    res.z=__hmul_sat_ru(a.z,b.z);
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



__device__ half_gpu_st __hmul_sat(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_sat_ru(a.x,b);
  else res.x=__hmul_sat_rd(a.x,b);
  if (random>>1) {
    res.y=__hmul_sat_ru(a.y,b);
    res.z=__hmul_sat_rd(a.z,b);;
  }
  else {
    res.y=__hmul_sat_rd(a.y,b);
    res.z=__hmul_sat_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}



__device__ half_gpu_st __hmul_sat(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;



  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hmul_sat_ru(a,b.x);
  else res.x=__hmul_sat_rd(a,b.x);
  if (random>>1) {
    res.y=__hmul_sat_ru(a,b.y);
    res.z=__hmul_sat_rd(a,b.z);;
  }
  else {
    res.y=__hmul_sat_rd(a,b.y);
    res.z=__hmul_sat_ru(a,b.z);
  }
  res.error=b.error;
  return res;
}



///////////////////////////////////////////

__device__ half_gpu_st operator/(const half_gpu_st& a, const half_gpu_st& b)
{
  unsigned int inst;
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a.x,b.x);
  else res.x=__hdiv_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hdiv_ru(a.y,b.y);
    res.z=__hdiv_rd(a.z,b.z);;
  }
  else {
    res.y=__hdiv_rd(a.y,b.y);
    res.z=__hdiv_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  res.error=a.error | b.error | inst;

  return res;
}


__device__ half_gpu_st operator/(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a.x,b);
  else res.x=__hdiv_rd(a.x,b);
  if (random>>1) {
    res.y=__hdiv_ru(a.y,b);
    res.z=__hdiv_rd(a.z,b);;
  }
  else {
    res.y=__hdiv_rd(a.y,b);
    res.z=__hdiv_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}


__device__ half_gpu_st operator/(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a,b.x);
  else res.x=__hdiv_rd(a,b.x);
  if (random>>1) {
    res.y=__hdiv_ru(a,b.y);
    res.z=__hdiv_rd(a,b.z);;
  }
  else {
    res.y=__hdiv_rd(a,b.y);
    res.z=__hdiv_ru(a,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_DIV : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}
__device__ half_gpu_st operator/=(half_gpu_st& a, const half_gpu_st& b)
{
  unsigned int inst;
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hdiv_ru(a.x,b.x);
  else a.x=__hdiv_rd(a.x,b.x);
  if (random>>1) {
    a.y=__hdiv_ru(a.y,b.y);
    a.z=__hdiv_rd(a.z,b.z);;
  }
  else {
    a.y=__hdiv_rd(a.y,b.y);
    a.z=__hdiv_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  a.error=a.error | b.error | inst;

  return a;
}


__device__ half_gpu_st operator/=(half_gpu_st& a, const half& b)
{
  //half_gpu_st res;
  unsigned char random;


  //res.accuracy=DIGIT_NOT_COMPUTED;
  //res.error=0;
  random = RANDOMGPU();
  if (random&1) a.x=__hdiv_ru(a.x,b);
  else a.x=__hdiv_rd(a.x,b);
  if (random>>1) {
    a.y=__hdiv_ru(a.y,b);
    a.z=__hdiv_rd(a.z,b);;
  }
  else {
    a.y=__hdiv_rd(a.y,b);
    a.z=__hdiv_ru(a.z,b);
  }
  //res.error=a.error;

  return a;
}


__device__ half_gpu_st __hdiv(const half_gpu_st& a, const half_gpu_st& b)
{
  unsigned int inst;
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a.x,b.x);
  else res.x=__hdiv_rd(a.x,b.x);
  if (random>>1) {
    res.y=__hdiv_ru(a.y,b.y);
    res.z=__hdiv_rd(a.z,b.z);;
  }
  else {
    res.y=__hdiv_rd(a.y,b.y);
    res.z=__hdiv_ru(a.z,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  inst=( b.accuracy==0    ) ? CADNA_DIV : 0;
  res.error=a.error | b.error | inst;

  return res;
}


__device__ half_gpu_st __hdiv(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a.x,b);
  else res.x=__hdiv_rd(a.x,b);
  if (random>>1) {
    res.y=__hdiv_ru(a.y,b);
    res.z=__hdiv_rd(a.z,b);;
  }
  else {
    res.y=__hdiv_rd(a.y,b);
    res.z=__hdiv_ru(a.z,b);
  }
  res.error=a.error;

  return res;
}


__device__ half_gpu_st __hdiv(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hdiv_ru(a,b.x);
  else res.x=__hdiv_rd(a,b.x);
  if (random>>1) {
    res.y=__hdiv_ru(a,b.y);
    res.z=__hdiv_rd(a,b.z);;
  }
  else {
    res.y=__hdiv_rd(a,b.y);
    res.z=__hdiv_ru(a,b.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_DIV : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half_gpu_st __hfma(const half_gpu_st& a, const half_gpu_st& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a.x,b.x,c.x);
  else res.x=__hfma_rd(a.x,b.x,c.x);
  if (random>>1) {
    res.y=__hfma_ru(a.y,b.y,c.y);
    res.z=__hfma_rd(a.z,b.z,c.z);;
  }
  else {
    res.y=__hfma_rd(a.y,b.y,c.y);
    res.z=__hfma_ru(a.z,b.z,c.z);
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

__device__ half_gpu_st __hfma(const half_gpu_st& a, const half& b,const half& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a.x,b,c);
  else res.x=__hfma_rd(a.x,b,c);
  if (random>>1) {
    res.y=__hfma_ru(a.y,b,c);
    res.z=__hfma_rd(a.z,b,c);;
  }
  else {
    res.y=__hfma_rd(a.y,b,c);
    res.z=__hfma_ru(a.z,b,c);
  }
  res.error=a.error;

  return res;
}
__device__ half_gpu_st __hfma(const half_gpu_st& a, const half& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a.x,b,c.x);
  else res.x=__hfma_rd(a.x,b,c.x);
  if (random>>1) {
    res.y=__hfma_ru(a.y,b,c.y);
    res.z=__hfma_rd(a.z,b,c.z);;
  }
  else {
    res.y=__hfma_rd(a.y,b,c.y);
    res.z=__hfma_ru(a.z,b,c.z);
  }
  res.error=a.error;

  return res;
}
__device__ half_gpu_st __hfma(const half_gpu_st& a, const half_gpu_st& b, const half& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a.x,b.x,c);
  else res.x=__hfma_rd(a.x,b.x,c);
  if (random>>1) {
    res.y=__hfma_ru(a.y,b.y,c);
    res.z=__hfma_rd(a.z,b.z,c);;
  }
  else {
    res.y=__hfma_rd(a.y,b.y,c);
    res.z=__hfma_ru(a.z,b.z,c);
  }
  res.error=a.error;

  return res;
}

__device__ half_gpu_st __hfma(const half& a, const half_gpu_st& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a,b.x,c.x);
  else res.x=__hfma_rd(a,b.x,c.x);
  if (random>>1) {
    res.y=__hfma_ru(a,b.y,c.y);
    res.z=__hfma_rd(a,b.z,c.z);;
  }
  else {
    res.y=__hfma_rd(a,b.y,c.y);
    res.z=__hfma_ru(a,b.z,c.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}


__device__ half_gpu_st __hfma(const half& a, const half& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a,b,c.x);
  else res.x=__hfma_rd(a,b,c.x);
  if (random>>1) {
    res.y=__hfma_ru(a,b,c.y);
    res.z=__hfma_rd(a,b,c.z);;
  }
  else {
    res.y=__hfma_rd(a,b,c.y);
    res.z=__hfma_ru(a,b,c.z);
  }

  
  res.error=c.error;
  return res;
}

__device__ half_gpu_st __hfma(const half& a, const half_gpu_st& b, const half& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_ru(a,b.x,c);
  else res.x=__hfma_rd(a,b.x,c);
  if (random>>1) {
    res.y=__hfma_ru(a,b.y,c);
    res.z=__hfma_rd(a,b.z,c);;
  }
  else {
    res.y=__hfma_rd(a,b.y,c);
    res.z=__hfma_ru(a,b.z,c);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half_gpu_st __hfma_sat(const half_gpu_st& a, const half_gpu_st& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  unsigned int inst;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a.x,b.x,c.x);
  else res.x=__hfma_sat_rd(a.x,b.x,c.x);
  if (random>>1) {
    res.y=__hfma_sat_ru(a.y,b.y,c.y);
    res.z=__hfma_sat_rd(a.z,b.z,c.z);;
  }
  else {
    res.y=__hfma_sat_rd(a.y,b.y,c.y);
    res.z=__hfma_sat_ru(a.z,b.z,c.z);
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

__device__ half_gpu_st __hfma_sat(const half_gpu_st& a, const half& b,const half& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a.x,b,c);
  else res.x=__hfma_sat_rd(a.x,b,c);
  if (random>>1) {
    res.y=__hfma_sat_ru(a.y,b,c);
    res.z=__hfma_sat_rd(a.z,b,c);;
  }
  else {
    res.y=__hfma_sat_rd(a.y,b,c);
    res.z=__hfma_sat_ru(a.z,b,c);
  }
  res.error=a.error;

  return res;
}
__device__ half_gpu_st __hfma_sat(const half_gpu_st& a, const half& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a.x,b,c.x);
  else res.x=__hfma_sat_rd(a.x,b,c.x);
  if (random>>1) {
    res.y=__hfma_sat_ru(a.y,b,c.y);
    res.z=__hfma_sat_rd(a.z,b,c.z);;
  }
  else {
    res.y=__hfma_sat_rd(a.y,b,c.y);
    res.z=__hfma_sat_ru(a.z,b,c.z);
  }
  res.error=a.error;

  return res;
}
__device__ half_gpu_st __hfma_sat(const half_gpu_st& a, const half_gpu_st& b, const half& c)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a.x,b.x,c);
  else res.x=__hfma_sat_rd(a.x,b.x,c);
  if (random>>1) {
    res.y=__hfma_sat_ru(a.y,b.y,c);
    res.z=__hfma_sat_rd(a.z,b.z,c);;
  }
  else {
    res.y=__hfma_sat_rd(a.y,b.y,c);
    res.z=__hfma_sat_ru(a.z,b.z,c);
  }
  res.error=a.error;

  return res;
}

__device__ half_gpu_st __hfma_sat(const half& a, const half_gpu_st& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a,b.x,c.x);
  else res.x=__hfma_sat_rd(a,b.x,c.x);
  if (random>>1) {
    res.y=__hfma_sat_ru(a,b.y,c.y);
    res.z=__hfma_sat_rd(a,b.z,c.z);;
  }
  else {
    res.y=__hfma_sat_rd(a,b.y,c.y);
    res.z=__hfma_sat_ru(a,b.z,c.z);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}


__device__ half_gpu_st __hfma_sat(const half& a, const half& b, const half_gpu_st& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a,b,c.x);
  else res.x=__hfma_sat_rd(a,b,c.x);
  if (random>>1) {
    res.y=__hfma_sat_ru(a,b,c.y);
    res.z=__hfma_sat_rd(a,b,c.z);;
  }
  else {
    res.y=__hfma_sat_rd(a,b,c.y);
    res.z=__hfma_sat_ru(a,b,c.z);
  }

  
  res.error=c.error;
  return res;
}

__device__ half_gpu_st __hfma_sat(const half& a, const half_gpu_st& b, const half& c)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hfma_sat_ru(a,b.x,c);
  else res.x=__hfma_sat_rd(a,b.x,c);
  if (random>>1) {
    res.y=__hfma_sat_ru(a,b.y,c);
    res.z=__hfma_sat_rd(a,b.z,c);;
  }
  else {
    res.y=__hfma_sat_rd(a,b.y,c);
    res.z=__hfma_sat_ru(a,b.z,c);
  }

  if (b.accuracy==DIGIT_NOT_COMPUTED)
    b.approx_digit();
  
  res.error=b.error |((b.accuracy==0)  ? CADNA_MUL : 0); //FJ 19 June 2017
  // res.error=b.error |(b.accuracy ? CADNA_DIV : 0); //old
  return res;
}

///////////////////////////////////////
__device__ half_gpu_st __hneg(const half_gpu_st& a)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.error=0;
  random = RANDOMGPU();
  if (random&1) res.x=__hneg_ru(a.x);
  else res.x=__hneg_rd(a.x);
  if (random>>1) {
    res.y=__hneg_ru(a.y);
    res.z=__hneg_rd(a.z);;
  }
  else {
    res.y=__hneg_rd(a.y);
    res.z=__hneg_ru(a.z);
  }

  res.error=a.error;
 return res;	
}






///////////////////////////////////////

__device__ int operator==(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int operator==(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int operator==(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return res.computedzero();
}


__device__ int __heq(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int __heq(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int __heq(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return res.computedzero();
}

__device__ int __hequ(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return res.computedzero();
}

__device__ int __hequ(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return res.computedzero();
}

__device__ int __hequ(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return res.computedzero();
}

///////////////////////////////////////

///////////////////////////////////////

__device__ int operator!=(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int operator!=(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int operator!=(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return !res.computedzero();
}

__device__ int __hne(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int __hne(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int __hne(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return !res.computedzero();
}

__device__ int __hneu(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  return !res.computedzero();
}

__device__ int __hneu(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  return !res.computedzero();
}

__device__ int __hneu(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  return !res.computedzero();
}



///////////////////////////////////////

__device__ int operator>(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( (__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int operator>(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }


  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) > 3*__half2float(b) );
}



__device__ int operator>(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  >	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hgt(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( a.x + a.y + a.z ) >	( b.x + b.y + b.z ));
}


__device__ int __hgt(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }


  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) > 3*__half2float(b) );
}



__device__ int __hgt(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  >	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hgtu(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( a.x + a.y + a.z ) >	( b.x + b.y + b.z ));
}


__device__ int __hgtu(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }


  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) > 3*__half2float(b) );
}



__device__ int __hgtu(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  >	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

///////////////////////////////////////


__device__ int operator>=(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int operator>=(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	3*__half2float(b));
}



__device__ int operator>=(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hge(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hge(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	3*__half2float(b));
}



__device__ int __hge(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hgeu(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hgeu(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();

  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) >=	3*__half2float(b));
}



__device__ int __hgeu(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  >=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}
///////////////////////////////////////

__device__ int operator<(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int operator<(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) < 3*__half2float(b) );
}



__device__ int operator<(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hlt(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hlt(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) < 3*__half2float(b) );
}



__device__ int __hlt(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hltu(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return !r && ( ( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hltu(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;

  }

  return !r && ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) < 3*__half2float(b) );
}



__device__ int __hltu(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    b.error |= CADNA_BRANCHING;
  }


  return !r && ( 3*__half2float(a)  <	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}
///////////////////////////////////////


__device__ int operator<=(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int operator<=(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) <=	3*__half2float(b));
}



__device__ int operator<=(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hle(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hle(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) <=	3*__half2float(b));
}



__device__ int __hle(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}

__device__ int __hleu(const half_gpu_st& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b.x);
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b.y);
    res.z=__hsub_rd(a.z,b.z);;
  }
  else {
    res.y=__hsub_rd(a.y,b.y);
    res.z=__hsub_ru(a.z,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
    b.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}


__device__ int __hleu(const half_gpu_st& a, const half& b)
{
  half_gpu_st res;
  unsigned char random;


  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a.x,b);
  else res.x=__hsub_rd(a.x,b);

  if (random>>1) {
    res.y=__hsub_ru(a.y,b);
    res.z=__hsub_rd(a.z,b);;
  }
  else {
    res.y=__hsub_rd(a.y,b);
    res.z=__hsub_ru(a.z,b);
  }
  int r=res.isnumericalnoise();
  if (r) {
    a.error |= CADNA_BRANCHING;
  }

  return r || ( (__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) <=	3*__half2float(b));
}



__device__ int __hleu(const half& a, const half_gpu_st& b)
{
  half_gpu_st res;
  unsigned char random;

  res.accuracy=DIGIT_NOT_COMPUTED;
  random = RANDOMGPU();
  if (random&1) res.x=__hsub_ru(a,b.x);
  else res.x=__hsub_rd(a,b.x);

  if (random>>1) {
    res.y=__hsub_ru(a,b.y);
    res.z=__hsub_rd(a,b.z);;
  }
  else {
    res.y=__hsub_rd(a,b.y);
    res.z=__hsub_ru(a,b.z);
  }
  int r=res.isnumericalnoise();
  if (r) {

    b.error |= CADNA_BRANCHING;
  }

  return r || ( 3*__half2float(a)  <=	( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) ));
}
///////////////////////////////////////
 __device__  half_gpu_st fabsf(const  half_gpu_st& a) 
{ 
   half_gpu_st res; 
   res.x = __float2half(fabsf(__half2float(a.x))); 
   res.y = __float2half(fabsf(__half2float(a.y))); 
   res.z = __float2half(fabsf(__half2float(a.z))); 
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

 __device__  half_gpu_st fabs(const  half_gpu_st& a) 
{ 
   half_gpu_st res; 
   res.x = __float2half(fabs(__half2float(a.x))); 
   res.y = __float2half(fabs(__half2float(a.y))); 
   res.z = __float2half(fabs(__half2float(a.z))); 
 /*  res.x = fabs(a.x); 
   res.y = fabs(a.y); 
   res.z = fabs(a.z); */
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half_gpu_st hsqrt(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__hsqrt_ru(a.x);	    
  else res.x=__hsqrt_rd(a.x);

  if (random>>1) {
     res.y=__hsqrt_ru(a.y);				 
     res.z=__hsqrt_rd(a.z);				 
  }							 
  else {							 
    res.y=__hsqrt_rd(a.y);				 
    res.z=__hsqrt_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


///////////////////////////////////////

__device__  half_gpu_st hsin(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hsin_ru(a.x);	    
  else res.x=hsin_rd(a.x);

  if (random>>1) {
     res.y=hsin_ru(a.y);				 
     res.z=hsin_rd(a.z);				 
  }							 
  else {							 
    res.y=hsin_rd(a.y);				 
    res.z=hsin_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half_gpu_st hcos(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hcos_ru(a.x);	    
  else res.x=hcos_rd(a.x);

  if (random>>1) {
     res.y=hcos_ru(a.y);				 
     res.z=hcos_rd(a.z);				 
  }							 
  else {							 
    res.y=hcos_rd(a.y);				 
    res.z=hcos_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half_gpu_st hexp(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hexp_ru(a.x);	    
  else res.x=hexp_rd(a.x);

  if (random>>1) {
     res.y=hexp_ru(a.y);				 
     res.z=hexp_rd(a.z);				 
  }							 
  else {							 
    res.y=hexp_rd(a.y);				 
    res.z=hexp_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half_gpu_st hlog(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hlog_ru(a.x);	    
  else res.x=hlog_rd(a.x);

  if (random>>1) {
     res.y=hlog_ru(a.y);				 
     res.z=hlog_rd(a.z);				 
  }							 
  else {							 
    res.y=hlog_rd(a.y);				 
    res.z=hlog_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}


__device__  half_gpu_st hlog2(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hlog2_ru(a.x);	    
  else res.x=hlog2_rd(a.x);

  if (random>>1) {
     res.y=hlog2_ru(a.y);				 
     res.z=hlog2_rd(a.z);				 
  }							 
  else {							 
    res.y=hlog2_rd(a.y);				 
    res.z=hlog2_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half_gpu_st hlog10(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hlog10_ru(a.x);	    
  else res.x=hlog10_rd(a.x);

  if (random>>1) {
     res.y=hlog10_ru(a.y);				 
     res.z=hlog10_rd(a.z);				 
  }							 
  else {							 
    res.y=hlog10_rd(a.y);				 
    res.z=hlog10_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

__device__  half_gpu_st hrcp(const  half_gpu_st& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=hrcp_ru(a.x);	    
  else res.x=hrcp_rd(a.x);

  if (random>>1) {
     res.y=hrcp_ru(a.y);				 
     res.z=hrcp_rd(a.z);				 
  }							 
  else {							 
    res.y=hrcp_rd(a.y);				 
    res.z=hrcp_ru(a.z);					
  }				
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   return(res); 
}

///////////////////////////////////////
__device__  half_gpu_st fmaxf(const half& a, const half_gpu_st& b) 
{ 
  half_gpu_st res;   
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a,b.x);			 
  else res.x=__hsub_rd(a,b.x);
  if (random>>1) {						 
    res.y=__hsub_ru(a,b.y);				 
    res.z=__hsub_rd(a,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a,b.y);				 
    res.z=__hsub_ru(a,b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*__half2float(a) > (__half2float(b.x) + __half2float(b.y) + __half2float(b.z))) {
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


__device__  half_gpu_st fmaxf(const  half_gpu_st& a, const  half& b) 
{ 
  half_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a.x,b);			 
  else res.x=__hsub_rd(a.x,b);
  if  (random>>1) {
    res.y=__hsub_ru(a.y,b);				 
    res.z=__hsub_rd(a.z,b);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a.y,b);				 
    res.z=__hsub_ru(a.z,b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ((__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) > 3.f*__half2float(b) ) {
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


__device__  half_gpu_st fmaxf(const  half_gpu_st& a, const  half_gpu_st& b) 
{ 
  half_gpu_st res;  
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a.x,b.x);			 
  else res.x=__hsub_rd(a.x,b.x);				 
  if (random>>1) {						 
    res.y=__hsub_ru(a.y,b.y);				 
    res.z=__hsub_rd(a.z,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a.y,b.y);				 
    res.z=__hsub_ru(a.z,b.z);					
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
    if ((__half2float( a.x) + __half2float(a.y) + __half2float(a.z) ) > ( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

//////

__device__  half_gpu_st fminf(const half& a, half_gpu_st& b) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a,b.x);			 
  else res.x=__hsub_rd(a,b.x);
  if (random>>1) {						 
    res.y=__hsub_ru(a,b.y);				 
    res.z=__hsub_rd(a,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a,b.y);				 
    res.z=__hsub_ru(a,b.z);					
  }			
  if (res.isnumericalnoise()){
      res.x=a; 
      res.y=a; 
      res.z=a; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if ( 3.f*__half2float(a) < (__half2float(b.x) + __half2float(b.y) + __half2float(b.z))) {
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


__device__  half_gpu_st fminf(const  half_gpu_st& a, const  half& b) 
{ 
  half_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a.x,b);			 
  else res.x=__hsub_rd(a.x,b);
  if (random>>1) {
    res.y=__hsub_ru(a.y,b);				 
    res.z=__hsub_rd(a.z,b);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a.y,b);				 
    res.z=__hsub_ru(a.z,b);					
  }			
  if (res.isnumericalnoise()){
      res.x=b; 
      res.y=b; 
      res.z=b; 
      res.accuracy=3;
      res.error=CADNA_BRANCHING;
  }
  else { 
    if (( __half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) < 3.f*__half2float(b) ) {
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



__device__  half_gpu_st fminf(const  half_gpu_st& a, const  half_gpu_st& b) 
{ 
  half_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();

  if (random&1) res.x=__hsub_ru(a.x,b.x);			 
  else res.x=__hsub_rd(a.x,b.x);

  if (random>>1) {				 
    res.y=__hsub_ru(a.y,b.y);				 
    res.z=__hsub_rd(a.z,b.z);				 
  }					       	 
  else {							 
    res.y=__hsub_rd(a.y,b.y);				 
    res.z=__hsub_ru(a.z,b.z);					
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
    if ((__half2float(a.x) + __half2float(a.y) + __half2float(a.z) ) < ( __half2float(b.x) + __half2float(b.y) + __half2float(b.z) )) {
	res=a;	
    }
    else {								 
	res=b;	
    }
  }
  return(res); 
}

///////////////////////////////////////

__device__  half_gpu_st __float2half_gpu_st(const  float& a) 
{ 
  half_gpu_st res; 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x = __float2half_ru(a);	    
  else res.x=__float2half_rd(a);

  if (random>>1) {
     res.y=__float2half_ru(a);				 
     res.z=__float2half_rd(a);				 
  }							 
  else {							 
    res.y=__float2half_rd(a);				 
    res.z=__float2half_ru(a);					
  }				
   //res.accuracy=DIGIT_NOT_COMPUTED; 
   //res.error=a.error;
   return(res); 
}

__device__  float __half_gpu_st2float(const  half_gpu_st& a) 
{ 
  float res_f = __half2float(a.x);
 /* half_gpu_st res;
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=half2float_ru(a.x);	    
  else res.x=half2float_rd(a.x);

  if (random>>1) {
     res.y=half2float_ru(a.y);				 
     res.z=half2float_rd(a.z);				 
  }							 
  else {							 
    res.y=half2float_rd(a.y);				 
    res.z=half2float_ru(a.z);					
  }
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   */
   return(res_f); 
}


__device__  float_gpu_st __half_gpu_st2float_gpu_st(const half_gpu_st& a) 
{ 
 
  float_gpu_st res;
  res.x = __half2float(a.x);
  res.y = __half2float(a.y);
  res.z = __half2float(a.z);
 /* unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__half2float(a.x);	    
  else res.x=__half2float(a.x);

  if (random>>1) {
     res.y=__half2float(a.y);				 
     res.z=__half2float(a.z);				 
  }							 
  else {							 
    res.y=__half2float(a.y);				 
    res.z=__half2float(a.z);					
  }*/
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   
   return(res); 
}

__device__  half_gpu_st __float_gpu_st2half_gpu_st(const float_gpu_st& a) 
{ 
 
  half_gpu_st res;
 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__float2half_ru(a.x);	    
  else res.x=__float2half_rd(a.x);

  if (random>>1) {
     res.y=__float2half_ru(a.y);				 
     res.z=__float2half_rd(a.z);				 
  }							 
  else {							 
    res.y=__float2half_rd(a.y);				 
    res.z=__float2half_ru(a.z);					
  }
  res.accuracy=DIGIT_NOT_COMPUTED; 
  res.error=a.error;
   return(res); 
}
__device__  float_gpu_st __half_gpu_st2float_gpu_st(half_gpu_st& a) 
{ 
 
  float_gpu_st res;
  res.x = __half2float(a.x);
  res.y = __half2float(a.y);
  res.z = __half2float(a.z);
 /* unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__half2float(a.x);	    
  else res.x=__half2float(a.x);

  if (random>>1) {
     res.y=__half2float(a.y);				 
     res.z=__half2float(a.z);				 
  }							 
  else {							 
    res.y=__half2float(a.y);				 
    res.z=__half2float(a.z);					
  }*/
   res.accuracy=DIGIT_NOT_COMPUTED; 
   res.error=a.error;
   
   return(res); 
}

__device__  half_gpu_st __float_gpu_st2half_gpu_st(float_gpu_st& a) 
{ 
 
  half_gpu_st res;
 
  unsigned char random;
  random = RANDOMGPU();
 
  if (random&1) res.x=__float2half_ru(a.x);	    
  else res.x=__float2half_rd(a.x);

  if (random>>1) {
     res.y=__float2half_ru(a.y);				 
     res.z=__float2half_rd(a.z);				 
  }							 
  else {							 
    res.y=__float2half_rd(a.y);				 
    res.z=__float2half_ru(a.z);					
  }
  res.accuracy=DIGIT_NOT_COMPUTED; 
  res.error=a.error;
   return(res); 
}


///////////////////////////////////////

__device__ void half_gpu_st::modify(const int &a)
{
  accuracy |=a;
}


__device__ half_gpu_st& half_gpu_st::operator=(const half &a)
{
  x=a;
  y=a;
  z=a;
  accuracy=3;
  error=0;
  return *this ;
}

__device__ int  half_gpu_st::nb_significant_digit() const
{
  float x0,x1,x2,xx;
	float x_h,y_h,z_h;
	x_h = __half2float(x); 	
	y_h = __half2float(y); 	
	z_h = __half2float(z); 	
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



