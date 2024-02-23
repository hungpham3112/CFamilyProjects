#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>

#ifndef N
//#define N 0x2000000
#define N 0x1000000
//#define N 4
#endif

#ifndef TAILLE_BLOC_X
//#define TAILLE_BLOC_X 160
#define TAILLE_BLOC_X 4
#endif

#ifndef NLOOP
#define NLOOP 1
#endif

#ifndef CPT
#define CPT 1
#endif


#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu.cu"
#define DATATYPE float
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU half2_gpu_st
#define DATATYPECADNAGPUF float_gpu_st
#else
#include "cuda_fp16.h"
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU half2
#define DATATYPECADNAGPUF float
#endif

#define SEED 1


using namespace std;

// Génération aléatoire du tableau :
__host__ void aleaTabSt(DATATYPECADNA* T, int n){
  long int i;
  srand(SEED);

  for(i = 0; i < n; i++){
#ifdef CADNA
    T[i] =DATATYPECADNA(0.5f + (DATATYPE)rand()/(DATATYPE)RAND_MAX);
#else
    T[i] =(0.5f + (DATATYPE)rand()/(DATATYPE)RAND_MAX);
#endif
  }
}

// Mesures :
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

__host__ void compute(DATATYPECADNA *a, DATATYPECADNA *b){
  for (int i = 0; i < N/2; i++){
    DATATYPECADNA f = a[i];
    DATATYPECADNA g = b[i];
    for (int j = 0; j < NLOOP; j++)
      f = g + f;
    a[i] = f;
  }
}
/*
__global__ void computeGPU(DATATYPECADNAGPUF *a, DATATYPECADNAGPUF *b){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();

  if(i<N){
    DATATYPECADNAGPU f = __float_gpu_st2half2_gpu_st(a[i]);
    DATATYPECADNAGPU g = __float_gpu_st2half2_gpu_st(b[i]);
    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    a[i] = __half2_gpu_st2float_gpu_st(f);
  }

#else
  if(i<N){
    DATATYPECADNAGPU f = __float2half2_rn(a[i]);
    DATATYPECADNAGPU g = __float2half2_rn(b[i]);
    for(int j=0; j<NLOOP; j++){
      f = __hadd2(g , f);
      //f = g + f;
    }
    a[i] = __high2float(f);
  }
#endif
}
*/




















/*
__forceinline__ __device__ void reduceInShared(half2 * const v)
{
	if(threadIdx.x<64)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+64]);
	__syncthreads();
	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+32]);
	__syncthreads();
	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+16]);
	__syncthreads();
	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+8]);				        __syncthreads();
       	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+4]);
	__syncthreads();
	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+2]);
	__syncthreads();
	if(threadIdx.x<32)
		v[threadIdx.x] = __hadd2( v[threadIdx.x], v[threadIdx.x+1]);
	__syncthreads();
}
*/


__global__ void computeGPU(DATATYPECADNAGPUF *a0, DATATYPECADNAGPUF *a1, DATATYPECADNAGPUF *b0, DATATYPECADNAGPUF *b1){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();

  if(i<N){
    DATATYPECADNAGPU f = __float_gpu_st2half2_gpu_st(a0[i], a1[i]);
    DATATYPECADNAGPU g = __float_gpu_st2half2_gpu_st(b0[i], b1[i]);
   // printf(" thread %d   a[%d] = %f , a[%d] = %f \n", threadIdx.x,  i, (float)a[i], i+1, (float)a[i+1]);
    //printf(" thread %d   b[%d] = %f , b[%d] = %f \n", threadIdx.x, i, (float)b[i], i+1, (float)b[i+1]);

    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    
    a0[i] = __half2_gpu_st2float_gpu_st_low(f);
    a1[i] = __half2_gpu_st2float_gpu_st_high(f);

    //printf(" low + high thread %d   a[%d] = %f  \n", threadIdx.x,  i, (float)a[i]);
    }
#else
  if(i<N){
    DATATYPECADNAGPU f = __floats2half2_rn(a0[i], a1[i]);
    DATATYPECADNAGPU g = __floats2half2_rn(b0[i], b1[i]);
    for(int j=0; j<NLOOP; j++){
      f = __hadd2(g , f);
      //f = g + f;
    }
    a0[i] = __low2float(f);
    a1[i] = __high2float(f);
  }
#endif
}



/*
__global__ void computeGPU(DATATYPECADNAGPUF *a, DATATYPECADNAGPUF *b){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();

  if(i<N && (i%2 == 0)){
    DATATYPECADNAGPU f = __float_gpu_st2half2_gpu_st(a[i], a[i+1]);
    DATATYPECADNAGPU g = __float_gpu_st2half2_gpu_st(b[i], b[i+1]);
   // printf(" thread %d   a[%d] = %f , a[%d] = %f \n", threadIdx.x,  i, (float)a[i], i+1, (float)a[i+1]);
    //printf(" thread %d   b[%d] = %f , b[%d] = %f \n", threadIdx.x, i, (float)b[i], i+1, (float)b[i+1]);

 
    a[i] = __half2_gpu_st2float_gpu_st_low(f);
    a[i+1] = __half2_gpu_st2float_gpu_st_high(f);

    //printf(" low + high thread %d   a[%d] = %f  \n", threadIdx.x,  i, (float)a[i]);
  }
#else
  if(i<N && (i%2 == 0)){
    DATATYPECADNAGPU f = __floats2half2_rn(a[i], a[i+1]);
    DATATYPECADNAGPU g = __floats2half2_rn(b[i], b[i+1]);

    a[i] = __low2float(f);
    a[i+1] = __high2float(f);
  }
#endif
}
*/




int main(int argc, char **argv){

  cudaError_t sync_error;
#ifdef CADNA
  cadna_init(-1, CADNA_INTRINSIC | CADNA_CANCEL);
#endif
  int ic;
#ifndef CADNA
  long int taille_totale = (N)*sizeof(DATATYPE);
#else
  long int taille_totale = (N)*sizeof(DATATYPECADNA);
#endif

  double t_startGPU, t_endGPU;
  double t_GPU;

  DATATYPECADNAGPUF *d_a0, *d_b0, *d_a1, *d_b1;

  DATATYPECADNA *a0 = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *a1 = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *b0 = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *b1 = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *a0_GPU = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *a1_GPU = (DATATYPECADNA*)malloc(taille_totale);

  aleaTabSt(a0, N);
  aleaTabSt(a1, N);
  aleaTabSt(b0, N);
  aleaTabSt(b1, N);

  /* Allocation GPU : */
  cudaMalloc((void **) &d_a0, taille_totale);
  cudaMalloc((void **) &d_a1, taille_totale);
  cudaMalloc((void **) &d_b0, taille_totale);
  cudaMalloc((void **) &d_b1, taille_totale);

  /* Transferts CPU -> GPU (synchrones) : */
  cudaMemcpy(d_a0, a0, taille_totale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a1, a1, taille_totale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, b0, taille_totale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, b1, taille_totale, cudaMemcpyHostToDevice);

  /* Lancement de kernel (asynchrone) : */
  dim3 threadsParBloc(TAILLE_BLOC_X);
  //dim3 tailleGrille(ceil((N/(float) TAILLE_BLOC_X))); // Nombre de blocs
  dim3 tailleGrille(ceil((N/(float) TAILLE_BLOC_X))); // Nombre de blocs


  t_startGPU = my_gettimeofday();
  for(ic=0; ic<CPT;ic++){
    computeGPU<<<tailleGrille, threadsParBloc>>>(d_a0, d_a1, d_b0, d_b1);
    cudaDeviceSynchronize();
  }
  t_endGPU =  my_gettimeofday();

  sync_error = cudaGetLastError();
  if(sync_error != cudaSuccess) {
    fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",
	    __FILE__ , __LINE__, cudaGetErrorString(sync_error));
    exit(EXIT_FAILURE);
  }

#ifdef NUMCHECK
  double t_startCPU, t_endCPU;
  t_startCPU = my_gettimeofday();
  for(ic=0; ic<CPT;ic++){
    compute(a0,b0);

  }
  t_endCPU = my_gettimeofday();
#endif

  cudaMemcpy(a0_GPU, d_a0, taille_totale, cudaMemcpyDeviceToHost);
  cudaMemcpy(a1_GPU, d_a1, taille_totale, cudaMemcpyDeviceToHost);

  t_GPU = (t_endGPU - t_startGPU)/CPT;
#ifdef NUMCHECK
  double t_CPU;
  t_CPU = (t_endCPU - t_startCPU)/CPT;

  DATATYPECADNA max, moy;

  max = (DATATYPE)0.f;
  moy = (DATATYPE)0.f;
  int i;
  for(i = 0; i < N; i++) {
    if (a0[i] != (DATATYPE)0.f) {
      DATATYPECADNA relerr = fabsf((a0[i]-a0_GPU[i])/a0[i]);
      moy = moy + relerr;
      if (relerr > max)
	max = relerr;
    }
  }

  moy = moy / (DATATYPE)N;

  cerr << TAILLE_BLOC_X << " " << max << " " << moy << " " << t_GPU << " " << t_CPU << endl;
#else
  cerr << TAILLE_BLOC_X << " " << t_GPU << endl;
#endif

#ifdef CADNA
  cadna_end();
#endif
  /* Libération mémoire GPU et CPU : */
  cudaFree(d_a0); cudaFree(d_b0);
  cudaFree(d_a1); cudaFree(d_b1);
  free(a0); free(b0); free(a0_GPU);
  free(a1); free(b1); free(a1_GPU);

  // fclose(fic);
  return EXIT_SUCCESS;
}
