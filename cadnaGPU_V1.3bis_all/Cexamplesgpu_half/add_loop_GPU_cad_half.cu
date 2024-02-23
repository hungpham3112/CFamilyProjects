#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>


#ifndef N
#define N 0x2000000

//#define N 32
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

#ifdef DOUBLE
#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu.cu"
#define DATATYPE double
#define DATATYPECADNA double_st
#define DATATYPECADNAGPU double_gpu_st
#else
#define DATATYPE double
#define DATATYPECADNA double
#define DATATYPECADNAGPU double
#endif
#elif FLOAT
#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu.cu"
#define DATATYPE float
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU float_gpu_st
#else
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU float
#endif
#else
#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu_half.cu"
#include "cadna_gpu_float.h"
#include "cadna_gpu_float.cu"
#define DATATYPE float
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU half_gpu_st
#define DATATYPECADNAGPUF float_gpu_st
#else
#include "cuda_fp16.h"
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU half
#define DATATYPECADNAGPUF float
#endif
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
  for (int i = 0; i < N; i++){
    DATATYPECADNA f = a[i];
    DATATYPECADNA g = b[i];
    for (int j = 0; j < NLOOP; j++)
      f = g + f;
    a[i] = f;
  }
}



__global__ void computeGPU(DATATYPECADNAGPUF *a, DATATYPECADNAGPUF *b){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();

  if(i<N){
    DATATYPECADNAGPU f = __float_gpu_st2half_gpu_st(a[i]);
    DATATYPECADNAGPU g = __float_gpu_st2half_gpu_st(b[i]);
    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    a[i] = __half_gpu_st2float_gpu_st(f);
  }

#else
  if(i<N){
    DATATYPECADNAGPU f = __float2half(a[i]);
    DATATYPECADNAGPU g = __float2half(b[i]);
    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    a[i] = __half2float(f);
  }
#endif
}


/*
__global__ void computeGPU(DATATYPECADNAGPUF *a, DATATYPECADNAGPUF *b){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();

  if(i<N){
    DATATYPECADNAGPU f = __float_gpu_st2half_gpu_st(a[i]);
    DATATYPECADNAGPU g = __float_gpu_st2half_gpu_st(b[i]);
   
    a[i] = __half_gpu_st2float_gpu_st(f);
  }

#else
  if(i<N){
    DATATYPECADNAGPU f = __float2half(a[i]);
    DATATYPECADNAGPU g = __float2half(b[i]);
    
    a[i] = __half2float(f);
  }
#endif
}
*/
/*
__device__ void computeGPUcon(DATATYPECADNAGPU f, DATATYPECADNAGPU g){
#ifdef CADNA
    
    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    //a[i] = f;
  //}

#else
  if(i<N){
    DATATYPECADNAGPU f = a[i];
    DATATYPECADNAGPU g = b[i];
    for(int j=0; j<NLOOP; j++){
      f = g + f;
    }
    a[i] = f;
  }
#endif
}

//double t_startGPU, t_endGPU;



__global__ void computeGPU(DATATYPECADNAGPUF *a, DATATYPECADNAGPUF *b)
{

  int i = blockDim.x * blockIdx.x + threadIdx.x ;
  
  
  if(i<N){
	cadna_init_gpu();
	DATATYPECADNAGPU f = __float_gpu_st2half_gpu_st(a[i]);
	DATATYPECADNAGPU g = __float_gpu_st2half_gpu_st(b[i]);
  //t_startGPU = 0;
  
	computeGPUcon(f, g);

  //t_endGPU =  my_gettimeofday();
  	a[i] = __half_gpu_st2float_gpu_st(f);


  }

}
*/


int main(int argc, char **argv){

  cudaError_t sync_error;
#ifdef CADNA
  cadna_init(-1, CADNA_INTRINSIC | CADNA_CANCEL);
#endif
  int ic;
#ifndef CADNA
  long int taille_totale = N*sizeof(DATATYPE);
#else
  long int taille_totale = N*sizeof(DATATYPECADNA);
#endif

  double t_startGPU, t_endGPU;
  double t_GPU;

  DATATYPECADNAGPUF *d_a, *d_b;

  DATATYPECADNA *a = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *b = (DATATYPECADNA*)malloc(taille_totale);
  DATATYPECADNA *a_GPU = (DATATYPECADNA*)malloc(taille_totale);

  aleaTabSt(a, N);
  aleaTabSt(b, N);

  /* Allocation GPU : */
  cudaMalloc((void **) &d_a, taille_totale);
  cudaMalloc((void **) &d_b, taille_totale);

  /* Transferts CPU -> GPU (synchrones) : */
  cudaMemcpy(d_a, a, taille_totale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, taille_totale, cudaMemcpyHostToDevice);

  /* Lancement de kernel (asynchrone) : */
  dim3 threadsParBloc(TAILLE_BLOC_X);
  dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X)); // Nombre de blocs



  t_startGPU = my_gettimeofday();
  for(ic=0; ic<CPT;ic++){
    computeGPU<<<tailleGrille, threadsParBloc>>>(d_a, d_b);
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
    compute(a,b);

  }
  t_endCPU = my_gettimeofday();
#endif

  cudaMemcpy(a_GPU, d_a, taille_totale, cudaMemcpyDeviceToHost);

  t_GPU = (t_endGPU - t_startGPU)/CPT;
#ifdef NUMCHECK
  double t_CPU;
  t_CPU = (t_endCPU - t_startCPU)/CPT;

  DATATYPECADNA max, moy;

  max = (DATATYPE)0.f;
  moy = (DATATYPE)0.f;
  int i;
  for(i = 0; i < N; i++) {
    if (a[i] != (DATATYPE)0.f) {
      DATATYPECADNA relerr = fabsf((a[i]-a_GPU[i])/a[i]);
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
  cudaFree(d_a); cudaFree(d_b);
  free(a); free(b); free(a_GPU);

  // fclose(fic);
  return EXIT_SUCCESS;
}
