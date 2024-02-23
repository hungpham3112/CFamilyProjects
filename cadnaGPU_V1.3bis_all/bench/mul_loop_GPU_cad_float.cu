#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>


#ifndef N
#define N 0x2000000
#endif

#ifndef TAILLE_BLOC_X
#define TAILLE_BLOC_X 160
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
#define DATATYPECADNAGPU float_gpu_st
#else
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU float
#endif

#define SEED 1


using namespace std;

// Génération aléatoire du tableau :
__host__ void aleaTabSt(DATATYPECADNA* T, int n){
  long int i;
  srand(SEED);

  for(i = 0; i < n; i++){
#ifdef CADNA
    T[i] = DATATYPECADNA(1.0f + 0.01f*((DATATYPE)rand()/(DATATYPE)RAND_MAX));
#else
    T[i] = 1.0f + 0.01f*((DATATYPE)rand()/(DATATYPE)RAND_MAX);
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
      f = g * f;
    a[i] = f;
  }
}

__global__ void computeGPU(DATATYPECADNAGPU *a, DATATYPECADNAGPU *b){
  int i = blockDim.x * blockIdx.x + threadIdx.x ;
#ifdef CADNA
  cadna_init_gpu();
#endif

  if(i<N){
    DATATYPECADNAGPU f = a[i];
    DATATYPECADNAGPU g = b[i];
    for(int j=0; j<NLOOP; j++){
      f = g * f;
    }
    a[i] = f;
  }

}


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

  DATATYPECADNAGPU *d_a, *d_b;

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

  max = 0.f;
  moy = 0.f;
  int i;
  for(i = 0; i < N; i++) {
    if (a[i] != 0.f) {
      DATATYPECADNA relerr = fabsf((a[i]-a_GPU[i])/a[i]);
      moy = moy + relerr;
      if (relerr > max)
	max = relerr;
    }
  }

  moy = moy / (float)N;

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
