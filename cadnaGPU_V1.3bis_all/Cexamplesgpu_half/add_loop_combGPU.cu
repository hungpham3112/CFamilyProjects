#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>
#include "cuda_fp16.h"


#ifndef N
//#define N 0x10000000
#define N 4
#endif

#ifndef TAILLE_BLOC_X
#define TAILLE_BLOC_X 32
#endif

#ifndef NLOOP
#define NLOOP 1
#endif

#ifndef CPT
#define CPT 1
#endif


// CADNA
#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu.cu"
#define DATATYPE half
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU half_gpu_st
#else
#define DATATYPE half
#define DATATYPECADNA half
#define DATATYPECADNAGPU half
#endif



#define SEED 1

using namespace std;

// Génération aléatoire du tableau :
__host__ void aleaTabSt(float* T, int n){
    long int i;
    srand(SEED);
    //    posix_memalign((void **)&T, 64, N * sizeof(float));
    for(i = 0; i < n; i++){
        T[i] =(float)(0.5f + (float)rand()/(float)RAND_MAX);
    }
}

// Conversion :
__device__ void convertIntoSt(DATATYPE * T, DATATYPECADNA * Tst, int n){
     for(int i = 0; i < n; i++){
        Tst[i] = T[i];
    }
}

// Mesures :
double my_gettimeofday(){
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/*
__host__ void compute(float *a, float *b){
    for (int i = 0; i < N; i++){
        float f = a[i];
        for (int j = 0; j < NLOOP; j++)
            f = b[i] + f;
        a[i] = f;
    }
}
*/
__global__ void computeGPU(float *a, float *b){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        half f = __float2half(a[i]);
        for(int j=0; j<NLOOP; j++){
            f = __hadd(__float2half(b[i]), f);
        }
        a[i] = __half2float(f);
    }
}




int main(int argc, char **argv){
    int i, ic;
    int taille_totale = N*sizeof(half);

#ifdef DEBUG
    int nb_oper =0;
#endif
    //double t_startCPU, t_endCPU;
    double t_startGPU, t_endGPU;


    float *d_a, *d_b;

    float *a = (float*)malloc(taille_totale);
    float *b = (float*)malloc(taille_totale);
    aleaTabSt(a, N);
    aleaTabSt(b, N);



 	float *a_GPU = (float*)malloc(taille_totale);

    // Concatène par défaut, écrase si nouvelle série.
    //FILE* fic=(TAILLE_BLOC_X == 32)?fopen("operBXaddComb.res", "w"):fopen("operBXaddComb.res", "a");



   // int valide = 1;




    /* Allocation GPU : */
    cudaMalloc((void **) &d_a, taille_totale);
    cudaMalloc((void **) &d_b, taille_totale);

    // Transferts CPU -> GPU (synchrones) :
    // Passage en DATATYPECADNAGPU implicite.
    cudaMemcpy(d_a, a, taille_totale, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, taille_totale, cudaMemcpyHostToDevice);


    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);
    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X)); // Nombre de blocs



    t_startGPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        computeGPU<<<tailleGrille, threadsParBloc>>>(d_a,d_b);
        cudaDeviceSynchronize();
    }
    t_endGPU =  my_gettimeofday();



    /*t_startCPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        compute(a,b);
    }
    t_endCPU = my_gettimeofday();
*/

    cudaMemcpy(a_GPU, d_a, taille_totale, cudaMemcpyDeviceToHost);


    /*for(i = 0; i < 5; i++)
        cerr << "Result[" << i << "] :" << a[i] << endl;

    cout << "Time: " << t_endCPU-t_startCPU << endl;
*/
    for(i = 0; i < 5; i++)
        cerr << "Result[" << i << "] :" << a[i] << endl;

    cout << "Time: " << t_endGPU-t_startGPU << endl;


  //  t_CPU = (t_endCPU - t_startCPU)/CPT;
   // t_GPU = (t_endGPU - t_startGPU)/CPT;
/*
    for(i = 0; i < 5; i++)
        if (a[i] != a_GPU[i]){
            valide = 0;
            cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;
            fprintf(fic, "%d %d %f %f %d %d\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU, N, NLOOP, CPT);
            exit(1);
        }
*/
  //  fprintf(fic, "%d %d %f %f %d %d\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU, N, NLOOP, CPT);
    //cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;

    /* Libération mémoire GPU et CPU : */
    //cudaFree(d_a); cudaFree(d_b);
    //free(a); free(b); free(a_GPU);


    //fclose(fic);
    return EXIT_SUCCESS;
}
