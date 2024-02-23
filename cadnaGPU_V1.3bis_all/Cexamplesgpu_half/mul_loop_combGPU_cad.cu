#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>
#include "cuPrintf.cuh"
#include "cuPrintf.cu"



#ifndef N
#define N 0x8000000
#endif

#ifndef TAILLE_BLOC_X
#define TAILLE_BLOC_X 160
#endif

#ifndef NLOOP
#define NLOOP 1024
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

#define CUDA_ERROR(cuda_call) {                 \
    cudaError_t error = cuda_call;              \
    if(error != cudaSuccess){                   \
        fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",      \
                __FILE__ , __LINE__, cudaGetErrorString(error));  \
        exit(EXIT_FAILURE);                 \
    }                               \
    /* CUDA_SYNC_ERROR();   */                   \
}

using namespace std;

//__device__ unsigned int* seed;


// Génération aléatoire du tableau :
__host__ void aleaTabSt(DATATYPECADNA* T, int n, int seed){
    long int i;
    srand(seed);

    //    posix_memalign((void **)&T, 64, N * sizeof(float));
    for(i = 0; i < n; i++){
#ifdef CADNA
        T[i] = float_st(1.0f + 0.01f*((float)rand()/(float)RAND_MAX));
#else
        T[i] =(0.5f + (float)rand()/(float)RAND_MAX);
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
        for (int j = 0; j < 1 /*NLOOP*/; j++)
            f = b[i] * f;
        a[i] = f;
    }
}

__global__ void computeGPU(DATATYPECADNAGPU *a, DATATYPECADNAGPU *b){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;

    if(i<N){
#ifdef DISPLAY
        //    if(i<500) cuPrintf("i=%d\n",i);//d_a[i]=%f\n",i, a[i]);
#endif
        DATATYPECADNAGPU f = a[i];
        for(int j=0; j<NLOOP; j++){
            f = b[i] * f;
        }
        a[i] = f;
    }

}


int main(int argc, char **argv){


    InitCuda(1, N);
    cudaError_t sync_error;
    int Cadna = 0;
#ifdef CADNA
    cadna_init(-1, CADNA_INTRINSIC);
    Cadna = 1;
#endif
    int i, ic;
#ifndef CADNA
    long int taille_totale = N*sizeof(DATATYPE);
#else
    long int taille_totale = N*sizeof(DATATYPECADNA);
    // sizeof(DATATYPECADNA) équiv sizeof(DATATYPECADNAGPU)
#endif

    double t_startCPU, t_endCPU;
    double t_startGPU, t_endGPU;
    double t_CPU, t_GPU;

    DATATYPECADNAGPU *d_a, *d_b;

#ifdef DISPLAY
    cout <<"N =  " << N <<endl;
    cout <<"size_t =  " << sizeof(size_t) <<endl;
    cout <<"taille DTC " << sizeof(DATATYPECADNA) <<endl;
#endif


    DATATYPECADNA *a = (DATATYPECADNA*)malloc(taille_totale);
    DATATYPECADNA *b = (DATATYPECADNA*)malloc(taille_totale);
    DATATYPECADNA *a_GPU = (DATATYPECADNA*)malloc(taille_totale);

#ifdef DISPLAY
    cout <<"Allocation CPU OK" << endl;
#endif
    // Concatène par défaut, écrase si nouvelle série.
    FILE* fic=fopen("mulCombBX.res", "a");

    int valide = 1;

    aleaTabSt(a, N, SEED);
    aleaTabSt(b, N, SEED+1);

#ifdef DISPLAY
    cout <<"Génération aléatoire de stochastiques OK" << endl;
#endif

    /* Allocation GPU : */
    cudaMalloc((void **) &d_a, taille_totale);
    cudaMalloc((void **) &d_b, taille_totale);

    /* Transferts CPU -> GPU (synchrones) : */
    cudaMemcpy(d_a, a, taille_totale, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, taille_totale, cudaMemcpyHostToDevice);

    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);
    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X)); // Nombre de blocs

#ifdef DISPLAY
    cout <<"Allocation et copie sur GPU OK" << endl;
    cout <<"Début du calcul sur GPU" << endl;
    cout <<"a[4] =" << a[4] << endl;
#endif


    t_startGPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        computeGPU<<<tailleGrille, threadsParBloc>>>(d_a, d_b);
        cudaDeviceSynchronize();
#ifdef DISPLAY
        cout << "ic = " << ic <<endl;
#endif
    }
    t_endGPU =  my_gettimeofday();
    sync_error = cudaGetLastError();
    if(sync_error != cudaSuccess) {
        fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",
                __FILE__ , __LINE__, cudaGetErrorString(sync_error));
        exit(EXIT_FAILURE);
    }
#ifdef DISPLAY
    cout <<"Début du calcul sur CPU" << endl;
#endif

    t_startCPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        compute(a,b);
#ifdef DISPLAY
        cout << "ic = " << ic <<endl;
#endif

    }
    t_endCPU = my_gettimeofday();

    cudaMemcpy(a_GPU, d_a, taille_totale, cudaMemcpyDeviceToHost);

#ifdef DISPLAY

#ifdef CADNA
    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << strp(a[i]) << endl;
    cout << "CPU time: " << t_endCPU-t_startCPU << endl;

    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << strp(a_GPU[i]) << endl; //" " << a_GPU[i].getx() << " " << a_GPU[i].gety() << " " << a_GPU[i].getz() << " " << a_GPU[i].nb_significant_digit() << endl;
    cout << "GPU time: " << t_endGPU-t_startGPU << endl;
#else
    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << fixed << a[i] << endl;
    cout << "CPU time: " << t_endCPU-t_startCPU << endl;

    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << fixed << a_GPU[i] << endl; // voir notation "scientific" au lieu de "fixed"
    cout << "GPU time: " << t_endGPU-t_startGPU << endl;

#endif //CADNA
#endif // DISPLAY

    t_CPU = (t_endCPU - t_startCPU)/CPT;
    t_GPU = (t_endGPU - t_startGPU)/CPT;

    for(i = 0; i < 5; i++)
        if (a[i] != a_GPU[i]){
            valide = 0;
            cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;
            fprintf(fic, "%d %d %f %f %d %d %d\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU, N, NLOOP, CPT, Cadna);
            exit(1);
        }

    fprintf(fic, "%d %d %f %f %d %d %d\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU, N, NLOOP, CPT, Cadna);
#ifdef DISPLAY
    cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;
    /*
       cudaMemcpy(seedCPU, seed, 1024*sizeof(unsigned int), cudaMemcpyDeviceToHost);
       for(i = 0; i < 160; i++){
       cout<< i << "seedCPU[]: " << seedCPU[i] << endl;
       cout<< i+160 << "seedCPU[]: " << seedCPU[i+160] << endl;
       }
     */

#endif
#ifdef CADNA
    cadna_end();
#endif
    /* Libération mémoire GPU et CPU : */
    cudaFree(d_a); cudaFree(d_b);
    free(a); free(b); free(a_GPU);

#ifdef CUDA
    CUDA_ERROR(cudaFree(seed));
#endif //CUDA
    fclose(fic);
    return EXIT_SUCCESS;
}
