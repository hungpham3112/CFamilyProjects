#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>

#ifndef N
#define N 0x10000000
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
/*
int InitCuda(int seeProps){
    cudaError_t err;

    cudaSetDevice(0);
    err = cudaGetLastError();

    if (err != cudaSuccess)
        return 1;

    if (seeProps){
        int devID = -1;
        cudaDeviceProp sprops;
        size_t free_mem = 0, tot_mem=0;
        // get number of SMs on this GPU
        cudaGetDevice(&devID);
        cudaGetDeviceProperties(&sprops, devID);
        printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, sprops.name, sprops.major, sprops.minor);
        cudaMemGetInfo(&free_mem, &tot_mem);
        printf("Memoire disponible/totale(bytes):%d/%d\n",(int)free_mem, (int)tot_mem);

    }
    return 0;
}
*/

// Génération aléatoire du tableau :
__host__ void aleaTabSt(DATATYPECADNA* T, int n){
    long int i;
    srand(SEED);

    //    posix_memalign((void **)&T, 64, N * sizeof(float));
    for(i = 0; i < n; i++){
#ifdef CADNA
        T[i] =float_st(0.5f + (float)rand()/(float)RAND_MAX);
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
        float f = a[i];
        for (int j = 0; j < NLOOP; j++)
            f = b[i] + f;
        a[i] = f;
    }
}

__global__ void computeGPU(DATATYPECADNAGPU *a, DATATYPECADNAGPU *b){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#ifdef CADNA
 //   cadna_init_gpu();
#endif
#ifdef DEBUG
    if(i==4) printf("d_a[4]=%f\n",a[i]);
#endif

    if(i<N){
        float f = a[i];
        for(int j=0; j<NLOOP; j++){
            f = b[i] + f;
        }
        a[i] = f;
    }
}


int main(int argc, char **argv){

//    InitCuda(1);
    int Cadna = 0;
#ifdef CADNA
    cadna_init(-1);
    Cadna = 1;
#endif
    int i, ic;
#ifndef CADNA
    long int taille_totale = N*sizeof(DATATYPE);
#else
    long int taille_totale = N*sizeof(DATATYPECADNA);
    // RQ: sizeof(DATATYPECADNA) équiv sizeof(DATATYPECADNAGPU)
#endif

    double t_startCPU, t_endCPU;
    double t_startGPU, t_endGPU;
    double t_CPU, t_GPU;

    DATATYPECADNAGPU *d_a, *d_b;

#ifdef DEBUG
    cout <<"N =  " << N <<endl;
    cout <<"size_t =  " << sizeof(size_t) <<endl;
    cout <<"taille DTC " << sizeof(DATATYPECADNA) <<endl;
#endif


    DATATYPECADNA *a = (DATATYPECADNA*)malloc(taille_totale);
    DATATYPECADNA *b = (DATATYPECADNA*)malloc(taille_totale);
    DATATYPECADNA *a_GPU = (DATATYPECADNA*)malloc(taille_totale);

#ifdef DEBUG
    cout <<"Allocation CPU OK" << endl;
#endif
    // Concatène par défaut, écrase si nouvelle série.
    FILE* fic=(TAILLE_BLOC_X == 32)?fopen("operBXaddComb.res", "w"):fopen("operBXaddComb.res", "a");

    int valide = 1;

    aleaTabSt(a, N);
    aleaTabSt(b, N);

#ifdef DEBUG
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


#ifdef DEBUG
    cout <<"Allocation et copie sur GPU OK" << endl;
    cout <<"Début du calcul sur GPU" << endl;
    cout <<"a[4] =" << a[4] << endl;
#endif


    t_startGPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        computeGPU<<<tailleGrille, threadsParBloc>>>(d_a, d_b);
        cudaDeviceSynchronize();
#ifdef DEBUG
        cout << "ic = " << ic <<endl;
#endif
    }
    t_endGPU =  my_gettimeofday();


#ifdef DEBUG
    cout <<"Début du calcul sur CPU" << endl;
#endif

    t_startCPU = my_gettimeofday();
    for(ic=0; ic<CPT;ic++){
        compute(a,b);
#ifdef DEBUG
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
        cerr << "result[" << i << "] :" << strp(a_GPU[i]) << endl;
    cout << "GPU time: " << t_endGPU-t_startGPU << endl;
#else
    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << a[i] << endl;
    cout << "CPU time: " << t_endCPU-t_startCPU << endl;

    for(i = 0; i < 5; i++)
        cerr << "result[" << i << "] :" << a_GPU[i] << endl;
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
#endif
#ifdef CADNA
    cadna_end();
#endif
    /* Libération mémoire GPU et CPU : */
    cudaFree(d_a); cudaFree(d_b);
    free(a); free(b); free(a_GPU);

    fclose(fic);
    return EXIT_SUCCESS;
}
