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
#define CPT 100
#endif


#define SEED 1

using namespace std;

// Génération aléatoire du tableau :
__host__ void aleaTab(float* T, int n){
    long int i;
    srand(SEED);

    //    posix_memalign((void **)&T, 64, N * sizeof(float));
    for(i = 0; i < n; i++){
        T[i] = 0.5f + (float)rand()/(float)RAND_MAX;

    }
}

// Mesures :
double my_gettimeofday(){
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


__host__ void compute(float *a, float *b){
    for (int i = 0; i < N; i++){
        float f = a[i];
        for (int j = 0; j < NLOOP; j++)
            f = b[i] * f;
        a[i] = f;
    }
}

__global__ void computeGPU(float *a, float *b){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float f = a[i];
    for(int j=0; j<NLOOP; j++){
        f = b[i] * f;
    }
    a[i] = f;
}




int main(int argc, char **argv){
    int i;
    int taille_totale = N*sizeof(float);

    double t_startCPU, t_endCPU;
    double t_startGPU, t_endGPU;
    double t_CPU, t_GPU;

    float *d_a, *d_b;

    float *a = (float*)malloc(taille_totale);
    float *b = (float*)malloc(taille_totale);
    float *a_GPU = (float*)malloc(taille_totale);

    int valide = 1;

    //cout << "N = " << N << endl;


    aleaTab(a, N);
    aleaTab(b, N);

    /* Allocation GPU : */

    // cout<<0<<endl;

    cudaMalloc((void **) &d_a, taille_totale);
    cudaMalloc((void **) &d_b, taille_totale);

    // cout<<0.1<<endl;
    /* Transferts CPU -> GPU (synchrones) : */
    cudaMemcpy(d_a, a, taille_totale, cudaMemcpyHostToDevice);


    cudaMemcpy(d_b, b, taille_totale, cudaMemcpyHostToDevice);


    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);


    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X));


    // cout<<1<<endl;

    t_startGPU = my_gettimeofday();
    for(i=0; i<CPT;i++){
        computeGPU<<<tailleGrille, threadsParBloc>>>(d_a, d_b);
        cudaDeviceSynchronize();
    }
    t_endGPU =  my_gettimeofday();



    t_startCPU = my_gettimeofday();
    compute(a, b);
    t_endCPU = my_gettimeofday();

    cudaMemcpy(a_GPU, d_a, taille_totale, cudaMemcpyDeviceToHost);


    /*
       for(i = 0; i < 5; i++)
       cerr << "Result[" << i << "] :" << a[i] << endl;

       cout << "Time: " << t_endCPU-t_startCPU << endl;

       for(i = 0; i < 5; i++)
       cerr << "Result[" << i << "] :" << a_GPU[i] << endl;

       cout << "Time: " << t_endGPU-t_startGPU << endl;
     */

    t_CPU = t_endCPU - t_startCPU;
    t_GPU = (t_endGPU - t_startGPU)/CPT;

    for(i = 0; i < 5; i++)
        if (a[i] != a_GPU[i]){
            valide = 0;
            cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;
            //printf("%d %d %f %f\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU);
            exit(1);
        }

    //printf("%d %d %f %f\n", TAILLE_BLOC_X, valide, t_CPU, t_GPU);
    cout<< TAILLE_BLOC_X << " " << valide << " " << t_CPU << " " << t_GPU << endl;

    /* Libération mémoire GPU et CPU : */
    cudaFree(d_a); cudaFree(d_b);
    free(a); free(b); free(a_GPU);


    return EXIT_SUCCESS;
}
