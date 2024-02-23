#include <cstdlib>
#include <ctime>
/*#include <stdio.h>
#include <time.h>*/
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <assert.h>

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
/*
// Génération aléatoire du tableau :
__host__ void aleaTab(__half* T, int n){
    long int i;
    srand(SEED);

    //    posix_memalign((void **)&T, 64, N * sizeof(float));
    for(i = 0; i < n; i++){
        T[i] = 0.5 + (__half)rand()/(__half)RAND_MAX;

    }
}
*/
// Mesures :
double my_gettimeofday(){
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


/*
__global__ void add(float a, float b,float res)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	half a_h = __float2half(a);
	half b_h = __float2half(b);
	half res_h = __hadd(a_h,b_h);
	res = __half2float(res_h);
}
*/

/*
__device__ void add()
{
	half x = __hadd(0.,0.);
	printf("");
}
*/

__device__ void sub()
{

	float2 a,b,c;
       a.x = 1.01f;
       a.y = 2.f; 
	b.x = 3.f;
	b.y = 4.f;
	c = {1,2};
	half2 res_a, res_b,res_c;
	res_a = __float22half2_rn(a);
	res_b = __float22half2_rn(b);
	res_c = __float22half2_rn(c);
	
	__half2 res = __hsub2(res_a, res_c);
	
	printf("res = %f, a = %f\n",__half22float2(res), (float2)a);


}



/*
__device__ void mul()
{
	half z = __hmul(0.,0.);
}

__device__ void div()
{
	half d = __hdiv(0.,0.);
}
*/
/*
__global__ void add_kernel()
{
	add();
}
*/
__global__ void sub_kernel()
{
	sub();
}
/*
__global__ void mul_kernel()
{
	mul();
}

__global__ void div_kernel()
{
	div();
}

*/
int main(int argc, char **argv){
    double t_startGPU, t_endGPU;

  /*  float *a, *b, *res;
    a = (float*)malloc(1*sizeof(float));
    b = (float*)malloc(1*sizeof(float));
    res = (float*)malloc(1*sizeof(float));
    a[0] = (float)0.126;
    b[0] = (float)0.523;
    res[0] = 0;
    float a_h,b_h,res_h;
    cudaMalloc(&a_h,1*sizeof(float));
    cudaMalloc(&b_h,1*sizeof(float));
    cudaMalloc(&res_h,1*sizeof(float));
	cudaMemcpy(a_h,a,1*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(b_h,b,1*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(res_h,res,1*sizeof(float),cudaMemcpyHostToDevice);
    */
    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);


    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X));

 
    t_startGPU = my_gettimeofday();
    //add<<<1,1>>>(a_h,b_h,res_h);
	//add_kernel<<<1,1>>>();
	sub_kernel<<<1,1>>>();
	//mul_kernel<<<1,1>>>();
	//div_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
 //   cudaMemcpy(res,res_h,1,cudaMemcpyDeviceToHost);
   
    t_endGPU =  my_gettimeofday();

    	//printf("a =%f, b= %f, res = %f\n",a,b,res);
	printf("Time = %d", t_endGPU - t_startGPU);
	/*cudaFree(a);
	cudaFree(b);
	cudaFree(res);
*/
    return EXIT_SUCCESS;
}
