#include <cstdlib>
#include <ctime>
#include <stdio.h>
//#include <time.h>
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

#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu_half.cu"
#include "cadna_gpu_float.h"
#include "cadna_gpu_float.cu"
#define DATATYPE half
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU half_gpu_st
#else
#define DATATYPE half
#define DATATYPECADNA half
#define DATATYPECADNAGPU half
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

// Mesures :
double my_gettimeofday(){
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}





__device__ void add(half_gpu_st *a, half_gpu_st *b)
{
	//half_gpu_st a =__float2half(-2.1564f);
	//half_gpu_st b = __float2half(2.f);
	half_gpu_st c,d;
	c= a[0]+b[0];
	d = __hadd(a[0],b[0]);
	printf("a = %f b = %f\n", __half_gpu_st2float(a[0]), __half_gpu_st2float(b[0]));
	printf("+ c.x=%f\n",__half_gpu_st2float(c.x));
	printf("+ c.y=%f\n",__half_gpu_st2float(c.y));
	printf("+ c.z=%f\n",__half_gpu_st2float(c.z));
	printf("+ c=%f\n",__half_gpu_st2float(c));

/*	printf("hadd d.x=%f\n",__half2float(d.x));
	printf("hadd d.y=%f\n",__half2float(d.y));
	printf("hadd d.z=%f\n",__half2float(d.z));
	printf("+ d=%f\n",__half2float(d));
/////////////////
	float_gpu_st f,g,ff,res_float;
	f =__half_gpu_st2float_gpu_st(a);
	ff = -2.1564f;
	g = 2.f;
	res_float = f+g;
	float_gpu_st res_float_f = ff+g;
	printf("float_gpu_st f = %f,f.x = %f,float_gpu_st g=%f\n half_gpu_st -> float_gpu_st : f+g = %f half_gpu_st -> float_gpu_st : ff+g :%f\n",(float)f, (float)f.x, (float)g, (float)res_float, (float)res_float_f);
	half_gpu_st h,i;
	h = __float_gpu_st2half_gpu_st(f);
	i = b;
	printf("half_gpu_st h =%f half_gpu_st i=%f\n h+i float_gpu_st -> half_gpu_st : %f\n", __half2float(h),__half2float(i),__half2float(h+i));
*/
}
/*
__device__ void sub()
{
	half_gpu_st a =__float2half(0.346f);
	half_gpu_st b = __float2half(2.f);
	half a_h = __float2half(0.346f);
	half b_h = __float2half(2.f);
	half_gpu_st res_h, res, res2, res3, res4, res5, res6;
	res_h = a_h-b_h;
	res = __hsub(a_h,b_h);
	res2 = __hsub(a,b_h);
	res3 = __hsub(a_h,b);
	res4 = __hsub_sat(a_h,b_h);
	res5 = __hsub_sat(a,b_h);
	res6 = __hsub_sat(a_h,b);
	
	half_gpu_st cons,cons2;
	cons.x = __float2half(1.13f);
	cons.y = __float2half(1.12f);
	cons.z = __float2half(1.12f);
	printf("TEST : significant digit :%d\n", cons.nb_significant_digit());
	printf("TEST : accuracy :%d\n", cons.accuracy);
	printf("TEST : error :%d\n", cons.error);

	cons2.x = __float2half(1.13f);
	cons2.y = __float2half(1.12f);
	cons2.z = __float2half(1.12f);
	
	printf("TEST : @.0 :%f\n", __half_gpu_st2float(cons2*cons));





	printf("half half a = %f, half b = %f, res_h = %f\n",__half2float(a_h), __half2float(b_h), __half2float(res_h));
	printf("hsub half_gpu_st a = %f, half_gpu_st b = %f, res_h = %f\n",__half2float(a), __half2float(b), __half2float(res));
	printf("hsub  half_gpu_st a = %f, half b = %f, res = %f\n",__half2float(a), __half2float(b_h), __half2float(res2));
	printf("hsub  half a = %f, half_gpu_st b = %f, res = %f\n",__half2float(a_h), __half2float(b), __half2float(res3));

	printf("hsub_sat half_gpu_st a = %f, half_gpu_st b = %f, res_h = %f\n",__half2float(a), __half2float(b), __half2float(res4));
	printf("hsub_sat  half_gpu_st a = %f, half b = %f, res = %f\n",__half2float(a), __half2float(b_h), __half2float(res5));
	printf("hsub_sat  half a = %f, half_gpu_st b = %f, res = %f\n",__half2float(a_h), __half2float(b), __half2float(res6));




}

__device__ void mul()
{
	half_gpu_st a =__float2half(0.346f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st c;
	c= a*b;
	printf("c.x=%f\n",__half2float(c.x));
	printf("c.y=%f\n",__half2float(c.y));
	printf("c.z=%f\n",__half2float(c.z));

	half a_h = __float2half(0.346f);
	half b_h = __float2half(2.f);
	half_gpu_st  res, res2, res3, res4, res5, res6;
	res = __hmul(a_h,b_h);
	res2 = __hmul(a,b_h);
	res3 = __hmul(a_h,b);
	res4 = __hmul_sat(a_h,b_h);
	res5 = __hmul_sat(a,b_h);
	res6 = __hmul_sat(a_h,b);

	printf("half half a = %f, half b = %f, res_h = %f\n",__half2float(a_h), __half2float(b_h), __half2float(c));
	printf("hmul half_gpu_st a = %f, half_gpu_st b = %f, res_h = %f\n",__half2float(a), __half2float(b), __half2float(res));
	printf("hmul  half_gpu_st a = %f, half b = %f, res = %f\n",__half2float(a), __half2float(b_h), __half2float(res2));
	printf("hmul  half a = %f, half_gpu_st b = %f, res = %f\n",__half2float(a_h), __half2float(b), __half2float(res3));

	printf("hmul_sat half_gpu_st a = %f, half_gpu_st b = %f, res_h = %f\n",__half2float(a), __half2float(b), __half2float(res4));
	printf("hmul_sat  half_gpu_st a = %f, half b = %f, res = %f\n",__half2float(a), __half2float(b_h), __half2float(res5));
	printf("hmul_sat  half a = %f, half_gpu_st b = %f, res = %f\n",__half2float(a_h), __half2float(b), __half2float(res6));







}

__device__ void div()
{
	half_gpu_st a =__float2half(2.f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st c,d;
	c= a/b;
	d= __hdiv(a,b);
	printf("c.x=%f\n",__half2float(c.x));
	printf("c.y=%f\n",__half2float(c.y));
	printf("c.z=%f\n",__half2float(c.z));
	printf("c=%f\n",__half2float(c));
	printf("d=%f\n",__half2float(d));


}

__device__ void fma()
{
	half_gpu_st a = __float2half(0.4f);
	half_gpu_st b = __float2half(3.5f);
	half_gpu_st c = __float2half(0.5678f);
	half a_h = __float2half(0.5678f);
	half b_h = __float2half(3.5f);
	half c_h = __float2half(0.5678f);

	half res_h = __hfma(a_h,b_h,c_h);
	half_gpu_st res = __hfma(a,b,c);
	half_gpu_st res2 = __hfma(a,b_h,c_h);
	half_gpu_st res3 = __hfma(a,b_h,c);
	half_gpu_st res4 = __hfma(a,b,c_h);
	half_gpu_st res5 = __hfma(a_h,b,c);
	half_gpu_st res6 = __hfma(a_h,b,c_h);
	half_gpu_st res7 = __hfma(a_h,b_h,c);

	printf("fma : half a = %f, half b=%f, half c =%f res_h = %f\n",__half2float(a_h), __half2float(b_h),__half2float(c_h), __half2float(res_h));
	printf("fma : half_gpu_st a = %f, half_gpu_st b=%f, half_gpu_st c =%f res = %f\n",__half2float(a), __half2float(b),__half2float(c), __half2float(res));
	printf("fma : half_gpu_st a = %f, half b=%f, half c =%f res = %f\n",__half2float(a), __half2float(b_h),__half2float(c_h), __half2float(res2));
	printf("fma : half_gpu_st a = %f, half b=%f, half_gpu_st c =%f res = %f\n",__half2float(a), __half2float(b_h),__half2float(c), __half2float(res3));
	printf("fma : half_gpu_st a = %f, half_gpu_st b=%f, half c =%f res = %f\n",__half2float(a), __half2float(b),__half2float(c_h), __half2float(res4));
	printf("fma : half a = %f, half_gpu_st b=%f, half_gpu_st c =%f res = %f\n",__half2float(a_h), __half2float(b),__half2float(c), __half2float(res5));
	printf("fma : half a = %f, half_gpu_st b=%f, half c =%f res = %f\n",__half2float(a_h), __half2float(b),__half2float(c_h), __half2float(res6));
	printf("fma : half a = %f, half b=%f, half_gpu_st c =%f res = %f\n",__half2float(a_h), __half2float(b_h),__half2float(c), __half2float(res7));
	
	half res_h_s = __hfma_sat(a_h,b_h,c_h);
	half_gpu_st res_s = __hfma_sat(a,b,c);
	half_gpu_st res2_s = __hfma_sat(a,b_h,c_h);
	half_gpu_st res3_s = __hfma_sat(a,b_h,c);
	half_gpu_st res4_s = __hfma_sat(a,b,c_h);
	half_gpu_st res5_s = __hfma_sat(a_h,b,c);
	half_gpu_st res6_s = __hfma_sat(a_h,b,c_h);
	half_gpu_st res7_s = __hfma_sat(a_h,b_h,c);
	
	printf("fma_sat in\n");
	printf("fma_sat : half a = %f, half b=%f, half c =%f res_h = %f\n",__half2float(a_h), __half2float(b_h),__half2float(c_h), __half2float(res_h_s));
	printf("fma_sat : half_gpu_st a = %f, half_gpu_st b=%f, half_gpu_st c =%f res = %f\n",__half2float(a), __half2float(b),__half2float(c), __half2float(res_s));
	printf("fma_sat : half_gpu_st a = %f, half b=%f, half c =%f res = %f\n",__half2float(a), __half2float(b_h),__half2float(c_h), __half2float(res2_s));
	printf("fma_sat : half_gpu_st a = %f, half b=%f, half_gpu_st c =%f res = %f\n",__half2float(a), __half2float(b_h),__half2float(c), __half2float(res3_s));
	printf("fma_sat : half_gpu_st a = %f, half_gpu_st b=%f, half c =%f res = %f\n",__half2float(a), __half2float(b),__half2float(c_h), __half2float(res4_s));
	printf("fma_sat : half a = %f, half_gpu_st b=%f, half_gpu_st c =%f res = %f\n",__half2float(a_h), __half2float(b),__half2float(c), __half2float(res5_s));
	printf("fma_sat : half a = %f, half_gpu_st b=%f, half c =%f res = %f\n",__half2float(a_h), __half2float(b),__half2float(c_h), __half2float(res6_s));
	printf("fma_sat : half a = %f, half b=%f, half_gpu_st c =%f res = %f\n",__half2float(a_h), __half2float(b_h),__half2float(c), __half2float(res7_s));
	



}



__device__ void egalite()
{
	half_gpu_st a = __float2half(2.f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st c = __float2half(5.f);
	if(a == b)
	{
		printf("a == b\n");
	}
	if(a != c)
	{
		printf("a != c\n");
	}
	half a_h = __float2half(2.f);
	half res_h = __hneg(a_h);
	printf("hneq a_h =%f, res_h = %f\n", __half2float(a_h), __half2float(res_h));
	half_gpu_st res = __hneg(a);
	printf("hneq a =%f, res = %f\n", __half2float(a), __half2float(res));

	int res_eq = a==c;
	bool res_heq = __heq(a_h,__hdiv(__float2half(0.f), __float2half(0.f)));
	bool res_hequ = __hequ(a_h,__hdiv(__float2half(0.f), __float2half(0.f)));
	printf("heq a == b, res = %d\n",__half2float(res_eq));
	printf("heq , res = %d\n",__half2float(res_heq));
	printf("hequ , res = %d\n",__half2float(res_hequ));




}
__device__ void comparaison()
{
	half_gpu_st a = __float2half(2.f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st c = __float2half(5.f);
	if(a > b)
	{
		printf("a > b\n");
	}
	if(a >= c)
	{
		printf("a != c\n");
	}
	if(a < c)
	{
		printf("a < c\n");
	}
	if(a <= c)
	{
		printf("a <= c\n");
	}

}

__device__ void absolu()
{
	half_gpu_st a = __float2half(2.f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st c = __float2half(5.f);
	a = fabsf(a);
	b = fabs(b);
	c = hsqrt(c);
	printf("fabsf a = %f\n",__half2float(a));
	printf("fabs b = %f\n",__half2float(b));
	printf("sqrt c = %f\n",__half2float(c));
}
__device__ void maxmin()
{
	half_gpu_st a = __float2half(2.f);
	half_gpu_st b = __float2half(3.5f);
	half_gpu_st min,max;
	max = fmaxf(a,b);
	min = fminf(a,b);
	printf("max a,b = %f\n",__half2float(max));
	printf("min a,b = %f\n",__half2float(min));
}

__device__ void add_sat()
{
	half_gpu_st a = __float2half(0.4f);
	half_gpu_st b = __float2half(3.5f);
	half c = __float2half(0.5f);
	half_gpu_st res = __hadd_sat(a,b);
	half_gpu_st res2 = __hadd_sat(a,c);
	half_gpu_st res3 = __hadd_sat(c,b);
	printf("add_sat : half_gpu_st a = %f, half_gpu_st b=%f res = %f\n",__half2float(a), __half2float(b), __half2float(res));
	printf("add_sat : half_gpu_st a=%f, half c =%f res = %f\n",__half2float(a), __half2float(c), __half2float(res2));
	printf("add_sat : half c=%f, half__gpu_st b = %f res = %f\n",__half2float(c), __half2float(b), __half2float(res3));
	
}


__device__ void math()
{
	half_gpu_st a =__float2half(0.346f);
	half_gpu_st b = __float2half(2.f);
	half_gpu_st sin,cos,exp,log,log2,log10,rcp;
	
	sin = hsin(a);
	cos = hcos(a);
	exp = hexp(a);
	log = hlog(a);
	log2 = hlog2(a);
	log10 = hlog10(a);
	rcp = hrcp(a);
	
	
	printf("a = %f, sin = %f cos = %f, exp = %f, log = %f, log2 = %f, log10 = %f, rcp = %f\n",__half2float(a), __half2float(sin), __half2float(cos), __half2float(exp), __half2float(log), __half2float(log2), __half2float(log10), __half2float(rcp));


}

__device__ void conversion()
{
	half_gpu_st a =__float2half_gpu_st(0.346f);
	float b = __half_gpu_st2float(a);


	printf("a = %f, a.x = %f, a.y = %f, a.z = %f, b= %f\n",__half2float(a), __half2float(a.x), __half2float(a.y), __half2float(a.z), b);


}

__device__ void table()
{
	half_gpu_st* a[5];
	int i;
	for(i=0;i<5;i++)
	{
		a[i]= __float2half_gpu_st(i*1.f);
		//printf("a[%d] = %f\n",i, __half_gpu_st2float(a[i]));
	}	



}
*/

//////////////////////////////////////

__global__ void add_kernel(half_gpu_st* a, half_gpu_st* b)
{	
	cadna_init_gpu();


	printf("-------------Add :---------------\n");
	add(a, b);
}
/*
__global__ void sub_kernel()
{
	cadna_init_gpu();
	printf("-------------Sub :----------------\n");
	sub();
}

__global__ void mul_kernel()
{
	cadna_init_gpu();
	printf("-------------Mul :-----------------\n");
	mul();
}

__global__ void div_kernel()
{
	cadna_init_gpu();
	printf("-------------Div :------------------\n");
	div();
}

__global__ void egalite_kernel()
{
	cadna_init_gpu();
	printf("-------------Egalite :---------------\n");
	egalite();
}

__global__ void comparaison_kernel()
{
	cadna_init_gpu();
	printf("-------------Comparaison :-----------\n");
	comparaison();
}


__global__ void absolu_kernel()
{
	cadna_init_gpu();
	printf("--------------Absolu :--------------\n");
	absolu();
}

__global__ void maxmin_kernel()
{
	cadna_init_gpu();
	printf("-------------Maxmin :-------------\n");
	maxmin();
}

__global__ void add_sat_kernel()
{
	cadna_init_gpu();
	printf("-------------Add_sat :-------------\n");
	add_sat();
}

__global__ void fma_kernel()
{
	cadna_init_gpu();
	printf("-------------Fma :-------------\n");
	fma();
}

__global__ void math_kernel()
{
	cadna_init_gpu();
	printf("-------------Math :-------------\n");
	math();
}

__global__ void conversion_kernel()
{
	cadna_init_gpu();
	printf("-------------Conversion :-------------\n");
	conversion();
}

__global__ void table_kernel()
{
	cadna_init_gpu();
	printf("-------------Conversion :-------------\n");
	table();
}
*/



//////////////////////////////////////////////



int main(int argc, char **argv){
    double t_startGPU, t_endGPU;
   	//InitCuda(1,N);
	//cudaError_t sync_error;
	//int Cadna = 0;

#ifdef CADNA
	    cadna_init(-1, CADNA_INTRINSIC);
	    //Cadna = 1;
#endif

#ifdef DISPLAY
		printf("CADNA execute\n");
#endif
    float_gpu_st *af, *bf;
    cudaMalloc( &af, sizeof(float_st));
    cudaMalloc( &bf, sizeof(float_st));

	half_gpu_st * ah, *bh;

    cudaMalloc( &bf, sizeof(float_st));
    cudaMalloc( &af, sizeof(float_st));
	




    float_st * a, *b;
    a = (float_st*)malloc(sizeof(float_st));
    b = (float_st*)malloc(sizeof(float_st));
 	a[0] = float_st(2.1564f);
 	b[0] = float_st(2.f);

	cudaMemcpy(af, a , sizeof(float_st), cudaMemcpyHostToDevice);
	cudaMemcpy(bf, b , sizeof(float_st), cudaMemcpyHostToDevice);

    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);

    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X));



	ah[0] = __float_gpu_st2half_gpu_st(af[0]);

	bh[0] = __float_gpu_st2half_gpu_st(bf[0]);



    t_startGPU = my_gettimeofday();
	add_kernel<<<1,1>>>(ah, bh);
/*	sub_kernel<<<1,1>>>();
	mul_kernel<<<1,1>>>();
	div_kernel<<<1,1>>>();
	egalite_kernel<<<1,1>>>();
	comparaison_kernel<<<1,1>>>();
	absolu_kernel<<<1,1>>>();
	maxmin_kernel<<<1,1>>>();
	add_sat_kernel<<<1,1>>>();
	fma_kernel<<<1,1>>>();
	math_kernel<<<1,1>>>();
	conversion_kernel<<<1,1>>>();
	*/
    cudaDeviceSynchronize();

	cudaMemcpy(a, af , sizeof(float), cudaMemcpyHostToDevice);
#ifdef DISPLAY
    printf("CADNA calcul fini\n");
#endif

    t_endGPU =  my_gettimeofday();

#ifdef CADNA
	//printf("CADNA Time = %d", t_endGPU - t_startGPU);
    cout << "GPU time: " << t_endGPU-t_startGPU << endl;
#else

	printf("CUDA Time = %s", t_endGPU - t_startGPU);
#endif

#ifdef CADNA
	cadna_end();
#endif
    return EXIT_SUCCESS;
}
