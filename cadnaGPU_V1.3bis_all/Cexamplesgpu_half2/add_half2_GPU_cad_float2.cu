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
#include "cadna_gpu_half2_float2.cu"
#include "cadna_gpu_float2.h"
#include "cadna_gpu_float2.cu"
#define DATATYPE half2
#define DATATYPECADNA float2_st
#define DATATYPECADNAGPU half2_gpu_st
#else
#define DATATYPE half2
#define DATATYPECADNA half2
#define DATATYPECADNAGPU half2
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





__device__ void add()
{
	half2_gpu_st a =__float22half2(-2.1564f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c,d;
	c= a+b;
	d = __hadd(a,b);
	printf("a = %f b = %f\n", __half22float2(a), __half22float2(b));
	printf("+ c.x=%f\n",__half22float2(c.x));
	printf("+ c.y=%f\n",__half22float2(c.y));
	printf("+ c.z=%f\n",__half22float2(c.z));
	printf("+ c=%f\n",__half22float2(c));

	printf("hadd d.x=%f\n",__half22float2(d.x));
	printf("hadd d.y=%f\n",__half22float2(d.y));
	printf("hadd d.z=%f\n",__half22float2(d.z));
	printf("+ d=%f\n",__half22float2(d));
/////////////////
	float2_gpu_st f,g,ff,res_float2;
	f =__half2_gpu_st2float2_gpu_st(a);
	ff = -2.1564f;
	g = 2.f;
	res_float2 = f+g;
	float2_gpu_st res_float2_f = ff+g;
	printf("float2_gpu_st f = %f,f.x = %f,float2_gpu_st g=%f\n half2_gpu_st -> float2_gpu_st : f+g = %f half2_gpu_st -> float2_gpu_st : ff+g :%f\n",(float2)f, (float2)f.x, (float2)g, (float2)res_float2, (float2)res_float2_f);
	half2_gpu_st h,i;
	h = __float2_gpu_st2half2_gpu_st(f);
	i = b;
	printf("half2_gpu_st h =%f half2_gpu_st i=%f\n h+i float2_gpu_st -> half2_gpu_st : %f\n", __half22float2(h),__half22float2(i),__half22float2(h+i));

}

__device__ void sub()
{
	half2_gpu_st a =__float22half2(0.346f);
	half2_gpu_st b = __float22half2(2.f);
	half2 a_h = __float22half2(0.346f);
	half2 b_h = __float22half2(2.f);
	half2_gpu_st res_h, res, res2, res3, res4, res5, res6;
	res_h = a_h-b_h;
	res = __hsub(a_h,b_h);
	res2 = __hsub(a,b_h);
	res3 = __hsub(a_h,b);
	res4 = __hsub_sat(a_h,b_h);
	res5 = __hsub_sat(a,b_h);
	res6 = __hsub_sat(a_h,b);
	
	half2_gpu_st cons,cons2;
	cons.x = __float22half2(1.13f);
	cons.y = __float22half2(1.12f);
	cons.z = __float22half2(1.12f);
	printf("TEST : significant digit :%d\n", cons.nb_significant_digit());
	printf("TEST : accuracy :%d\n", cons.accuracy);
	printf("TEST : error :%d\n", cons.error);

	cons2.x = __float22half2(1.13f);
	cons2.y = __float22half2(1.12f);
	cons2.z = __float22half2(1.12f);
	
	printf("TEST : @.0 :%f\n", __half2_gpu_st2float2(cons2*cons));





	printf("half2 half2 a = %f, half2 b = %f, res_h = %f\n",__half22float2(a_h), __half22float2(b_h), __half22float2(res_h));
	printf("hsub half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__half22float2(a), __half22float2(b), __half22float2(res));
	printf("hsub  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__half22float2(a), __half22float2(b_h), __half22float2(res2));
	printf("hsub  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__half22float2(a_h), __half22float2(b), __half22float2(res3));

	printf("hsub_sat half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__half22float2(a), __half22float2(b), __half22float2(res4));
	printf("hsub_sat  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__half22float2(a), __half22float2(b_h), __half22float2(res5));
	printf("hsub_sat  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__half22float2(a_h), __half22float2(b), __half22float2(res6));




}

__device__ void mul()
{
	half2_gpu_st a =__float22half2(0.346f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c;
	c= a*b;
	printf("c.x=%f\n",__half22float2(c.x));
	printf("c.y=%f\n",__half22float2(c.y));
	printf("c.z=%f\n",__half22float2(c.z));

	half2 a_h = __float22half2(0.346f);
	half2 b_h = __float22half2(2.f);
	half2_gpu_st  res, res2, res3, res4, res5, res6;
	res = __hmul(a_h,b_h);
	res2 = __hmul(a,b_h);
	res3 = __hmul(a_h,b);
	res4 = __hmul_sat(a_h,b_h);
	res5 = __hmul_sat(a,b_h);
	res6 = __hmul_sat(a_h,b);

	printf("half2 half2 a = %f, half2 b = %f, res_h = %f\n",__half22float2(a_h), __half22float2(b_h), __half22float2(c));
	printf("hmul half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__half22float2(a), __half22float2(b), __half22float2(res));
	printf("hmul  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__half22float2(a), __half22float2(b_h), __half22float2(res2));
	printf("hmul  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__half22float2(a_h), __half22float2(b), __half22float2(res3));

	printf("hmul_sat half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__half22float2(a), __half22float2(b), __half22float2(res4));
	printf("hmul_sat  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__half22float2(a), __half22float2(b_h), __half22float2(res5));
	printf("hmul_sat  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__half22float2(a_h), __half22float2(b), __half22float2(res6));







}

__device__ void div()
{
	half2_gpu_st a =__float22half2(2.f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c,d;
	c= a/b;
	d= __hdiv(a,b);
	printf("c.x=%f\n",__half22float2(c.x));
	printf("c.y=%f\n",__half22float2(c.y));
	printf("c.z=%f\n",__half22float2(c.z));
	printf("c=%f\n",__half22float2(c));
	printf("d=%f\n",__half22float2(d));


}

__device__ void fma()
{
	half2_gpu_st a = __float22half2(0.4f);
	half2_gpu_st b = __float22half2(3.5f);
	half2_gpu_st c = __float22half2(0.5678f);
	half2 a_h = __float22half2(0.5678f);
	half2 b_h = __float22half2(3.5f);
	half2 c_h = __float22half2(0.5678f);

	half2 res_h = __hfma(a_h,b_h,c_h);
	half2_gpu_st res = __hfma(a,b,c);
	half2_gpu_st res2 = __hfma(a,b_h,c_h);
	half2_gpu_st res3 = __hfma(a,b_h,c);
	half2_gpu_st res4 = __hfma(a,b,c_h);
	half2_gpu_st res5 = __hfma(a_h,b,c);
	half2_gpu_st res6 = __hfma(a_h,b,c_h);
	half2_gpu_st res7 = __hfma(a_h,b_h,c);

	printf("fma : half2 a = %f, half2 b=%f, half2 c =%f res_h = %f\n",__half22float2(a_h), __half22float2(b_h),__half22float2(c_h), __half22float2(res_h));
	printf("fma : half2_gpu_st a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a), __half22float2(b),__half22float2(c), __half22float2(res));
	printf("fma : half2_gpu_st a = %f, half2 b=%f, half2 c =%f res = %f\n",__half22float2(a), __half22float2(b_h),__half22float2(c_h), __half22float2(res2));
	printf("fma : half2_gpu_st a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a), __half22float2(b_h),__half22float2(c), __half22float2(res3));
	printf("fma : half2_gpu_st a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__half22float2(a), __half22float2(b),__half22float2(c_h), __half22float2(res4));
	printf("fma : half2 a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a_h), __half22float2(b),__half22float2(c), __half22float2(res5));
	printf("fma : half2 a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__half22float2(a_h), __half22float2(b),__half22float2(c_h), __half22float2(res6));
	printf("fma : half2 a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a_h), __half22float2(b_h),__half22float2(c), __half22float2(res7));
	
	half2 res_h_s = __hfma_sat(a_h,b_h,c_h);
	half2_gpu_st res_s = __hfma_sat(a,b,c);
	half2_gpu_st res2_s = __hfma_sat(a,b_h,c_h);
	half2_gpu_st res3_s = __hfma_sat(a,b_h,c);
	half2_gpu_st res4_s = __hfma_sat(a,b,c_h);
	half2_gpu_st res5_s = __hfma_sat(a_h,b,c);
	half2_gpu_st res6_s = __hfma_sat(a_h,b,c_h);
	half2_gpu_st res7_s = __hfma_sat(a_h,b_h,c);
	
	printf("fma_sat in\n");
	printf("fma_sat : half2 a = %f, half2 b=%f, half2 c =%f res_h = %f\n",__half22float2(a_h), __half22float2(b_h),__half22float2(c_h), __half22float2(res_h_s));
	printf("fma_sat : half2_gpu_st a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a), __half22float2(b),__half22float2(c), __half22float2(res_s));
	printf("fma_sat : half2_gpu_st a = %f, half2 b=%f, half2 c =%f res = %f\n",__half22float2(a), __half22float2(b_h),__half22float2(c_h), __half22float2(res2_s));
	printf("fma_sat : half2_gpu_st a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a), __half22float2(b_h),__half22float2(c), __half22float2(res3_s));
	printf("fma_sat : half2_gpu_st a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__half22float2(a), __half22float2(b),__half22float2(c_h), __half22float2(res4_s));
	printf("fma_sat : half2 a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a_h), __half22float2(b),__half22float2(c), __half22float2(res5_s));
	printf("fma_sat : half2 a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__half22float2(a_h), __half22float2(b),__half22float2(c_h), __half22float2(res6_s));
	printf("fma_sat : half2 a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__half22float2(a_h), __half22float2(b_h),__half22float2(c), __half22float2(res7_s));
	



}



__device__ void egalite()
{
	half2_gpu_st a = __float22half2(2.f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c = __float22half2(5.f);
	if(a == b)
	{
		printf("a == b\n");
	}
	if(a != c)
	{
		printf("a != c\n");
	}
	half2 a_h = __float22half2(2.f);
	half2 res_h = __hneg(a_h);
	printf("hneq a_h =%f, res_h = %f\n", __half22float2(a_h), __half22float2(res_h));
	half2_gpu_st res = __hneg(a);
	printf("hneq a =%f, res = %f\n", __half22float2(a), __half22float2(res));

	int res_eq = a==c;
	bool res_heq = __heq(a_h,__hdiv(__float22half2(0.f), __float22half2(0.f)));
	bool res_hequ = __hequ(a_h,__hdiv(__float22half2(0.f), __float22half2(0.f)));
	printf("heq a == b, res = %d\n",__half22float2(res_eq));
	printf("heq , res = %d\n",__half22float2(res_heq));
	printf("hequ , res = %d\n",__half22float2(res_hequ));




}
__device__ void comparaison()
{
	half2_gpu_st a = __float22half2(2.f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c = __float22half2(5.f);
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
	half2_gpu_st a = __float22half2(2.f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st c = __float22half2(5.f);
	a = fabsf(a);
	b = fabs(b);
	c = hsqrt(c);
	printf("fabsf a = %f\n",__half22float2(a));
	printf("fabs b = %f\n",__half22float2(b));
	printf("sqrt c = %f\n",__half22float2(c));
}
__device__ void maxmin()
{
	half2_gpu_st a = __float22half2(2.f);
	half2_gpu_st b = __float22half2(3.5f);
	half2_gpu_st min,max;
	max = fmaxf(a,b);
	min = fminf(a,b);
	printf("max a,b = %f\n",__half22float2(max));
	printf("min a,b = %f\n",__half22float2(min));
}

__device__ void add_sat()
{
	half2_gpu_st a = __float22half2(0.4f);
	half2_gpu_st b = __float22half2(3.5f);
	half2 c = __float22half2(0.5f);
	half2_gpu_st res = __hadd_sat(a,b);
	half2_gpu_st res2 = __hadd_sat(a,c);
	half2_gpu_st res3 = __hadd_sat(c,b);
	printf("add_sat : half2_gpu_st a = %f, half2_gpu_st b=%f res = %f\n",__half22float2(a), __half22float2(b), __half22float2(res));
	printf("add_sat : half2_gpu_st a=%f, half2 c =%f res = %f\n",__half22float2(a), __half22float2(c), __half22float2(res2));
	printf("add_sat : half2 c=%f, half2__gpu_st b = %f res = %f\n",__half22float2(c), __half22float2(b), __half22float2(res3));
	
}


__device__ void math()
{
	half2_gpu_st a =__float22half2(0.346f);
	half2_gpu_st b = __float22half2(2.f);
	half2_gpu_st sin,cos,exp,log,log2,log10,rcp;
	
	sin = hsin(a);
	cos = hcos(a);
	exp = hexp(a);
	log = hlog(a);
	log2 = hlog2(a);
	log10 = hlog10(a);
	rcp = hrcp(a);
	
	
	printf("a = %f, sin = %f cos = %f, exp = %f, log = %f, log2 = %f, log10 = %f, rcp = %f\n",__half22float2(a), __half22float2(sin), __half22float2(cos), __half22float2(exp), __half22float2(log), __half22float2(log2), __half22float2(log10), __half22float2(rcp));


}

__device__ void conversion()
{
	half2_gpu_st a =__float22half2_gpu_st(0.346f);
	float2 b = __half2_gpu_st2float2(a);	

	printf("a = %f, a.x = %f, a.y = %f, a.z = %f, b= %f\n",__half22float2(a), __half22float2(a.x), __half22float2(a.y), __half22float2(a.z), b);


}
/*
__device__ void table()
{
	half2_gpu_st* a[5];
	int i;
	for(i=0;i<5;i++)
	{
		a[i]= __float22half2_gpu_st(i*1.f);
		//printf("a[%d] = %f\n",i, __half2_gpu_st2float2(a[i]));
	}	



}
*/

//////////////////////////////////////

__global__ void add_kernel()
{	
	cadna_init_gpu();
	printf("-------------Add :---------------\n");
	add();
}

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
/*
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
  
    /* Lancement de kernel (asynchrone) : */
    dim3 threadsParBloc(TAILLE_BLOC_X);

    dim3 tailleGrille(ceil(N/(float2) TAILLE_BLOC_X));

    t_startGPU = my_gettimeofday();
	add_kernel<<<1,1>>>();
	sub_kernel<<<1,1>>>();
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
    cudaDeviceSynchronize();

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
