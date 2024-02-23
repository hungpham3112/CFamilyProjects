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
#include "cadna_gpu.cu"
#define DATATYPE half2
//#define DATATYPECADNA float_st
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


/*
__global__ void add(float a, float b,float res)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	half2 a_h = __float2half2_rn(a);
	half2 b_h = __float2half2_rn(b);
	half2 res_h = __hadd(a_h,b_h);
	res = __low2float()+__high2float()(res_h);
}
*/


__device__ void add()
{
	//half2_gpu_st a =__float2half2_rn{-2.1564f,0.f};
	//half2_gpu_st b = __float2half2_rn{2.f,0.f};
	half2_gpu_st a =__float2half2_rn(-2.1564f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c,d;
	c= a+b;
	d = __hadd2(a,b);
	
	printf("+ c.x=%f\n",__low2float(c.x));
	printf("+ c.y=%f\n",__low2float(c.y));
	printf("+ c.z=%f\n",__low2float(c.z));
	printf("+ c=%f\n",__low2float(c));

	printf("hadd d.x=%f\n",__low2float(d.x));
	printf("hadd d.y=%f\n",__low2float(d.y));
	printf("hadd d.z=%f\n",__low2float(d.z));
	printf("+ d=%f\n",__low2float(d));
	
}

__device__ void sub()
{
	half2_gpu_st a =__float2half2_rn(0.346f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2 a_h = __float2half2_rn(0.346f);
	half2 b_h = __float2half2_rn(2.f);
	half2_gpu_st res_h, res, res2, res3, res4, res5, res6;
	//res_h = a_h-b_h;
	res = __hsub2(a_h,b_h);
	res2 = __hsub2(a,b_h);
	res3 = __hsub2(a_h,b);
	res4 = __hsub2_sat(a_h,b_h);
	res5 = __hsub2_sat(a,b_h);
	res6 = __hsub2_sat(a_h,b);
	
	half2_gpu_st cons,cons2;
	cons.x = __float2half2_rn(1.13f);
	cons.y = __float2half2_rn(1.12f);
	cons.z = __float2half2_rn(1.12f);
	printf("TEST : significant digit :%d\n", cons.nb_significant_digit());
	cons2.x = __float2half2_rn(1.13f);
	cons2.y = __float2half2_rn(1.12f);
	cons2.z = __float2half2_rn(1.12f);
	
	printf("TEST : @.0 :%f\n", __half2_gpu_st2float(cons2*cons));





	//printf("half2 half2 a = %f, half2 b = %f, res_h = %f\n",__low2float(a_h)+__high2float(a_h), __low2float(b_h)+__high2float(b_h), __low2float(res_h)+__high2float(res_h)(res_h));
	printf("hsub half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__low2float(a), __low2float(b), __low2float(res));
	printf("hsub  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__low2float(a), __low2float(b_h), __low2float(res2));
	printf("hsub  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__low2float(a_h), __low2float(b), __low2float(res3));

	printf("hsub_sat half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__low2float(a), __low2float(b), __low2float(res4));
	printf("hsub_sat  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__low2float(a), __low2float(b_h), __low2float(res5));
	printf("hsub_sat  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__low2float(a_h), __low2float(b), __low2float(res6));




}

__device__ void mul()
{
	half2_gpu_st a =__float2half2_rn(0.346f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c;
	c= a*b;
	printf("c.x=%f\n",__low2float(c.x)+__high2float(c.x));
	printf("c.y=%f\n",__low2float(c.y)+__high2float(c.y));
	printf("c.z=%f\n",__low2float(c.z)+__high2float(c.z));

	half2 a_h = __float2half2_rn(0.346f);
	half2 b_h = __float2half2_rn(2.f);
	half2_gpu_st  res, res2, res3, res4, res5, res6;
	res = __hmul2(a_h,b_h);
	res2 = __hmul2(a,b_h);
	res3 = __hmul2(a_h,b);
	res4 = __hmul2_sat(a_h,b_h);
	res5 = __hmul2_sat(a,b_h);
	res6 = __hmul2_sat(a_h,b);

	printf("half2 half2 a = %f, half2 b = %f, res_h = %f\n",__low2float(a_h), __low2float(b_h), __low2float(c));
	printf("hmul half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__low2float(a), __low2float(b), __low2float(res));
	printf("hmul  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__low2float(a), __low2float(b_h), __low2float(res2));
	printf("hmul  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__low2float(a_h), __low2float(b), __low2float(res3));

	printf("hmul_sat half2_gpu_st a = %f, half2_gpu_st b = %f, res_h = %f\n",__low2float(a), __low2float(b), __low2float(res4));
	printf("hmul_sat  half2_gpu_st a = %f, half2 b = %f, res = %f\n",__low2float(a), __low2float(b_h), __low2float(res5));
	printf("hmul_sat  half2 a = %f, half2_gpu_st b = %f, res = %f\n",__low2float(a_h), __low2float(b), __low2float(res6));







}

__device__ void div()
{
	half2_gpu_st a =__float2half2_rn(2.f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c,d;
	c= a/b;
	d= __h2div(a,b);
	printf("c.x=%f\n",__low2float(c.x));
	printf("c.y=%f\n",__low2float(c.y));
	printf("c.z=%f\n",__low2float(c.z));
	printf("c=%f\n",__low2float(c));
	printf("d=%f\n",__low2float(d));


}

__device__ void fma()
{
	half2_gpu_st a = __float2half2_rn(0.4f);
	half2_gpu_st b = __float2half2_rn(3.5f);
	half2_gpu_st c = __float2half2_rn(0.5678f);
	half2 a_h = __float2half2_rn(0.5678f);
	half2 b_h = __float2half2_rn(3.5f);
	half2 c_h = __float2half2_rn(0.5678f);

	half2 res_h = __hfma2(a_h,b_h,c_h);
	half2_gpu_st res = __hfma2(a,b,c);
	half2_gpu_st res2 = __hfma2(a,b_h,c_h);
	half2_gpu_st res3 = __hfma2(a,b_h,c);
	half2_gpu_st res4 = __hfma2(a,b,c_h);
	half2_gpu_st res5 = __hfma2(a_h,b,c);
	half2_gpu_st res6 = __hfma2(a_h,b,c_h);
	half2_gpu_st res7 = __hfma2(a_h,b_h,c);

	printf("fma : half2 a = %f, half2 b=%f, half2 c =%f res_h = %f\n",__low2float(a_h), __low2float(b_h),__low2float(c_h), __low2float(res_h));
	printf("fma : half2_gpu_st a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a), __low2float(b),__low2float(c), __low2float(res));
	printf("fma : half2_gpu_st a = %f, half2 b=%f, half2 c =%f res = %f\n",__low2float(a), __low2float(b_h),__low2float(c_h), __low2float(res2));
	printf("fma : half2_gpu_st a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a), __low2float(b_h),__low2float(c), __low2float(res3));
	printf("fma : half2_gpu_st a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__low2float(a), __low2float(b),__low2float(c_h), __low2float(res4));
	printf("fma : half2 a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a_h), __low2float(b),__low2float(c), __low2float(res5));
	printf("fma : half2 a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__low2float(a_h), __low2float(b),__low2float(c_h), __low2float(res6));
	printf("fma : half2 a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a_h), __low2float(b_h),__low2float(c), __low2float(res7));
	
	half2 res_h_s = __hfma2_sat(a_h,b_h,c_h);
	half2_gpu_st res_s = __hfma2_sat(a,b,c);
	half2_gpu_st res2_s = __hfma2_sat(a,b_h,c_h);
	half2_gpu_st res3_s = __hfma2_sat(a,b_h,c);
	half2_gpu_st res4_s = __hfma2_sat(a,b,c_h);
	half2_gpu_st res5_s = __hfma2_sat(a_h,b,c);
	half2_gpu_st res6_s = __hfma2_sat(a_h,b,c_h);
	half2_gpu_st res7_s = __hfma2_sat(a_h,b_h,c);
	
	printf("fma_sat in\n");
	printf("fma_sat : half2 a = %f, half2 b=%f, half2 c =%f res_h = %f\n",__low2float(a_h), __low2float(b_h),__low2float(c_h), __low2float(res_h_s));
	printf("fma_sat : half2_gpu_st a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a), __low2float(b),__low2float(c), __low2float(res_s));
	printf("fma_sat : half2_gpu_st a = %f, half2 b=%f, half2 c =%f res = %f\n",__low2float(a), __low2float(b_h),__low2float(c_h), __low2float(res2_s));
	printf("fma_sat : half2_gpu_st a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a), __low2float(b_h),__low2float(c), __low2float(res3_s));
	printf("fma_sat : half2_gpu_st a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__low2float(a), __low2float(b),__low2float(c_h), __low2float(res4_s));
	printf("fma_sat : half2 a = %f, half2_gpu_st b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a_h), __low2float(b),__low2float(c), __low2float(res5_s));
	printf("fma_sat : half2 a = %f, half2_gpu_st b=%f, half2 c =%f res = %f\n",__low2float(a_h), __low2float(b),__low2float(c_h), __low2float(res6_s));
	printf("fma_sat : half2 a = %f, half2 b=%f, half2_gpu_st c =%f res = %f\n",__low2float(a_h), __low2float(b_h),__low2float(c), __low2float(res7_s));
	



}



__device__ void egalite()
{
	half2_gpu_st a = __float2half2_rn(2.f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c = __float2half2_rn(5.f);
	if(a == b)
	{
		printf("a == b\n");
	}
	if(a != c)
	{
		printf("a != c\n");
	}
	half2 a_h = __float2half2_rn(2.f);
	half2 res_h = __hneg2(a_h);
	printf("hneq a_h =%f, res_h = %f\n", __low2float(a_h), __low2float(res_h));
	half2_gpu_st res = __hneg2(a);
	printf("hneq a =%f, res = %f\n", __low2float(a), __low2float(res));

	int res_eq = a==c;
	bool res_heq = __hbeq2(a_h,__h2div(__float2half2_rn(0.f), __float2half2_rn(0.f)));
	bool res_hequ = __hbequ2(a_h,__h2div(__float2half2_rn(0.f), __float2half2_rn(0.f)));
	printf("heq a == b, res = %d\n",res_eq);
	printf("heq , res = %d\n",res_heq);
	printf("hequ , res = %d\n",res_hequ);




}
__device__ void comparaison()
{
	half2_gpu_st a = __float2half2_rn(2.f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c = __float2half2_rn(5.f);
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
	half2_gpu_st a = __float2half2_rn(2.f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st c = __float2half2_rn(5.f);
	a = fabsf(a);
	b = fabs(b);
	c = h2sqrt(c);
	printf("fabsf a = %f\n",__low2float(a));
	printf("fabs b = %f\n",__low2float(b));
	printf("sqrt c = %f\n",__low2float(c));
}
__device__ void maxmin()
{
	half2_gpu_st a = __float2half2_rn(2.f);
	half2_gpu_st b = __float2half2_rn(3.5f);
	half2_gpu_st min,max;
	max = fmaxf(a,b);
	min = fminf(a,b);
	printf("max a,b = %f\n",__low2float(max));
	printf("min a,b = %f\n",__low2float(min));
}

__device__ void add_sat()
{
	half2_gpu_st a = __float2half2_rn(0.4f);
	half2_gpu_st b = __float2half2_rn(3.5f);
	half2 c = __float2half2_rn(0.5f);
	half2_gpu_st res = __hadd2_sat(a,b);
	half2_gpu_st res2 = __hadd2_sat(a,c);
	half2_gpu_st res3 = __hadd2_sat(c,b);
	printf("add_sat : half2_gpu_st a = %f, half2_gpu_st b=%f res = %f\n",__low2float(a), __low2float(b), __low2float(res));
	printf("add_sat : half2_gpu_st a=%f, half2 c =%f res = %f\n",__low2float(a), __low2float(c), __low2float(res2));
	printf("add_sat : half2 c=%f, half2__gpu_st b = %f res = %f\n",__low2float(c), __low2float(b), __low2float(res3));
	
}


__device__ void math()
{
	half2_gpu_st a =__float2half2_rn(0.346f);
	half2_gpu_st b = __float2half2_rn(2.f);
	half2_gpu_st sin,cos,exp,log,log2,log10,rcp;
	
	sin = h2sin(a);
	cos = h2cos(a);
	exp = h2exp(a);
	log = h2log(a);
	log2 = h2log2(a);
	log10 = h2log10(a);
	rcp = h2rcp(a);
	
	
	printf("a = %f, sin = %f cos = %f, exp = %f, log = %f, log2 = %f, log10 = %f, rcp = %f\n",__low2float(a), __low2float(sin), __low2float(cos), __low2float(exp), __low2float(log), __low2float(log2), __low2float(log10), __low2float(rcp));


}

__device__ void conversion()
{
	half2_gpu_st a =__float2half2_gpu_st(0.346f);
	float b = __half2_gpu_st2float(a);	

	printf("a = %f, a.x = %f, a.y = %f, a.z = %f, b= %f\n",__low2float(a)+__high2float(a), __low2float(a.x)+__high2float(a.x), __low2float(a.y)+__high2float(a.y), __low2float(a.z)+__high2float(a.z), b);


}

/*
__device__ void table()
{
	half2_gpu_st* a[5];
	int i;
	for(i=0;i<5;i++)
	{
		a[i]= __float2half2_rn_gpu_st(i*1.f);
		//printf("a[%d] = %f\n",i, __half2_gpu_st2float(a[i]));
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

    dim3 tailleGrille(ceil(N/(float) TAILLE_BLOC_X));

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
