#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>


#ifndef TAILLE_BLOC_X
#define TAILLE_BLOC_X 32
#endif

#ifndef TAILLE_BLOC_Y
#define TAILLE_BLOC_Y 6
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
#include "cadna_gpu.cu"
#define DATATYPE float
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU half_gpu_st
#define DATATYPECADNAGPUF float_gpu_st
#else
#include <cuda_fp16.h>
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU half
#define DATATYPECADNAGPUF float
#endif

#endif

#ifndef SEED
#define SEED 1
#endif


using namespace std;


// Mesures :
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

// /* Write a PPM image file with the image of the Mandelbrot set */
// static void
// writePPM(int *buf, int width, int height, const char *fn) {
//   FILE *fp = fopen(fn, "wb");
//   fprintf(fp, "P6\n");
//   fprintf(fp, "%d %d\n", width, height);
//   fprintf(fp, "255\n");
//   for (int i = 0; i < width*height; ++i) {
//     // Map the iteration count to colors by just alternating between
//     // two greys.
//     // char c = (buf[i] & 0x1) ? 240 : 20;
//     char c = buf[i] % 256;
//     for (int j = 0; j < 3; ++j)
//       fputc(c, fp);
//   }
//   fclose(fp);
//   printf("Wrote image file %s\n", fn);
// }
/*
__global__ void mandelbrotKernel(int *d_buf, int width, int height, float x0, float y0, float dx, float dy, int maxIterations){
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
#ifdef CADNA
  cadna_init_gpu();

  if (i < width && j < height){
    DATATYPECADNAGPU x = __float_gpu_st2half_gpu_st(DATATYPECADNAGPUF(x0 + i * dx));
    DATATYPECADNAGPU y = __float_gpu_st2half_gpu_st(DATATYPECADNAGPUF(y0 + j * dy));
    int k;
    DATATYPECADNAGPU z_re = DATATYPECADNAGPU(x);
    DATATYPECADNAGPU z_im = DATATYPECADNAGPU(y);

    for (k = 0; k < maxIterations; ++k) {
      if (z_re * z_re + z_im * z_im > __float2half((DATATYPE)4.f))
	break;
      DATATYPECADNAGPU new_re = DATATYPECADNAGPU(z_re*z_re - z_im*z_im);
      DATATYPECADNAGPU new_im = DATATYPECADNAGPU(__float2half((DATATYPE)2.f) * z_re * z_im);
      z_re = x + new_re;
      z_im = y + new_im;
    }
    
    int index = (j * width + i);
    d_buf[index] = k;
  }
#else
  if (i < width && j < height){
    DATATYPECADNAGPU x = __float2half(DATATYPECADNAGPUF(x0 + i * dx));
    DATATYPECADNAGPU y = __float2half(DATATYPECADNAGPUF(y0 + j * dy));
    int k;
    DATATYPECADNAGPU z_re = DATATYPECADNAGPU(x);
    DATATYPECADNAGPU z_im = DATATYPECADNAGPU(y);

    for (k = 0; k < maxIterations; ++k) {
      if (z_re * z_re + z_im * z_im > __float2half((DATATYPE)4.f))
	break;
      DATATYPECADNAGPU new_re = DATATYPECADNAGPU(z_re*z_re - z_im*z_im);
      DATATYPECADNAGPU new_im = DATATYPECADNAGPU(__float2half((DATATYPE)2.f) * z_re * z_im);
      z_re = x + new_re;
      z_im = y + new_im;
    }
    
    int index = (j * width + i);
    d_buf[index] = k;
  }
#endif
}
*/
/*
__global__ void mandelbrotKernel(int *d_buf, int width, int height, float x0, float y0, float dx, float dy, int maxIterations){
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
#ifdef CADNA
  cadna_init_gpu();
  if (i < width && j < height){
    DATATYPECADNAGPU x = __float2half_gpu_st(x0 + i * dx);
    DATATYPECADNAGPU y = __float2half_gpu_st(y0 + j * dy);
    int k;
    DATATYPECADNAGPU z_re = x;
    DATATYPECADNAGPU z_im = y;
//	printf("x = %f y = %f z_re = %f z_im = %f\n", __half_gpu_st2float(x), __half_gpu_st2float(y), __half_gpu_st2float(z_re), __half_gpu_st2float(z_im));
    for (k = 0; k < maxIterations; ++k) {
      if (z_re * z_re + z_im * z_im > (__float2half_gpu_st((DATATYPE)4.f)))
      {
	      break;
      }
      DATATYPECADNAGPU new_re = z_re*z_re - z_im*z_im;
      DATATYPECADNAGPU new_im = (__float2half_gpu_st((DATATYPE)2.f)) * z_re * z_im;
      z_re = x + new_re;
      z_im = y + new_im;
//	printf("Dans itération K : new_re = %f new_im = %f z_re = %f z_im = %f\n", __half_gpu_st2float(new_re), __half_gpu_st2float(new_im), __half_gpu_st2float(z_re), __half_gpu_st2float(z_im));
    }
    
    int index = (j * width + i);
    d_buf[index] = k;
  }
#else
  if (i < width && j < height){
    DATATYPECADNAGPU x = __float2half(x0 + i * dx);
    DATATYPECADNAGPU y = __float2half(y0 + j * dy);
    int k;
    DATATYPECADNAGPU z_re = x;
    DATATYPECADNAGPU z_im = y;

    for (k = 0; k < maxIterations; ++k) {
      if (z_re * z_re + z_im * z_im > __float2half((DATATYPE)4.f))
	break;
      DATATYPECADNAGPU new_re = z_re*z_re - z_im*z_im;
      DATATYPECADNAGPU new_im = __float2half((DATATYPE)2.f) * z_re * z_im;
      z_re = x + new_re;
      z_im = y + new_im;
    }
    
    int index = (j * width + i);
    d_buf[index] = k;
  }
#endif
}
*/


__global__ void mandelbrotKernel(int *d_buf, int width, int height, float x0, float y0, float dx, float dy, int maxIterations){
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;
#ifdef CADNA
  cadna_init_gpu();
  if (i < width && j < height){
    DATATYPECADNAGPU x = __float2half_gpu_st(x0 + i * dx);
    DATATYPECADNAGPU y = __float2half_gpu_st(y0 + j * dy);
    int k;
    DATATYPECADNAGPU z_re = x;
    DATATYPECADNAGPU z_im = y;
//	printf("x = %f y = %f z_re = %f z_im = %f\n", __half_gpu_st2float(x), __half_gpu_st2float(y), __half_gpu_st2float(z_re), __half_gpu_st2float(z_im));
        int index = (j * width + i);
    d_buf[index] = k;
  }
#else
  if (i < width && j < height){
    DATATYPECADNAGPU x = __float2half(x0 + i * dx);
    DATATYPECADNAGPU y = __float2half(y0 + j * dy);
    int k;
    DATATYPECADNAGPU z_re = x;
    DATATYPECADNAGPU z_im = y;

        int index = (j * width + i);
    d_buf[index] = k;
  }
#endif
}
__host__ void mandelbrotCPU(int *buf, int width, int height, float x0, float y0, float dx, float dy, int maxIterations){
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; ++i) {
      DATATYPECADNA x = DATATYPECADNA(x0 + i * dx);
      DATATYPECADNA y = DATATYPECADNA(y0 + j * dy);
      int k;
      DATATYPECADNA z_re = DATATYPECADNA(x);
      DATATYPECADNA z_im = DATATYPECADNA(y);

      for (k = 0; k < maxIterations; ++k) {
	if (z_re * z_re + z_im * z_im > (DATATYPE)4.f)
	  break;
	DATATYPECADNA new_re = DATATYPECADNA(z_re*z_re - z_im*z_im);
	DATATYPECADNA new_im = DATATYPECADNA((DATATYPE)2.f * z_re * z_im);
	z_re = x + new_re;
	z_im = y + new_im;
      }

      int index = (j * width + i);
      buf[index] = k;
    }
  }
}



int main(int argc, char **argv){


  unsigned int width = 768;
  unsigned int height = 512;

  float x0 = -2;
  float x1 = 1;
  float y0 = -1;
  float y1 = 1;
  
   int maxIterations = 4096;

  double t_startGPU, t_stopGPU, t_GPU;

  printf("Usage : %s [iter [width [height]]]\n", argv[0]);

  if (argc > 1) {
    maxIterations=atoi(argv[1]);
  }

  if (argc > 2) {
    width=atoi(argv[2]);
  }

  if (argc > 3) {
    height=atoi(argv[3]);
  }

  int size = width*height;

  int *d_buf;

  cudaMalloc((void **) &d_buf, size*sizeof(int));

  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  /* Lancement de kernel (asynchrone) : */
  dim3 threadsParBloc(TAILLE_BLOC_X, TAILLE_BLOC_Y);
  dim3 tailleGrille(ceil(width/(float) TAILLE_BLOC_X), ceil(height/(float) TAILLE_BLOC_Y));

#ifdef CADNA
  cadna_init(-1, CADNA_INTRINSIC | CADNA_CANCEL);
#endif

  t_startGPU = my_gettimeofday();
  mandelbrotKernel<<< tailleGrille , threadsParBloc>>>(d_buf, width, height, x0, y0, dx, dy, maxIterations);
  cudaThreadSynchronize();
  t_stopGPU = my_gettimeofday();


  t_GPU  = t_stopGPU - t_startGPU;

#ifdef NUMCHECK
  // Rapatriement de C:
  int *buf = new int[width*height];
  cudaMemcpy(buf, d_buf, size, cudaMemcpyDeviceToHost);
  int count_CPU = 0;
  int count_GPU = 0;
  for (int i = 0; i < size ; i++)
    count_GPU += buf[i];
  double t_startCPU, t_stopCPU, t_CPU;
  t_startCPU = my_gettimeofday();
  mandelbrotCPU(buf, width, height, x0, y0, dx, dy, maxIterations);
  t_stopCPU = my_gettimeofday();
  t_CPU  = t_stopCPU - t_startCPU;

  for (int i = 0; i < size ; i++)
    count_CPU += buf[i];

  delete[] buf;
  double err = fabs((count_GPU - count_CPU) / (double) count_CPU);
  cerr << TAILLE_BLOC_X << " " << TAILLE_BLOC_Y << " " << err << " " << t_GPU << " " << t_CPU << endl;
#endif
  cerr << TAILLE_BLOC_X << " " << TAILLE_BLOC_Y << " " << t_GPU << endl;


#ifdef CADNA
  cadna_end();
#endif
  cudaFree(d_buf);
  return EXIT_SUCCESS;
}
