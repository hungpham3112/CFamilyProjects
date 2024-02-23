

#include "cadna.h"
//#include <../src/cadna_gpu.h>
//#include <../src/cadna.cu>

#include <stdio.h>

// Constants -----------------------------------------------------------
// - alpha param
#define ALPHA (float)2.0

// - 1D float vector size: < GPU-RAM/3  &&  <= (65535 * BLOCK_SIZE_X) elements
#define VECTOR_SIZE 4

// - 1D Block size (on X): <= 512
#define BLOCK_SIZE_X  4


// // Function definitions -----------------------------------------------
// extern void InitCPU(void);
// extern void CalculCPU(void);
// extern void CalculGPUWrapper(void);
// extern void DataTransferToGPU(void);
// extern void ResultTransferFromGPU(void);
// extern bool IsEqualResult(void);

#include "cadna_gpu.cu"


__device__ float_gpu_st X_GPU[VECTOR_SIZE];
__device__ float_gpu_st Y_GPU[VECTOR_SIZE];
__device__ float_gpu_st S_GPU[VECTOR_SIZE];



float_st X_CPU[VECTOR_SIZE];
float_st Y_CPU[VECTOR_SIZE];
float_st S_CPU[VECTOR_SIZE];
float_st S_CPU_fromGPU[VECTOR_SIZE];

// Scalar variable declaration --------------------------------------

// Initialisation of the vectors /////////////////////////////////////
void InitCPU(void)
{
 // Init vectors in order to have: S=2.X+Y == VECTOR_SIZE

  X_CPU[0] = (float)5.5;
  Y_CPU[0] = (float)61./11;
  S_CPU[0] = (float)1051.0;


  for (int i = 1; i < VECTOR_SIZE; i++){
    X_CPU[i] = (float)i;
    Y_CPU[i] = (float)i+1;
    S_CPU[i] = (float)1051.0;
  }

  // Set the GPU table of results to -1.0 (just to improve debugging)
  for (int i = 0; i < VECTOR_SIZE; i++) {
    S_CPU_fromGPU[i] = (float)-1.0;
  }
  cudaMemcpyToSymbol(S_GPU,&S_CPU_fromGPU[0],sizeof(float_st)*VECTOR_SIZE,0,cudaMemcpyHostToDevice);

  // Set the CPU buffer of GPU results to -2.0 (just to improve debugging)
  for (int i = 0; i < VECTOR_SIZE; i++){
    S_CPU_fromGPU[i] = (float)-2.0;
  }
}


// Compute the "saxpy" on CPU /////////////////////////////////////////
void CalculCPU(void)
{
  for (int i = 0; i < VECTOR_SIZE; i++){
    S_CPU[i] =  X_CPU[i] + Y_CPU[i];

  }
}


// Compute the "saxpy" on GPU /////////////////////////////////////////
// Kernel computing saxpy on the GPU ----------------------------------
__global__ void CalculGPUKernel(void)
{
  int idx;               // Index of the element processed by the thread
  float_gpu_st a,b,c;
  int i;
  //  float_gpu_st mille, troismille, centonze;

//  cadna_init_gpu();

  idx = threadIdx.x + blockIdx.x*BLOCK_SIZE_X;

  a = X_GPU[idx];
  //  S_GPU [idx] = X_GPU[idx];
  b = Y_GPU[idx];

  for(i=3;i<=15;i++){
    c = b;
    //printf("i%i idx=%i str(b)=%s\n", i, str(b));
    b = 111.f - 1130.f/b + 3000.f/(a*b);
    a = c;
  }

  S_GPU[idx] = b;
}

// Wrapper running kernel on the GPU ----------------------------------
void CalculGPUWrapper(void)
{
 // Grid and Block definitions (to run a grid of blocks of threads on the GPU)
 dim3 Dg, Db;
 // Set the Block definition
 Db.x = BLOCK_SIZE_X;
 Db.y = 1;
 Db.z = 1;
 // Set the Grid definition
 if (VECTOR_SIZE % BLOCK_SIZE_X == 0) {
   Dg.x = VECTOR_SIZE/BLOCK_SIZE_X;
 } else {
   Dg.x = VECTOR_SIZE/BLOCK_SIZE_X + 1;
 }
 Dg.y = 1;
 Dg.z = 1;
 // Call the kernel on the GPU
 CalculGPUKernel<<< Dg,Db >>>();
}


// Transfer data to GPU and retrieve results from GPU ////////////////
// Copy data on the GPU ----------------------------------------------
void DataTransferToGPU(void)
{
  cudaMemcpyToSymbol(X_GPU,&X_CPU[0],sizeof(float_st)*VECTOR_SIZE,0,
		     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y_GPU,&Y_CPU[0],sizeof(float_st)*VECTOR_SIZE,0,
		     cudaMemcpyHostToDevice);
}


// Copy results from the GPU -----------------------------------------
void ResultTransferFromGPU(void)
{
  cudaMemcpyFromSymbol(&S_CPU_fromGPU[0],S_GPU,sizeof(float_st)*VECTOR_SIZE,0,
		       cudaMemcpyDeviceToHost);

}



// Compare CPU and GPU final results /////////////////////////////////


// Main function ////////////////////////////////////////////////////
int main (int argc, char **argv)
{
    //InitCuda(1, 512);
  char s[128];


  cadna_init(-1);
  // Beginning of the program
  printf(" pgm started\n");

  // Data initialization (on the CPU)
  printf("- Initialization\n");
  InitCPU();

  // CPU computation
  printf("- CPU computations.\n");

  CalculCPU();

  // GPU computation
  printf("- GPU computations.\n");
  // - transfer data on GPU
  printf("  + data transfer on GPU\n");
  DataTransferToGPU();

  // - run GPU computations
  printf("  + run computations on GPU\n");
  CalculGPUWrapper();

  // - transfer results on CPU
  printf("  + result transfer on CPU\n");
  ResultTransferFromGPU();


  for (int i = 0; i < VECTOR_SIZE /*VECTOR_SIZE*/; i++) {
    printf("  S_CPU[%d] = ",i);
    S_CPU[i].display();
    printf("\n  S_fromGPU[%d] = ",i);
    S_CPU_fromGPU[i].display();
    printf("%s \n",S_CPU_fromGPU[i].str(s));
  }

  printf(" pgm finished\n");
  return(EXIT_SUCCESS);
}

