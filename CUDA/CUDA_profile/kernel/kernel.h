#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARR_LEN 10000

void generateRandomUnsignedIntArray(unsigned int *array, int length,
                                    unsigned int lower, unsigned int upper) {
  unsigned int range = upper - lower + 1;

  for (int i = 0; i < length; i++) {
    array[i] = lower + (unsigned int)(rand() % range);
  }
}

void ShowArr(unsigned int *array, int length, const char *name) {
  for (int i = 0; i < length; ++i) {
    printf("%s[%d]: %u\n", name, i, array[i]);
  }
}

__host__ void performApproxTest(unsigned int *h_approx, unsigned int *h_expect,
                                int numElements) {

  // This is the implemetation for isclose() function in pytorch
  // ∣input−other∣ ≤ atol + rtol × ∣other∣
  // it can be applied to verify that the result vector is similar
  // to original vector after copying via GPU
  float atol = 1e-8f;
  float rtol = 1e-5f;
  for (int i = 0; i < numElements; ++i) {
    float abs_err = fabs((double)(h_approx[i] - h_expect[i]));
    if (abs_err > (rtol * fabs(((double)h_expect[i] + atol))) / 2.0) {
      fprintf(stderr, "Result verification failed at element %i!\n", i);
      printf("h_approx[%u]: %i, h_expect[%u]: %i\n", i, h_approx[i], i,
             h_expect[i]);
      exit(EXIT_FAILURE);
    }
  }
  fprintf(stdout, "Result approx verification succesfully!\n");
}

__host__ void performIdenticalTest(unsigned int *h_approx,
                                   unsigned int *h_expect, int numElements) {

  for (int i = 0; i < numElements; ++i) {
    if (h_approx[i] != h_expect[i]) {
      fprintf(stderr, "Result verification failed at element %i!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  fprintf(stdout, "Result identical verification succesfully!\n");
}

__global__ void CopyBufferOrigin(unsigned int *src, unsigned int *dst);
__global__ void CopyBufferOpt1(const unsigned int *src, unsigned int *dst);
__global__ void CopyBufferOpt2(const unsigned int *src, unsigned int *dst);
