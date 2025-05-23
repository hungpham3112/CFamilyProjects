#define CL_TARGET_OPENCL_VERSION 300

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define ARR_LEN 100

void performApproxTest(unsigned int *h_approx, unsigned int *h_expect,
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

void performIdenticalTest(unsigned int *h_approx, unsigned int *h_expect,
                          int numElements) {

  for (int i = 0; i < numElements; ++i) {
    if (h_approx[i] != h_expect[i]) {
      fprintf(stderr, "Result verification failed at element %i!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  fprintf(stdout, "Result identical verification succesfully!\n");
}

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

const char *readKernelSourceFromFile(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    return NULL; // File not found or couldn't be opened
  }

  // Find the size of the file
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  rewind(file);

  // Allocate memory for the file content plus a null-terminating character
  char *source_code = (char *)malloc(file_size + 1);
  if (source_code == NULL) {
    fclose(file);
    return NULL; // Memory allocation failed
  }

  // Read the file content
  size_t read_size = fread(source_code, 1, file_size, file);
  fclose(file);

  if (read_size != file_size) {
    free(source_code);
    return NULL; // Reading the file failed
  }

  // Null-terminate the string
  source_code[file_size] = '\0';

  return source_code;
}
