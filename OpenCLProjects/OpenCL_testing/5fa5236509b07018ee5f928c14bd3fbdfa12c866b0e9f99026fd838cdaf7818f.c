#define CL_TARGET_OPENCL_VERSION 300

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <limits.h>
void printBuildLog(cl_program program, cl_device_id device) {
  size_t logSize;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &logSize);

  char *log = (char *)malloc(logSize);
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log,
                        NULL);

  printf("Build Log:\n%s\n", log);

  free(log);
}

void generateRandomFloatArray(float array[], unsigned long size) {
  for (unsigned long i = 0; i < size; ++i) {
    array[i] = ((float)rand() / (float)RAND_MAX);
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

int main() {
  srand((unsigned int)time(NULL));
  // Get platform
  cl_platform_id *platforms;
  cl_int err;
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't get platform id\n");
    exit(1);
  }
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // Get device
  cl_device_id *devices;
  cl_uint num_devices;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err < 0) {
    perror("Couldn't get device id\n");
    exit(1);
  }
  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

  // Create 1 context to store 1 above device.
  cl_context context;
  context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
  if (err < 0) {
    perror("Couldn't create context\n");
    exit(EXIT_FAILURE);
  }

  // Create program in context
  cl_program program;
  const char *kernelSource = readKernelSourceFromFile(
      "5fa5236509b07018ee5f928c14bd3fbdfa12c866b0e9f99026fd838cdaf7818f.cl");
  if (kernelSource == NULL) {
    printf("Failed to read the kernel source from the file.\n");
    exit(EXIT_FAILURE);
  }

  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error creating program: %d\n", err);
    exit(EXIT_FAILURE);
  }

  // Create queue
  cl_command_queue queue;
  queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create queue");
    exit(EXIT_FAILURE);
  }

  // Set flag and build program
  const char options[] =
      "-cl-std=CL3.0 -cl-mad-enable -Werror -DCL_TARGET_OPENCL_VERSION=300";
  err = clBuildProgram(program, 1, &devices[0], options, NULL, NULL);
  if (err != CL_SUCCESS) {
    perror("Couldn't build program");
    printBuildLog(program, devices[0]);
    exit(EXIT_FAILURE);
  }

  // This will create with specific kernel name
  cl_kernel kernel;
  kernel = clCreateKernel(program, "A", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    // exit(1);
  }

  // Create buffer as kernel container
  cl_mem s_buffer = NULL, f_buffer = NULL;
  size_t gsize = 32, lsize = 16;
  int s_bound = (int)ceil(1.0000000000000002 * gsize -
                          5.398695250599143e-16 * lsize - 1.0000000000004547);
  int f_bound = (int)ceil(1.39 * gsize - 5.3986952505 * lsize + 123.2);

  printf("s_bound:  %d\n", s_bound);
  printf("f_bound: %d\n", f_bound);
  float *s = (float *)malloc(sizeof(float) * s_bound);
  float *f = (float *)malloc(sizeof(float) * f_bound);
  generateRandomFloatArray(s, s_bound);
  generateRandomFloatArray(f, f_bound);
  /* printf("s before: \n"); */
  /* for (int i = 0; i < s_bound; ++i) { */
  /*   printf("s[%d]: %f\n", i, s[i]); */
  /* } */
  printf("f before: \n");
  for (int i = 0; i < f_bound; ++i) {
    printf("f[%d]: %f\n", i, f[i]);
  }

  unsigned int l = 100;
  unsigned int e = 13;
  unsigned int b = 64;
  unsigned int g = 1;

  s_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * s_bound, s, &err);
  f_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * f_bound, f, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, s_buffer, CL_TRUE, 0, sizeof(float) * s_bound, s,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(queue, f_buffer, CL_TRUE, 0, sizeof(float) * f_bound, f,
                       0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &s_buffer);
  clSetKernelArg(kernel, 1, sizeof(unsigned int), &l);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &e);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &f_buffer);
  clSetKernelArg(kernel, 4, sizeof(unsigned int), &b);
  clSetKernelArg(kernel, 5, sizeof(unsigned int), &g);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, f_buffer, CL_TRUE, 0, sizeof(float) * f_bound, f,
                      0, NULL, NULL);

  printf("f after: \n");
  for (int i = 0; i < f_bound + 20; i++) {
    printf("f[%d]: %lf\n", i, f[i]);
  }

  free(s);
  free(f);
  clReleaseMemObject(s_buffer);
  clReleaseMemObject(f_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
