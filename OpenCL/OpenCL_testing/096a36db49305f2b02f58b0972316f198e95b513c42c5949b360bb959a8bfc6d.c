#include <stddef.h>
#include <sys/types.h>
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
typedef unsigned short ushort;
void generateRandomUshortArray(ushort *arr, unsigned long size) {
  for (unsigned long i = 0; i < size; ++i) {
    arr[i] = (ushort)(rand() % (USHRT_MAX + 1));
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
  printf("number of platforms: %d\n", num_platforms);
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
  printf("num devices: %d\n", num_devices);
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
      "096a36db49305f2b02f58b0972316f198e95b513c42c5949b360bb959a8bfc6d.cl");
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
    char *log;
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size,
                          &log, NULL);
    printf("Build log:\n%s\n", log);
    free(log);
    exit(1);
  }

  // Create kernel
  cl_kernel kernel;
  // This will create with specific kernel name
  kernel = clCreateKernel(program, "A", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // Create buffer as kernel container
  cl_mem e_buffer, c_buffer;
  size_t gsize = 20, lsize = 2.0;
  unsigned long e_bound =
      (unsigned long)ceil(0.9423256014692764 * gsize -
                          0.15445459227213437 * lsize + 12.37652173913284);
  unsigned long c_bound =
      (unsigned long)ceil(0.9423256014692764 * gsize -
                          0.15445459227213437 * lsize - 12.37652173913284);
  printf("m_bound is: %lu \n", e_bound);
  printf("l_bound is: %lu \n", c_bound);
  ushort *e = (ushort *)malloc(sizeof(ushort) * e_bound);
  ushort *c = (ushort *)malloc(sizeof(ushort) * c_bound);
  unsigned int g = 4, d = 9;
  generateRandomUshortArray(e, e_bound);
  generateRandomUshortArray(c, c_bound);
  printf("e before: \n");
  for (int i = 0; i < e_bound; ++i) {
    printf("e[%d]: %hu\n", i, e[i]);
  }

  printf("c before: \n");
  for (int i = 0; i < c_bound; ++i) {
    printf("c[%d]: %hu\n", i, c[i]);
  }

  e_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(ushort) * e_bound, e, &err);
  c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(ushort) * c_bound, c, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, e_buffer, CL_TRUE, 0, sizeof(ushort) * e_bound, e,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(queue, c_buffer, CL_TRUE, 0, sizeof(ushort) * c_bound, c,
                       0, NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &e_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_buffer);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &g);
  clSetKernelArg(kernel, 3, sizeof(unsigned int), &d);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, e_buffer, CL_TRUE, 0, sizeof(ushort) * e_bound, e,
                      0, NULL, NULL);
  clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 1, sizeof(ushort) * c_bound, c,
                      0, NULL, NULL);

  printf("e after: \n");
  for (int i = 0; i < gsize; i++) {
    printf("e[%d]: %hu\n", i, e[i]);
  }
  printf("c after: \n");
  for (int i = 0; i < gsize + 100; i++) {
    printf("c[%d]: %hu\n", i, c[i]);
  }

  free(e);
  free(c);
  clReleaseMemObject(e_buffer);
  clReleaseMemObject(c_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  free(platforms);
}
