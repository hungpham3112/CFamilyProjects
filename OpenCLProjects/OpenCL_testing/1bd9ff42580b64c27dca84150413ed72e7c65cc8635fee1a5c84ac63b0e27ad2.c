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
      "1bd9ff42580b64c27dca84150413ed72e7c65cc8635fee1a5c84ac63b0e27ad2.cl");
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
  cl_mem f_buffer, y_buffer, d_buffer;
  size_t gsize = 10, lsize = 5;
  int f_bound =
      (int)ceil(31.00000000000001 * gsize - 1.6790433018269837e-14 * lsize -
                1.4551915228366852e-11);
  int y_bound =
      (int)ceil(31.00000000000001 * gsize - 1.6790433018269837e-14 * lsize -
                1.4551915228366852e-11);
  int d_bound =
      (int)ceil(31.00000000000001 * gsize - 1.6790433018269837e-14 * lsize -
                1.4551915228366852e-11);
  printf("f_bound is: %d \n", f_bound);
  printf("y_bound is: %d \n", y_bound);
  printf("d_bound is: %d \n", d_bound);
  float *f = (float *)malloc(sizeof(float) * f_bound);
  float *y = (float *)malloc(sizeof(float) * y_bound);
  float *d = (float *)malloc(sizeof(float) * d_bound);
  generateRandomFloatArray(f, f_bound);
  generateRandomFloatArray(y, y_bound);
  generateRandomFloatArray(d, d_bound);
  /* printf("l before: \n"); */
  /* for (int i = 0; i < f_bound; ++i) { */
  /*   printf("l[%d]: %f\n", i, f[i]); */
  /* } */

  /* printf("y before: \n"); */
  /* for (int i = 0; i < y_bound; ++i) { */
  /*   printf("y[%d]: %f\n", i, y[i]); */
  /* } */

  /* printf("d before: \n"); */
  /* for (int i = 0; i < d_bound; ++i) { */
  /*   printf("d[%d]: %f\n", i, d[i]); */
  /* } */
  unsigned int v = 2;
  unsigned int o = 9;
  unsigned int j = 1;
  unsigned int g = 3;
  unsigned int m = 4;
  unsigned int n = 6;
  unsigned int h = 7;
  unsigned int l = 9;
  unsigned int x = 14;
  unsigned int k = 8;
  unsigned int s = 41;
  unsigned int e = 0;
  unsigned int w = 41;
  unsigned int z = 94;

  // because of using dynamic allocation so in here we have to use sizeof(float)
  // * f_bound to prevent the bug. When you accidentally use sizeof(f), it imply
  // size of a pointer which is 8 bytes, therefore when printing out array
  // information inside kernel, you can only print the two first number in array
  // because of each element is float = 4 bytes.
  f_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * f_bound, f, &err);
  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * y_bound, y, &err);
  d_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * d_bound, d, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }
  clEnqueueWriteBuffer(queue, f_buffer, CL_TRUE, 0, sizeof(float) * f_bound, f,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * y_bound, y,
                       0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &f_buffer);
  clSetKernelArg(kernel, 1, sizeof(unsigned int), &v);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &o);
  clSetKernelArg(kernel, 3, sizeof(unsigned int), &j);
  clSetKernelArg(kernel, 4, sizeof(unsigned int), &g);
  clSetKernelArg(kernel, 5, sizeof(unsigned int), &m);
  clSetKernelArg(kernel, 6, sizeof(unsigned int), &n);
  clSetKernelArg(kernel, 7, sizeof(unsigned int), &h);
  clSetKernelArg(kernel, 8, sizeof(unsigned int), &l);
  clSetKernelArg(kernel, 9, sizeof(cl_mem), &y_buffer);
  clSetKernelArg(kernel, 10, sizeof(unsigned int), &x);
  clSetKernelArg(kernel, 11, sizeof(unsigned int), &k);
  clSetKernelArg(kernel, 12, sizeof(unsigned int), &s);
  clSetKernelArg(kernel, 13, sizeof(cl_mem), &d_buffer);
  clSetKernelArg(kernel, 14, sizeof(unsigned int), &e);
  clSetKernelArg(kernel, 15, sizeof(unsigned int), &w);
  clSetKernelArg(kernel, 16, sizeof(unsigned int), &z);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, d_buffer, CL_TRUE, 0, sizeof(float) * d_bound, d,
                      0, NULL, NULL);

  /* printf("d after: \n"); */
  /* for (int i = 0; i < d_bound; i++) { */
  /*   printf("d[%d]: %lf\n", i, d[i]); */
  /* } */

  clReleaseMemObject(f_buffer);
  clReleaseMemObject(y_buffer);
  clReleaseMemObject(d_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
