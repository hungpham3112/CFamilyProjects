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

void generateRandomUnsignedIntArray(unsigned int array[], int size,
                                    unsigned int lower, unsigned int upper) {
  unsigned int range = upper - lower + 1;

  for (int i = 0; i < size; i++) {
    array[i] = lower + (unsigned int)(rand() % range);
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
      "8e355ff536c99be01f759aaac95003072f5943261072f97999895d94ef774f4b.cl");
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
    exit(EXIT_FAILURE);
  }

  cl_kernel kernel;
  // This will create with specific kernel name
  kernel = clCreateKernel(program, "A", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // Create buffer as kernel container
  cl_mem w_buffer;
  size_t gsize = 100, lsize = 10;
  unsigned int w_bound =
      (int)ceil(1.9999999999999996 * gsize + 4.489579651123267e-16 * lsize -
                0.9999999999990905);

  printf("w_bound is: %d \n", w_bound);

  unsigned int *w = (unsigned int *)malloc(sizeof(unsigned int) * w_bound);
  generateRandomUnsignedIntArray(w, w_bound, 100, 200);
  /* printf("w before: \n"); */
  /* for (int i = 0; i < w_bound; ++i) { */
  /*   printf("w[%d]: %u\n", i, w[i]); */
  /* } */

  // In this code because of y is static allocation so you can use sizeof(y)
  // instead of sizeof(unsigned int) * y_bound
  w_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(unsigned int) * w_bound, w, &err);

  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  unsigned int t = 2, v = 3;
  clEnqueueWriteBuffer(queue, w_buffer, CL_TRUE, 0,
                       sizeof(unsigned int) * w_bound, w, 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &w_buffer);
  clSetKernelArg(kernel, 1, sizeof(unsigned int), &t);
  clSetKernelArg(kernel, 2, sizeof(unsigned int), &v);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, w_buffer, CL_TRUE, 0,
                      sizeof(unsigned int) * w_bound, w, 0, NULL, NULL);

  /* printf("w after: \n"); */
  /* for (int i = 0; i < w_bound; i++) { */
  /*   printf("w[%d]: %u\n", i, w[i]); */
  /* } */

  free(w);
  clReleaseMemObject(w_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
