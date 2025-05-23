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

void generateRandomIntArray(int *array, int size, int min, int max) {
  for (int i = 0; i < size; ++i) {
    array[i] = rand() % (max - min + 1) + min;
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
      "f9c28d8e908897b4b97c60b9787e49d79070653c3f5503a19630189f68e43c12.cl");
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
    fprintf(stderr, "Error building program: %d\n", err);

    // Get the build log size
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    // Allocate memory for the build log
    char *log = (char *)malloc(log_size);

    // Get the build log
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);

    // Print the build log
    fprintf(stderr, "Build log:\n%s\n", log);

    // Free the memory
    free(log);

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
  cl_mem v_buffer = NULL, g_buffer = NULL;
  size_t gsize = 64, lsize = 8;

  int v_bound = (int)ceil(9.12 * gsize + 0.12 * lsize + 12);
  int g_bound = (int)ceil(12 * gsize - 12 * lsize + 90);

  int *v = (int *)malloc(sizeof(int) * v_bound);
  int *g = (int *)malloc(sizeof(int) * g_bound);

  printf("g_bound is: %d \n", g_bound);
  printf("v_bound is: %d \n", v_bound);

  generateRandomIntArray(g, g_bound, -100, 100);
  generateRandomIntArray(v, v_bound, -100, 100);

  printf("g before: \n");
  for (int i = 0; i < g_bound; ++i) {
    printf("g[%d]: %d\n", i, g[i]);
  }

  /* printf("c before: \n"); */
  /* for (int i = 0; i < c_bound; ++i) { */
  /*   printf("c[%d]: %d\n", i, c[i]); */
  /* } */

  v_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * v_bound, v, &err);
  g_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * g_bound, g, &err);

  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, g_buffer, CL_TRUE, 0, sizeof(int) * g_bound, g, 0,
                       NULL, NULL);

  clEnqueueWriteBuffer(queue, v_buffer, CL_TRUE, 0, sizeof(int) * v_bound, v, 0,
                       NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_buffer);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, g_buffer, CL_TRUE, 0, sizeof(int) * g_bound, g, 0,
                      NULL, NULL);
  clFinish(queue);

  /* if (g[0] == c[0]) { */
  /*   printf("Run kernel successfully\n"); */
  /* } */

  printf("g after: \n");
  for (int i = 0; i < g_bound - 1; ++i) {
    printf("g[%d]: %d\n", i, g[i]);
  }

  free(g);
  free(v);
  clReleaseMemObject(g_buffer);
  clReleaseMemObject(v_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
