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

void generateRandomDoubleArray(double array[], int size) {
  for (int i = 0; i < size; ++i) {
    array[i] = ((double)rand() / (double)RAND_MAX);
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
      "a6752c4648be7242a7be6521d417d2e49080813e75d65fc5ff1f693d9c9781e5.cl");
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
  cl_mem f_buffer;
  size_t gsize = 32, lsize = 8;
  int f_bound = (int)ceil(31.000430888857068 * gsize +
                          0.04877639447721133 * lsize - 3.702857142830908);
  printf("f_bound is: %d \n", f_bound);

  double *f = (double *)malloc(sizeof(double) * f_bound);
  generateRandomDoubleArray(f, f_bound);

  printf("f before: \n");
  for (int i = 0; i < f_bound; ++i) {
    printf("f[%d]: %.7lf\n", i, f[i]);
  }

  f_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * f_bound, f, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  int u = 4, c = 9, g = -1, h = 92;
  clEnqueueWriteBuffer(queue, f_buffer, CL_TRUE, 0, sizeof(double) * f_bound, f,
                       0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(int), &u);
  clSetKernelArg(kernel, 1, sizeof(int), &c);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &f_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), &g);
  clSetKernelArg(kernel, 4, sizeof(int), &h);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, f_buffer, CL_TRUE, 0, sizeof(double) * f_bound, f,
                      0, NULL, NULL);
  clFinish(queue);

  printf("f after: \n");
  for (int i = 0; i < f_bound; i++) {
    printf("f[%d]: %.7lf\n", i, f[i]);
  }
  free(f);
  clReleaseMemObject(f_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
