
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
  printf("file size: %lu\n", file_size);
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

  err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
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
      "f3fd829de44e87ff6e96b6e1b9fc1cd8b3ab258c6d7cc372450af5c02a185ea6.cl");
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
  cl_mem y_buffer = NULL;
  unsigned long gsize = 64, lsize = 8.0;
  int y_bound =
      (int)ceil(31.00000000000001 * gsize + 6.152800796892673e-14 * lsize -
                1.4551915228366852e-11);
  printf("y_bound is: %d \n", y_bound);
  float *y = (float *)malloc(sizeof(float) * y_bound);
  int i = 4, k = 5, j = 3, e = 7;
  /* printf("d is: %d\nc is: %d\no is: %d\n", s, c, o); */
  generateRandomFloatArray(y, y_bound);

  /* printf("y before: \n"); */
  /* for (int i = 0; i < y_bound; ++i) { */
  /*   printf("y[%d]: %f\n", i, y[i]); */
  /* } */
  printf("origin: %f\n", y[7]);

  y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * y_bound, y, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * y_bound, y,
                       0, NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(int), &i);
  clSetKernelArg(kernel, 1, sizeof(int), &k);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &y_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), &j);
  clSetKernelArg(kernel, 4, sizeof(int), &e);

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL,
                               NULL);
  if (err != CL_SUCCESS) {
    printf("wrong\n");
    exit(1);
  }
  clFinish(queue);

  err =
      clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * y_bound,
                          (cl_float4 *)y, 0, NULL, NULL);

  if (err != CL_SUCCESS) {
    printf("wrong\n");
    exit(1);
  }

  if (y[7] == 0.0f && y[14] == 0.0f) {
    printf("Kernel run successfully\n");
  } else {
    printf("y: %f", y[7]);
  }
  /* printf("m after: \n"); */
  /* for (int i = 0; i < m_bound * s + (m_bound / c * o); i++) { */
  /*   printf("m[%d]: %f\n", i, m[i]); */
  /* } */
  /* printf("l after: \n"); */
  /* for (int i = 0; i < l_bound * s + (l_bound / c * o); i++) { */
  /*   printf("m[%d]: %f\n", i, m[i]); */
  /* } */

  free(y);
  clReleaseMemObject(y_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
