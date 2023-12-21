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
      "035f57cf2b4d3f44af0cfab15a4f64bf363d94fed0d1bab20cb7095b34493674.cl");
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
  cl_mem m_buffer, l_buffer;
  unsigned long gsize = 2000, lsize = 2.0;
  unsigned long m_bound =
      (unsigned long)ceil(16439.232743560806 * gsize -
                          48449.26202661346 * lsize - 28335196.605217382);
  unsigned long l_bound =
      (unsigned long)ceil(0.9423256014692764 * gsize -
                          0.15445459227213437 * lsize + 158.37652173913284);
  /* printf("m_bound is: %lu \n", m_bound); */
  /* printf("l_bound is: %lu \n", l_bound); */
  float *m = (float *)malloc(sizeof(float) * m_bound);
  float *l = (float *)malloc(sizeof(float) * l_bound);
  int s = 4, c = 5, o = 3;
  /* printf("d is: %d\nc is: %d\no is: %d\n", s, c, o); */
  generateRandomFloatArray(m, m_bound);
  generateRandomFloatArray(l, l_bound);
  /* printf("m before: \n"); */
  /* for (int i = 0; i < m_bound; ++i) { */
  /*   printf("m[%d]: %f\n", i, m[i]); */
  /* } */

  /* printf("l before: \n"); */
  /* for (int i = 0; i < l_bound; ++i) { */
  /*   printf("l[%d]: %f\n", i, l[i]); */
  /* } */

  m_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(m), m, &err);
  l_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(l), l, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, m_buffer, CL_TRUE, 0, sizeof(m), m, 0, NULL,
                       NULL);
  clEnqueueWriteBuffer(queue, l_buffer, CL_TRUE, 0, sizeof(l), l, 0, NULL,
                       NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &l_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), &s);
  clSetKernelArg(kernel, 3, sizeof(int), &c);
  clSetKernelArg(kernel, 4, sizeof(int), &o);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL, NULL);

  clEnqueueReadBuffer(queue, m_buffer, CL_TRUE, 0, sizeof(m), m, 0, NULL, NULL);

  if (12 / (c * s) < s) {
    printf("%f\n", m[12 * s + (int)(12 / (c * o))]);
    printf("%f\n", l[(int)(12 % (s * c))]);
    if (m[12 * s + (int)(12 / (c * o))] == l[(int)(12 % (s * c))]) {
      printf("Kernel run successfully\n");
    } else {
      printf("NO\n");
    }
  }
  /* printf("m after: \n"); */
  /* for (int i = 0; i < m_bound * s + (m_bound / c * o); i++) { */
  /*   printf("m[%d]: %f\n", i, m[i]); */
  /* } */
  /* printf("l after: \n"); */
  /* for (int i = 0; i < l_bound * s + (l_bound / c * o); i++) { */
  /*   printf("m[%d]: %f\n", i, m[i]); */
  /* } */

  free(m);
  free(l);
  clReleaseMemObject(m_buffer);
  clReleaseMemObject(l_buffer);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  // Free heap
  clReleaseDevice(devices[0]);
  free(platforms);
}
