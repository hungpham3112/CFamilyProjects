#include "util.h"
#include <stdlib.h>

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
  const char *kernelSource =
      readKernelSourceFromFile("./kernel/"
                               "6dde730a-514d-4785-925d-64fe58878cab.cl");
  if (kernelSource == NULL) {
    printf("Failed to read the kernel source from the file.\n");
    exit(EXIT_FAILURE);
  }

  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error creating program: %d\n", err);
    /* exit(EXIT_FAILURE); */
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
  kernel = clCreateKernel(program, "CopyBufferOpt1", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // Create buffer as kernel container
  cl_mem d_A, d_B;
  size_t gsize = ARR_LEN;
  size_t size = sizeof(unsigned int) * ARR_LEN;

  unsigned int *h_A = (unsigned int *)malloc(size);
  unsigned int *h_B = (unsigned int *)malloc(size);
  generateRandomUnsignedIntArray(h_A, ARR_LEN, 100, 200);
  /* ShowArr(h_A, ARR_LEN, "h_A"); */

  d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size,
                       h_A, &err);
  d_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);

  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }

  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size, h_A, 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);

  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, NULL, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL);

  /* ShowArr(h_B, ARR_LEN, "h_B"); */
  performApproxTest(h_B, h_A, ARR_LEN);
  performIdenticalTest(h_B, h_A, ARR_LEN);

  free(h_A);
  free(h_B);
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(devices[0]);
  free(platforms);
}
