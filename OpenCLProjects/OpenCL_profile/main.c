#include "util.h"

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
      "0a1ba12659a64a6479f02924131ffd7aefc810b3188946429cb24cf607e47280.cl");
  if (kernelSource == NULL) {
    fprintf(stderr, "Failed to read the kernel source from the file: %d\n",
            err);
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
  kernel = clCreateKernel(program, "CopyBufferOrigin", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // Create buffer as kernel container
  cl_mem h_A_buffer, h_B_buffer;
  size_t gsize = 100, lsize = 0;
  size_t size = sizeof(unsigned int) * ARR_LEN;

  unsigned int *h_A = (unsigned int *)malloc(size);
  generateRandomUnsignedIntArray(h_A, ARR_LEN, 100, 200);
  ShowArr(h_A, ARR_LEN, "h_A");

  /* y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
   */
  /*                           sizeof(unsigned int) * y_bound, y, &err); */
  /* c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
   * CL_MEM_COPY_HOST_PTR, */
  /*                           sizeof(unsigned int) * c_bound, c, &err); */

  /* if (err != CL_SUCCESS) { */
  /*   perror("Couldn't create buffer"); */
  /*   exit(1); */
  /* } */

  /* clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, */
  /*                      sizeof(unsigned int) * y_bound, y, 0, NULL, NULL); */
  /* clEnqueueWriteBuffer(queue, c_buffer, CL_TRUE, 0, */
  /*                      sizeof(unsigned int) * c_bound, c, 0, NULL, NULL); */

  /* clSetKernelArg(kernel, 0, sizeof(int), &p); */
  /* clSetKernelArg(kernel, 1, sizeof(int), &w); */
  /* clSetKernelArg(kernel, 2, sizeof(float), &b); */
  /* clSetKernelArg(kernel, 3, sizeof(cl_mem), &y_buffer); */
  /* clSetKernelArg(kernel, 4, sizeof(cl_mem), &c_buffer); */

  /* clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsize, &lsize, 0, NULL,
   * NULL); */
  /* clFinish(queue); */

  /* clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, */
  /*                     sizeof(unsigned int) * c_bound, c, 0, NULL, NULL); */

  /* printf("c after: \n"); */
  /* for (int i = 0; i < w; i++) { */
  /*   printf("c[%d]: %u\n", i, c[i]); */
  /* } */

  /* clReleaseMemObject(y_buffer); */
  /* clReleaseMemObject(c_buffer); */
  /* clReleaseKernel(kernel); */
  /* clReleaseCommandQueue(queue); */
  /* clReleaseProgram(program); */
  /* clReleaseContext(context); */

  /* // Free heap */
  /* clReleaseDevice(devices[0]); */
  /* free(platforms); */
}
