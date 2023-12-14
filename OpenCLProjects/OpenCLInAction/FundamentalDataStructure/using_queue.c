#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define PROGRAM_FILE "kernel.cl"

int main() {
  // Get platform
  cl_platform_id *platforms;
  cl_uint num_platforms;
  cl_int err;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS) {
    perror("Couldn't indentify platforms");
    exit(EXIT_FAILURE);
  } else {
    printf("Num platforms: %u\n", num_platforms);
  }
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // Get devices
  cl_device_id *devices;
  cl_uint num_devices;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err != CL_SUCCESS) {
    perror("Couldn't indentify devices");
    exit(EXIT_FAILURE);
  } else {
    printf("Num devices: %u\n", num_devices);
  }
  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

  // Create context
  cl_context context;
  context = clCreateContext(NULL, sizeof(devices) / sizeof(cl_device_id), &devices[0], NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create context");
    exit(EXIT_FAILURE);
  } 

  // Load kernel file to buffer as text
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;

  program_handle = fopen(PROGRAM_FILE, "r");
  fseek(program_handle, 0, SEEK_END);
  program_size = (const size_t) ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char *)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  // Create program
  cl_program program;
  program = clCreateProgramWithSource(context, 1, (const char **) &program_buffer, &program_size, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create program");
    exit(EXIT_FAILURE);
  } else {
    free(program_buffer);
  }

  // Build program
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Load specific kernel
  char *kernel_name = "vec_add";
  cl_kernel kernel;
  kernel = clCreateKernel(program, kernel_name, &err);

  // Create queue;
  cl_command_queue queue;
  const cl_queue_properties properties = {CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
  queue = clCreateCommandQueueWithProperties(context, devices[0], &properties, &err);

  // Free heap
  free(devices);
  free(platforms);


}
