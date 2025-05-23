#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


int main() {
  cl_platform_id *platforms;
  cl_uint num_platforms, err;

  // Identify # of platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't find any platforms");
    exit(1);
  }

  // Malloc for multiple platforms
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  cl_device_id *devices;
  cl_uint num_devices;
  // Identify # of devices
  if (num_platforms == 1) {
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err < 0) {
      perror("Couldn't find any devices");
      exit(1);
    }
    devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
  }

  cl_context context;
  cl_uint context_reference_count;
  // In here context allocation does not require.
  context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
  if (err < 0) {
    perror("Couldn't create context\n");
    exit(1);
  } else {
    printf("Create context successfully\n");
  }

  clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(context_reference_count), &context_reference_count, NULL);
  printf("Context reference count: %u\n", context_reference_count);
  clRetainContext(context);
  clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(context_reference_count), &context_reference_count, NULL);
  printf("Context reference count: %u\n", context_reference_count);
  clReleaseContext(context);
  clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(context_reference_count), &context_reference_count, NULL);
  printf("Context reference count: %u\n", context_reference_count);

  // Free heap
  free(devices);
  free(platforms);
}
