#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <stdio.h>

int main() {
  cl_device_id my_device_id;
  int result;
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &my_device_id, NULL);
  if (result != CL_SUCCESS) {
    printf("Something went wrong\n");
    return 1;
  } else {
    printf("Good");
  }
}
