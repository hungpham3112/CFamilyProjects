#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h> 

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {   
  // Identify platform
  cl_platform_id *platforms;
  cl_uint num_entries = 2, err = CL_SUCCESS, num_platforms;
  // The first run to identify how many number of platforms available
  err = clGetPlatformIDs(num_entries, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't find any platforms");
    exit(1);
  } else {
    printf("Num platforms: %d\n", num_platforms);
  }
  // Memory allocation for platform based on # of platforms above
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
  // In here you pass num_platforms to num_entries because you want to know all IDs.
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // Identify devices id
  cl_device_id *devices;
  cl_uint num_devices;
  // The first time run to determine how many number of device avaiable in platform
  // we only need the # of device so we set our expectation(# of entries = 0).
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err < 0) {
    perror("Couldn't find any devices");
    exit(1);
  } else {
    printf("Num devices: %d\n", num_devices);
  }
  // Then we need to allocate devices
  devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

  // Identify devices info
  cl_device_type device_type;
  cl_uint device_vendor_id, device_max_compute_units;
  char *device_extensions;
  size_t device_extension_size;
  for (int i = 0; i < num_devices; ++i) {
      clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(device_vendor_id), &device_vendor_id, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_max_compute_units), &device_max_compute_units, NULL);
      clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &device_extension_size);
      device_extensions = (char *) malloc(device_extension_size);
      clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, device_extension_size, device_extensions, NULL);
      // Using & not && in here because device_type is long unsigned int
      // and CL_DEVICE_TYPE_GPU != 0 so if use && in here the condition always
      // true -> wrong. In here we have 2 number so we need bitwise & operator.
      if (device_type == CL_DEVICE_TYPE_GPU) {
        printf("Device type: GPU\n");
      }
      printf("Device vendor id: %d\n", device_vendor_id);
      printf("Device max_compute_units: %d\n", device_max_compute_units);
      printf("Device extension(s): %s\n", device_extensions);
      free(device_extensions);
  }

  // Clean heap
  free(devices);
  free(platforms);

}
