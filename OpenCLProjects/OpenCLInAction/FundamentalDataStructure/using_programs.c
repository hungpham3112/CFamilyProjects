#include <stdio.h>

#include <CL/cl.h>

int main() {
  // Get platform
  cl_platform_id *platforms;
  cl_int err, num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't get platform id\n");
    exit(1);
  } else {
    printf("Number of platforms: %u\n", num_platforms);
  }
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // Get device
  cl_device_id *devices;
  cl_uint num_devices;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (err < 0) {
    perror("Couldn't get platform id\n");
    exit(1);
  } else {
    printf("Number of device: %u\n", num_devices);
  }
  devices = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

  // Create 1 context to store 1 above device.
  cl_context context;
  cl_uint num_reference_count;
  context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
  if (err < 0) {
    perror("Couldn't create context\n");
    exit(1);
  } else {
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(num_reference_count), &num_reference_count, NULL);
    printf("Number of context reference: %u\n", num_reference_count);
  }

  // Create program in context
  cl_program program;
  FILE* program_handle;
  char* program_buffer;
  size_t program_size;
  program_handle = fopen("kernel.cl", "rb");
  if (!program_handle) {
      perror("Error opening file");
      exit(EXIT_FAILURE);
  }

  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);

  program_buffer = (char*) malloc(program_size + 1);
  if (!program_buffer) {
      perror("Error allocating memory");
      fclose(program_handle);
      exit(EXIT_FAILURE);
  }

  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  // Now split the program_buffer into lines
  const char* program_strings[1];
  program_strings[0] = program_buffer;

  program = clCreateProgramWithSource(context, 1, program_strings, &program_size, &err);

  // The second way to explicit return ERROR, avoid using magic number
  if (err != CL_SUCCESS) {
      fprintf(stderr, "Error creating program: %d\n", err);
      free(program_buffer);
      exit(EXIT_FAILURE);
  }

  // Extract program info
  // in here if * in front of name varible -> pointer -> asterisk must be added explicitly
  char *source_code, *kernel_name;
  size_t num_kernels;
  clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(num_kernels), &num_kernels, NULL);
  clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(kernel_name), kernel_name, NULL);
  printf("Source code:\n%lu\n", num_kernels);
  printf("Kernel name:\n%s\n", kernel_name);

  //Set flag and build program
  //const char options[] = "-cl-std=CL3.0 -cl-mad-enable -Werror -DCL_TARGET_OPENCL_VERSION=300";
  //clBuildProgram(program, 1, &devices[0], options, NULL, NULL);



}
