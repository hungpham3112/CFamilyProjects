#include <stdio.h>

#include <CL/cl.h>

const char* readKernelSourceFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        return NULL; // File not found or couldn't be opened
    }

    // Find the size of the file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate memory for the file content plus a null-terminating character
    char* source_code = (char*)malloc(file_size + 1);
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
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
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
  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
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
  const char* kernelSource = readKernelSourceFromFile("kernel.cl");

  if (kernelSource == NULL) {
      printf("Failed to read the kernel source from the file.\n");
      exit(EXIT_FAILURE);
  }

  program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);

  // The second way to explicit return ERROR, avoid using magic number
  if (err != CL_SUCCESS) {
      fprintf(stderr, "Error creating program: %d\n", err);
      exit(EXIT_FAILURE);
  }

  // Extract program info
  // in here if * in front of name varible -> pointer -> asterisk must be added explicitly
  char *kernel_name, *source_code;
  size_t num_kernels, kernel_name_size, source_code_size;
  cl_uint program_reference_count;
  
  clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &source_code_size);
  source_code = (char *)malloc(source_code_size);
  clGetProgramInfo(program, CL_PROGRAM_SOURCE, source_code_size, source_code, NULL);
  printf("Kernel source code:\n%s\n", source_code);

  clGetProgramInfo(program, CL_PROGRAM_REFERENCE_COUNT, sizeof(program_reference_count), &program_reference_count, NULL);
  printf("Program reference count: %u\n", program_reference_count);
  free(source_code);

  //Set flag and build program
  const char options[] = "-cl-std=CL3.0 -cl-mad-enable -Werror -DCL_TARGET_OPENCL_VERSION=300";
  err = clBuildProgram(program, 1, &devices[0], options, NULL, NULL);
  if (err != CL_SUCCESS) {
    perror("Couldn't build program");
    exit(EXIT_FAILURE);
  } else {
    // Extract build info
    // Build status
    cl_build_status build_status;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL);
    printf("Build Status: %d\n", build_status);

    // Build options
    char *build_options;
    size_t build_options_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_OPTIONS, 0, build_options, &build_options_size);
    build_options = (char *) malloc(build_options_size);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_OPTIONS, build_options_size , build_options, NULL);
    printf("Build options size: %ld\n", build_options_size);
    printf("Build options: %s\n", build_options);
    free(build_options);

    // Build info
    char *build_log;
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = (char*) calloc(log_size+1, sizeof(char));
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    printf("Log: %s\n", build_log);
    free(build_log);
  }
  
  // Free heap
  free(devices);
  free(platforms);

}
