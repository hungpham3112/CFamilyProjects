#define CL_TARGET_OPENCL_VERSION 300
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
  cl_uint num_reference_count;
  context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
  if (err < 0) {
    perror("Couldn't create context\n");
    exit(EXIT_FAILURE);
  }

  // Create program in context
  cl_program program;
  const char* kernelSource = readKernelSourceFromFile("kernel.cl");
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

  //Set flag and build program
  const char options[] = "-cl-std=CL3.0 -cl-mad-enable -Werror -DCL_TARGET_OPENCL_VERSION=300";
  err = clBuildProgram(program, 1, &devices[0], options, NULL, NULL);
  if (err != CL_SUCCESS) {
    perror("Couldn't build program");
    exit(EXIT_FAILURE);
  }

  cl_kernel kernel1;
  // This will create with specific kernel name
  kernel1 = clCreateKernel(program, "matvec_mult", &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // get kernel info
  char *kernel_name;
  size_t kernel_name_size;
  clGetKernelInfo(kernel1, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
  kernel_name = (char *)malloc(kernel_name_size);
  clGetKernelInfo(kernel1, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
  printf("Kernel1 name: %s\n", kernel_name);
  free(kernel_name);

  // Create buffer as kernel container
  cl_mem input_buffer, output_buffer;
  float vec[32];
  input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vec), vec, &err);
  output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(vec), NULL, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create buffer");
    exit(1);
  }
  clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input_buffer);
  clSetKernelArg(kernel1, 1, sizeof(cl_mem), &input_buffer);

  // Get info about buffer objects
  size_t input_size, output_size;
  err = clGetMemObjectInfo(input_buffer, CL_MEM_SIZE, 0, NULL, &input_size);
  err = clGetMemObjectInfo(output_buffer, CL_MEM_SIZE, 0, NULL, &output_size);
  printf("Input size: %lu Byte(s)\n", input_size);
  printf("Output size: %lu Byte(s)\n", input_size);

  clReleaseMemObject(output_buffer);
  clReleaseMemObject(input_buffer);

  // Free heap
  free(devices);
  free(platforms);

}
