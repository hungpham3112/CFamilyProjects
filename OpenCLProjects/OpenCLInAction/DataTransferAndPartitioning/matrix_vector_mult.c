#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <time.h>
#define ROWS 4
#define COLS 4
#define KERNEL_NAME "A"

int bound_var(

int randInt() {
    return rand();
}

void randVector(int n, int vector[]) {
    for (int i = 0; i < n; ++i) {
        vector[i] = rand();
    }
}

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

void initializeMatrixAndVector(float matrix[ROWS * COLS], float vector[COLS]) {
    srand(time(NULL));

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            matrix[i * COLS + j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Initialize the vector with random values
    for (int i = 0; i < COLS; ++i) {
        vector[i] = (float)rand() / (float)RAND_MAX;
    }
}

void printMatrixAndVector(float matrix[ROWS * COLS], float vector[COLS]) {
    // Print the matrix
    printf("Matrix:\n");
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            printf("%f\t", matrix[i * COLS + j]);
        }
        printf("\n");
    }

    // Print the vector
    printf("\nVector:\n");
    for (int i = 0; i < COLS; ++i) {
        printf("%f\n", vector[i]);
    }
}

int main() {
  cl_platform_id platform;
  cl_int err;
  cl_uint num_platform = 1;
  err = clGetPlatformIDs(num_platform, &platform, NULL);
  if (err != CL_SUCCESS) {
    perror("Couldn't get platform");
    exit(EXIT_FAILURE);
  } 

  // In here we imply only 1 device;
  cl_device_id device;
  cl_uint num_device = 1;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_device, &device, NULL);
  if (err != CL_SUCCESS) {
    perror("Couldn't get device");
    exit(EXIT_FAILURE);
  } 

  // Create context
  cl_context context;
  context = clCreateContext(NULL, num_device, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    perror("Couldn't create context");
    exit(EXIT_FAILURE);
  } 

  // Gen matrix and vector;
  float mat[ROWS * COLS], vec[COLS], result[COLS];
  initializeMatrixAndVector(mat, vec);


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

  //Set flag and build program
  const char options[] = "-cl-std=CL3.0 -cl-mad-enable -Werror -DCL_TARGET_OPENCL_VERSION=300";
  err = clBuildProgram(program, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't build program. OpenCL error code: %d\n", err);
    exit(EXIT_FAILURE);
  }

  // Create kernel
  cl_kernel kernel;
  // This will create with specific kernel name
  kernel = clCreateKernel(program, KERNEL_NAME , &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }

  // Create memory objects for kernels to operate on
  cl_command_queue queue;
  queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

  cl_mem mat_buff, vec_buff, res_buff;
  mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*16, mat, &err);
  vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*4, vec, &err);
  res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*4, NULL, &err);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);

  const size_t work_units_per_kernel = 4;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel, NULL, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(res_buff), result, 0, NULL, NULL);
  err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(result), result, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't create kernel. OpenCL error code: %d\n", err);
    exit(1);
  }
  clFinish(queue);


  printf("\nResult: \n");
  for (int i = 0; i < COLS; ++i) {
    printf("%f\n", result[i]);
  }

  clReleaseMemObject(mat_buff);
  clReleaseMemObject(vec_buff);
  clReleaseMemObject(res_buff);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
  clReleaseDevice(device);
  free(platform);




}
