#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10

// Function to read the content of a file
char* readKernelSource(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* content = (char*)malloc(length + 1);
    fread(content, 1, length, file);
    content[length] = '\0';

    fclose(file);
    return content;
}

int main() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    int v[N];
    for (int i = 0; i < N; ++i) {
        v[i] = i;
      printf("v[%d] = %d\n",i, v[i]);
    }
    printf("v[-1] = %d\n", v[-1]);

    int g[N];

    cl_mem vBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, v, NULL);
    cl_mem gBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N, NULL, NULL);

    // Load kernel from external file
    const char* kernelSource = readKernelSource("test_kernel.cl");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "A", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gBuffer);

    size_t globalWorkSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, gBuffer, CL_TRUE, 0, sizeof(int) * N, g, 0, NULL, NULL);

    // Print the result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", g[i]);
    }
    printf("\n");

    // Cleanup
    free((void*)kernelSource);
    clReleaseMemObject(vBuffer);
    clReleaseMemObject(gBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

