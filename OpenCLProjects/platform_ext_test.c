#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
  cl_platform_id *platforms;
  cl_uint num_platforms;
  cl_int i, err, platform_index = -1;
  char *ext_data;
  size_t ext_size;
  const char icd_ext[] = "cl_khr_icd";
  err = clGetPlatformIDs(1, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't find any platforms.");
    exit(1);
  }
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);
  for (i = 0; i < num_platforms; i++) {
    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL,
                            &ext_size);
    if (err < 0) {
      perror("Couldn't read extension data.");
      exit(1);
    }
    ext_data = (char *)malloc(ext_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, ext_size, ext_data,
                      NULL);
    printf("Platform %d supports extensions: %s\n", i, ext_data);
    if (strstr(ext_data, icd_ext) != NULL) {
      free(ext_data);
      platform_index = i;
      break;
    }
    free(ext_data);
  }
  if (platform_index > -1)
    printf("Platform %d supports the %s extension.\n", platform_index, icd_ext);
  else
    printf("No platforms support the %s extension.\n", icd_ext);
  free(platforms);
  return 0;
}
