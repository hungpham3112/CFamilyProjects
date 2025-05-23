#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h> 

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {   
  // Identify platform id
  cl_platform_id *platforms;
  cl_uint err = 0, num_platforms;
  // The first parameter in here is num_entries, it's only meaningful after determining
  // the number of platform first, so it will be set to 0.
  // platforms at least = 0 because logic, you should add at least 0 entries to platforms
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err < 0) {
    perror("Couldn't find any platforms");
    exit(1);
  } else {
    printf("Found %d platform(s)\n", num_platforms);
  }
  // Because you can run script in many type of platforms, therefore, good practice
  // is allocate memory based on previous # platform, if not script will return 
  // segmentfault.
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
  // Now your platforms is determined, you wish to get ID of all platform so pass
  // num_platforms to the first parameter of clGetPlatformIDs as num_entries.
  clGetPlatformIDs(num_platforms, platforms, NULL);

  // Identify multi platform info
  char platform_name[100], platform_vendor[100], platform_profile[100], platform_version[100], platform_extensions[2048];
  for (int i = 0; i < num_platforms; i++) {
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, sizeof(platform_profile), platform_profile, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, sizeof(platform_extensions), platform_extensions, NULL);
    printf("Platform name: %s\n", platform_name);
    printf("Platform vendor: %s\n", platform_vendor);
    printf("Platform version: %s\n", platform_version);
    printf("Platform profile: %s\n", platform_profile);
    printf("Platform extension: %s\n", platform_extensions);
  }
  // Free heap
  free(platforms);
}
