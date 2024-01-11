// OpenCL kernel optimized 1

extern kernel void CopyBuffer(global unsigned int* src, global unsigned int* dst) {
  int id = get_global_id(0);
  int coarseningFactor = 4; // Example coarsening factor, can be adjusted based on performance testing
  int coarsenedId = id * coarseningFactor;

  // Perform coarsened copy using the coarsening factor
  for (int i = 0; i < coarseningFactor; i++) {
    dst[coarsenedId + i] = src[coarsenedId + i];
  }
}
