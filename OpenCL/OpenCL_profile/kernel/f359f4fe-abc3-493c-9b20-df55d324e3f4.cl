// OpenCL kernel optimized 2

kernel void CopyBufferOpt2(global unsigned int* src, global unsigned int* dst) {
  int id = (int)get_global_id(0);
  dst[id] = (id < 512) ? src[id] : 0;
}

