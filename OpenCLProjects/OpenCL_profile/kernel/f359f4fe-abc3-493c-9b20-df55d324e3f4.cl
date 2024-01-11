// OpenCL kernel optimized 2

extern kernel void CopyBuffer(global unsigned int* src, global unsigned int* dst) {
  int id = (int)get_global_id(0);
  dst[id] = (id < 512) ? src[id] : 0;
}
