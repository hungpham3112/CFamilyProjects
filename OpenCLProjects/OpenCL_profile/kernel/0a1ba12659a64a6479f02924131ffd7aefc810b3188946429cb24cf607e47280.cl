// OpenCL kernel unoptimized

extern kernel void CopyBuffer(global unsigned int* src, global unsigned int* dst) {
  int id = (int)get_global_id(0);
  dst[id] = src[id];
}
