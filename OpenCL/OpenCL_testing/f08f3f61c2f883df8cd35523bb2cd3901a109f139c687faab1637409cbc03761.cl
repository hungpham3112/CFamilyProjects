kernel void A(global const float* d, const unsigned int q, const unsigned int e, global float* m, const unsigned int b, const unsigned int c) {
  m[b + get_global_id(0) * c] = log2(d[q + get_global_id(0) * e]);
printf("m[%u]: %f, log2(d[%u]): %f\n", b + get_global_id(0) * c, m[b + get_global_id(0) * c], q + get_global_id(0) * e, log2(d[q + get_global_id(0) * e]) );
}
