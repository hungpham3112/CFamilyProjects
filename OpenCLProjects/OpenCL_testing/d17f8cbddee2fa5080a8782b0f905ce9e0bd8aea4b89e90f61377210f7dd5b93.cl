kernel void A(global const float* v, global float* y) {
  int w = get_global_id(0);
  y[w * 2] = rsqrt(v[w * 2 + 1]);
printf("y[%u]: %f, v[%u]: %f\n", w*2, y[w*2], w * 2 + 1, v[w*2+1]);
}
