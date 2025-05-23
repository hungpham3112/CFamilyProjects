kernel void A(const int w, global const float* a, const int e, global float* v, const int b) {
  const int o = get_global_id(0);
 const int g = get_global_id(1);
 v[g * w + o + b] = a[g * e + o + b];
printf("a: %u\n", g * e + o + b);
printf("v[%u]: %f\n", g * w + o + b, v[g * w + o + b]);
}
