kernel void A(global float16* g, unsigned int c, unsigned int t, unsigned int e, float m) {
  unsigned int i = e / 16;
  for (unsigned int j = get_global_id(0); j < i; j += get_global_size(0)) {
    printf("flsdjalf\n");
    printf("g[%u]: %f , m: %f\n", j * t + c, g[j * t + c], m);
    g[j * t + c] *= m;
    }
}
