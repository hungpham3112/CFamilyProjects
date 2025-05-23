kernel void A(const global float* f, global float* b) {
    for (int i =0; i < 20; i++) {
     printf("f[%u]: %f\n", i, f[i]);
 }
  unsigned int h = get_global_id(0);
  vstore3(normalize(vload3(h, f)), h, b);
}
