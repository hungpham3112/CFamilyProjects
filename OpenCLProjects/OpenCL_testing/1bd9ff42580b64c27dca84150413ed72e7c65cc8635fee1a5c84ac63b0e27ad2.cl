kernel void A(global const float* f, unsigned int v, unsigned int o,
                     unsigned int j, unsigned int g, unsigned int m,
                     unsigned int n, unsigned int h, unsigned int l,
                     global const float* y, unsigned int x, unsigned int k,
                     unsigned int s, global float* d, unsigned int e,
                     unsigned int w, unsigned int z) {

  printf("f[5] not in the loop: %u\n", f[5]);
  printf("y[5] not in the loop: %u\n", y[5]);
  printf("d[5] not in the loop: %u\n", d[5]);
  for (unsigned int c = get_global_id(0); c < m; c += get_global_size(0)) {
    printf("f[%u]: %f\n", c, f[c]);
    float r = 0;
      printf("y[5] Outer loop: %u\n", y[5]);
    for (unsigned int i = 0; i < n; ++i){
      printf("First sum: %u\n", (c * j + o) + (i * g + v) * h);
      printf("x: %u, k: %u, i: %u\n",x, k, i);
      printf("Second sum: %u ,y[%u]: %f\n",x + k * i, x +k *i,  y[x + k * i]);
      printf("y[5] Inner loop: %u\n", y[5]);
      printf("R inner: %u\n", r);
      r += f[(c * j + o) + (i * g + v) * h] * y[x + k * i];
  }
    d[c * w + e] = r;
    printf("r: %lu\n", r);
    printf("c * w + e: %u\n", c * w + e);
    printf("d[c * w + e]%u\n", d[c * w + e]);
  }
}
