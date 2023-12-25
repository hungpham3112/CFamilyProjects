kernel void A(global const float* f, unsigned int v, unsigned int o,
                     unsigned int j, unsigned int g, unsigned int m,
                     unsigned int n, unsigned int h, unsigned int l,
                     global const float* y, unsigned int x, unsigned int k,
                     unsigned int s, global float* d, unsigned int e,
                     unsigned int w, unsigned int z) {
  for (unsigned int c = get_global_id(0); c < m; c += get_global_size(0)) {
    float r = 0;
    for (unsigned int i = 0; i < n; ++i){
      r += f[(c * j + o) + (i * g + v) * h] * y[x + k * i];
      }
    d[c * w + e] = r;
    printf("d[%u]: %f", c* w + e,d[c*w +e]);
  }
}
