kernel void A(int p, int w, float b, global unsigned int* y, global unsigned int* c) {
  unsigned int k[256];
  unsigned int j[512];
  for (int o = 0; o < w; o++) {
    k[o] = 0;
  }
  for (int g = 0; g < p / 512; g++) {
    for (int s = 0; s < 512; s++)
      j[s] = y[g * 512 + s];
    for (int o = 0; o < 512; o++) {
      int s = (j[o] * w) >> 12;
      k[s]++;
    }
  }
  for (int o = 0; o < w; o++) {
    c[o] += k[o];
  }
}

