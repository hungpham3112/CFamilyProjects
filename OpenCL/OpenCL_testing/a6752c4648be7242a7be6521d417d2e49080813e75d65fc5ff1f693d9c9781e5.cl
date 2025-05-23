kernel void A(int u, int c, global double* f, int g, int h) {
  int q = get_group_id(0) * 64;
  int k = get_group_id(1) * 32;
  int p = q + get_local_id(0);
  f += g + p + k * h;
  double x;
  x = 0.0;
  for (int n = 0; n < 32; n++)
    if (k + n < c && p < u)
      f[n * h] = x;
}
