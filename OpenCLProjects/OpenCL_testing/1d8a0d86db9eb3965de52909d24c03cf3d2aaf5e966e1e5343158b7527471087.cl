kernel void A(int c, int o, global float2* l, int q, int k) {
  int m = get_group_id(0) * 64;
  int d = get_group_id(1) * 32;
  int g = m + get_local_id(0);
  l += q + g + d * k;
  float2 w;
  w = (float2)(0.0, 0.0);
  for (int v = 0; v < 32; v++)
    if (d + v < o && g < c) {
      l[v * k] = w;
    }
}
