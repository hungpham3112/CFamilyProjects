kernel void A(global int* s, long n, int k) {
  global int* g = s + get_global_id(0) * k;
  int j, p;
  int l;
  j = g[0];
  p = g[1];
  l = j;
  while (n-- > 1) {
    j = j + l;
    p = p + l;
    l = l + l;
  }
  g[0] = j;
  g[1] = p;
}
