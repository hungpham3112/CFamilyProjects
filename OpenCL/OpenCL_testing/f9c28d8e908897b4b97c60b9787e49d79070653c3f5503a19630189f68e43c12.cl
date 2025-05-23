kernel void A(global const int* v, global int* g) {
  int b = get_global_id(0);
  g[b] = (v[b - 1] + v[b] + v[b + 1]) / 3;
printf("here: g[%u]: %d, \n", b, g[b]);
}
