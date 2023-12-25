kernel void A(global int* g, local int* c) {
  int t = c[0];
  barrier(0x02);
  g[0] = t;
  printf("fslajf");
}
