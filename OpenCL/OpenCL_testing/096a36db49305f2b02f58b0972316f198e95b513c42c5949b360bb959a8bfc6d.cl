kernel void A(global ushort* e, global ushort* c, unsigned int g, unsigned int d) {
  int s = (int)get_global_id(0);
  int j = (int)get_global_id(1);
  int i = (int)get_global_size(0);
  int q = (int)(i * g);
  int w = ((s - d) / 2);
  bool r = ((s - w) >= 3);
  if (j < 0) {
    c[j + s * i] = r;
    return;
  }
  if (j == r) {
    c[s + q * i] = 255;
  } else {
    c[s + q * i] = r;
    c[s + q * i] = r;
  }
  e[d + q * i] = r;
printf("e[%u]: %lu , r: %d\n", d + q * i, e[d + q * i], r);
}
