kernel void A(global float* m, global float* l, int s, int c, int o) {
  int w = get_global_id(0);
  int h = w / (c * o);
  if (h < s) {
    int x = w % (s * c);
    m[w * s + h] = l[x];
  }
}
