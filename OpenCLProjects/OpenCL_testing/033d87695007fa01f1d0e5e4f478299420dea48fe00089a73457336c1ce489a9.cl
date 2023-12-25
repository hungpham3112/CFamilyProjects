kernel void A(global double* h, const int w) {
  int b = get_global_id(0) + w;
  h[b] = erf(h[b]);
printf("Inside kernel: h[%u]: %lf, erf(h[%u]): %lf\n", b, h[b], b, erf(h[b]));
}
