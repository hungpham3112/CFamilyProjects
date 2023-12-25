kernel void A(global unsigned int* w, unsigned int t, unsigned int v) {
  w[get_global_id(0) + t] = v;
printf("%u\n", get_global_id(0));
}
