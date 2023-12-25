kernel void A(global int* u) {
  int q = get_global_id(0) * 4;
  u[q++] = 0;
  u[q++] = 0;
  u[q++] = 2;
printf("execute\n");
}
