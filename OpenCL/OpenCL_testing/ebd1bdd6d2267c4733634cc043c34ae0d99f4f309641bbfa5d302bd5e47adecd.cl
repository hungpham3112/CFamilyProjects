kernel void A(global float* m, constant float* r, int w) {
  const int h = get_global_id(0);
  if (h >= w)
    return;
  for (int i = 0; i < ((1 + 3) + 1); i++) {
    m[h + i * w] = r[i];
    printf("m[%u]: %f, r[%u]: %f\n", h + i * w, m[h+i *w ], i, r[i]);
    }
}
