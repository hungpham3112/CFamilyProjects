kernel void A(global const float16* r, unsigned int h, unsigned int x, unsigned int j, global const float* f, unsigned int z, unsigned int i, unsigned int w, global float16* b, unsigned int g, unsigned int l, unsigned int y) {
printf("INSIDE KERNEL: \n");
int oo = get_global_id(0);
         printf("r[%u]: %f\n", oo, r[oo]);
  unsigned int u = j / 16;
  for (unsigned int v = get_global_id(0); v < u; v += get_global_size(0)) {
        b[v * l + g] = r[v * x + h] + f[v * i + z];
printf("b[%u]: %f\n", v * l + g, b[v * l + g]);
//printf("r[%u]: %f, f[%u]: %f\n", v * x + h, r[v * x + h], v * i + z, f[v * i + z]);
//printf("r[%u] + f[%u] = %f\n",  v * x + h, v * i + z, r[v * x + h] + f[v * i + z]));
    }
}
