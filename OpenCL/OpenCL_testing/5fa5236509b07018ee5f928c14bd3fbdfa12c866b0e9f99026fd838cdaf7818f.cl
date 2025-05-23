kernel void A(global const float* s, const unsigned int l, const unsigned int e, global float* f, const unsigned int b, const unsigned int g) {
        for (int i =0; i < 20; i++) {
     printf("s[%u]: %f", i, s[i]);
         }
  f[b + get_global_id(0) * g] = log2(s[l + get_global_id(0) * e]);
         printf("f[%u]: %f and log2(s[%u]): %f\n", b + get_global_id(0) * g,f[b + get_global_id(0) * g], l + get_global_id(0) * e), log2(s[l + get_global_id(0) * e]);
}
