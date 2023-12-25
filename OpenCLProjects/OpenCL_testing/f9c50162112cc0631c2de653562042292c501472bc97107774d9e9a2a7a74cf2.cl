kernel void A(global const float4* o, global float* s) {
  int i = get_global_id(0);
  float4 n = o[i];
printf("n: %f\n",i, n.xyz);
  vstore3(n.xyz, i, s);
}
