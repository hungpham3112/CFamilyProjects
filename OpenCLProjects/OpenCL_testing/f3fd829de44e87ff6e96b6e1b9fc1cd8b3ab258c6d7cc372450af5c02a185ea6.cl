kernel void A(int i, int k, global float2* y, int j, int e) {
  int g = get_group_id(0) * 64;
  int s = get_group_id(1) * 32;
  int o = g + get_local_id(0);
  y += (s + o);
  j += (i + o);
  float2 z;
  z = (float2)(0.0, 0.0);
  for (int w = 0; w < 32; w++) 
           if (s + w < k && o < j && o < w) {
               y[w * e] = z;
           }
    //y = (float *)y;
    printf("y[7]: %f\n",((float*)(y))[7]);
}
