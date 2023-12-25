kernel void A(int g, global float* b, int k, int a, int p, int t, global float* z) {
  int x = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * get_local_size(0) + get_local_id(0);
  if (x >= g)
    return;
  int m = x % a;
  x = x / a;
  int f = x % k;
  x = x / k;
  int o = x;
  int r = m % a;
  int e = f / a;
  int d = r;
  int i = r * k + o;
  int q = r * g + d;
  int s = r % g;
  int h = q / g;
  int v = q + p * k;
  int j = h + s;
  z[i + p * a] = v - b[j + o * a];
  float c = exp(-(1.f / t / t));
  z[i + o * a] = c * (1 - f / t / (1 - t + c));
printf()
}
