kernel void A(global long* i, global long* v, int w) {
printf("hello\n");
    long c;
    for (int a = 0; a < w; a++) {
      if (i[a] != 0) {
        v[c] = i[a];
        c++;
      }
    }
}
