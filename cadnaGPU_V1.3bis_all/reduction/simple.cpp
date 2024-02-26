#include <cadna.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  cadna_init(-1);
  float x = 0.5f;
  float_st y = 0.5f;
  printf("Deteministic: %f\n", x);
  printf("Stochastic: %s with %d significant figures\n", strp(y),
         y.nb_significant_digit());
  cadna_end();
  return 0;
}
