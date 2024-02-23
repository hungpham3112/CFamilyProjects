#include <cadna.h>
#include <iostream>
#include <stdio.h>

#define SEED 100

int main() {
  cadna_init(-1);
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("|  with CADNA                            |\n");
  printf("------------------------------------------\n");

  float_st x = 77617.;
  float_st y = 33096.;
  float_st res;

  res = 333.75f * y * y * y * y * y * y +
        x * x *
            (11.f * x * x * y * y - y * y * y * y * y * y -
             121.f * y * y * y * y - 2.0f) +
        5.5f * y * y * y * y * y * y * y * y + x / (2.f * y);

  printf("res=%s %d\n", strp(res), res.nb_significant_digit());
  float a, b, c;
  a = res.getx();
  b = res.gety();
  c = res.getz();
  // printf("res=%f %d\n", (float)res, res.nb_significant_digit());
  std::cout << "x = " << a << " y = " << b << " z = "
            << " " << c << std::endl;
  cadna_end();
  return 0;
}
