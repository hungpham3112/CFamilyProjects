#include <cadna.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  cadna_init(-1);
  float_st gpu = (float_st)0.163757e5;
  double_st cpu = (double_st)0.163757695403231e5;
  double_st diff = fabs(static_cast<double_st>(gpu) - cpu);
  std::cout << "GPU: " << gpu << std::endl;
  std::cout << "CPU: " << cpu << std::endl;
  printf("Stochastic: %s with %d significant figures\n", strp(diff),
         diff.nb_significant_digit());
  cadna_end();
  return 0;
}
