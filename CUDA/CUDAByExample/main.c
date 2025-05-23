#include <stdio.h>

void func(void **A) { *A = 2; }

int main(int argc, char *argv[]) {
  int *A = 1;
  func((void **)&A);
  printf("%d\n", A);
  return 0;
}
