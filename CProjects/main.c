#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

  int *a = malloc(sizeof(int));
  float *b = malloc(sizeof(float));
  double *c = malloc(sizeof(double));
  for (int i = 0; i < sizeof(int); ++i) {
    a[i] = i * i;
    printf("%d\n", a[i]);
  }
  for (int i = 0; i < sizeof(float); ++i) {
    b[i] = i * i;
    printf("%f\n", b[i]);
  }
  for (int i = 0; i < sizeof(double); ++i) {

    c[i] = i * i;
    printf("%lf\n", c[i]);
  }

  printf("Size of %lu\n", sizeof(int));
  printf("Size of %lu\n", sizeof(float));
  printf("Size of %lu\n", sizeof(double));
  printf("%p\n", a);
  free(a);
  a = NULL;
  printf("%p\n", a);
  return 0;
}
