

#include <cadna.h>
#include <math.h>
#include <stdio.h>

main()
{
  int i, nmax=100;
  float_st y, x;

  cadna_init(-1);

  printf("-------------------------------------------------------------\n");
  printf("|  Computation of a root of a polynomial by Newton's method |\n");
  printf("|  with CADNA                                               |\n");
  printf("-------------------------------------------------------------\n");


  y = 0.5f;
  
  for(i = 1;i<=nmax;i++){
    x = y;
      y = x-(1.47f*pow(x,3)+1.19f*pow(x,2)-1.83f*x+0.45f)/
      (4.41f*pow(x,2)+2.38f*x-1.83f);
      if (fabs(x-y)<1.e-12) break;
  }
  printf("x(%3d) = %s\n",i-1,strp(x));
  printf("x(%3d) = %s\n",i,strp(y));

  cadna_end();
}




