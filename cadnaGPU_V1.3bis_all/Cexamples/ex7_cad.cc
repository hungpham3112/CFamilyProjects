#include "stdio.h"
#include <math.h>
#include "cadna.h"
main()
{
  cadna_init(-1);
  float_st r,r1,x,y,z;
  int i, nloop, ierr;
  printf("Enter the number of iterations: ");
  scanf("%d",&nloop);
  ierr = 0;
  for(i=0;i<nloop;i++){
    x=6.83561e+5f;
    y=6.83560e+5f;
    z=1.00000000007f;
    r = z - x;
    r1 = z - y;
    r = r + y;
    r1 = r1 + x;
    r1 = r1 - 2.f;
    r = r + r1;
    //      r = ((z-x)+y) + ((z-y)+x-2)
    if(r != 1.4e-10f) ierr = ierr + 1;
  }
  printf("r = %s   ierr = %d \n", strp(r) ,ierr);
  cadna_end();
}

