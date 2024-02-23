#include <math.h>
#include <stdio.h>

main()
{
  double  r,r1,x,y,z;
  x=6.83561e+5;
  y=6.83560e+5;
  z=1.00000000007;
  r = z - x;
  r1 = z - y;
  r = r + y;
  r1 = r1 + x;
  r1 = r1 - 2;
  r = r + r1;
  //      r = ((z-x)+y) + ((z-y)+x-2)
  printf("r= %+.14e\n", r);
}
