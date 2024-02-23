#include <stdio.h>
#include <math.h>
#include <cadna.h>

main()
{      

  cadna_init(-1);

  float_st a = 0.3;
  float_st b = -2.1;
  float_st c = 3.675;
  float_st d, x1,x2;

  printf("----------------------------------\n");
  printf("|  Second order equation         |\n");
  printf("|  with CADNA                    |\n");
  printf("----------------------------------\n");

  //
  //      CASE: A = 0
  //
  if (a==0.f)
    if (b==0.f) {
      if (c==0.f) printf("Every complex value is solution.\n");
      else printf("There is no solution.\n");
    }
    else {
      x1 = - c/b;
      printf("The equation is degenerated.\n");
      printf("There is one real solution %s\n",strp(x1));
    }
  else {
    //
    //     CASE: A /= 0
    //
    b = b/a;
    c = c/a;
    d = b*b - 4.0f*c;
    printf("d = %s\n",strp(d));
    d.display();
    //
    //   DISCRIMINANT = 0
    //
    if (d==0.f) {
      x1 = -b*0.5f;
      printf("Discriminant is zero.\n");
      printf("The double solution is %s\n",strp(x1));
    }
    else {
      //
      //      DISCRIMINANT > 0
      //
      if (d>0.f) {
          x1 = ( - b - sqrtf(d))*0.5f;
          x2 = ( - b + sqrtf(d))*0.5f;
	  printf("There are two real solutions.\n");
	  printf("x1 = %s x2 = %s\n",strp(x1),strp(x2));
      }
      else {
	//
	//      DISCRIMINANT < 0
	//
	x1 = - b*0.5f;
	x2 = sqrtf(-d)*0.5f;
	printf("There are two complex solutions.\n");
	printf("z1 = %s  +  i * %s\n",strp(x1),strp(x2));
	printf("z2 = %s  +  i * %s\n",strp(x1), strp(-x2));
      }
    }
  }

  cadna_end();
}








