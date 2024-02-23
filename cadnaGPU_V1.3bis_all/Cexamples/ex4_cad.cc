#include <cadna.h>
#include <math.h>
#include <stdio.h>

using  namespace std;

main()
{
  int i;
  float_st a,b,c;

  cadna_init(-1);

  printf("-------------------------------------\n");
  printf("| A second order recurrent sequence |\n");
  printf("| without CADNA                     |\n");
  printf("-------------------------------------\n");

  a = 5.5f;
  b = 61.f/11.f;
  for(i=3;i<=30;i++){
    c = b;
    b = 111.f - 1130.f/b + 3000.f/(a*b);
    printf("%d : %s\n",i,strp(b)); 
    a = c;
  }
  printf("The exact limit is 6.\n");

  cadna_end();
}






