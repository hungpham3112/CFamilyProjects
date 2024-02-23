// Copyright 2019  J.-M. Chesneaux, F. Jezequel, J.-L. Lamotte

// This file is part of CADNA.

// CADNA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// CADNA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with CADNA.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////

#include <string.h>
#include <stdlib.h>
#include "cadna.h"
#include "cadna_private.h"

#include <cmath>
#include <iostream>
#include <cstdio>

#define MAXCHAINE 256
static char chstr[MAXCHAINE][64];
static int numstr=0; 

using namespace std;

//////////////////////////////////////////////////////////
//****m* cadna_str/display
//    NAME
//      display
//
//    SYNOPSIS
//      display()
//      display(char *)
//
//    FUNCTION
//      The display method prints the triplet associated with 
//      a stochastic variable.
//    INPUTS
//   
//    RESULT
//      void
//    SEE ALSO
//      str(3)
//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /



void double_st::display() const 
{ 
  unsigned int inst;
  char t[40]="";

  
  if (error & CADNA_DIV       ) strcat(t,"DIV-");
  if (error & CADNA_MUL       ) strcat(t,"MUL");
  if (error & CADNA_POWER     ) strcat(t,"POW-");
  if (error & CADNA_MATH      ) strcat(t,"MATH");
  if (error & CADNA_BRANCHING ) strcat(t,"BRANC");
  if (error & CADNA_INTRINSIC ) strcat(t,"INTRIS");
  if (error & CADNA_CANCEL    ) strcat(t,"CANCEL");

  printf(" %+.8e -- %+.8e -- %+.8e %s",x,y,z,t);
}

//////////////////////////////////////////////////////////
void double_st::display(const char *s) const 
{ 
  char t[40]="";
   
  if (error & CADNA_DIV       ) strcat(t,"DIV-");
  if (error & CADNA_MUL       ) strcat(t,"MUL");
  if (error & CADNA_POWER     ) strcat(t,"POW-");
  if (error & CADNA_MATH      ) strcat(t,"MATH");
  if (error & CADNA_BRANCHING ) strcat(t,"BRANC");
  if (error & CADNA_INTRINSIC ) strcat(t,"INTRIS");
  if (error & CADNA_CANCEL    ) strcat(t,"CANCEL");

  printf(" %s %+.8e -- %+.8e -- %+.8e %s",s,x,y,z,t);
}

//////////////////////////////////////////////////////////


ostream& operator <<(ostream& s, const double_st &a) 
{
  char ch[64];
  return s << a.str(ch);
}


istream& operator >>(istream& s, double_st& a)
{
  double d;
  s >> d;
  a.x = d;
  a.y = d;
  a.z = d;
  a.accuracy=DIGIT_NOT_COMPUTED;
  return s;
}


char* double_st::str(char *s) const 
{
  double aux=1.0;
  int b, nn, nres, naux, acc;
  double fract_res, res;
  int tmp;
  char *t;

  // save the rounding mode
  // to compute the number of significant digits always
  // with the same rounding mode
  b=(aux+1.e-20)==1.0 ?  1 : 0;
  rnd_arr();

  //  printf("str  acc %2x  error %2x \n", accuracy, error);
  if (accuracy==DIGIT_NOT_COMPUTED || accuracy==NOT_NUMERICAL_NOISE) {
    this->nb_significant_digit();

    //  cout << "#################" << endl;
    //  this-> display();
    //  cout << endl << "accuracy="<< (int)accuracy<< endl;
    //  cout << "#################" << endl;
  }
  

  //        12345678901234567890123
  strcpy(s,"                       ");
  if (accuracy==0) {
    strncpy(s," @.0",4);
  }
  else {
    acc=(accuracy<15) ? accuracy : 15;
    res=( (x) +(y) +(z) )/(double)(3);
    if (res<0.0) strncpy(s,"-0.",3);
    else strncpy(s," 0.",3);
    res=fabs(res);
    
    if (res==0.0) {
      nn=0;
      fract_res=0.0;
    }
    else{
      if (res>=1.0) nn=(int)log10(res)+1;
      else  nn= (int)log10(res);
      fract_res=res*pow(10.,-nn);
      if (fract_res<0.1) fract_res=0.1;
      if (fract_res>=1.0) {
	fract_res=0.1;
	nn=nn+1;
      }
    }
    naux=acc+3;
    t=s+3;
    for(int i=4;i<naux;i++){
      nres=(int)(fract_res*10.);
      *t++=48+nres;
      fract_res=10.0*fract_res-nres;
    }
    tmp = (int)(fract_res*10.0);
    nres = tmp < 9 ? tmp : 9 ;
    *t++=48+nres;
    *t++='E';
    if(nn<0) *t++='-';
    else *t++='+';
    nn=abs(nn);
    *t++=48+(nn/100);
    nn=nn%100;        
    *t++=48+(nn/10);
    *t++=48+(nn%10);
    *t='\0';
  }

  if (error & CADNA_DIV       ) strcat(s,"DIV-");
  if (error & CADNA_MUL       ) strcat(s,"MUL");
  if (error & CADNA_POWER     ) strcat(s,"POW-");
  if (error & CADNA_MATH      ) strcat(s,"MATH");
  if (error & CADNA_BRANCHING ) strcat(s,"BRANC");
  if (error & CADNA_INTRINSIC ) strcat(s,"INTRIS");
  if (error & CADNA_CANCEL    ) strcat(s,"CANCEL");


  // to restore the rounding mode
  if (b) rnd_moinf();
  else rnd_plinf();

  //  printf("str: %23.15e %23.15e %23.15e  acc=%d  error=%d \n", x,y,z, accuracy, error);

  return(s);
}




char* str(char *s, double_st& a)
{
  return(a.str(s));
}

//////////////////////////////////////////////////////////



//****f* cadna_str/strp
//    NAME
//     strp
//
//    SYNOPSIS
//      char* strp(double_st&)
//      char* strp(double_st&)
//    FUNCTION
//      The output string contains the scientific notation of the 
//      stochastic argument; only the exact significant digits appear 
//      in the string. The strp function must be used only with
//      the family of printf functions. The only restriction is that
//      it is not possible to have more than 256 calls to the strp
//      function in one call to the printf function.
//    INPUTS
//      The strp function has a stochastic argument.  
//    RESULT
//      It returns a string.
//    SEE ALSO
//      str(3)
//*****
//   You can use this space for remarks that should not be included
//   in the documentation.
//    EXAMPLE
//      
//  
//    NOTES
//  
//  
//    BUGS
//  
//  
//      
//      
//  /


char* strp(const double_st& a)
{
  char *s;
  
  s=chstr[numstr];
  if ((++numstr) == MAXCHAINE)  numstr=0; 
  return a.str(s);
}

//////////////////////////////////////////////////////////



  

//////////////////////////////////////////:
