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
#include <cmath>

#include "cadna.h"
#include "cadna_private.h"


// ----------------------------------------


// save the rounding mode to compute the number of significant digits
// always with the same rounding mode


//****f* cadna_digitnumber/digitnumber
//    NAME
//      digitnumber
//    SYNOPSIS
//      res = digitnumber(x) 
//      res = x.digitnumber() 
//    FUNCTION
//      The digitnumber() function returns the number of significant
//      digits of a stochastic x
//      
//      
//    INPUTS
//      x           - a stochastic number
//    RESULT
//      res         - an integer value
//             
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
//    SEE ALSO
//      
//      
//  /




 
 
int double_st::nb_significant_digit() const 
{ 
  int b; 
  double aux=1.0; 
  double x0,x1,x2,xx; 
  unsigned int acc=0;
 
  b=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
 
  xx=x+y+z; 
 
  if (xx==0.0){ 
    if ((x==y) &&(x==z) ) acc=15;  
  } 
  else { 
    xx=3./xx; 
    x0=x*xx-1.; 
    x1=y*xx-1.; 
    x2=z*xx-1.; 
    double yy=sqrt(x0*x0+x1*x1+x2*x2)*1.7565495069643402; 
     
    if (yy<=1.e-15)  acc=15; 
    else { 
      yy= -log10(yy); 
      if (yy>=0.) acc=(int)(yy+0.5); 
    } 
  } 

  accuracy = acc;

  if (b) rnd_moinf(); 
  else rnd_plinf(); 
 
  return accuracy;  
} 
 
 
int double_st::approx_digit() const 
{
  int b; 
  double aux=1.0; 
  double x0,x1,x2,xx; 
  unsigned int acc=0;
  int res;
  
  b=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
    
  accuracy=NOT_NUMERICAL_NOISE;
  xx=(x+y+z);
  
  if (xx==(double)0.0) {
    if ((x != y ) || (x != z)) accuracy=0;
  }
  else {
    xx=(double)3./xx;
    x0=x*xx-(double)1;
    x1=y*xx-(double)1;
    x2=z*xx-(double)1;
    if (((x0*x0+x1*x1+x2*x2)*(double)3.0854661704166664) > (double)0.1)
      accuracy=0;
  }
  
  if (b) rnd_moinf(); 
  else rnd_plinf(); 
 
  return accuracy;  
}
  



  







