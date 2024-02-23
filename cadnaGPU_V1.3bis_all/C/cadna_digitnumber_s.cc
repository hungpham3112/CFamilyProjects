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




 
 
int float_st::nb_significant_digit() const 
{ 
  int b; 
  double aux=1.0; 
  double x0,x1,x2,xx; 
  unsigned int acc=0;
 
  b=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
 
  xx=(x+y+z); 
 
  if (xx==0.0){ 
    if ((x==y) &&(x==z) ) acc=7;  
  } 
  else { 
    xx=3.f/xx; 
    x0=x*xx-1.0f; 
    x1=y*xx-1.0f; 
    x2=z*xx-1.0f; 
    float yy=sqrtf(x0*x0+x1*x1+x2*x2)*1.7565495069643402f; 
     
    if (yy<=1.e-7f)  acc=7; 
    else { 
      yy= -log10f(yy); 
      if (yy>=0.f) acc=(int)(yy+0.5f); 
    } 
  } 
  // on recupere les anciennes erreurs et on 
  // ajoute la precisio
  accuracy = acc;

  if (b) rnd_moinf(); 
  else rnd_plinf(); 
 
  return accuracy;  
} 
 
 

int float_st::approx_digit() const 
{
  int b; 
  float aux=1.0; 
  float x0,x1,x2,xx; 
  unsigned int acc=0;
  int res;
  
  b=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
    
  accuracy=NOT_NUMERICAL_NOISE;
  xx=(x+y+z);
  
  if (xx==0.0f) {
    if ((x != y ) || (x != z)) accuracy=0;
  }
  else {
    xx=3.f/xx;
    x0=x*xx-1.f;
    x1=y*xx-1.f;
    x2=z*xx-1.f;
    if (((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f)
      accuracy=0;
  }
  
  if (b) rnd_moinf(); 
  else rnd_plinf(); 
 
  return accuracy;  
}
  
  








  







