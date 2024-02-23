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
#include <cmath>

#include "cadna.h"
#include "cadna_private.h"

 
 
int double_st::computedzero() const 
{ 
  int b,res; 
  double aux=1.0; 
  double x0,x1,x2,xx; 
 
  b=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
 
  xx=(x+y+z); 
 
  if (xx==0.0) res=1; 
  else { 
    xx=3./xx; 
    x0=x*xx-1.; 
    x1=y*xx-1; 
    x2=z*xx-1; 
    res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664) > 0.1; 
  } 
  if (b) rnd_moinf(); 
  else rnd_plinf(); 
 
  return res;  
} 
 
 




  







