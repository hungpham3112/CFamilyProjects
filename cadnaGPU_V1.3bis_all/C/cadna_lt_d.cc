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
#include "cadna.h"
#include "cadna_private.h"


 
int operator<(const double& a, const double_st& b) 
{ 
  double_st res; 
   
  if (RANDOM) rnd_switch(); res.x=a-b.x; 
  if (RANDOM) rnd_switch(); res.y=a-b.y; 
  rnd_switch();   res.z=a-b.z; 
   
  int r=res.approx_digit()==0;  
  if (_cadna_branching_tag && r) instability(&_cadna_branching_count); 
  return !r && ( ( a+a+a ) < (  b.x + b.y + b.z )); 
}
  
 
int operator<(const double_st& a, const  double& b) 
{ 
  double_st res; 
   
  if (RANDOM) rnd_switch(); res.x=a.x-b; 
  if (RANDOM) rnd_switch(); res.y=a.y-b; 
  rnd_switch(); res.z=a.z-b; 
   
  int r=res.approx_digit()==0;  
  if (_cadna_branching_tag && r) instability(&_cadna_branching_count); 
  return !r && (  (a.x + a.y + a.z ) < (b+b+b )); 
}
  
 
int operator<(const double_st& a, const double_st& b) 
{ 
  double_st res; 
   
  if (RANDOM) rnd_switch(); res.x=a.x-b.x; 
  if (RANDOM) rnd_switch(); res.y=a.y-b.y; 
  rnd_switch(); res.z=a.z-b.z; 
   
  int r=res.approx_digit()==0; 
  if (_cadna_branching_tag && r) instability(&_cadna_branching_count); 
  return !r && ( ( a.x + a.y + a.z ) <	( b.x + b.y + b.z )); 
}
 
 

