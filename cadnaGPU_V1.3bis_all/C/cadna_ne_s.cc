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


 
int operator!=(const float& a, const float_st& b) 
{ 
  float_st res; 
  if (RANDOM) rnd_switch();  
  res.x=a-b.x; 
  if (RANDOM) rnd_switch();  
  res.y=a-b.y; 
  rnd_switch();  
  res.z=a-b.z; 
  return ~res.computedzero(); 
}
 
 
int operator!=(const float_st& a, const float& b) 
{ 
  float_st res; 
  if (RANDOM) rnd_switch();  
  res.x=a.x-b; 
  if (RANDOM) rnd_switch();  
  res.y=a.y-b; 
  rnd_switch();  
  res.z=a.z-b; 
  return !(res.computedzero());			  
}
 
 
int operator!=(const float_st& a, const float_st& b) 
{ 
  float_st res; 
  if (RANDOM) rnd_switch(); 
  res.x=a.x-b.x; 
  if (RANDOM) rnd_switch(); 
  res.y=a.y-b.y; 
  rnd_switch(); 
  res.z=a.z-b.z; 
  return !res.computedzero(); 
}
 
 



 








