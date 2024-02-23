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


/////////////////////////////////////////////////////////////////////////

 
float_st operator/(const  float_st& a, const  float& b) 
{ 
  float_st res; 
 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();   
  res.x=a.x/b; 
  if (RANDOM) rnd_switch();   
  res.y=a.y/b; 
  rnd_switch();   
  res.z=a.z/b; 

  res.error=a.error;
  return res ; 
}
 
float_st operator/(const float& a, const float_st& b) 
{ 
  float_st res; 
  if(_cadna_div_tag) { 
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();	 
    if (b.accuracy==0) 
      instability(&_cadna_div_count); 
  } 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();   
  res.x=a/b.x; 
  if (RANDOM) rnd_switch();   
  res.y=a/b.y; 
  rnd_switch();   
  res.z=a/b.z;

  res.error=b.error;				  
  return res ; 
}


float_st operator/(const float_st& a, const float_st& b) 
{ 
  float_st res; 
  if(_cadna_div_tag) { 
    if (b.accuracy==DIGIT_NOT_COMPUTED) b.approx_digit();	 
    if (b.accuracy==0) 
      instability(&_cadna_div_count); 
  } 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();   
  res.x=a.x/b.x; 
  if (RANDOM) rnd_switch();   
  res.y=a.y/b.y; 
  rnd_switch();   
  res.z=a.z/b.z;

  res.error=a.error | b.error;
  return res ; 
}
 
 
 

/////////////////////////////////////////////////////////////////////////
