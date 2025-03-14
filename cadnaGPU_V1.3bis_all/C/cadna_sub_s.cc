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
//****m* cadna_sub/operator-
//    NAME
//      operator-
//    SYNOPSIS
//      res = a - b 
//    FUNCTION
//      Define all the functions involving at least one argument
//      of stochastic type which overload the "-" operator
//      in a statement such as "a-b".
//    INPUTS
//      a           - an integer, a float, a double or a stochastic number
//      b           - an integer, a float, a double or a stochastic number 
//      At least one argument must be of stochastic type.
//    RESULT
//      res         - a stochastic number
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

#include "cadna.h"
#include  "cadna_private.h"


float_st float_st::operator-() const  
{ 
  float_st res; 
  res.x=-x; 
  res.y=-y; 
  res.z=-z; 
  return res; 
}
 
 

 
 

/////////////////////////////////////////////////////////////////////////


 

/////////////////////////////////////////////////////////////////////////


 
 
float_st operator-(const  float& a, const float_st& b) 
{ 
  float_st res; 
 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();  
  res.x=a-b.x; 
  if (RANDOM) rnd_switch();  
  res.y=a-b.y; 
  rnd_switch();  
  res.z=a-b.z; 
   
//   if (_cadna_cancel_tag){				 
//     if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit(); 
//     if (b.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value) 
//       instability(&_cadna_cancel_count); 
//   }						 
  res.error = b.error;
  return res; 
}
 
 
 
 
float_st operator-(const  float_st& a, const  float& b) 
{ 
  float_st res; 
 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();  
  res.x=a.x-b; 
  if (RANDOM) rnd_switch();  
  res.y=a.y-b; 
  rnd_switch();  
  res.z=a.z-b; 
   
//   if (_cadna_cancel_tag){				 
//     if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit(); 
//     if (a.accuracy >= 	res.nb_significant_digit()+_cadna_cancel_value) 
//       instability(&_cadna_cancel_count); 
//   }						 

  res.error = a.error;				 
  return res; 
}
 
 
 
float_st operator-(const float_st& a, const float_st& b) 
{ 
  float_st res; 
 
  res.accuracy=DIGIT_NOT_COMPUTED; 
  if (RANDOM) rnd_switch();  
  res.x=a.x-b.x; 
  if (RANDOM) rnd_switch();  
  res.y=a.y-b.y; 
  rnd_switch();  
  res.z=a.z-b.z; 
 
   
//   if (_cadna_cancel_tag){				 
//     if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit(); 
//     if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit(); 
//     if ((a.accuracy < b.accuracy ? a.accuracy: b.accuracy) >=  
// 	res.nb_significant_digit()+_cadna_cancel_value) 
//       instability(&_cadna_cancel_count); 
//  }						 

  res.error = a.error  | b.error ;				 
  return res; 
}
 
 

/////////////////////////////////////////////////////////////////////////

