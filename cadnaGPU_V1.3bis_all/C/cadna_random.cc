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

// variables globales 
#include <stdio.h>
 # include "cadna_private.h"
static unsigned int   recurrence = 987654321; 


// 0x40000 -> 262144
#define _CADNA_MAX_RANDOMTABLE 0x40000

unsigned short _cadna_random_table[_CADNA_MAX_RANDOMTABLE];
unsigned int   _cadna_table_counter=0;


// The random generator uses a sequence of _CADNA_MAX_RANDOMTABLE*16
// bits. During the initialisation of the library, a table of
// _CADNA_MAX_RANDOMTABLE element is set with random generated
// unsigned short value.  To obtain a zero or a not zero value, a 32
// counter is used. In the next figure the bits are labelled 0, 1 or
// 2.
//
// The 32 bits: 00000000001111111111111111112222
//
// The bits labelled 0 are not used.
//
// The bits labelled 1 are extracted with a mask and a shift. They
// give the element number in the table _cadna_random_table.
//
// The bits labelled 2 are extracted with a mask. They give the bit
// number.
//
// To obtain the next bit, it is just necessary to increment the
// counter of 1.
//
// This random generator allows to obtain better performance in
// computation time than a classical one which generate a new random
// number for each step.


//****if* cadna_random/_cadna_random_function
//    NAME
//     _cadna_random_function
//
//    SYNOPSIS
//      static unsigned short _cadna_random_function()
//
//    FUNCTION
//        internal random generator of the CADNA library.
//      
//    INPUTS
//   
//    RESULT

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


static unsigned short _cadna_random_function()
{
  static const int IA=16087;
  static const int IM=2147483647;
  static const int IQ=127773;
  static const int IR=2836;
  static const int MASK=123459876;
  int k;

  recurrence^=MASK;
  k=recurrence/IQ;
  recurrence=IA*(recurrence-k*IQ)-IR*k;
  if (recurrence <0) recurrence+=IM;
  recurrence^=MASK;
 
 // We need only a 16 bits number.
  // The lower and higher bits are not used.

  return (recurrence&0x00FFFF00)>>8;
}


//****if* cadna_random/_cadna_init_table
//    NAME
//     _cadna_init_table
//
//    SYNOPSIS
//      void _cadna_init_table(unsigned int a)
//
//    FUNCTION
//      
//    INPUTS
//   
//    RESULT

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

void _cadna_init_table(unsigned int a)
{
  int i;

  recurrence+=a;

  for(i=0;i<_CADNA_MAX_RANDOMTABLE;i++){
    _cadna_random_table[i]=_cadna_random_function();
  }
}

