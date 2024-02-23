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

#ifndef __CADNA_PRIVATE__
#define __CADNA_PRIVATE__

#include "cadna_rounding.h"

extern int _cadna_max_instability, _cadna_instability_detected;

//  global variables 

// INSTABILITIES MANAGEMENT

extern unsigned long _cadna_div_count;
extern unsigned long _cadna_mul_count;  
extern unsigned long _cadna_power_count;  
extern unsigned long _cadna_math_count; 
extern unsigned long _cadna_intrinsic_count;  
extern unsigned long _cadna_cancel_count;  
extern unsigned long _cadna_branching_count; 

extern unsigned int _cadna_div_tag; 
extern unsigned int _cadna_mul_tag;  
extern unsigned int _cadna_power_tag;  
extern unsigned int _cadna_math_tag; 
extern unsigned int _cadna_intrinsic_tag; 
extern unsigned int _cadna_cancel_tag; 
extern unsigned int _cadna_branching_tag;

extern unsigned int _cadna_div_change; 
extern unsigned int _cadna_mul_change;  
extern unsigned int _cadna_power_change;  
extern unsigned int _cadna_math_change; 
extern unsigned int _cadna_intrinsic_change; 
extern unsigned int _cadna_cancel_change; 
extern unsigned int _cadna_branching_change;

extern int _cadna_cancel_value;
void  instability(unsigned long*);



// The RANDOM BIT GENERATOR
extern unsigned short  _cadna_random_table[];
extern unsigned int _cadna_table_counter;

void _cadna_init_table(unsigned int);

#define RANDOM_BIT_NUMBER 0x0000000F
#define RANDOM_ELEMENT_NUMBER 0x003FFFF0
#define RANDOM_ELEMENT_SHIFT 4

#define RANDOM (_cadna_random_table[(_cadna_table_counter&RANDOM_ELEMENT_NUMBER)>>RANDOM_ELEMENT_SHIFT] & (1<<((_cadna_table_counter++)&RANDOM_BIT_NUMBER)))


#endif


