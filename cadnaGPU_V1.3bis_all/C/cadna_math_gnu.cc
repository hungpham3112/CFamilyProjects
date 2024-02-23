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
#include "cadna.h"
#include "cadna_private.h"
using namespace std;

//****f* cadna_math/pow
//    NAME
//      pow
//    SYNOPSIS
//      res = pow(x,y) 
//    FUNCTION
//      The pow() functions compute x raised to the power y.
//      
//      
//    INPUTS
//      a           - double_st
//      b           - double_st
//    RESULT
//      res         - double_st
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

//****f* cadna_math/powf
//    NAME
//      powf
//    SYNOPSIS
//      res = powf(x,y) 
//    FUNCTION
//      The powf() functions compute x raised to the power y.
//      
//      
//    INPUTS
//      a           - float_st
//      b           - float_st
//    RESULT
//      res         - float_st
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



  double_st  pow(const  double& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  float& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  unsigned long long& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  long long& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  unsigned long& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  long& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  unsigned int& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  int& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  unsigned short& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  short& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  unsigned char& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  double_st  pow(const  char& a, const  double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.x);		 
    res.y= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.y);		 
    res.z= pow((double_st::TYPEBASE)a,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 

  float_st  powf(const  double& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  float& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  unsigned long long& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  long long& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  unsigned long& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  long& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  unsigned int & a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  int& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  unsigned short& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  short& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  unsigned char& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 
  float_st  powf(const  char& a, const  float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;						 
    int bb;							 
    								 
    									 
    if(_cadna_power_tag!=0){						 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (b.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;					 
    rnd_arr();								 
    res.x= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.x);		 
    res.y= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.y);		 
    res.z= powf((float_st::TYPEBASE)a,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }									
 
 



  double_st  pow(const  double_st& a, const  double& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  float& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  unsigned long long& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  long long& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  unsigned long& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  long& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  unsigned int & b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  int & b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  unsigned short& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  short& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  unsigned char& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  double_st  pow(const  double_st& a, const  char& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b);		 
    res.y= pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b);		 
    res.z= pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 

  float_st  powf(const  float_st& a, const  double& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  float& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  unsigned long long& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  long long& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  unsigned long& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  long& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  unsigned int & b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  int & b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  unsigned short& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  short& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  unsigned char& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 
  float_st  powf(const  float_st& a, const  char& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;					 
    						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (a.accuracy==0) instability(&_cadna_power_count);		 
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();					 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x= powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b);		 
    res.y= powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b);		 
    res.z= powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b);		 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }								
 
 


/////////////////////////////////////////////////////////////////////////


  double_st pow(const double_st& a, const double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (a.accuracy==0 || b.accuracy==0) instability(&_cadna_power_count);  
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b.x);		 
    res.y=pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b.y);		 
    res.z=pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  double_st pow(const double_st& a, const float_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (a.accuracy==0 || b.accuracy==0) instability(&_cadna_power_count);  
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b.x);		 
    res.y=pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b.y);		 
    res.z=pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  double_st pow(const float_st& a, const double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    int bb;						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (a.accuracy==0 || b.accuracy==0) instability(&_cadna_power_count);  
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=pow((double_st::TYPEBASE)a.x, (double_st::TYPEBASE)b.x);		 
    res.y=pow((double_st::TYPEBASE)a.y, (double_st::TYPEBASE)b.y);		 
    res.z=pow((double_st::TYPEBASE)a.z, (double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  float_st powf(const float_st& a, const float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    int bb;						 
    if(_cadna_power_tag!=0){						 
      if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();	 
      if (b.accuracy==DIGIT_NOT_COMPUTED) b.nb_significant_digit();	 
      if (a.accuracy==0 || b.accuracy==0) instability(&_cadna_power_count);  
    }									 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=powf((float_st::TYPEBASE)a.x, (float_st::TYPEBASE)b.x);		 
    res.y=powf((float_st::TYPEBASE)a.y, (float_st::TYPEBASE)b.y);		 
    res.z=powf((float_st::TYPEBASE)a.z, (float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/log
//    NAME
//      log
//
//    SYNOPSIS
//      res = log(x) 
//
//    FUNCTION
//       The log() function computes the value of the natural
//       logarithm of argument x.
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)       
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

double_st  log(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;

  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log(a.x);
  res.y=log(a.y);
  res.z=log(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/logf
//    NAME
//      logf
//    SYNOPSIS
//      res = logf(x) 
//    FUNCTION
//      
//      The logf() function computes the value of the natural
//      logarithm of argument x.
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3), powf(3)             
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

float_st  logf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  
  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=logf(a.x);
  res.y=logf(a.y);
  res.z=logf(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/log2
//    NAME
//      log2
//    SYNOPSIS
//      res = log2(x) 
//    FUNCTION
//      The log2() function computes the value of the logarithm of
//      argument x to base 2.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)             
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

double_st  log2(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  
  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log2(a.x);
  res.y=log2(a.y);
  res.z=log2(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/log2f
//    NAME
//      log2f
//    SYNOPSIS
//      res = log2f(x) 
//    FUNCTION
//      The log2f() function computes the value of the logarithm of
//      argument x to base 2.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3), powf(3)                    
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

float_st  log2f(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log2(a.x);
  res.y=log2(a.y);
  res.z=log2(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/log10
//    NAME
//       log10
//    SYNOPSIS
//      res =  log1(x) 
//    FUNCTION
//      The log10() function computes the value of the logarithm of
//      argument x to base 10.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)
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
double_st  log10(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log10(a.x);
  res.y=log10(a.y);
  res.z=log10(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/log10f
//    NAME
//      log10f
//    SYNOPSIS
//      res = log10f(x) 
//    FUNCTION
//      The log10f() function computes the value of the logarithm of
//      argument x to base 10.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3), powf(3)             
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

float_st  log10f(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log10(a.x);
  res.y=log10(a.y);
  res.z=log10(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}


//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/log1p
//    NAME
//       log1p
//    SYNOPSIS
//      res =  log1p(x) 
//    FUNCTION
//      The log1p() function computes the value of log(1+x) accurately
//      even for very small values of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)
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
double_st  log1p(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <=-1){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log1p(a.x);
  res.y=log1p(a.y);
  res.z=log1p(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/log1pf
//    NAME
//      log1pf
//    SYNOPSIS
//      res = log1pf(x) 
//    FUNCTION
//      The log1pf() function computes the value of log(1+x)
//      accurately even for very small values of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//       log2() , log10() , log1p(), exp(3), exp2(3), expm1(3),
//       powf(3)
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

float_st  log1pf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <=-1){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=log1p(a.x);
  res.y=log1p(a.y);
  res.z=log1p(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////
//****f* cadna_math/logb
//    NAME
//      logb
//    SYNOPSIS
//      res = logb(x) 
//    FUNCTION
//      The logb() functions return the exponent of x, represented as
//      a floating-point number.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
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
double_st  logb(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <=-1){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=logb(a.x);
  res.y=logb(a.y);
  res.z=logb(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}
//****f* cadna_math/logbf
//    NAME
//      logbf
//    SYNOPSIS
//      res = logbf(x) 
//    FUNCTION
//      The logbf() functions return the exponent of x, represented as
//      a floating-point number.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
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

float_st  logbf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0 || a.x <=-1){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=logb(a.x);
  res.y=logb(a.y);
  res.z=logb(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/exp
//    NAME
//       exp
//    SYNOPSIS
//      res =  exp(x) 
//    FUNCTION
//      The exp() function computes e**x, the base-e exponential of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//       log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)
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
double_st  exp(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr();   

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=exp(a.x);
  res.y=exp(a.y);
  res.z=exp(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 
//****f* cadna_math/expf
//    NAME
//      expf
//    SYNOPSIS
//      res = expf(x) 
//    FUNCTION
//      The expf() function computes e**x, the base-e exponential of
//      x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3),
//      powf(3)
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

float_st  expf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=exp(a.x);
  res.y=exp(a.y);
  res.z=exp(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//////////////////////////////////////////////////////////////////////////////
//****f* cadna_math/exp2
//    NAME
//       exp2
//    SYNOPSIS
//      res = exp2(x) 
//    FUNCTION
//      The exp2() function computes 2**x, the base-2 exponential of
//      x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)
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
double_st  exp2(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=exp2(a.x);
  res.y=exp2(a.y);
  res.z=exp2(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//****f* cadna_math/exp2f
//    NAME
//      exp2f
//    SYNOPSIS
//      res = exp2f(x) 
//    FUNCTION
//      The exp2f() function computes 2**x, the base-2 exponential of
//      x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3),
//      powf(3)
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
float_st  exp2f(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }

  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=exp2(a.x);
  res.y=exp2(a.y);
  res.z=exp2(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 


//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/expm1
//    NAME
//       expm1
//    SYNOPSIS
//      res =  expm1(x) 
//    FUNCTION
//      The expm1() function computes the base-e exponential of x ,
//      minus 1 accurately even for very small values of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      log2() , log10() , log1p(), exp(3), exp2(3), expm1(3), pow(3)
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
double_st  expm1(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=expm1(a.x);
  res.y=expm1(a.y);
  res.z=expm1(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//****f* cadna_math/expm1f
//    NAME
//      expm1f
//    SYNOPSIS
//      res = expm1f(x) 
//    FUNCTION
//      The expm1f() function computes the base-e exponential of x ,
//      minus 1 accurately even for very small values of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      log2f() , log10f() , log1pf(), expf(3), exp2f(3), expm1f(3),
//      powf(3)
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

float_st  expm1f(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=expm1(a.x);
  res.y=expm1(a.y);
  res.z=expm1(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 


//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/sqrt
//    NAME
//      sqrt
//    SYNOPSIS
//      res = sqrt(x) 
//    FUNCTION
//      The sqrt() function compute the non-negative square root of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      cbrt(3)
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
//      
//  /
double_st  sqrt(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sqrt(a.x);
  res.y=sqrt(a.y);
  res.z=sqrt(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//****f* cadna_math/sqrtf
//    NAME
//      sqrtf
//    SYNOPSIS
//      res = sqrtf(x) 
//    FUNCTION
//      The sqrtf() function compute the non-negative square root of
//      x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      cbrt(3)
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

float_st  sqrtf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sqrt(a.x);
  res.y=sqrt(a.y);
  res.z=sqrt(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//////////////////////////////////////////////////////////////////////////////


//****f* cadna_math/cbrt
//    NAME
//      cbrt
//    SYNOPSIS
//      res = cbrt(x) 
//    FUNCTION
//      The cbrt() function computes the cube root of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      sqrt(3)
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
double_st  cbrt(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cbrt(a.x);
  res.y=cbrt(a.y);
  res.z=cbrt(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//****f* cadna_math/cbrtf
//    NAME
//      cbrtf
//    SYNOPSIS
//      res = cbrtf(x) 
//    FUNCTION
//      The cbrtf() function computes the cube root of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      sqrt(3)
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

float_st  cbrtf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  if(_cadna_math_tag!=0){
    if (a.accuracy==DIGIT_NOT_COMPUTED) a.nb_significant_digit();
    if (a.accuracy==0) instability(&_cadna_math_count);
  }
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 

  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cbrt(a.x);
  res.y=cbrt(a.y);
  res.z=cbrt(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/sin
//    NAME
//      sin
//    SYNOPSIS
//      res = sin(x) 
//    FUNCTION
//      The sin() function computes the sine of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
//      
//  /
double_st  sin(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sin(a.x);
  res.y=sin(a.y);
  res.z=sin(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/sinf
//    NAME
//      sinf
//    SYNOPSIS
//      res = sinf(x) 
//    FUNCTION
//      The sinf() function computes the sine of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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

float_st  sinf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sin(a.x);
  res.y=sin(a.y);
  res.z=sin(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/cos
//    NAME
//      cos
//    SYNOPSIS
//      res = cos(x) 
//    FUNCTION
//      The cos() function computes the cosine of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  cos(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cos(a.x);
  res.y=cos(a.y);
  res.z=cos(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/cosf
//    NAME
//      cosf
//    SYNOPSIS
//      res = cosf(x) 
//    FUNCTION
//      The cosf() function computes the cosine of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  cosf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cos(a.x);
  res.y=cos(a.y);
  res.z=cos(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/tan
//    NAME
//      tan
//    SYNOPSIS
//      res = tan(x) 
//    FUNCTION
//      The tan() function computes the tangent of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  tan(const double_st& a)
{
  double_st res;
  double aux;
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=tan(a.x);
  res.y=tan(a.y);
  res.z=tan(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//****f* cadna_math/tanf
//    NAME
//      tanf
//    SYNOPSIS
//      res = tanf(x) 
//    FUNCTION
//      The tanf() function computes the tangent of x (measured in
//      radians).
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  tanf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=tan(a.x);
  res.y=tan(a.y);
  res.z=tan(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/asin
//    NAME
//      asin
//    SYNOPSIS
//      res = asin(x) 
//    FUNCTION
//      The asin() function computes the principal value of the arc
//      sine of x.  The result is in the range [-pi/2, +pi/2].
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  asin(const double_st& a)
{
  double_st res;
  double aux;
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=asin(a.x);
  res.y=asin(a.y);
  res.z=asin(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}



//****f* cadna_math/asinf
//    NAME
//      asinf
//    SYNOPSIS
//      res = asinf(x) 
//    FUNCTION
//      The asin() function computes the principal value of the arc
//      sine of x.  The result is in the range [-pi/2, +pi/2].
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE A
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  asinf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=asin(a.x);
  res.y=asin(a.y);
  res.z=asin(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/acos
//    NAME
//      acos
//    SYNOPSIS
//      res = acos(x) 
//    FUNCTION
//      The acos() function computes the principle value of the arc
//      cosine of x.  The result is in the range [0, pi].
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  acos(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=acos(a.x);
  res.y=acos(a.y);
  res.z=acos(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/acosf
//    NAME
//      acosf
//    SYNOPSIS
//      res = acosf(x) 
//    FUNCTION
//      The acosf() function computes the principle value of the arc
//      cosine of x.  The result is in the range [0, pi].
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  acosf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=acos(a.x);
  res.y=acos(a.y);
  res.z=acos(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/atan
//    NAME
//      atan
//    SYNOPSIS
//      res = atan(x) 
//    FUNCTION
//      The atan() function computes the principal value of the arc
//      tangent of x.  The result is in the range [-pi/2, +pi/2].
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  atan(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=atan(a.x);
  res.y=atan(a.y);
  res.z=atan(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/atanf
//    NAME
//      atanf
//    SYNOPSIS
//      res = atanf(x) 
//    FUNCTION
//      The atanf() function computes the principal value of the arc
//      tangent of x.  The result is in the range [-pi/2, +pi/2].
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  atanf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=atan(a.x);
  res.y=atan(a.y);
  res.z=atan(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
} 
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////



//****f* cadna_math/atan2
//    NAME
//      atan2
//    SYNOPSIS
//      res = atan2(x,y) 
//    FUNCTION
//      The atan2() function computes the principal value of the arc
//      tangent of y/x, using the signs of both arguments to determine
//      the quadrant of the return value.
//      
//      
//    INPUTS
//      x           - double_st 
//      y           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
//  / ****f* cadna_math/atan2f 
//      NAME 
//        atan2f 
//      SYNOPSIS 
//        res = atan2f(x,y)
//      FUNCTION The atan2() function computes the principal value of
//      the arc tangent of y/x, using the signs of both arguments to
//      determine the quadrant of the return value.
//      
//    INPUTS
//      a           - float_st  
//      b           - float_st 
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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



  double_st atan2(const double_st& a, const double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();								 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=atan2((double_st::TYPEBASE)a.x,(double_st::TYPEBASE)b.x);		 
    res.y=atan2((double_st::TYPEBASE)a.y,(double_st::TYPEBASE)b.y);		 
    res.z=atan2((double_st::TYPEBASE)a.z,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  double_st atan2(const double_st& a, const float_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();								 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=atan2((double_st::TYPEBASE)a.x,(double_st::TYPEBASE)b.x);		 
    res.y=atan2((double_st::TYPEBASE)a.y,(double_st::TYPEBASE)b.y);		 
    res.z=atan2((double_st::TYPEBASE)a.z,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  double_st atan2(const float_st& a, const double_st& b)	 
  {						 
    double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();								 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=atan2((double_st::TYPEBASE)a.x,(double_st::TYPEBASE)b.x);		 
    res.y=atan2((double_st::TYPEBASE)a.y,(double_st::TYPEBASE)b.y);		 
    res.z=atan2((double_st::TYPEBASE)a.z,(double_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 
  float_st atan2f(const float_st& a, const float_st& b)	 
  {						 
    float_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;		 
    rnd_arr();								 
    res.accuracy=DIGIT_NOT_COMPUTED;					 
    res.x=atan2f((float_st::TYPEBASE)a.x,(float_st::TYPEBASE)b.x);		 
    res.y=atan2f((float_st::TYPEBASE)a.y,(float_st::TYPEBASE)b.y);		 
    res.z=atan2f((float_st::TYPEBASE)a.z,(float_st::TYPEBASE)b.z);		 
    									 
    if (bb) rnd_moinf();						 
    else rnd_plinf();							 
    									 
    return res;								 
  }
 
 

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/sinh
//    NAME
//      sinh
//    SYNOPSIS
//      res = sinh(x) 
//    FUNCTION
//      The sinh() function computes the hyperbolic sine of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  sinh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sinh(a.x);
  res.y=sinh(a.y);
  res.z=sinh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/sinhf
//    NAME
//      sinhf
//    SYNOPSIS
//      res = sinhf
//   FUNCTION
//     The sinhf() function computes the hyperbolic sine of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  sinhf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=sinh(a.x);
  res.y=sinh(a.y);
  res.z=sinh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/cosh
//    NAME
//      cosh
//    SYNOPSIS
//      res = cosh(x) 
//    FUNCTION
//      The cosh() function computes the hyperbolic cosine of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  cosh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cosh(a.x);
  res.y=cosh(a.y);
  res.z=cosh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/coshf
//    NAME
//      coshf
//    SYNOPSIS
//      res = coshf(x) 
//    FUNCTION
//      The coshf() function computes the hyperbolic cosine of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  coshf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=cosh(a.x);
  res.y=cosh(a.y);
  res.z=cosh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/tanh
//    NAME
//      tanh
//    SYNOPSIS
//      res = tanh(x) 
//    FUNCTION
//      The tanh() function computes the hyperbolic tangent of x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  tanh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=tanh(a.x);
  res.y=tanh(a.y);
  res.z=tanh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/tanhf
//    NAME
//      tanhf
//    SYNOPSIS
//      res = tanhf(x) 
//    FUNCTION
//      The tanhf() function computes the hyperbolic tangent of x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  tanhf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=tanh(a.x);
  res.y=tanh(a.y);
  res.z=tanh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/asinh
//    NAME
//      asinh
//    SYNOPSIS
//      res = asinh(x) 
//    FUNCTION
//      The asinh() function computes the inverse hyperbolic sine of
//      the double_st argument
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  asinh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=asinh(a.x);
  res.y=asinh(a.y);
  res.z=asinh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/asinhf
//    NAME
//      asinhf
//    SYNOPSIS
//      res = asinhf(x) 
//    FUNCTION
//      The asinhf() function computes the inverse hyperbolic sine of
//      the float_st argument
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  asinhf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=asinh(a.x);
  res.y=asinh(a.y);
  res.z=asinh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}
//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/acosh
//    NAME
//      acosh
//    SYNOPSIS
//      res = acosh(x) 
//    FUNCTION
//      The acosh() function computes the inverse hyperbolic cosine of
//      the double argument.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  acosh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=acosh(a.x);
  res.y=acosh(a.y);
  res.z=acosh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/acoshf
//    NAME
//      acoshf
//    SYNOPSIS
//      res = acoshf(x) 
//    FUNCTION
//      The acoshf() function computes the inverse hyperbolic cosine
//      of the double argument.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  acoshf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=acosh(a.x);
  res.y=acosh(a.y);
  res.z=acosh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}
//////////////////////////////////////////////////////////////////////////////

//****f* cadna_math/atanh
//    NAME
//      atanh
//    SYNOPSIS
//      res = atanh(x) 
//    FUNCTION
//      The atanh() function computes the inverse hyperbolic tangent
//      of the double_st argument x.
//      
//      
//    INPUTS
//      a           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      acos(3), asin(3), atan(3), atan2(3), cos(3), cosh(3), sinh(3),
//      tan(3), tanh(3)
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
double_st  atanh(const double_st& a)
{
  double_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=atanh(a.x);
  res.y=atanh(a.y);
  res.z=atanh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//****f* cadna_math/atanhf
//    NAME
//      atanhf
//    SYNOPSIS
//      res = atanhf(x) 
//    FUNCTION
//      The atanhf() function computes the inverse hyperbolic tangent
//      of the float_st argument x.
//      
//      
//    INPUTS
//      a           - float_st
//    RESULT
//      res         - float_st
//    SEE ALSO
//      acosf(3), asinf(3), atanf(3), atan2f(3), cosf(3), coshf(3),
//      sinhf(3), tanf(3), tanhf(3)
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
float_st  atanhf(const float_st& a)
{
  float_st res;
  double aux=1.0; 
  
  int bb;
  bb=(aux+1.e-20)==1.0 ?  1 : 0; 
  rnd_arr(); 
  
  res.accuracy=DIGIT_NOT_COMPUTED;
  res.x=atanh(a.x);
  res.y=atanh(a.y);
  res.z=atanh(a.z);
  
  if (bb) rnd_moinf(); 
  else rnd_plinf(); 
  
  return res;
}

//////////////////////////////////////////////////////////////////////////////
//****f* cadna_math/hypot
//    NAME
//      hypot
//    SYNOPSIS
//      res = hypot(x,y) 
//    FUNCTION
//      The hypot() function computes the sqrt(x*x+y*y) without undue
//      overflow or underflow.
//      
//      
//    INPUTS
//      a           - double_st
//      b           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      sqrtf(3)
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


//****f* cadna_math/hypotf
//    NAME
//      hypotf
//
//    SYNOPSIS
//      res = hypotf(x,y) 
//
//    FUNCTION
//      The hypotf() function computes the sqrt(x*x+y*y) without undue
//    overflow or underflow.  
//
//    SEE ALSO 
//       sqrtf(3)
//      
//      
//    INPUTS
//      a           - float_st
//      b           - float_st
//    RESULT
//      res         - float_st
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

//      
//  /




   double_st hypot(const  double_st& a, const  double_st& b)	 
  {						 
     double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;			 
    res.x=hypot(a.x,b.x);				 
    res.y=hypot(a.y,b.y);				 
    res.z=hypot(a.z,b.z);				 
    							 
    if (bb) rnd_moinf();				 
    else rnd_plinf();					 
    							 
    return res;						 
  }
 
 
   double_st hypot(const  double_st& a, const  float_st& b)	 
  {						 
     double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;			 
    res.x=hypot(a.x,b.x);				 
    res.y=hypot(a.y,b.y);				 
    res.z=hypot(a.z,b.z);				 
    							 
    if (bb) rnd_moinf();				 
    else rnd_plinf();					 
    							 
    return res;						 
  }
 
 
   double_st hypot(const  float_st& a, const  double_st& b)	 
  {						 
     double_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;			 
    res.x=hypot(a.x,b.x);				 
    res.y=hypot(a.y,b.y);				 
    res.z=hypot(a.z,b.z);				 
    							 
    if (bb) rnd_moinf();				 
    else rnd_plinf();					 
    							 
    return res;						 
  }
 
 
   float_st hypotf(const  float_st& a, const  float_st& b)	 
  {						 
     float_st res;					 
    double aux=1.0;				 
    						 
    int bb;					 
    bb=(aux+1.e-20)==1.0 ?  1 : 0;			 
    rnd_arr();						 
    							 
    res.accuracy=DIGIT_NOT_COMPUTED;			 
    res.x=hypotf(a.x,b.x);				 
    res.y=hypotf(a.y,b.y);				 
    res.z=hypotf(a.z,b.z);				 
    							 
    if (bb) rnd_moinf();				 
    else rnd_plinf();					 
    							 
    return res;						 
  }
 
 


///////////////////////////////////


//****f* cadna_math/fmax
//    NAME
//      fmax
//    SYNOPSIS
//      res = fmax(x,y) 
//    FUNCTION
//      The fmax() functions return x or y, whichever is larger.
//      
//      
//    INPUTS
//      a           - double_st 
//      b           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      fmin(3)
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

//      
//  /
//****f* cadna_math/fmaxf
//    NAME
//      fmaxf
//    SYNOPSIS
//      res = fmaxf(x,y) 
//    FUNCTION
//      The fmaxf() functions return x or y, whichever is larger.
//      
//      
//    INPUTS
//      a           - float_st 
//      b           - float_st 
//    RESULT
//      res         - float_st
//    SEE ALSO
//      fminf(3)
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


//   double_st fmax(const  double& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  float& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  unsigned long long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  long long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  unsigned long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  unsigned int & a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  int & a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  unsigned short& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  short& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  unsigned char& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmax(const  char& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=double_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 

//   float_st fmaxf(const  double& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  float& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  unsigned long long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  long long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  unsigned long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  unsigned int & a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  int & a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  unsigned short& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  short& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  unsigned char& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fmaxf(const  char& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
//     if (r){					 
//       res=float_st(a);				 
//     }						 
//     else {					 
//       if ( 3*a  > ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 


//    double_st fmax(const  double_st& a, const  double& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  float& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  unsigned long long& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  long long& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  unsigned long& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  long& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  unsigned int & b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  int & b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  unsigned short& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  short& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  unsigned char& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    double_st fmax(const  double_st& a, const  char& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 

//    float_st fmaxf(const  float_st& a, const  double& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  float& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  unsigned long long& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  long long& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  unsigned long& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  long& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  unsigned int & b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  int & b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  unsigned short& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  short& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  unsigned char& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 
//    float_st fmaxf(const  float_st& a, const  char& b)	 
//   {						 
//      float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) > b+b+b)		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }						 
  
 
 


//    double_st fmax(const  double_st& a, const  double_st& b)	 
//   {						 
//      double_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b.y;	 
//     rnd_switch(); res.z=a.z-b.z;		 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r){							 
//       if (a.nb_significant_digit()>b.nb_significant_digit())	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     else {							 
//       if ( ( a.x + a.y + a.z ) >	( b.x + b.y + b.z ))	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     return res;							 
//   }
 
 
 
//   float_st fmaxf(const  float_st& a, const  float_st& b)	 
//   {						 
//     float_st res;					 
    						 
//     if (RANDOM) rnd_switch(); res.x=a.x-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b.y;	 
//     rnd_switch(); res.z=a.z-b.z;		 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r){							 
//       if (a.nb_significant_digit()>b.nb_significant_digit())	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     else {							 
//       if ( ( a.x + a.y + a.z ) >	( b.x + b.y + b.z ))	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     return res;							 
//   }
 
 


///////////////////////////////////


//****f* cadna_math/fmin
//    NAME
//      fmin
//    SYNOPSIS
//      res = fmin(x,y) 
//    FUNCTION
//      The fmin() functions return x or y, whichever is smaller.
//      
//      
//    INPUTS
//      a           - double_st 
//      b           - double_st
//    RESULT
//      res         - double_st
//    SEE ALSO
//      fmax(3)
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
//****f* cadna_math/fminf
//    NAME
//      fminf
//    SYNOPSIS
//      res = fminf(x,y) 
//    FUNCTION
//      The fminf() functions return x or y, whichever is smaller.

//      
//      
//    INPUTS
//      a           - float_st 
//      b           - float_st 
//    RESULT
//      res         - float_st
//    SEE ALSO
//      fmaxf(3)
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




//   double_st fmin(const  double& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  float& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  unsigned long long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  long long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  unsigned long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  long& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  unsigned int & a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  int & a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  unsigned short& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  short& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  unsigned char& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   double_st fmin(const  char& a, const double_st& b)	 
//   {						 
//     double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=double_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=double_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 

//   float_st fminf(const  double& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  float& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  unsigned long long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  long long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  unsigned long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  long& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  unsigned int & a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  int & a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  unsigned short& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  short& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  unsigned char& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 
//   float_st fminf(const  char& a, const float_st& b)	 
//   {						 
//     float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a-b.y;	 
//     rnd_switch(); res.z=a-b.z;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res=float_st(a);			 
//     else {					 
//       if ( 3*a < ( b.x + b.y + b.z ))		 
// 	res=float_st(a);				 
//       else					 
// 	res=b;					 
//     }						 
//     return res;					 
//   }
 
 



//    double_st fmin(const  double_st& a, const  double& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  float& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  unsigned long long& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  long long& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  unsigned long& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  long& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  unsigned int & b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  int & b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  unsigned short& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  short& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  unsigned char& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    double_st fmin(const  double_st& a, const  char& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= double_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= double_st(b);				 
//     }						 
//     return res;					 
//   }
 
 

//    float_st fminf(const  float_st& a, const  double& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  float& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  unsigned long long& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  long long& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  unsigned long& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  long& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  unsigned int & b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  int & b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  unsigned short& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  short& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  unsigned char& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 
//    float_st fminf(const  float_st& a, const  char& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b;	 
//     rnd_switch(); res.z=a.z-b;			 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r) res= float_st(b);			 
//     else {					 
//       if ( ( a.x + a.y + a.z ) < 3*b )		 
// 	res=a;					 
//       else					 
// 	res= float_st(b);				 
//     }						 
//     return res;					 
//   }
 
 



//    double_st fmin(const  double_st& a, const  double_st& b)	 
//   {						 
//      double_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b.y;	 
//     rnd_switch(); res.z=a.z-b.z;		 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r){							 
//       if (a.nb_significant_digit()>b.nb_significant_digit())	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     else {							 
//       if ( ( a.x + a.y + a.z ) <	( b.x + b.y + b.z ))	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     return res;							 
//   }
 
 
 
//    float_st fminf(const  float_st& a, const  float_st& b)	 
//   {						 
//      float_st res;					 
//     if (RANDOM) rnd_switch(); res.x=a.x-b.x;	 
//     if (RANDOM) rnd_switch(); res.y=a.y-b.y;	 
//     rnd_switch(); res.z=a.z-b.z;		 
    						 
//     int r=res.computedzero();			 
    						 
//     if (_cadna_branching_tag && r)		 
//       instability(&_cadna_branching_count);	 
    						 
//     if (r){							 
//       if (a.nb_significant_digit()>b.nb_significant_digit())	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     else {							 
//       if ( ( a.x + a.y + a.z ) <	( b.x + b.y + b.z ))	 
// 	res=a;							 
//       else							 
// 	res=b;							 
//     }								 
//     return res;							 
//   }
 
 

