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

float_st& float_st::operator=(const float_st &a)
{
    x=a.x;
    y=a.y;
    z=a.z;
    accuracy=a.accuracy;
    error=a.error;
    return *this;
}



float_st& float_st::operator=(const double &a) 
{ 
    x=a; 
    y=a; 
    z=a; 
    accuracy=7; 
    error=0;
    return *this ; 
}
 
 
float_st& float_st::operator=(const float &a) 
{ 
    x=a; 
    y=a; 
    z=a; 
    accuracy=7;
    error=0;
    return *this ; 
}
 
 
// float_st& float_st::operator=(const unsigned long long &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const long long &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const unsigned long &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const long &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const unsigned int &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const int &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const unsigned short &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const short &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const unsigned char &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 
// float_st& float_st::operator=(const char &a) 
// { 
//     x=a; 
//     y=a; 
//     z=a; 
//     accuracy=7; 
//     return *this ; 
// }
 
 






