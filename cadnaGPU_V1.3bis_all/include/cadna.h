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

#ifndef __CADNA__
#define __CADNA__


#include <iostream>
#include <cmath>
#include "cadna_commun.h"

using namespace std;

class double_st;

class float_st {
  friend class double_st;

protected:
  float x,y,z;

  typedef float TYPEBASE;

private:
  mutable unsigned char accuracy;
  mutable unsigned char error;
  unsigned char pad1, pad2;

public:

  inline float_st(void) :x(0),y(0),z(0),accuracy(7), error(0) {};
  inline float_st(const double &a, const double &b, const double &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};

  inline float_st(const float &a): x(a),y(a),z(a),accuracy(7), error(0) {};

  float getx() {return(this->x);};
  float gety() {return(this->y);};
  float getz() {return(this->z);};


  // ASSIGNMENT

  float_st& operator=(const double&);
  float_st& operator=(const float_st&);
  float_st& operator=(const float&);



  // ADDITION
  float_st operator+() const ;

  friend float_st operator+(const float_st&, const float_st&);

  friend float_st operator+(const float_st&, const float&);
  friend float_st operator+(const float&, const float_st&);


  // SUBTRACTION

  float_st operator-() const ;

  friend float_st operator-(const float_st&, const float_st&);


  friend float_st operator-(const float_st&, const float&);
  friend float_st operator-(const float&, const float_st&);

  // PRODUCT

  friend float_st operator*(const float_st&, const float_st&);
  friend float_st operator*(const float_st&, const float&);
  friend float_st operator*(const float&, const float_st&);

  // DIVISION


  float_st& operator/=(const unsigned char&);
  float_st& operator/=(const char&);

  friend float_st operator/(const float_st&, const float_st&);
  friend float_st operator/(const float_st&, const float&);
  friend float_st operator/(const float&, const float_st&);

  // RELATIONAL OPERATORS

  friend int operator==(const float_st&, const float_st&);
  friend int operator==(const float_st&, const float&);
  friend int operator==(const float&, const float_st&);

  friend int operator!=(const float_st&, const float_st&);
  friend int operator!=(const float_st&, const float&);
  friend int operator!=(const float&, const float_st&);

  friend int operator<=(const float_st&, const float_st&);
  friend int operator<=(const float_st&, const float&);
  friend int operator<=(const float&, const float_st&);

  friend int operator>=(const float_st&, const float_st&);
  friend int operator>=(const float_st&, const float&);
  friend int operator>=(const float&, const float_st&);

  friend int operator<(const float_st&, const float_st&);
  friend int operator<(const float_st&, const float&);
  friend int operator<(const float&, const float_st&);

  friend int operator>(const float_st&, const float_st&);
  friend int operator>(const float_st&, const float&);
  friend int operator>(const float&, const float_st&);

  // MATHEMATICAL FUNCTIONS

  friend float_st  logf(const float_st&);
  friend float_st  log2f(const float_st&);
  friend float_st  log10f(const float_st&);
  friend float_st  log1pf(const float_st&);
  friend float_st  logbf(const float_st&);
  friend float_st  expf(const float_st&);
  friend float_st  exp2f(const float_st&);
  friend float_st  expm1f(const float_st&);
  friend float_st  sqrtf(const float_st&);
  friend float_st  cbrtf(const float_st&);
  friend float_st  sinf(const float_st&);
  friend float_st  cosf(const float_st&);
  friend float_st  tanf(const float_st&);
  friend float_st  asinf(const float_st&);
  friend float_st  acosf(const float_st&);
  friend float_st  atanf(const float_st&);
  friend double_st atan2(const double_st&, const float_st&);
  friend double_st atan2(const float_st&, const double_st&);
  friend float_st  atan2f(const float_st&, const float_st&);
  friend float_st  sinhf(const float_st&);
  friend float_st  coshf(const float_st&);
  friend float_st  tanhf(const float_st&);
  friend float_st  asinhf(const float_st&);
  friend float_st  acoshf(const float_st&);
  friend float_st  atanhf(const float_st&);
  friend double_st  hypot(const double_st&, const float_st&);
  friend double_st  hypot(const float_st&, const double_st&);
  friend float_st  hypotf(const float_st&, const float_st&);

  friend double_st pow(const double_st&, const float_st&);
  friend double_st pow(const float_st&, const double_st&);
  friend float_st powf(const float_st&, const float_st&);

  friend float_st powf(const float_st&, const double&);
  friend float_st powf(const float_st&, const float&);
  friend float_st powf(const float_st&, const unsigned long long&);
  friend float_st powf(const float_st&, const long long&);
  friend float_st powf(const float_st&, const unsigned long&);
  friend float_st powf(const float_st&, const long&);
  friend float_st powf(const float_st&, const unsigned int&);
  friend float_st powf(const float_st&, const int&);
  friend float_st powf(const float_st&, const unsigned short&);
  friend float_st powf(const float_st&, const short&);
  friend float_st powf(const float_st&, const unsigned char&);
  friend float_st powf(const float_st&, const char&);

  friend float_st powf(const double&, const float_st&);
  friend float_st powf(const float&, const float_st&);
  friend float_st powf(const unsigned long long&, const float_st&);
  friend float_st powf(const long long&, const float_st&);
  friend float_st powf(const unsigned long&, const float_st&);
  friend float_st powf(const long&, const float_st&);
  friend float_st powf(const unsigned int&, const float_st&);
  friend float_st powf(const int&, const float_st&);
  friend float_st powf(const unsigned short&, const float_st&);
  friend float_st powf(const short&, const float_st&);
  friend float_st powf(const unsigned char&, const float_st&);
  friend float_st powf(const char&, const float_st&);


  friend float_st fmaxf(const float_st&, const float_st&);
  friend double_st fmax(const double_st&, const float_st&);
  friend double_st fmax(const float_st&, const double_st&);

  friend float_st fmaxf(const float_st&, const double&);
  friend float_st fmaxf(const float_st&, const float&);
  friend float_st fmaxf(const float_st&, const unsigned long long&);
  friend float_st fmaxf(const float_st&, const long long&);
  friend float_st fmaxf(const float_st&, const unsigned long&);
  friend float_st fmaxf(const float_st&, const long&);
  friend float_st fmaxf(const float_st&, const unsigned int&);
  friend float_st fmaxf(const float_st&, const int&);
  friend float_st fmaxf(const float_st&, const unsigned short&);
  friend float_st fmaxf(const float_st&, const short&);
  friend float_st fmaxf(const float_st&, const unsigned char&);
  friend float_st fmaxf(const float_st&, const char&);

  friend float_st fmaxf(const double&, const float_st&);
  friend float_st fmaxf(const float&, const float_st&);
  friend float_st fmaxf(const unsigned long long&, const float_st&);
  friend float_st fmaxf(const long long&, const float_st&);
  friend float_st fmaxf(const unsigned long&, const float_st&);
  friend float_st fmaxf(const long&, const float_st&);
  friend float_st fmaxf(const unsigned int&, const float_st&);
  friend float_st fmaxf(const int&, const float_st&);
  friend float_st fmaxf(const unsigned short&, const float_st&);
  friend float_st fmaxf(const short&, const float_st&);
  friend float_st fmaxf(const unsigned char&, const float_st&);
  friend float_st fmaxf(const char&, const float_st&);

  friend float_st fminf(const float_st&, const float_st&);
  friend double_st fmin(const double_st&, const float_st&);
  friend double_st fmin(const float_st&, const double_st&);

  friend float_st fminf(const float_st&, const double&);
  friend float_st fminf(const float_st&, const float&);
  friend float_st fminf(const float_st&, const unsigned long long&);
  friend float_st fminf(const float_st&, const long long&);
  friend float_st fminf(const float_st&, const unsigned long&);
  friend float_st fminf(const float_st&, const long&);
  friend float_st fminf(const float_st&, const unsigned int&);
  friend float_st fminf(const float_st&, const int&);
  friend float_st fminf(const float_st&, const unsigned short&);
  friend float_st fminf(const float_st&, const short&);
  friend float_st fminf(const float_st&, const unsigned char&);
  friend float_st fminf(const float_st&, const char&);

  friend float_st fminf(const double&, const float_st&);
  friend float_st fminf(const float&, const float_st&);
  friend float_st fminf(const unsigned long long&, const float_st&);
  friend float_st fminf(const long long&, const float_st&);
  friend float_st fminf(const unsigned long&, const float_st&);
  friend float_st fminf(const long&, const float_st&);
  friend float_st fminf(const unsigned int&, const float_st&);
  friend float_st fminf(const int&, const float_st&);
  friend float_st fminf(const unsigned short&, const float_st&);
  friend float_st fminf(const short&, const float_st&);
  friend float_st fminf(const unsigned char&, const float_st&);
  friend float_st fminf(const char&, const float_st&);



  // INTRINSIC FUNCTIONS
  friend float_st fabs(const float_st&);
  friend float_st fabsf(const float_st&);
  friend float_st floorf(const float_st&);
  friend float_st ceilf(const float_st&);
  friend float_st truncf(const float_st&);
  friend float_st nearbyintf(const float_st&);
  friend float_st rintf(const float_st&);
  friend long int  lrintf(const float_st&);
  friend long long int  llrintf(const float_st&);

  // Conversion
  operator char();
  operator unsigned char();
  operator short();
  operator unsigned short();
  operator int();
  operator unsigned();
  operator long();
  operator unsigned long();
  operator long long();
  operator unsigned long long();
  operator float();
  operator double();


  int nb_significant_digit() const;
  int approx_digit() const;
  int computedzero() const;
  void display() const ;
  void display(const char *) const ;
  char* str( char *)  const ;

  friend char* strp(const float_st&);
  friend char* str( char *, const float_st&);

  friend std::istream& operator >>(std::istream& s, float_st& );

  void data_st();
  void data_st(const double&, const int &);

};



class double_st {

  friend class float_st;

protected:
  double x,y,z;

  typedef double TYPEBASE;

private:
  mutable unsigned char accuracy;
  mutable unsigned char error;
  unsigned char pad1, pad2;

public:

  inline double_st(void) :x(0),y(0),z(0),accuracy(15),error(0) {};
  inline double_st(const double &a, const double &b, const double &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED),error(0) {};

  inline double_st(const double &a): x(a),y(a),z(a),accuracy(15),error(0) {};

  double getx() {return(this->x);};
  double gety() {return(this->y);};
  double getz() {return(this->z);};


  // AFFECTATION

  double_st& operator=(const double&);
  double_st& operator=(const float&);
  double_st& operator=(const double_st&);

  // ADDITION
  double_st operator+() const ;

  friend double_st operator+(const double_st&, const double_st&);

  friend double_st operator+(const double_st&, const double&);
  friend double_st operator+(const double&, const double_st&);


  // SUBSTRACTION

  double_st operator-() const ;

  friend double_st operator-(const double_st&, const double_st&);


  friend double_st operator-(const double_st&, const double&);
  friend double_st operator-(const double&, const double_st&);

  // PRODUCT

  friend double_st operator*(const double_st&, const double_st&);
  friend double_st operator*(const double_st&, const double&);
  friend double_st operator*(const double&, const double_st&);

  // DIVISION


  double_st& operator/=(const unsigned char&);
  double_st& operator/=(const char&);

  friend double_st operator/(const double_st&, const double_st&);
  friend double_st operator/(const double_st&, const double&);
  friend double_st operator/(const double&, const double_st&);

  // RELATIONAL OPERATORS

  friend int operator==(const double_st&, const double_st&);
  friend int operator==(const double_st&, const double&);
  friend int operator==(const double&, const double_st&);

  friend int operator!=(const double_st&, const double_st&);
  friend int operator!=(const double_st&, const double&);
  friend int operator!=(const double&, const double_st&);

  friend int operator<=(const double_st&, const double_st&);
  friend int operator<=(const double_st&, const double&);
  friend int operator<=(const double&, const double_st&);

  friend int operator>=(const double_st&, const double_st&);
  friend int operator>=(const double_st&, const double&);
  friend int operator>=(const double&, const double_st&);

  friend int operator<(const double_st&, const double_st&);
  friend int operator<(const double_st&, const double&);
  friend int operator<(const double&, const double_st&);

  friend int operator>(const double_st&, const double_st&);
  friend int operator>(const double_st&, const double&);
  friend int operator>(const double&, const double_st&);

  // MATHEMATICAL FUNCTIONS

  friend double_st  log(const double_st&);
  friend double_st  log2(const double_st&);
  friend double_st  log10(const double_st&);
  friend double_st  log1p(const double_st&);
  friend double_st  logb(const double_st&);
  friend double_st  exp(const double_st&);
  friend double_st  exp2(const double_st&);
  friend double_st  expm1(const double_st&);
  friend double_st  sqrt(const double_st&);
  friend double_st  cbrt(const double_st&);
  friend double_st  sin(const double_st&);
  friend double_st  cos(const double_st&);
  friend double_st  tan(const double_st&);
  friend double_st  asin(const double_st&);
  friend double_st  acos(const double_st&);
  friend double_st  atan(const double_st&);
  friend double_st  atan2(const double_st&, const double_st&);
  friend double_st  atan2(const double_st&, const float_st&);
  friend double_st  atan2(const float_st&, const double_st&);
  friend double_st  sinh(const double_st&);
  friend double_st  cosh(const double_st&);
  friend double_st  tanh(const double_st&);
  friend double_st  asinh(const double_st&);
  friend double_st  acosh(const double_st&);
  friend double_st  atanh(const double_st&);
  friend double_st  hypot(const double_st&, const double_st&);
  friend double_st  hypot(const double_st&, const float_st&);
  friend double_st  hypot(const float_st&, const double_st&);

  friend double_st pow(const double_st&, const double_st&);
  friend double_st pow(const double_st&, const float_st&);
  friend double_st pow(const float_st&, const double_st&);
  friend double_st pow(const double_st&, const double&);
  friend double_st pow(const double_st&, const float&);
  friend double_st pow(const double_st&, const unsigned long long&);
  friend double_st pow(const double_st&, const long long&);
  friend double_st pow(const double_st&, const unsigned long&);
  friend double_st pow(const double_st&, const long&);
  friend double_st pow(const double_st&, const unsigned int&);
  friend double_st pow(const double_st&, const int&);
  friend double_st pow(const double_st&, const unsigned short&);
  friend double_st pow(const double_st&, const short&);
  friend double_st pow(const double_st&, const unsigned char&);
  friend double_st pow(const double_st&, const char&);

  friend double_st pow(const double&, const double_st&);
  friend double_st pow(const float&, const double_st&);
  friend double_st pow(const unsigned long long&, const double_st&);
  friend double_st pow(const long long&, const double_st&);
  friend double_st pow(const unsigned long&, const double_st&);
  friend double_st pow(const long&, const double_st&);
  friend double_st pow(const unsigned int&, const double_st&);
  friend double_st pow(const int&, const double_st&);
  friend double_st pow(const unsigned short&, const double_st&);
  friend double_st pow(const short&, const double_st&);
  friend double_st pow(const unsigned char&, const double_st&);
  friend double_st pow(const char&, const double_st&);


  friend double_st fmax(const double_st&, const double_st&);
  friend double_st fmax(const double_st&, const float_st&);
  friend double_st fmax(const float_st&, const double_st&);
  friend double_st fmax(const double_st&, const double&);
  friend double_st fmax(const double_st&, const float&);
  friend double_st fmax(const double_st&, const unsigned long long&);
  friend double_st fmax(const double_st&, const long long&);
  friend double_st fmax(const double_st&, const unsigned long&);
  friend double_st fmax(const double_st&, const long&);
  friend double_st fmax(const double_st&, const unsigned int&);
  friend double_st fmax(const double_st&, const int&);
  friend double_st fmax(const double_st&, const unsigned short&);
  friend double_st fmax(const double_st&, const short&);
  friend double_st fmax(const double_st&, const unsigned char&);
  friend double_st fmax(const double_st&, const char&);

  friend double_st fmax(const double&, const double_st&);
  friend double_st fmax(const float&, const double_st&);
  friend double_st fmax(const unsigned long long&, const double_st&);
  friend double_st fmax(const long long&, const double_st&);
  friend double_st fmax(const unsigned long&, const double_st&);
  friend double_st fmax(const long&, const double_st&);
  friend double_st fmax(const unsigned int&, const double_st&);
  friend double_st fmax(const int&, const double_st&);
  friend double_st fmax(const unsigned short&, const double_st&);
  friend double_st fmax(const short&, const double_st&);
  friend double_st fmax(const unsigned char&, const double_st&);
  friend double_st fmax(const char&, const double_st&);


  friend double_st fmin(const double_st&, const double_st&);
  friend double_st fmin(const double_st&, const float_st&);
  friend double_st fmin(const float_st&, const double_st&);

  friend double_st fmin(const double_st&, const double&);
  friend double_st fmin(const double_st&, const float&);
  friend double_st fmin(const double_st&, const unsigned long long&);
  friend double_st fmin(const double_st&, const long long&);
  friend double_st fmin(const double_st&, const unsigned long&);
  friend double_st fmin(const double_st&, const long&);
  friend double_st fmin(const double_st&, const unsigned int&);
  friend double_st fmin(const double_st&, const int&);
  friend double_st fmin(const double_st&, const unsigned short&);
  friend double_st fmin(const double_st&, const short&);
  friend double_st fmin(const double_st&, const unsigned char&);
  friend double_st fmin(const double_st&, const char&);

  friend double_st fmin(const double&, const double_st&);
  friend double_st fmin(const float&, const double_st&);
  friend double_st fmin(const unsigned long long&, const double_st&);
  friend double_st fmin(const long long&, const double_st&);
  friend double_st fmin(const unsigned long&, const double_st&);
  friend double_st fmin(const long&, const double_st&);
  friend double_st fmin(const unsigned int&, const double_st&);
  friend double_st fmin(const int&, const double_st&);
  friend double_st fmin(const unsigned short&, const double_st&);
  friend double_st fmin(const short&, const double_st&);
  friend double_st fmin(const unsigned char&, const double_st&);
  friend double_st fmin(const char&, const double_st&);



  // INTRINSIC FUNCTIONS
  friend double_st fabs(const double_st&);
  friend double_st fabsf(const double_st&);
  friend double_st floorf(const double_st&);
  friend double_st ceilf(const double_st&);
  friend double_st truncf(const double_st&);
  friend double_st nearbyintf(const double_st&);
  friend double_st rintf(const double_st&);
  friend long int  lrintf(const double_st&);
  friend long long int  llrintf(const double_st&);

  // Conversion
  operator char();
  operator unsigned char();
  operator short();
  operator unsigned short();
  operator int();
  operator unsigned();
  operator long();
  operator unsigned long();
  operator long long();
  operator unsigned long long();
  operator float();
  operator double();


  int nb_significant_digit() const;
  int approx_digit() const;
  int computedzero() const;
  void display() const ;
  void display(const char *) const ;
  char* str( char *)  const ;

  friend char* strp(const double_st&);
  friend char* str( char *, const double_st&);

  friend std::istream& operator >>(std::istream& s, double_st& );

  void data_st();
  void data_st(const double&, const int &);

};


void  cadna_init(int, unsigned int, unsigned int, unsigned int);
void  cadna_init(int, unsigned int);
void  cadna_init(int);
void  cadna_end();

void cadna_enable(unsigned int);
void cadna_disable(unsigned int);

std::ostream& operator<<(std::ostream&, const double_st&);
std::istream& operator >>(std::istream&, double_st& );

std::ostream& operator<<(std::ostream&, const float_st&);
std::istream& operator >>(std::istream&, float_st& );



#endif











