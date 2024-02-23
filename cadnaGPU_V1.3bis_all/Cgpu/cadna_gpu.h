// Copyright 2019   J.-M. Chesneaux, P. Eberhart, F. Jezequel, J.-L. Lamotte, S. Zhou

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
#include <cadna_commun.h>
#include <cuda_fp16.h>

#ifndef __CADNA_GPU__

#define __CADNA_GPU__


class half_gpu_st {

    public:
        half x,y,z;

        typedef half TYPEBASE;

    public:
        //    mutable int accuracy;

        mutable unsigned char accuracy;
        mutable unsigned char error;

    public:

__device__  inline half_gpu_st(void){};


__device__  inline half_gpu_st(const half &a, const half &b, const half &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};
__device__  inline half_gpu_st(const half &a): x(a),y(a),z(a),accuracy(3), error(0) {};




        __device__ int nb_significant_digit() const ;
        __device__ void modify(const int&) ;
        __device__ half_gpu_st& operator=(const half&);
        __device__ operator half() {  float x_f,y_f,z_f,res_f;
		x_f = __half2float(x);
		y_f = __half2float(y);
		z_f = __half2float(z);
		res_f = (float)(x_f+y_f+z_f)/3.f;
	     	return __float2half(res_f);  }
/*
        inline __device__ int computedzero() const{
            int res;
            half x0,x1,x2,xx;

            xx=x+y+z;

            if (xx==(float)0.0) res=1;
            else {
                xx=(float)3./xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*(double)3.0854661704166664) > 0.1f; //FJ 2014
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }
*/

        inline __device__ int computedzero() const{
            int res;
            float x0,x1,x2,xx;
		float x_f,y_f,z_f;
		x_f = __half2float(x);
		y_f = __half2float(y);
		z_f = __half2float(z);
            xx=x_f+y_f+z_f;

            if (xx==(float)0.0) res=1;
            else {
                xx=(float)3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*(double)3.0854661704166664) > 0.1f; //FJ 2014
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }
/*
         inline __device__ int isnumericalnoise() const{
            float x0,x1,x2,xx;
		
	
            int res;

            xx=x+y+z;

            if (xx==0.0) { //FJ
                if ((x != y ) || (x != z)) res=1;
                else res=0;
            }
            else {
                xx=3./xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f;
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }


        inline __device__ int approx_digit() const{
            float x0,x1,x2,xx;
	   
	    accuracy=NOT_NUMERICAL_NOISE; 
            xx=x+y+z;

            if (xx==0.0) { //FJ
                if ((x != y ) || (x != z)) accuracy=0;
            }
            else {
                xx=(float)3./xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                if (x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02)
                    accuracy=0;
            }
            return accuracy; 
        }
*/
       inline __device__ int isnumericalnoise() const{
            float x0,x1,x2,xx;
		float x_f,y_f,z_f;
		x_f = __half2float(x);
		y_f = __half2float(y);
		z_f = __half2float(z);

	
            int res;

            xx=x_f+y_f+z_f;

            if (xx==0.0) { //FJ
                if ((x_f != y_f ) || (x_f != z_f)) res=1;
                else res=0;
            }
            else {
                xx=3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f;
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }


        inline __device__ int approx_digit() const{
            float x0,x1,x2,xx;
	   	float x_f,y_f,z_f;
		x_f = __half2float(x);
		y_f = __half2float(y);
		z_f = __half2float(z);

	    accuracy=NOT_NUMERICAL_NOISE; 
            xx=x_f+y_f+z_f;

            if (xx==0.0) { //FJ
                if ((x_f != y_f ) || (x_f != z_f)) accuracy=0;
            }
            else {
                xx=(float)3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                if (x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02)
                    accuracy=0;
            }
            return accuracy; 
        }


        // refus de NVCC en 3.0 beta
        //   friend double_gpu_st operator+(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator-(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator*(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator/(const double_gpu_st&, const double_gpu_st&);

};

class half2_gpu_st {

    public:
        half2 x,y,z;

        typedef half2 TYPEBASE;

    public:
        //    mutable int accuracy;

        mutable unsigned char accuracy;
        mutable unsigned char error;

    public:

__device__  inline half2_gpu_st(void){};


__device__  inline half2_gpu_st(const half2 &a, const half2 &b, const half2 &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};
__device__  inline half2_gpu_st(const half2 &a): x(a),y(a),z(a),accuracy(3), error(0) {};




        __device__ int nb_significant_digit() const ;
        __device__ void modify(const int&) ;
        __device__ half2_gpu_st& operator=(const half2&);
        __device__ operator half2() {  float x_f,y_f,z_f,res_f;
		x_f = __low2float(x)+__high2float(x);
		y_f = __low2float(y)+__high2float(y);
		z_f = __low2float(z)+__high2float(z);
		res_f = (float)(x_f+y_f+z_f)/3.f;
	     	return __float2half2_rn(res_f);  }

	inline __device__ int computedzero() const{
            int res;
            float x0,x1,x2,xx;
		float x_f,y_f,z_f;
		x_f = __low2float(x)+__high2float(x);
		y_f = __low2float(y)+__high2float(y);
		z_f = __low2float(z)+__high2float(z);

            xx=x_f+y_f+z_f;

            if (xx==(float)0.0) res=1;
            else {
                xx=(float)3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*(double)3.0854661704166664) > 0.1f; //FJ 2014
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }

 	inline __device__ int isnumericalnoise() const{
            float x0,x1,x2,xx;
		float x_f,y_f,z_f;
		x_f = __low2float(x)+__high2float(x);
		y_f = __low2float(y)+__high2float(y);
		z_f = __low2float(z)+__high2float(z);

	
            int res;

            xx=x_f+y_f+z_f;

            if (xx==0.0) { //FJ
                if ((x_f != y_f ) || (x_f != z_f)) res=1;
                else res=0;
            }
            else {
                xx=3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f;
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }


        inline __device__ int approx_digit() const{
            float x0,x1,x2,xx;
	   	float x_f,y_f,z_f;
		x_f = __low2float(x)+__high2float(x);
		y_f = __low2float(y)+__high2float(y);
		z_f = __low2float(z)+__high2float(z);

	    accuracy=NOT_NUMERICAL_NOISE; 
            xx=x_f+y_f+z_f;

            if (xx==0.0) { //FJ
                if ((x_f != y_f ) || (x_f != z_f)) accuracy=0;
            }
            else {
                xx=(float)3./xx;
                x0=x_f*xx-(float)1;
                x1=y_f*xx-(float)1;
                x2=z_f*xx-(float)1;
                if (x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02)
                    accuracy=0;
            }
            return accuracy; 
        }


        // refus de NVCC en 3.0 beta
        //   friend double_gpu_st operator+(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator-(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator*(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator/(const double_gpu_st&, const double_gpu_st&);

};


class float_gpu_st {

    public:
        float x,y,z;

        typedef float TYPEBASE;

    public:
        mutable unsigned char accuracy;
        mutable unsigned char error;
        unsigned char pad1, pad2;

    public:

__device__  inline float_gpu_st(void){};


__device__  inline float_gpu_st(const double &a, const double &b, const double &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};
__device__  inline float_gpu_st(const float &a): x(a),y(a),z(a),accuracy(7), error(0) {};

__device__  inline float_gpu_st(const float &a, const float &b, const float &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};
__device__  inline float_gpu_st(const double &a): x(a),y(a),z(a),accuracy(7), error(0) {};





        __device__ int nb_significant_digit() const ;
        __device__ void modify(const int&) ;
        __device__ float_gpu_st& operator=(const float&);
        __device__ operator float() {     return ((float)(x+y+z)/3.f);  }

        inline __device__ int computedzero() const{
            int res;
            float x0,x1,x2,xx;

            xx=x+y+z;

            if (xx==(float)0.0) res=1;
            else {
                xx=(float)3./xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*(float)3.0854661704166664) > 0.1f; //FJ 2014
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }
        inline __device__ int isnumericalnoise() const{
            float x0,x1,x2,xx;
            int res;

            xx=x+y+z;

            if (xx==0.0f) {
                if ((x != y ) || (x != z)) res=1;
                else res=0;
            }
            else {
                xx=(float)3.f/xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                //      res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f;
                res=x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }


        inline __device__ int approx_digit() const{
            float x0,x1,x2,xx;

            accuracy=NOT_NUMERICAL_NOISE;
            xx=(x+y+z);

            if (xx==0.0f) {
                if ((x != y ) || (x != z)) accuracy=0;
            }
            else {
                xx=(float)3.f/xx;
                x0=x*xx-(float)1;
                x1=y*xx-(float)1;
                x2=z*xx-(float)1;
                //  cuPrintf("approx %f\n",((x0*x0+x1*x1+x2*x2)*(float)3.081666666666666666));
                //      if (((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f)
                if (x0*x0+x1*x1+x2*x2 > (float)3.241001342318910E-02)
                    accuracy=0;
            }

            return accuracy;
        }



        // refus de NVCC en 3.0 beta
        //   friend float_gpu_st operator+(const float_gpu_st&, const float_gpu_st&);
        //   friend float_gpu_st operator-(const float_gpu_st&, const float_gpu_st&);
	//   friend float_gpu_st operator*(const float_gpu_st&, const float_gpu_st&);
        //   friend float_gpu_st operator/(const float_gpu_st&, const float_gpu_st&);

};

class double_gpu_st{

    public:
        double x,y,z;

        typedef double TYPEBASE;

    public:
        //    mutable int accuracy;

        mutable unsigned char accuracy;
        mutable unsigned char error;

        unsigned char pad1, pad2, pad3, pad4, pad5, pad6;

    public:

__device__  inline double_gpu_st(void){};


__device__  inline double_gpu_st(const double &a, const double &b, const double &c):x(a),y(b),z(c),accuracy(DIGIT_NOT_COMPUTED), error(0) {};
__device__  inline double_gpu_st(const double &a): x(a),y(a),z(a),accuracy(15), error(0) {};





        __device__ int nb_significant_digit() const ;
        __device__ void modify(const int&) ;
        __device__ double_gpu_st& operator=(const double&);
        __device__ operator double() {     return ((double)(x+y+z)/3.);  }

        inline __device__ int computedzero() const{
            int res;
            double x0,x1,x2,xx;

            xx=x+y+z;

            if (xx==(double)0.0) res=1;
            else {
                xx=(double)3./xx;
                x0=x*xx-(double)1;
                x1=y*xx-(double)1;
                x2=z*xx-(double)1;
                //      res=((x0*x0+x1*x1+x2*x2)*(double)3.0854661704166664) > 0.1f; //FJ 2014
                res=x0*x0+x1*x1+x2*x2 > (double)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }

        inline __device__ int isnumericalnoise() const{
            double x0,x1,x2,xx;
            int res;

            xx=x+y+z;

            if (xx==0.0) { //FJ
                if ((x != y ) || (x != z)) res=1;
                else res=0;
            }
            else {
                xx=3./xx;
                x0=x*xx-(double)1;
                x1=y*xx-(double)1;
                x2=z*xx-(double)1;
                //      res=((x0*x0+x1*x1+x2*x2)*3.0854661704166664f) > 0.1f;
                res=x0*x0+x1*x1+x2*x2 > (double)3.241001342318910E-02; //FJ 2014
            }
            return res;
        }


        inline __device__ int approx_digit() const{
            double x0,x1,x2,xx;
	   
	    accuracy=NOT_NUMERICAL_NOISE; 
            xx=x+y+z;

            if (xx==0.0) { //FJ
                if ((x != y ) || (x != z)) accuracy=0;
            }
            else {
                xx=(double)3./xx;
                x0=x*xx-(double)1;
                x1=y*xx-(double)1;
                x2=z*xx-(double)1;
                if (x0*x0+x1*x1+x2*x2 > (double)3.241001342318910E-02)
                    accuracy=0;
            }
            return accuracy; 
        }


        // refus de NVCC en 3.0 beta
        //   friend double_gpu_st operator+(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator-(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator*(const double_gpu_st&, const double_gpu_st&);
        //   friend double_gpu_st operator/(const double_gpu_st&, const double_gpu_st&);

};



__device__ void _cadna_init_table_gpu();


#endif
