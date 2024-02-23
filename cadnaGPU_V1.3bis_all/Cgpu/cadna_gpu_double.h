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

#ifndef __CADNA_GPU__

#define __CADNA_GPU__



class double_gpu_st {

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
