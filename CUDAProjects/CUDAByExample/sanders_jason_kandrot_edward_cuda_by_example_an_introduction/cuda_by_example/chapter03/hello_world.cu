/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include <iostream>
#include "../common/book.h"

__global__ void add(double a, double b, double* c){
    *c = a + b;
}

int main( void ) {
    double c;
    double* dev_c;
    cudaMalloc((void**) &dev_c, sizeof(double));
    add<<<1, 1>>>(2.4, 7.3, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(double), cudaMemcpyDeviceToHost);
    printf("2 + 7 = %f\n", c);
    cudaFree(dev_c);
    return 0;
}
