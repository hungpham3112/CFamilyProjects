#define __REDUCTION_H__
#ifndef __REDUCTION_H__

template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata);

template void reduce<float>(int size, int threads, int blocks, int whichKernel,
                            float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, int whichKernel,
                             double *d_idata, double *d_odata);
#endif
