This is a version of the CADNA library for GPU applications based on
CUDA environment. This version might not be bug free.

The distribution should be compiled with g++ on 64 bit systems and
has several directories. For 32 bit systems, you have to change the
makefile files and choose a suitable assembly file.


C directory 
----------- 
It contents a CADNA implementation for the CPU part of the code.
This CPU CADNA implementation is not optimized.
It is different from the optimized CPU CADNA implementation accessible from
http://cadna.lip6.fr

Assembly functions can generate problems.  2 assembly files are
proposed depending on the naming problem with g++. To choose one, you
have to type:
 
cp rnd_x86_00_64.s cadna_rounding_64.s
or
cp rnd_x86_10_64.s cadna_rounding_64.s


to compile: 

make 
and
make install

Cexamples directory
-------------------
It contains examples to test the C CPU implementation.
All examples should work before going further.


Cgpu directory
-------------- 
It contains the cadna implementation for the CUDA kernels.
Nothing to do.


CexamplesGPU directory
----------------------
muller.cu is a small example based on the Cexamples/ex4_cad.cc code.
cfmatmul.cu is a matrix product example


CexamplesGPU_half directory
----------------------
Some examples for GPU implementation of type half.


CexamplesGPU_half2 directory
----------------------
Some examples for GPU implementation of type half2.


bench directory
----------------------
Some examples for GPU implementation of type float and double to test the performance.


bench_half directory
----------------------
Some examples for GPU implementation of type half to test the performance.


bench_half2 directory
----------------------
Some examples for GPU implementation of type half2 to test the performance.

