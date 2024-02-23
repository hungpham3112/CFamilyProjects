

ARCHI=-m64
LIBFLAGS=-I../include -I../Cgpu
SMFLAG0=-arch=sm_60
SMFLAGTITAN=-arch=compute_60,code=sm_60
#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu  -arch=sm_13
#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu
CFLAGS =  $(ARCHI)  $(LIBFLAGS) $(SMFLAG0) #-I../include -I../Cgpu  -arch=sm_20
CFLAGSTITAN =  $(ARCHI)  $(LIBFLAGS) --generate-code  arch=compute_60,code=sm_60 #-I../include -I../Cgpu  -arch=sm_20


#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu  -O0 -Xopencc=-O0
CADNALDFLAGS= -L../lib -lcadnaC


all :   cfmatmul  muller 
#essai_rndg


muller : muller.cu
	nvcc -v $(CFLAGS) muller.cu -o muller   $(CADNALDFLAGS) 


#mullerTest : mullerTest.cu
#	nvcc -g -G -v $(CFLAGS) mullerTest.cu -o mullerTest   $(CADNALDFLAGS)


#cmatmul : cmatmul.cu matmul.h
#	nvcc  $(CFLAGS) $< -o $@   $(CADNALDFLAGS)


#fmatmul : cfmatmul.cu
#	nvcc  $(CFLAGS) $< -o $@


cfmatmul : cfmatmul.cu
	nvcc -g -G -DDISPLAY  $(CFLAGS) $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA


clean :
	rm -f *.o muller cfmatmul 
