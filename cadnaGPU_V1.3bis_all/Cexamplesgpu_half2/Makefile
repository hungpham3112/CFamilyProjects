

ARCHI=-m64
LIBFLAGS=-I../include -I../Cgpu
#SMFLAG0=-arch=sm_35
SMFLAG0=-arch=sm_60

#SMFLAGTITAN=-arch=compute_35,code=sm_35
SMFLAGTITAN=-arch=compute_60,code=sm_60
#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu  -arch=sm_13
#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu
CFLAGS =  $(ARCHI)  $(LIBFLAGS) $(SMFLAG0) #-I../include -I../Cgpu  -arch=sm_20
#CFLAGSTITAN =  $(ARCHI)  $(LIBFLAGS) --generate-code  arch=compute_35,code=sm_35 #-I../include -I../Cgpu  -arch=sm_20
CFLAGSTITAN =  $(ARCHI)  $(LIBFLAGS) --generate-code  arch=compute_60,code=sm_60 #-I../include -I../Cgpu  -arch=sm_20 


#CFLAGS =  $(ARCHI)  -I../include -I../Cgpu  -O0 -Xopencc=-O0
CADNALDFLAGS=  -L../lib -lcadnaC -DHALF



all :  add_loop_GPU_cad  add_half2_GPU_cad add_half2_GPU_cad  add_half2_GPU  test  add_half2_GPU add_loop_combGPU_cad_half2 add_loop_membGPU_cad_half2 mul_loop_membGPU_cad_half2 mul_loop_combGPU_cad_half2 #mullerTest muller #cfmatmul 
#essai_rndg


#half_ameliorate : half_ameliorate.cu
#	nvcc -g -G $(CFLAGSTITAN)  $< -o $@  $(CADNALDFLAGS) -O3 -DCADNA





add_loop_GPU_cad : add_loop_GPU_cad.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@  $(CADNALDFLAGS) -O3 -DCADNA

muller : muller.cu
	nvcc -v $(CFLAGS) muller.cu -o muller   $(CADNALDFLAGS) 

mullerTest : mullerTest.cu
	nvcc -g -G -v $(CFLAGS) mullerTest.cu -o mullerTest   $(CADNALDFLAGS)





#cmatmul : cmatmul.cu matmul.h
#	nvcc  $(CFLAGS) $< -o $@   $(CADNALDFLAGS)


fmatmul : cfmatmul.cu
	nvcc  $(CFLAGS) $< -o $@


cfmatmul : cfmatmul.cu
	nvcc -g -G -DDISPLAY  $(CFLAGS) $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA

test : test.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DDISPLAY -DCADNA 


add_half2_GPU : add_half2_GPU.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@ 
#add_half2_GPU_float2 : add_half2_GPU_float2.cu
#	nvcc -g -G $(CFLAGSTITAN)  $< -o $@ 


matmul_half_test : matmul_half_test.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DDISPLAY -DCADNA 










add_half2_GPU_cad : add_half2_GPU_cad.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DDISPLAY -DCADNA 
#add_half2_GPU_cad_float2 : add_half2_GPU_cad_float2.cu
#	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DDISPLAY -DCADNA 



add_loop_combGPU_cad_half2 : add_loop_GPU_cad_half2.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32
	#nvcc -g -G $(CFLAGSTITAN)  $< -o $@  -DNUMCHECK 



add_loop_membGPU_cad_half2 : add_loop_GPU_cad_half2.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1



mul_loop_combGPU_cad_half2 : mul_loop_GPU_cad_half2.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32 



mul_loop_membGPU_cad_half2 : mul_loop_GPU_cad_half2.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1







essai_rndg : essai_rndg.cu
	nvcc -v $(CFLAGS) essai_rndg.cu -o essai_rndg   $(CADNALDFLAGS)

clean :
	rm -f *.o  add_loop_GPU_cad add_half2_GPU_cad add_half2_GPU_cad  add_half2_GPU  test  add_half2_GPU cfmatmul  add_loop_combGPU_cad_half2 add_loop_membGPU_cad_half2 mul_loop_membGPU_cad_half2 mul_loop_combGPU_cad_half2 #mullerTest muller #essai_rndg

