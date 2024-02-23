ARCHI=-m64
LIBFLAGS=-I../include -I../Cgpu
SMFLAG0=-arch=sm_60
SMFLAGTITAN=-arch=compute_60,code=sm_60
CFLAGS =  $(ARCHI)  $(LIBFLAGS) $(SMFLAG0)
CFLAGSTITAN =  $(ARCHI)  $(LIBFLAGS) --generate-code  arch=compute_60,code=sm_60 

CADNALDFLAGS= -L../lib -lcadnaC -DHALF

all :   add_loop_combGPU_cad_half add_loop_membGPU_cad_half mul_loop_membGPU_cad_half mul_loop_combGPU_cad_half #mandelbrot_GPU_half

add_loop_combGPU_cad_half : add_loop_GPU_cad_half.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32
	#nvcc -g -G $(CFLAGSTITAN)  $< -o $@  -DNUMCHECK 


add_loop_membGPU_cad_half : add_loop_GPU_cad_half.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1


mul_loop_combGPU_cad_half : mul_loop_GPU_cad_half.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32


mul_loop_membGPU_cad_half : mul_loop_GPU_cad_half.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1


#mandelbrot_GPU_half : mandelbrot_GPU_half.cu
#	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK

clean :
	rm -f *.o  add_loop_combGPU_cad_half add_loop_membGPU_cad_half mul_loop_membGPU_cad_half mul_loop_combGPU_cad_half #mandelbrot_GPU_half


