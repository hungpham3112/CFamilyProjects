

ARCHI=-m64
LIBFLAGS=-I../include -I../Cgpu
SMFLAG0=-arch=sm_60
SMFLAGTITAN=-arch=compute_60,code=sm_60
CFLAGS =  $(ARCHI)  $(LIBFLAGS) $(SMFLAG0)
CFLAGSTITAN =  $(ARCHI)  $(LIBFLAGS) --generate-code  arch=compute_60,code=sm_60


CADNALDFLAGS= -L../lib -lcadnaC



all : add_loop_combGPU_cad_float add_loop_membGPU_cad_float mul_loop_membGPU_cad_float mul_loop_combGPU_cad_float mandelbrot_GPU_float add_loop_combGPU_cad_double add_loop_membGPU_cad_double mul_loop_membGPU_cad_double mul_loop_combGPU_cad_double mandelbrot_GPU_double



add_loop_combGPU_cad_float : add_loop_GPU_cad_float.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32

add_loop_membGPU_cad_float : add_loop_GPU_cad_float.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1

mul_loop_combGPU_cad_float : mul_loop_GPU_cad_float.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32

mul_loop_membGPU_cad_float : mul_loop_GPU_cad_float.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1

mandelbrot_GPU_float : mandelbrot_GPU_float.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK







add_loop_combGPU_cad_double : add_loop_GPU_cad_double.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32

add_loop_membGPU_cad_double : add_loop_GPU_cad_double.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1

mul_loop_combGPU_cad_double : mul_loop_GPU_cad_double.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=32

mul_loop_membGPU_cad_double : mul_loop_GPU_cad_double.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK -DNLOOP=1

mandelbrot_GPU_double : mandelbrot_GPU_double.cu
	nvcc -g -G $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DCADNA -DNUMCHECK


clean :
	rm -f *.o mandelbrot_GPU_float add_loop_combGPU_cad_float add_loop_membGPU_cad_float mul_loop_combGPU_cad_float mul_loop_membGPU_cad_float add_loop_combGPU_cad_double add_loop_membGPU_cad_double mul_loop_membGPU_cad_double mul_loop_combGPU_cad_double mandelbrot_GPU_double


