#! /bin/bash
LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"
PATH="/usr/local/cuda/bin:$PATH"

for i in {32..1024..32}; do
    for j in {1..32..1}; do
	if (($i*$j <= 1024)); then
	    #nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -L../lib -lcadnaC -O3 -DCADNA mandelbrot_GPU_half.cu -o mandelbrot_GPU_half -DTAILLE_BLOC_X="$i" -DTAILLE_BLOC_Y="$j" 
	    nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 mandelbrot_GPU_half.cu -o mandelbrot_GPU_half -DTAILLE_BLOC_X="$i" -DTAILLE_BLOC_Y="$j" 
	    ./mandelbrot_GPU_half 2>> tune_mandelbrot_half.log
	    #sleep 10
	    sleep 1
	fi
    done
done

