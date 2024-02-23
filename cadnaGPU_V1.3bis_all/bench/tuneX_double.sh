#! /bin/bash
LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"
PATH="/usr/local/cuda/bin:$PATH"

for i in {32..1024..32}; do
    #nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -L../lib -lcadnaC -O3 -DCADNA -DNLOOP=32 add_loop_GPU_cad_double.cu -o add_loop_membGPU_cad_double -DTAILLE_BLOC_X="$i"
    nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -DNLOOP=32 add_loop_GPU_cad_double.cu -o add_loop_membGPU_cad_double -DTAILLE_BLOC_X="$i"
    ./add_loop_membGPU_cad_double 2>> tune_add_loop_memb_double.log
    #sleep 10
    sleep 1
done

for i in {32..1024..32}; do
    #nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -L../lib -lcadnaC -O3 -DCADNA -DNLOOP=4096 add_loop_GPU_cad_double.cu -o add_loop_combGPU_cad_double -DTAILLE_BLOC_X="$i"
    nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -DNLOOP=4096 add_loop_GPU_cad_double.cu -o add_loop_combGPU_cad_double -DTAILLE_BLOC_X="$i"
    ./add_loop_combGPU_cad_double 2>> tune_add_loop_comb_double.log
    #sleep 10
    sleep 1
done

for i in {32..1024..32}; do
    #nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -L../lib -lcadnaC -O3 -DCADNA -DNLOOP=32 mul_loop_GPU_cad_double.cu -o mul_loop_membGPU_cad_double -DTAILLE_BLOC_X="$i"
    nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -DNLOOP=32 mul_loop_GPU_cad_double.cu -o mul_loop_membGPU_cad_double -DTAILLE_BLOC_X="$i"
    ./mul_loop_membGPU_cad_double 2>> tune_mul_loop_memb_double.log
    #sleep 10
    sleep 1
done

for i in {32..1024..32}; do
    #nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -L../lib -lcadnaC -O3 -DCADNA -DNLOOP=4096 mul_loop_GPU_cad_double.cu -o mul_loop_combGPU_cad_double -DTAILLE_BLOC_X="$i"
    nvcc -I../include -I../Cgpu --generate-code arch=compute_60,code=sm_60 -DNLOOP=4096 mul_loop_GPU_cad_double.cu -o mul_loop_combGPU_cad_double -DTAILLE_BLOC_X="$i"
    ./mul_loop_combGPU_cad_double 2>> tune_mul_loop_comb_double.log
    #sleep 10
    sleep 1
done
