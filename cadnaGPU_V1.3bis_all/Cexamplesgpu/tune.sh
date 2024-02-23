#! /bin/bash
LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"
PATH="/usr/local/cuda/bin:$PATH"

DATE=`date +%d%m`

rm addCombBX.dat addCombBX.res
for i in {32..1024..32}; do
    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35   add_loop_combGPU_cad.cu -o add_loop_combGPU_cad   -L../lib -lcadnaC -O3 -DDISPLAY -DCADNA -D TAILLE_BLOC_X="$i"
    echo `./add_loop_combGPU_cad` >> addCombBX.dat
    echo "$i"
done

rm addMembBX.dat addMembBX.res
for i in {32..1024..32}; do
    #nvcc -o add_loop_membGPU_cad add_loop_membGPU_cad.cu --generate-code arch=compute_35,code=sm_35 -O3 -D TAILLE_BLOC_X="$i" -D CADNA
    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35   add_loop_membGPU_cad.cu -o add_loop_membGPU_cad   -L../lib -lcadnaC -O3 -DDISPLAY -DCADNA -D TAILLE_BLOC_X="$i"
    echo `./add_loop_membGPU_cad` >> addMembBX.dat
    echo "$i"
done

rm mulCombBX.dat mulCombBX.res
for i in {32..1024..32}; do
    #nvcc -o mul_loop_combGPU_cad mul_loop_combGPU_cad.cu --generate-code arch=compute_35,code=sm_35 -O3 -D TAILLE_BLOC_X="$i" -D CADNA
     nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35   mul_loop_combGPU_cad.cu -o mul_loop_combGPU_cad   -L../lib -lcadnaC -O3 -DDISPLAY -DCADNA -D TAILLE_BLOC_X="$i"
    echo `./mul_loop_combGPU_cad` >> mulCombBX.dat
    echo "$i"
done

rm mulmembbx.dat mulmembbx.res
for i in {32..1024..32}; do
    #nvcc -o mul_loop_membGPU_cad mul_loop_membGPU_cad.cu --generate-code arch=compute_35,code=sm_35 -O3 -D TAILLE_BLOC_X="$i" -D CADNA
    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35   mul_loop_membGPU_cad.cu -o mul_loop_membGPU_cad   -L../lib -lcadnaC -O3 -DDISPLAY -DCADNA -D TAILLE_BLOC_X="$i"
    echo `./mul_loop_membGPU_cad` >> mulMembBX.dat
    echo "$i"
done

# MANDELBROT :
rm mandelbrotBY.dat mandelbrotBY.res
for i in {1..32}; do
	#nvcc $(CFLAGSTITAN)  mandelbrot.cu -o mandelbrot   $(CADNALDFLAGS) -O3 -DCADNA -D TAILLE_BLOC_Y="$i"
    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35   mandelbrot.cu -o mandelbrot   -L../lib -lcadnaC -O3 -DCADNA -D TAILLE_BLOC_Y="$i"

    echo `./mandelbrot` >> mandelbrotBY.dat
    echo "$i"
done

# Trensfert :
mkdir Tune$DATE && mv *.res *.dat Tune$DATE

echo "Terminé."
