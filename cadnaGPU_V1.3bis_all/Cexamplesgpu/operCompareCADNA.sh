#! /bin/bash
LD_LIBRARY_PATH="/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}"
PATH="/usr/local/cuda/bin:$PATH"

rm operCompareCADNA.dat
 	#nvcc $(CFLAGSTITAN)  $< -o $@   $(CADNALDFLAGS) -O3 -DDEBUG -DDISPLAY -DCADNA
    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35 add_loop_combGPU_cad.cu -o add_loop_combGPU_cad   -L../lib -lcadnaC -O3 -DDEBUG -DDISPLAY #-DCADNA
    echo `./add_loop_combGPU_cad` >> operCompareCADNA.dat
    echo "Add Comb Sans CADNA Fini."

    nvcc -m64  -I../include -I../Cgpu --generate-code  arch=compute_35,code=sm_35 add_loop_combGPU_cad.cu -o add_loop_combGPU_cad   -L../lib -lcadnaC -O3 -DDEBUG -DDISPLAY -DCADNA
    echo `./add_loop_combGPU_cad` >> operCompareCADNA.dat
    echo "Add Comb CADNA Fini."

echo "Terminé. ---------------------- OperCompareCADNA.sh"
