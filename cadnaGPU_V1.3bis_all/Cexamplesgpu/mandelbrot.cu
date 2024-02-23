#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"



#define N 1024

#ifndef TAILLE_BLOC_X
#define TAILLE_BLOC_X 32
#endif

#ifndef TAILLE_BLOC_Y
#define TAILLE_BLOC_Y 6 //32
#endif

#ifdef CADNA
#include <cadna.h>
#include "cadna_gpu.cu"
#define DATATYPE float
#define DATATYPECADNA float_st
#define DATATYPECADNAGPU float_gpu_st
#else
#define DATATYPE float
#define DATATYPECADNA float
#define DATATYPECADNAGPU float
#endif

#ifndef SEED
#define SEED 1
#endif


using namespace std;



// Mesures :
double my_gettimeofday(){
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}



/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width*height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        // char c = (buf[i] & 0x1) ? 240 : 20;
        char c = buf[i] % 256;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

__global__ void mandelbrotKernel(int *d_buf, int width, int height, float x0, float y0, float dx, float dy, int maxIterations){
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int j = blockDim.y*blockIdx.y+threadIdx.y;

    if (i < width && j < height){
        DATATYPECADNAGPU x = x0 + i * dx;
        DATATYPECADNAGPU y = y0 + j * dy;
        int k;
        DATATYPECADNAGPU z_re = x; //st
        DATATYPECADNAGPU z_im = y; //st

        for (k = 0; k < maxIterations; ++k) {
            if (z_re * z_re + z_im * z_im > 4.f)
                break;
            DATATYPECADNAGPU new_re = z_re*z_re - z_im*z_im; //st
            DATATYPECADNAGPU new_im = 2.f * z_re * z_im; //st
            z_re = x + new_re;
            z_im = y + new_im;
        }

        int index = (j * width + i);
        d_buf[index] = k;
    }
}


/*

   __host__ void mandelbrotCPU(int *buf, int width, int height){
   for (int j = 0; j < height; j++) {
   for (int i = 0; i < width; ++i) { /// 768
// for (int i = 0; i < width; ++i) {
float x = x0 + i * dx;
float y = y0 + j * dy;
int k;
float z_re = x;
float z_im = y;


for (k = 0; k < maxIterations; ++k) {
if (z_re * z_re + z_im * z_im > 4.f)
break;

float new_re = z_re*z_re - z_im*z_im;
float new_im = 2.f * z_re * z_im;
z_re = x + new_re;
z_im = y + new_im;
}

int index = (j * width + i);
buf[index] = k;

}
}
}

 */


int main(int argc, char **argv){

    int Cadna = 0;
#ifdef CADNA
    cadna_init(-1, CADNA_INTRINSIC);
    Cadna = 1;
#endif
    FILE* fic=fopen("mandelbrotBY.res", "a");

    double t_startGPU, t_endGPU, t_GPU;

    unsigned int width = 768;
    unsigned int height = 512;
    unsigned long int nbPix = width*height;

    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;
    int maxIterations = 256;

    printf("Usage : %s [iter [width [height]]]\n", argv[0]);

    if (argc > 1) {
        maxIterations=atoi(argv[1]);
    }

    if (argc > 2) {
        width=atoi(argv[2]);
    }

    if (argc > 3) {
        height=atoi(argv[3]);
    }

    long int size = width*height*sizeof(int);

    int *buf = new int[width*height];
    int *d_buf;

    cudaMalloc((void **) &d_buf, size);

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    dim3 threadsPerBlock(16,16);
    int nbbx = (int)ceil((float)nbPix/(float)16);
    int nbby = (int)ceil((float)nbPix/(float)16);
    dim3 numBlocks(nbbx , nbby);

     t_startGPU = my_gettimeofday();
    mandelbrotKernel<<< numBlocks , threadsPerBlock>>>(d_buf, width, height, x0, y0, dx, dy, maxIterations);
    cudaThreadSynchronize();
    //  cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    t_endGPU = my_gettimeofday();

    t_GPU = t_endGPU - t_startGPU;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) { /// 768
            // for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;
            int k;
            float z_re = x;
            float z_im = y;


            for (k = 0; k < maxIterations; ++k) {
                if (z_re * z_re + z_im * z_im > 4.f)
                    break;

                float new_re = z_re*z_re - z_im*z_im;
                float new_im = 2.f * z_re * z_im;
                z_re = x + new_re;
                z_im = y + new_im;
            }

            int index = (j * width + i);
            buf[index] = k;
        }
        }

//        writePPM(buf, width, height, "ieee.ppm");


        // Rapatriement de C:
        cudaMemcpy(buf, d_buf, size, cudaMemcpyDeviceToHost);




        writePPM(buf, width, height, "ieeeGPU.ppm");


        cout << "Time: " << t_GPU << endl;


    fprintf(fic, "%d %f %d %d\n", TAILLE_BLOC_Y, t_GPU, N, Cadna);
    fclose(fic);
#ifdef CADNA
        cadna_end();
#endif

return EXIT_SUCCESS;
    }
