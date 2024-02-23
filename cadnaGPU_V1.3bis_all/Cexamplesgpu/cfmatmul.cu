#define DIMMAT 4


#include <stdio.h>
#include <sys/time.h>


double my_gettimeofday()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

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

__global__ void matMulKernelNormal(DATATYPECADNAGPU* mat1,
        DATATYPECADNAGPU* mat2,
        DATATYPECADNAGPU* matRes,
        int dim) {

    unsigned int x = blockDim.x*blockIdx.x+threadIdx.x;
    unsigned int y = blockDim.y*blockIdx.y+threadIdx.y;

#ifdef CADNA
    cadna_init_gpu();
#endif

    if (x < dim && y < dim){
        DATATYPECADNAGPU temp;

        temp=0;
        for(int i=0; i<dim;i++){
            //cuPrintf("%3d    mat1 = %e mat2 =%e\n",i,mat1[y*dim+i].x,mat2[i*dim+x].x);
            temp = temp + mat1[y * dim + i] * mat2[i * dim + x];
            //     cuPrintf("%3d    mat1 =%d mat2 =%d\n",i,mat1[y*dim+i].accuracy,mat2[i*dim+x].accuracy);
        }
        matRes[y * dim + x] = temp;
    }
}

#define TILEDIM_MM 8

__global__ void matMulKernelSM(
        DATATYPECADNAGPU* mat1,
        DATATYPECADNAGPU* mat2,
        DATATYPECADNAGPU* matRes,
        int dim
        )
{

    //coordonnees du block dans la grille
    unsigned int bRow = blockIdx.y;
    unsigned int bCol = blockIdx.x;
    //coordonnees du thread dans le block
    unsigned int row = threadIdx.y;
    unsigned int col = threadIdx.x;

    if ((blockDim.x*blockIdx.x+threadIdx.x)<dim && (blockDim.y*blockIdx.y+threadIdx.y)<dim){
        DATATYPECADNAGPU* subRes = &matRes[bRow*dim*blockDim.x+bCol*blockDim.x];
        DATATYPECADNAGPU value;
        value=0;

        for (int k=0; k<dim/TILEDIM_MM; k++){
            DATATYPECADNAGPU* subMat1 = &mat1[bRow*dim*blockDim.x+k*blockDim.x];
            DATATYPECADNAGPU* subMat2 = &mat2[k*dim*blockDim.x+bCol*blockDim.x];

            __shared__ DATATYPECADNAGPU SM_M1[TILEDIM_MM][TILEDIM_MM];
            __shared__ DATATYPECADNAGPU SM_M2[TILEDIM_MM][TILEDIM_MM];

            SM_M1[row][col] = subMat1[row*dim+col];
            SM_M2[row][col] = subMat2[row*dim+col];

            __syncthreads();

            for (int s=0 ; s<TILEDIM_MM ; s++)
                value = value +  SM_M1[row][s]*SM_M2[s][col];

            __syncthreads();
        }
        subRes[row* dim+col] = value;
    }
}


DATATYPECADNA mat1[DIMMAT][DIMMAT], mat2[DIMMAT][DIMMAT],
              res[DIMMAT][DIMMAT], res1[DIMMAT][DIMMAT];

int main()
{
#ifdef CADNA
    cadna_init(-1);
#endif
    DATATYPECADNAGPU *d_mat1, *d_mat2, *d_res;
    double t1, t2;
    int i,j;

    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++){
            mat1[i][j]=(float)i*DIMMAT+j;
            mat2[i][j]=(float)1;
        }
    }

#ifdef CADNA
    mat1[2][1]=float_st(-101., 20., 25.);
    mat2[1][1]=float_st(-101., 1., 100.);
#else
    mat1[2][1]=(float)0;
    mat2[1][1]=(float)0;
#endif

    int size = DIMMAT * DIMMAT * sizeof(DATATYPECADNA);

    cudaMalloc((void **) &d_mat1, size);
    cudaMalloc((void **) &d_mat2, size);
    cudaMalloc((void **) &d_res,  size);

    dim3 threadsPerBlock(16,16);
    int nbbx = (int)ceil((float)DIMMAT/(float)16);
    int nbby = (int)ceil((float)DIMMAT/(float)16);
    dim3 numBlocks(nbbx , nbby);
    ////////////////
    t1=my_gettimeofday();
    cudaMemcpy(d_mat1, mat1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, size, cudaMemcpyHostToDevice);
    t2=my_gettimeofday();

    printf("Transfer host -> GPU %f\n",t2-t1);

    t1=my_gettimeofday();

    matMulKernelNormal<<< numBlocks , threadsPerBlock>>>(d_mat1, d_mat2, d_res, DIMMAT);
    cudaThreadSynchronize();

    t2=my_gettimeofday();
    printf("Computation by kernel,  time: %f   GFLOPS %f \n",t2-t1, (float)DIMMAT*DIMMAT*DIMMAT/(t2-t1)/1.e9);

    t1=my_gettimeofday();
    cudaMemcpy(res, d_res,  size, cudaMemcpyDeviceToHost);
    t2=my_gettimeofday();
    printf("Transfer GPU -> host %f\n",t2-t1);
    ////////////////////////

    t1=my_gettimeofday();
    matMulKernelSM<<< numBlocks , threadsPerBlock>>>(d_mat1, d_mat2, d_res, DIMMAT);
    cudaThreadSynchronize();
    t2=my_gettimeofday();

    printf("Computation by kernel SM, time: %f   GFLOPS %f \n",t2-t1, (float)DIMMAT*DIMMAT*DIMMAT/(t2-t1)/1.e9);

    t1=my_gettimeofday();
    cudaMemcpy(res1, d_res,  size, cudaMemcpyDeviceToHost);
    t2=my_gettimeofday();
    printf("Transfer GPU -> host %f\n",t2-t1);

#ifdef CADNA
    //  printf("fin calcul %s %s\n",strp(res[0][0]), strp(res1[0][0]));
#ifdef DISPLAY
   printf("mat1:\n");
   for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
            printf("%s ", strp(mat1[i][j]));
        printf("\n");
    }
   printf("\nmat2:\n");
    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
            printf("%s ", strp(mat2[i][j]));
        printf("\n");
    }
    printf("\nres:\n");
    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
           printf("%s ", strp(res[i][j]));
        printf("\n");
    }
#endif
    cadna_end();
#else
    //  printf("fin calcul %e %e\n",res[0][0],res1[0][0]);
#ifdef DISPLAY
    printf("mat1:\n");
    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
            printf("%e ", (mat1[i][j]));
        printf("\n");
    }
   printf("\nmat2:\n");
    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
            printf("%e ", (mat2[i][j]));
        printf("\n");
    }
   printf("\nres:\n");
    for(i=0;i<DIMMAT;i++){
        for(j=0;j<DIMMAT;j++)
            printf("%e ", (res[i][j]));
        printf("\n");
    }
#endif


#endif

    /*   for(i=0;i<DIMMAT;i++) */
    /*     for(j=0;j<DIMMAT;j++) */
    /*       if (res[i][j]!=res1[i][j])  */
    /* #ifdef CADNA */
    /* 	printf("%4d %4d %s %s\n",i,j,strp(res[i][j]), strp(res1[i][j]));  */
    /* #else	 */
    /*   printf("%4d %4d %f %f\n",i,j,res[i][j], res1[i][j]);  */
    /* #endif */
    //   cadna_end();
    return 0;
}
