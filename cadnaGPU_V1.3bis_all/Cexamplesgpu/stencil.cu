#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sys/time.h>

using namespace std;

void InitData(int Nx, int Ny, int Nz, float *A[2], float *vsq) {
    int offset = 0;
    for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x, ++offset) {
                A[0][offset] = (x < Nx / 2) ? x / float(Nx) : y / float(Ny);
                A[1][offset] = 0;
                vsq[offset] = x*y*z / float(Nx * Ny * Nz);
            }
}


int main(int argc, char **argv){
  int t;
  
  int Nx = 256, Ny = 256, Nz = 256;
  int width = 4;
  float *A[2];
  int maxIterations = 6;
  float coeff[4] = { 0.5, -.25, .125, -.0625 }; 

  struct timeval tval1, tval2;
  
  printf("Usage : %s [iter [Nx [Ny [Nz [width]]]]]\n", argv[0]);

  if (argc > 1) {
    maxIterations=atoi(argv[1]);
  }

  if (argc > 2) {
    Nx=atoi(argv[2]);
  }

  if (argc > 3) {
    Ny=atoi(argv[3]);
  }

  if (argc > 4) {
    Nz=atoi(argv[4]);
  }

  if (argc > 5) {
    width=atoi(argv[5]);
  }

  A[0] = new float [Nx * Ny * Nz];
  A[1] = new float [Nx * Ny * Nz];
  float *vsq = new float [Nx * Ny * Nz];

  InitData(Nx, Ny, Nz, A, vsq);

  gettimeofday(&tval1, NULL);

  for (int t = 0; t < maxIterations; ++t) {
    for (int z = width; z < Nz-width; ++z) {
      for (int y = width; y < Ny-width; ++y) {
	// for (int x = width; x < Nx-width; ++x) {
	for (int x = 4; x < 252; ++x) {
	  int index = (z * Ny * Nx) + (y * Nx) + x;
#define A_access(A, x, y, z) A[index + (x) + ((y) * Nx) + ((z) * Ny * Nx)]
	  float div = 
	    coeff[0] * A_access(A[t&1], 0, 0, 0) +
	    coeff[1] * (A_access(A[t&1], +1, 0, 0) + A_access(A[t&1], -1, 0, 0) +
		       A_access(A[t&1], 0, +1, 0) + A_access(A[t&1], 0, -1, 0) +
		       A_access(A[t&1], 0, 0, +1) + A_access(A[t&1], 0, 0, -1)) +
	    coeff[2] * (A_access(A[t&1], +2, 0, 0) + A_access(A[t&1], -2, 0, 0) +
		       A_access(A[t&1], 0, +2, 0) + A_access(A[t&1], 0, -2, 0) +
		       A_access(A[t&1], 0, 0, +2) + A_access(A[t&1], 0, 0, -2)) +
	    coeff[3] * (A_access(A[t&1], +3, 0, 0) + A_access(A[t&1], -3, 0, 0) +
		       A_access(A[t&1], 0, +3, 0) + A_access(A[t&1], 0, -3, 0) +
		       A_access(A[t&1], 0, 0, +3) + A_access(A[t&1], 0, 0, -3));

	  A_access(A[1-(t&1)], 0, 0, 0) = 
	    2. * A_access(A[t&1], 0, 0, 0) - 
	    A_access(A[1-(t&1)], 0, 0, 0) + 
	    vsq[index] * div;	  
	}
      }
    }
  }


  gettimeofday(&tval2, NULL);

  t = (tval2.tv_sec - tval1.tv_sec) * 1000000 + tval2.tv_usec - tval1.tv_usec;

  cout << "Time: " << t << endl;

  for (int z = width; z < width+4; ++z) {
    for (int y = width; y < width+4; ++y) {
      for (int x = width; x < width+4; ++x) {
	int index = (z * Ny * Nx) + (y * Nx) + x;
	cerr << "(" << x << "," << y << "," << z << ")" << " : " << A_access(A[1],0,0,0) << endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
