/* This may look like nonsense, but really is -*- mode: C -*- */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int i;
    double A[5] = {
        [0] = 9.0,
        [1] = 2.8,
        [4] = 3.E+25,
        [3] = .00004,
    };
    for (i = 0; i < 5; ++i)
    {
        printf("A[%d] is %g, its square is %g\n", i, A[i], A[i] * A[i]);
    }
    return EXIT_SUCCESS;
}
