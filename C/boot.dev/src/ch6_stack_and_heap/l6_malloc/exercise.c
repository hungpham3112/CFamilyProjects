#include "exercise.h"
#include <stdio.h>
#include <stdlib.h>

int *allocate_scalar_array(int size, int multiplier)
{
    int *arr = malloc(size * sizeof(int));
    if (arr == NULL)
    {
        return NULL;
    }
    for (int i = 0; i < size; ++i)
    {
        arr[i] = i * multiplier;
    }

    return arr;
}
