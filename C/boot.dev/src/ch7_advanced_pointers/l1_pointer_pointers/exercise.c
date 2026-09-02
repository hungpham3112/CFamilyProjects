#include "exercise.h"
#include "stdlib.h"

void allocate_int(int **pointer_pointer, int value)
{
    *pointer_pointer = malloc(sizeof(int));
    **pointer_pointer = value;
}
