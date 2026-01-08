#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    size_t max = SIZE_MAX;
    printf("%zu", max);
    return 0;
}
