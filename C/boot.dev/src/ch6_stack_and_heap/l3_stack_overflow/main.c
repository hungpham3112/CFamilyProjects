#include <stdio.h>

int main()
{
    const int pool_size = 1024 * 1024 * 10;
    char snek_pool[pool_size];
    snek_pool[0] = 's';
    snek_pool[1] = 'n';
    snek_pool[2] = 'e';
    snek_pool[3] = 'k';
    snek_pool[4] = '\0';

    printf("Size of pool: %d\n", pool_size);
    printf("Initial string: %s\n", snek_pool);
    return 0;
}
