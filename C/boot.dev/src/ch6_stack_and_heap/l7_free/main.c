#include "exercise.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

static void limit_memory(void)
{
    struct rlimit limit;

    // Limit virtual address space to 4 MiB.
    // 500 KiB is usually too small for a dynamically linked C program.
    limit.rlim_cur = 4 * 1024 * 1024;
    limit.rlim_max = 4 * 1024 * 1024;

    if (setrlimit(RLIMIT_AS, &limit) != 0)
    {
        perror("setrlimit");
        exit(1);
    }
}
int main()
{
    // limit_memory();
    const int num_lists = 500;
    for (int i = 0; i < num_lists; i++)
    {
        int *lst = allocate_scalar_list(50000, 2);
        if (lst == NULL)
        {
            printf("Failed to allocate list\n");
            return 1;
        }
        printf("Allocated list %d\n", i);
        free(lst);
        printf("%d", *lst);
    }
    return 0;
}
