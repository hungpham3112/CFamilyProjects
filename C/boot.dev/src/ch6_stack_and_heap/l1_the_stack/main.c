#include "exercise.h"
#include <stdio.h>

int main()
{
    printMessageOne();
    return 0;
}

// __attribute__((noinline)) helps the compiler behave; don't worry about it
__attribute__((noinline)) void printMessageOne(void)
{
    const char *message = "Dark mode?\n";
    printStackPointerDiff();
    printf("%s\n", message);
    printMessageTwo();
}

__attribute__((noinline)) void printMessageTwo(void)
{
    const char *message = "More like...\n";
    printStackPointerDiff();
    printf("%s\n", message);
    printMessageThree();
}

__attribute__((noinline)) void printMessageThree(void)
{
    const char *message = "dark roast.\n";
    printStackPointerDiff();
    printf("%s\n", message);
}

// don't touch below this line

void printStackPointerDiff(void)
{
    static void *last_sp = NULL;
    void *current_sp;
    current_sp = __builtin_frame_address(0);
    long diff;
    if (last_sp == NULL)
    {
        last_sp = current_sp;
        diff = 0;
    }
    else
    {
        diff = (char *)last_sp - (char *)current_sp;
    }
    printf("---------------------------------\n");
    printf("Stack pointer offset: %ld bytes\n", diff);
    printf("---------------------------------\n");
}
