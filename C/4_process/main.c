#include <stdio.h>     // printf
#include <stdlib.h>    // exit
#include <sys/types.h> // pid_t
#include <sys/wait.h>  // wait
#include <unistd.h>    // fork, getpid

int main()
{

    printf("Current process ID: %d\n", getpid());
    pid_t result = fork();

    if (result < 0)
    {
        perror("can't spawn process");
    }
    else if (result == 0)
    {
        pid_t pid = wait(NULL);
        printf("Child process ID: %d, parent process: %d\n", getpid(), getppid());
    }
    else
    {
        printf("Parent process ID: %d, child process: %d\n", getpid(), result);
        // printf("Parent process ID: %d, child process: %d\n", getpid(), pid);
    }

    return 0;
}
