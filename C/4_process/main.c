#include <stdio.h>     // printf
#include <stdlib.h>    // exit
#include <sys/types.h> // pid_t
#include <sys/wait.h>  // wait
#include <unistd.h>    // fork, getpid

int main() {
  /*printf("Original process: PID = %d\n", getpid());*/
  /*sleep(10);*/

  printf("1");
  pid_t result;
  result = fork(); // create new process
  printf("2\n");

  /*if (result < 0) {*/
  /*  perror("fork failed");*/
  /*  exit(1);*/
  /*} else if (result == 0) {*/
  /*  // This block is executed by child process*/
  /*  printf("Child process: PID = %d, PPID = %d\n", getpid(), getppid());*/
  /*  exit(0);*/
  /*} else {*/
  /*  // This block is executed by parent process*/
  /*  printf("Parent process: PID = %d, Child PID = %d\n", getpid(), result);*/
  /*  wait(NULL); // wait for child to finish*/
  /*}*/

  return 0;
}
