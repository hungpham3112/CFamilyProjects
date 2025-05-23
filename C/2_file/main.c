#include <fcntl.h>  // file control
#include <stdio.h>  // handling I/O
#include <string.h> // strlen()
#include <unistd.h> // API for POSIX compliant

int main(int argc, char *argv[]) {
  const char *meminfo = "/proc/meminfo";
  const char *new_file = "new_file.txt";
  int fd = open(new_file, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    perror("Can't create a file");
    return -1;
  }

  const char *content = "hello\n";

  ssize_t byte_written = write(fd, content, strlen(content));

  if (byte_written == -1) {
    perror("Error writing file");
    close(fd);
    return -1;
  }

  close(fd);
  return 0;
}
