#include <cstdint>
#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
  int64_t b = std::numeric_limits<int64_t>::max();
  int32_t c(b);
  if (c == b) {
    printf("Non\n");
  }
  return 0;
}
