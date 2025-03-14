#include <cstdint>
#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
  // DANGEOUS !!!
  // int64_t b = std::numeric_limits<int64_t>::max();
  // int32_t c(b);
  // if (c == b) {
  //   printf("Non\n");
  // }
  // SAFE
  int32_t x = 4;
  int64_t y{x};
  if (y == x) {
    printf("y = x\n");
  }
  int8_t myUnsignedByte = -128;
  std::cout << static_cast<int>(myUnsignedByte) << std::endl;

  return 0;
}
