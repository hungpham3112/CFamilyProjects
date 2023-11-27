#include <iostream>

int main(int argc, char *argv[]) {
  long int num;
  std::cin >> num;
  if (num == 1) {
    std::cout << num << " ";
  } else if (num == 2 || num == 3) {
    std::cout << "NO SOLUTION" << std::endl;
  } else if (num == 4) {
    std::cout << 2 << " " << 4 << " " << 1 << " " << 3 << std::endl;
  } else {
    std::cout << num << " ";
    for (long int i = 1; i < num; i++) {
      if (i % 2 == 0) {
        std::cout << i << " ";
      }
    }
    for (long int i = 1; i < num; i++) {
      if (i % 2 != 0) {
        std::cout << i << " ";
      }
    }
  }

  return 0;
}
