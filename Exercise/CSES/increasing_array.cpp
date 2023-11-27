#include <iostream>

int main(int argc, char *argv[]) {
  long int num, max, curr, sum{};
  std::cin >> num;
  std::cin >> curr;
  max = curr;
  for (int i = 1; i < num; i++) {
    std::cin >> curr;
    if (curr < max) {
      sum = sum + max - curr;
    } else {
      max = curr;
    }
  }
  std::cout << sum << std::endl;
  return 0;
}
