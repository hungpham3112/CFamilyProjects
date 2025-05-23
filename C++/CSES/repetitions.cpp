#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  long int num = 1;
  std::string arr;
  std::cin >> arr;
  for (long int i = 1; i < sizeof(arr) / sizeof(char); i++) {
    if (arr[i] == arr[i - 1]) {
      num += 1;
    } else {
      num = 1;
    }
  }
  std::cout << num << std::endl;
  return 0;
}
