#include <iostream>

#define ll long long int

using namespace std;

int main() {
  ll num;
  cin >> num;
  cout << num << " ";
  while (num != 1) {
    if (num % 2 == 0) {
      num /= 2;
      cout << num << " ";
    } else {
      num = num * 3 + 1;
      cout << num << " ";
    }
  }
  return 0;
}
