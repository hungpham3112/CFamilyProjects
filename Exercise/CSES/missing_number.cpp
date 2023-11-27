#include <iostream>

using namespace std;

int main() {
  long int num, sum = 0, var[200000];
  cin >> num;

  for (long int i = 0; i < num - 1; i++) {
    cin >> var[i];
    sum += var[i];
  }

  cout << (num * (num + 1) / 2) - sum;
  return 0;
}
