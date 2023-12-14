#include <iostream>
#include <string>

using std::cout;

int main() {
	std::string name;
	cout << "What is your name: ";
	std::cin >> name;
	cout << "Hi " << name << std::endl;
	return 0;
}

