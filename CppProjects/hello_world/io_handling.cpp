#include <iostream>
#include <string>
#include <print>

int main() {
    std::string name;
    std::cout << "What is your name\n";
    std::cin >> name;
    std::cout << "Hello " << name << "\n"; 
    std::string nhu{"nhu"};
    std::string lon{"lon"};
    std::print("hello world {0} {1}", nhu, lon); 
    return 0;
}


