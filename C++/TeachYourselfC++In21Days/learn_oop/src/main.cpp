#include <iostream>
#include "cat.hpp"


int main() {
    Cat tom(12, Gender::Male);
    std::cout << tom.name() << std::endl;
    return 0;
}