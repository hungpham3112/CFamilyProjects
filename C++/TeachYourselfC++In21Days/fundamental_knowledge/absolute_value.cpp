#include <iostream>

int absolute_value(int num) {
    if (num < 0) {
        return -num;
    } else {
        return num;
    }
}

int main() {
    int my_num, second_num;
    std::cout << "Type your number: "; 
    std::cin >> my_num >>  second_num;
    std::cout << "Your number is: " << my_num << " and absolute_value is: "<< absolute_value(my_num) << std::endl;
    std::cout << "number 1: " << my_num << ", number 2: "<< second_num << "their sum: "<< my_num + second_num << std::endl;
    return 0;
}
