#include <iostream>

// Pass by value - default
int add2_by_value(int num) {
    return num + 2;
}

// Pass by reference (use when pass the large data, not for primitive type) 
void add2(int& num) {
    num += 2;
}

// Pass by const reference
int add2_by_reference(const int& num) {
    int new_num = num;
    return new_num += 2;
}

// Pass by pointer
void add2(int *num) {
    *num += 2;
}

int main() {
    int i = 5;
    i = add2_by_value(i);
    std::cout << "a (pass by value): " << i << std::endl;
    add2(i);
    std::cout << "i (pass by reference): " << i << std::endl;
    add2(&i);
    std::cout << "i (pass by pointer): " << i  << std::endl;
    int new_num = add2_by_reference(i);
    std::cout << "i (pass by pointer): " << i  << std::endl;
    std::cout << "i (pass by pointer): " << new_num << std::endl;

}
