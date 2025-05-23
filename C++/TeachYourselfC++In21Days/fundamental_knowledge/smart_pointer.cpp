#include <iostream>
#include <memory>


int main(int argc, char *argv[])
{
    int a = 42;
    std::unique_ptr<int> int_ptr = std::make_unique<int>(a);
    std::cout << *int_ptr << std::endl;
    auto int_ptr_ptr = &int_ptr;
    std::cout << **int_ptr_ptr << std::endl;
    return 0;
}
