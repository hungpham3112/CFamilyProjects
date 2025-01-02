// 1. write an array, raw array in c-style and std::array
// 2. show type of array and element inside that containers
// 3. fill the array from 1 to 1000
#include <iostream>
#include <array>
#include <iterator>
#include <numeric>
#include <typeinfo>

int main() {
    // C++-style
    std::array<int, 5> arr{0, 5, 8, 9, 4};
    /*std::iota(std::begin(arr), std::end(arr), 10);*/
    /*for (auto& i: arr) {*/
    /*    std::cout << i << std::endl;*/
    /*}*/

    int first_element = arr[0];
    first_element++;

    std::cout << "first_element: " << first_element << std::endl;
    std::cout << "arr[0] origin: " << arr[0] << std::endl;

    int &ref_first_element = arr[0];
    ref_first_element++;
    std::cout << "arr[0] ref: " << ref_first_element << std::endl;

    int *ptr_first_element = &arr[0];
    std::cout << std::endl;
    std::cout << "arr[0] ptr: " << ptr_first_element << std::endl;
    std::cout << "arr[0] ptr: " << *ptr_first_element << std::endl;
    std::cout << "arr[0] ptr: " << &arr[0] << std::endl;
    std::cout << std::endl;

    ptr_first_element++;
    std::cout << "arr[0] ptr ++ : " << ptr_first_element << std::endl;
    std::cout << "arr[0] ptr ++ : " << *ptr_first_element << std::endl;
    std::cout << "arr[1] ptr: " << &arr[1] << std::endl;
    std::cout << std::endl;

    ptr_first_element++;
    std::cout << "arr[0] ptr ++ * 2: " << ptr_first_element << std::endl;
    std::cout << "arr[0] ptr ++ * 2: " << *ptr_first_element << std::endl;
    std::cout << "arr[2] ptr: " << &arr[2] << std::endl;

    int test = 1;
    int *ptr_test = &test;
    std::cout << "test: " << *ptr_test << std::endl;
    std::cout << "test ptr: " << ptr_test << std::endl;
    test++;
    std::cout << "test: " << *ptr_test << std::endl;
    std::cout << "test ptr: " << ptr_test << std::endl;
    return 0;

}
