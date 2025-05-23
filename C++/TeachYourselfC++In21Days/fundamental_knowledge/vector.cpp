#include "include/utils.h"
#include <array>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <vector>

std::vector<int> myVector;

int main(int argc, char *argv[])
{
    std::vector<int> vec;
    std::array<int, 4> arr{1, 2, 3, 4};
    vec.push_back(120);
    vec.push_back(139);
    vec.push_back(391);
    /* std::tuple<int> tup{1}; */
    /* std::cout << "type element of vec: " << typeid(vec[1]).name() << std::endl; */
    /* std::cout << "type of vec: " << typeid(vec).name() << std::endl; */
    /* std::cout << "type of tuple: " << typeid(tup).name() << std::endl; */
    /* std::cout << "type element of myVec: " << typeid(myVector).name() << std::endl; */

    auto begin_vec = std::begin(vec);
    auto begin_arr = std::begin(arr);
    auto begin_vec_type = typeid(begin_vec).name();
    auto begin_arr_type = typeid(begin_arr).name();

    std::cout << "type of begin vec: " << begin_vec_type << std::endl;
    std::cout << "type of begin arr: " << begin_arr_type << std::endl;

    std::cout << "demangled begin_vec_type: " << utils::demangle(begin_vec_type) << std::endl;
    std::cout << "demangled begin_arr_type: " << utils::demangle(begin_arr_type) << std::endl;

    if (vec[1] > 0)
    {
        std::cout << "hello" << std::endl;
    }
    else
    {
        std::cout << "lx" << std::endl;
    }

    /* const char *char_begin_vec_type = begin_vec_type; */
    /* std::cout << char_begin_vec_type << std::endl; */
    for (auto &i : vec)
    {
        std::cout << "&i: " << &i << std::endl;
    }

    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << &vec[i] << std::endl;
    }

    return 0;
}
