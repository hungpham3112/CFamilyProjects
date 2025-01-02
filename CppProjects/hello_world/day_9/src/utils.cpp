#ifndef UTILS_HEADER_H
#include "../include/utils.h"
#endif
#include <tuple>
#include <chrono>
#include <iostream>

void swap_by_reference(int &x, int &y) {
    int temp;
    temp = x;
    x = y;
    y = temp;
}

void swap_by_pointer(int *x, int *y) {
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

std::pair<int, int> swap_by_value(int x, int y) {
    int temp;
    temp = x;
    x = y;
    y = temp;
    return std::pair<int, int>(x, y);
}

void naive_benchmark() {
    using namespace std::chrono;
    int x{4}, y{5};
    // warm-up running, remember to run this before benchmarking
    // std::cout << "x before: " << x 
    //           << ", y before: " << y << std::endl;
    swap_by_pointer(&x, &y);
    swap_by_reference(x, y);
    std::tie(x, y) = swap_by_value(x, y);
    std::tie(x, y) = std::make_pair(y, x);

    auto start = high_resolution_clock::now();
    swap_by_reference(x, y);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    std::cout << "x after swap by reference: " << x 
              << ", y after swap by reference: " << y << std::endl;
    std::cout << "Time taken by swap by reference: " << duration.count() << " nanoseconds" << std::endl;

    start = high_resolution_clock::now();
    swap_by_pointer(&x, &y);
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start);
    std::cout << "x after swap by pointer: " << x 
              << ", y after swap by pointer: " << y << std::endl;
    std::cout << "Time taken by swap by pointer: " << duration.count() << " nanoseconds" << std::endl;
    
    // std::tie can be used to unpack the pair or tuple
    start = high_resolution_clock::now();
    std::tie(x, y) = swap_by_value(x, y);
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start);
    std::cout << "x after swap by value: " << x
              << ", y after swap by value: " << y << std::endl;
    std::cout << "Time taken by swap by value: " << duration.count() << " nanoseconds" << std::endl;

    // tips and trick with tie and make_pair
    start = high_resolution_clock::now();
    std::tie(x, y) = std::make_pair(y, x);
    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start);
    std::cout << "x after swap with tie and make_pair: " << x
              << ", y after swap with tie and make_pair: " << y << std::endl;
    std::cout << "Time taken by tie and make_pair: " << duration.count() << " nanoseconds" << std::endl;
}