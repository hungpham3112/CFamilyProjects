#ifndef UTILS_HEADER_h
#define UTILS_HEADER_h
#include <utility>

void swap_by_reference(int &x, int &y);
void swap_by_pointer(int *x, int *y);
std::pair<int, int> swap_by_value(int x, int y);
void naive_benchmark();

#endif