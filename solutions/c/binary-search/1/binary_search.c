#include "binary_search.h"

const int *binary_search(int value, const int *arr, size_t length) {
    if (length == 0) {
        return NULL;
    }
        size_t middle = (size_t)(length / 2);

    if (arr[middle] == value) {
        return arr + middle;
    }
    if (arr[middle] < value) {
        return binary_search(value, arr + middle + 1, length - middle - 1);
    }
    return binary_search(value, arr, middle);
}
