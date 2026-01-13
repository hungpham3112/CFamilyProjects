#include "eliuds_eggs.h"
#include <stddef.h>

size_t egg_count(int num) {
    size_t count = 0;
    while (num != 0) {
        if (num % 2 == 0) {
            num /= 2;
        } else {
            num = (num - 1) / 2;
            ++count;
        }
    }
    return count;
}