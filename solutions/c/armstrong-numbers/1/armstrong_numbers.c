#include "armstrong_numbers.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>



bool is_armstrong_number(int candidate) {
    uint32_t tmp = candidate;
    uint32_t tmp1 = candidate;
    
    uint8_t num_digit = 0;
    while (candidate != 0) {
        candidate /= 10;
        ++num_digit;
    }
    printf("%d", candidate);
    uint32_t result = 0;
    while (tmp != 0) {
        result += pow(tmp % 10, num_digit);
        tmp /= 10;
    }
    if (result == tmp1) {
        return true;
    }
    return false;
}
