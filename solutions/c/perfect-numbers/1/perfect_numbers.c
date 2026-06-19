#include "perfect_numbers.h"
#include <math.h>
#include <stdio.h>

kind classify_number(int num) {
    if (num == 1) {
        return DEFICIENT_NUMBER;
    }
    if (num <= 0) {
        return ERROR;
    }
    int sum = 0;
    for (int i = 1; i < num; ++i) {
        if (num % i == 0) {
            sum += i;
            printf("%d\n",i);
            
        }
    }
    if (sum == num) {
        return PERFECT_NUMBER;
    } else if (sum < num) {
        return DEFICIENT_NUMBER;
    } else {
        return ABUNDANT_NUMBER;
    }
}
