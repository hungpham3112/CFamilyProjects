#include "binary.h"
#include <stdlib.h>
#include <stddef.h>

int convert(const char *input) {
    int accumulate = 0;
    for (size_t i = 0; input[i] != '\0'; ++i) {
        if (input[i] != '0' && input[i] != '1') {
            return INVALID;
        }
        int bit = input[i] == '1' ? 1 : 0;
        accumulate = (accumulate << 1) | bit;
    }
    return accumulate;
}