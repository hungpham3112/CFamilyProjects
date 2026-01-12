#include "hamming.h"
#include <string.h>
size_t compute(const char *lhs, const char *rhs) {
    if (strlen(lhs) != strlen(rhs)) {
        return -1;
    }
    size_t count = 0;
    for (size_t i = 0; lhs[i] != '\0'; ++i) {
        if (lhs[i] != rhs[i]) {
            ++count;
        }
    }
    return count;
}
