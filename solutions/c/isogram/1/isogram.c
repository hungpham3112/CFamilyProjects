#include "isogram.h"
#include <stddef.h>
#include <ctype.h>

bool is_isogram(const char phrase[]) {
    if (phrase == NULL) return false;

    int tmp[26] = {0};
    for (size_t i = 0; phrase[i] != '\0'; ++i) {
        char c = tolower(phrase[i]);
        if (c >= 'a' && c <= 'z') {
            tmp[c - 'a'] += 1;
        }
    }
    for (int i = 0; i < 26; ++i) {
        if (tmp[i] > 1){
            return false;
        }
    }
    return true;
}
