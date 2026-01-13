#include "pangram.h"
#include <string.h>
#include <ctype.h>

bool is_pangram(const char *sentence) {
    if (sentence == NULL) return false;
    if (strlen(sentence) < 26) return false;
    char tmp[26] = {'\0'};

    for (size_t i = 0; sentence[i] != '\0'; ++i) {
        int idx = tolower(sentence[i]) - 'a';
        if (idx >= 0 && idx < 26)
            tmp[idx] = '0';
    }
    
    for (size_t i = 0; i < 26; ++i) {
        if (tmp[i] != '0') return false;
    }
    return true;
    
}
