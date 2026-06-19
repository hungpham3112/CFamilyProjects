#include "high_scores.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static int compare_ints(const void* a, const void* b)
{
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
 
    if (arg1 < arg2) return 1;
    if (arg1 > arg2) return -1;
    return 0;
}

int32_t latest(const int32_t *scores, size_t scores_len) {
    return scores[scores_len - 1];
}

int32_t personal_best(const int32_t *scores, size_t scores_len) {
    int32_t best_score = 0;
    for (size_t i = 0; i < scores_len; ++i) {
        if (*(scores + i) > best_score) {
            best_score = *(scores + i);
        }
    }
    return best_score;
}

size_t personal_top_three(const int32_t *scores, size_t scores_len,
                          int32_t *output) {
    int32_t tmp[scores_len];
    memcpy(tmp, scores, sizeof(int32_t) * scores_len);
    qsort(tmp, scores_len, sizeof(int32_t), compare_ints);
    if (scores_len > 3) {
        scores_len = 3;
    }
    for (size_t i = 0; i < scores_len; ++i) {
        output[i] = tmp[i];
        printf("%d\n", tmp[i]);
    }
    
    return scores_len; 
}
