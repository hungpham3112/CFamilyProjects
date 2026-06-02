#include "darts.h"
int score(coordinate_t p) {
    double length = sqrt(pow(p.x, 2) + pow(p.y, 2));
    if (length >= 0 && length <= 1) {
        return 10;
    } else if (length > 1 && length <= 5) {
        return 5;
    } else if (length > 5 && length <= 10) {
        return 1;
    }
    return 0;
}