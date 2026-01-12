#include "space_age.h"
#include <math.h>

float age(planet_t planet, int64_t seconds) {
    float num;
    switch (planet) {
        case MERCURY: num = 0.2408467f; break;
        case VENUS: num = 0.61519726f; break;
        case EARTH: num = 1.0f; break;
        case MARS: num = 1.8808158f; break;
        case JUPITER: num = 11.862615f; break;
        case SATURN: num = 29.447498f; break;
        case URANUS: num = 84.016846f; break;
        case NEPTUNE: num = 164.79132f; break;
        default: num = 0; break;
    }
    if (num == 0) {
        return -1;
    }
    num = roundf(seconds / 31557600 / num * 100) / 100;
    return num;
}
