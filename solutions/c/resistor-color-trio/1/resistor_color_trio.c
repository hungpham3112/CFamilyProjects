#include "resistor_color_trio.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>


resistor_value_t color_code(resistor_band_t band[]) {
    if (band == NULL) return (resistor_value_t) {
        .value = 0,
        .unit = OHMS,
    };
    resistor_value_t result;
    result.value = (double)(band[0] * 10 + band[1] ) * pow(10, band[2]);
    printf("%ld\n", result.value);
    printf("%d\n", band[0] * 10 + band[1]);
    printf("%f\n", pow(10, band[2]));
    result.unit = OHMS;
    if (result.value == 0) return (resistor_value_t) {
        .value = 0,
        .unit = OHMS,
    };
    if (result.value % 1000000000 == 0) {
        result.unit = GIGAOHMS;
        result.value /= 1000000000;
    }

    if (result.value % 1000000 == 0) {
        result.unit = MEGAOHMS;
        result.value /= 1000000;
    }

    if (result.value % 1000 == 0) {
        result.unit = KILOOHMS;
        result.value /= 1000;
    }
    
    return result;
}
