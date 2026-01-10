#include "resistor_color.h"

uint16_t color_code(resistor_band_t color) {
    return (uint16_t) color;
}

const resistor_band_t * colors() {
    static const resistor_band_t colors[] = {    BLACK,
    BROWN,
    RED,
    ORANGE,
    YELLOW,
    GREEN,
    BLUE,
    VIOLET,
    GREY,
    WHITE
};
    return colors;
}
